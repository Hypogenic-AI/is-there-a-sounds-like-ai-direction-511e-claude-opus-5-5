"""Shared utilities: paths, seeding, model loading, activation extraction, steering hooks."""
import os, re, json, random, contextlib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
RES = ROOT / "results"
FIG = ROOT / "figures"
ACTS = RES / "acts"
INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"
BASE = "Qwen/Qwen2.5-7B"
MAX_TOK = 128  # length cap for read-out texts (pairs are truncated to equal length)


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def normalize_hc3(t: str) -> str:
    """Undo HC3's pre-tokenised human answers (spaces before punctuation, split contractions)."""
    t = re.sub(r"\s+([,.!?;:%)\]])", r"\1", t)
    t = re.sub(r"([(\[$])\s+", r"\1", t)
    t = re.sub(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", t)
    t = re.sub(r'"\s+(.*?)\s+"', r'"\1"', t)
    t = re.sub(r"\s+-\s+", "-", t) if re.search(r"\w\s-\s\w", t) and t.count(" - ") > 2 else t
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def load_model(name, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map="cuda")
    model.eval()
    return model, tok


def layers_of(model):
    return model.model.layers


@torch.no_grad()
def mean_pooled_acts(model, tok, texts, batch_size=32, max_tok=MAX_TOK, skip_first=1):
    """Mean-pool the residual stream (output of each block; index 0 = embeddings) over text tokens.
    Returns float16 array [n, n_layers+1, d] and per-text mean token log-prob (fluency covariate)."""
    out, lps = [], []
    tok.padding_side = "right"
    for i in range(0, len(texts), batch_size):
        b = texts[i:i + batch_size]
        enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=max_tok).to(model.device)
        o = model(**enc, output_hidden_states=True)
        m = enc.attention_mask.clone()
        m[:, :skip_first] = 0  # first token acts as attention sink; exclude
        mf = m.unsqueeze(-1).to(torch.float32)
        hs = torch.stack(o.hidden_states, 1).float()  # [b, L+1, T, d]
        pooled = (hs * mf.unsqueeze(1)).sum(2) / mf.sum(1).unsqueeze(1)
        out.append(pooled.half().cpu().numpy())
        # mean log-prob of next tokens
        logits = o.logits[:, :-1].float()
        tgt = enc.input_ids[:, 1:]
        lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        am = enc.attention_mask[:, 1:].float()
        lps.append(((lp * am).sum(1) / am.sum(1)).cpu().numpy())
        del o, hs
    tok.padding_side = "left"
    return np.concatenate(out), np.concatenate(lps)


def truncate_tokens(tok, text, n):
    ids = tok(text, add_special_tokens=False).input_ids[:n]
    return tok.decode(ids)


class Steerer:
    """Context manager adding alpha * unit(v) to the residual stream output of layer `layer`
    (all positions), or ablating unit(v) at every layer (ablate=True)."""

    def __init__(self, model, vec=None, layer=None, coef=0.0, ablate=False, ablate_vecs=None):
        """ablate_vecs: optional array [n_layers, d] -> per-layer directional ablation (layer i uses ablate_vecs[i])."""
        self.model, self.layer, self.coef, self.ablate = model, layer, coef, ablate
        self.v = None if vec is None else torch.tensor(vec / np.linalg.norm(vec), dtype=torch.bfloat16, device=model.device)
        self.av = None if ablate_vecs is None else [torch.tensor(v / np.linalg.norm(v), dtype=torch.bfloat16, device=model.device) for v in ablate_vecs]
        self.handles = []

    def _add(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + self.coef * self.v
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def _abl(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h - (h @ self.v).unsqueeze(-1) * self.v
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def _hook_abl(self, v):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h @ v).unsqueeze(-1) * v
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return f

    def __enter__(self):
        L = layers_of(self.model)
        if self.av is not None:
            for l, v in zip(L, self.av):
                if torch.isnan(v).any(): continue  # zero vector -> no ablation at this layer
                self.handles.append(l.register_forward_hook(self._hook_abl(v)))
            return self
        if self.v is None or (self.coef == 0 and not self.ablate):
            return self
        if self.ablate:
            for l in L:
                self.handles.append(l.register_forward_hook(self._abl))
        else:
            self.handles.append(L[self.layer].register_forward_hook(self._add))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def generate(model, tok, prompts, system=None, max_new_tokens=160, batch_size=50, seed=0,
             temperature=0.7, top_p=0.95, chat=True):
    """Batched sampling generation with a fixed seed. prompts: list[str]."""
    outs = []
    for i in range(0, len(prompts), batch_size):
        b = prompts[i:i + batch_size]
        if chat:
            msgs = [([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}] for p in b]
            texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        else:
            texts = b
        tok.padding_side = "left"
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        torch.manual_seed(seed + i)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                           top_p=top_p, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
