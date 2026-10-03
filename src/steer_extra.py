"""E3b: mean-clamp interventions (a gentler alternative to zero-ablation).
At the hooked block(s) the component of the residual along unit(d) is *replaced* by a target value
(the mean projection of human texts, or of AI texts, measured on the read-out corpus with the same
mean-pooling protocol) instead of being zeroed. Zero-ablation moves activations far off-distribution
because the mean projection is not zero; clamping to the human mean is the in-distribution version of
"remove the AI-ness". Controls: the same clamp on random directions (clamped to their own human mean).
Positions with norm > 3x the typical residual norm (attention-sink tokens) are left untouched.
usage: python steer_extra.py"""
import json, time
import pandas as pd
from common import *
MAXNEW = 160  # same decoding as steer.py


@torch.no_grad()
def resid_norm(model, tok, prompts, layer):
    """Mean residual norm at the output of block `layer` on chat-formatted prompts (sink token excluded); as in steer.py."""
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    tok.padding_side = "right"
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1].float()
    m = enc.attention_mask.clone(); m[:, 0] = 0
    tok.padding_side = "left"
    return hs.norm(dim=-1)[m.bool()].mean().item()

set_seed(0)
cfg = json.load(open(RES / "steer_config.json")); L = cfg["layer"]
df = pd.read_parquet(RES / "texts_scored.parquet")
A = np.load(ACTS / "instruct.npy", mmap_mode="r")
DI = np.load(RES / "dirs_instruct.npy")[:, 0]  # [28 blocks, d]; index i = output of block i
tr = (df.split == "train") & df.set.isin(["hape", "hc3"])
hum = np.where(tr & (df.label == 0))[0]
ai = np.where(tr & df.src.isin(["gpt-4o", "llama70b-inst", "chatgpt"]))[0]


def proj_means(block, v):
    X_h = A[hum, block + 1].astype(np.float32); X_a = A[ai, block + 1].astype(np.float32)
    v = v / np.linalg.norm(v)
    return float((X_h @ v).mean()), float((X_a @ v).mean()), float((X_h @ v).std())


class Clamp:
    """Replace the projection on unit(v_b) by target t_b at each block b in `spec` = {b: (v_b, t_b)}."""

    def __init__(self, model, spec, rn):
        self.model, self.rn, self.h = model, rn, []
        self.spec = {b: (torch.tensor(v / np.linalg.norm(v), dtype=torch.bfloat16, device=model.device), t) for b, (v, t) in spec.items()}

    def _f(self, v, t):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            p = (h @ v).unsqueeze(-1)
            keep = (h.float().norm(dim=-1, keepdim=True) > 3 * self.rn)  # attention sink: leave alone
            h = torch.where(keep, h, h + (t - p) * v)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return f

    def __enter__(self):
        Ls = layers_of(self.model)
        for b, (v, t) in self.spec.items():
            self.h.append(Ls[b].register_forward_hook(self._f(v, t)))

    def __exit__(self, *a):
        for x in self.h: x.remove()


if __name__ == "__main__":
    model, tok = load_model(INSTRUCT)
    P = json.load(open(RES / "steer_prompts.json")); prompts = [p["q"] for p in P]
    rn = resid_norm(model, tok, prompts, L)
    rng = np.random.default_rng(11)
    meta = {}
    specs = {}
    mh, ma, sd = proj_means(L, DI[L]); meta[f"L{L}_ai"] = dict(human=mh, ai=ma, sd_human=sd)
    specs["clamp_L20_human"] = {L: (DI[L], mh)}
    specs["clamp_L20_ai"] = {L: (DI[L], ma)}
    # push one further: human mean minus one (human) s.d. -- still inside the human distribution's range
    specs["clamp_L20_human-1sd"] = {L: (DI[L], mh - sd)}
    mid = {}
    for b in range(10, DI.shape[0]):
        h_, a_, _ = proj_means(b, DI[b]); mid[b] = (DI[b], h_); meta[f"L{b}_ai"] = dict(human=h_, ai=a_)
    specs["clamp_mid_human"] = mid
    for k in range(3):  # random-direction controls, clamped to their own human-mean (single layer) / shifted by the same number of s.d.
        rv = rng.standard_normal(DI.shape[1]); h_, a_, s_ = proj_means(L, rv)
        specs[f"clamp_L20_random{k}_human-1sd"] = {L: (rv, h_ - s_)}
        meta[f"random{k}"] = dict(human=h_, ai=a_, sd_human=s_)
    jdump(dict(layer=L, resid_norm=rn, proj=meta), RES / "steer_extra_meta.json")
    out_path = RES / "gen_extra.json"
    res = json.load(open(out_path)) if out_path.exists() else []
    done = {r["cond"] for r in res}
    if "none" not in done:
        gens = generate(model, tok, prompts, max_new_tokens=MAXNEW, seed=1234)
        res += [dict(cond="none", method="none", alpha=0.0, layer=None, pid=i, text=g) for i, g in enumerate(gens)]
    for c, spec in specs.items():
        if c in done: continue
        t = time.time()
        with Clamp(model, spec, rn):
            gens = generate(model, tok, prompts, max_new_tokens=MAXNEW, seed=1234)
        meth = "clamp_random" if "random" in c else c
        res += [dict(cond=c, method=meth, alpha=0.0, layer=L, pid=i, text=g) for i, g in enumerate(gens)]
        jdump(res, out_path)
        print(c, f"{time.time()-t:.0f}s |", gens[0][:150].replace("\n", " "), flush=True)
    print(json.dumps(meta, indent=1)[:2000])
