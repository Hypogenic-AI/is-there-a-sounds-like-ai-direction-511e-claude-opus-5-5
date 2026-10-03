"""Score generations: supervised detectors (desklib, fakespot), formality, stylometrics,
content similarity (all-mpnet-base-v2 cosine to the unsteered answer for the same prompt and to the question),
and Binoculars (Falcon-7B observer / Falcon-7B-instruct performer) + Falcon-7B log-perplexity (coherence).
usage: python score_gens.py gen_main.json [gen_pilot.json ...]
Also scores the HC3 human & ChatGPT reference answers for the steering prompts (cond='ref_human'/'ref_chatgpt')."""
import sys, json, re
import pandas as pd
import torch
from common import *
from text_metrics import Scorers, stylometrics


def clean(t):
    t = t.split("\nQuestion:")[0]  # base-model completions run on into new Q/A pairs
    return t.strip()


@torch.no_grad()
def binoculars(texts, bs=16, max_len=256):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained("tiiuae/falcon-7b"); tk.pad_token = tk.eos_token; tk.padding_side = "right"
    obs = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", dtype=torch.bfloat16, device_map="cuda").eval()
    perf = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", dtype=torch.bfloat16, device_map="cuda").eval()
    B, PPL = [], []
    for i in range(0, len(texts), bs):
        enc = tk(texts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=max_len, return_token_type_ids=False).to("cuda")
        lo, lp = obs(**enc).logits[:, :-1].float(), perf(**enc).logits[:, :-1].float()
        y, m = enc.input_ids[:, 1:], enc.attention_mask[:, 1:].float()
        ce_perf = torch.nn.functional.cross_entropy(lp.transpose(1, 2), y, reduction="none")
        ce_obs = torch.nn.functional.cross_entropy(lo.transpose(1, 2), y, reduction="none")
        xent = -(torch.softmax(lo, -1) * torch.log_softmax(lp, -1)).sum(-1)  # cross-perplexity (observer probs, performer logprobs)
        ppl = (ce_perf * m).sum(1) / m.sum(1); x = (xent * m).sum(1) / m.sum(1)
        B.append((ppl / x).cpu().numpy()); PPL.append(((ce_obs * m).sum(1) / m.sum(1)).cpu().numpy())
    del obs, perf; torch.cuda.empty_cache()
    return np.concatenate(B), np.concatenate(PPL)


files = sys.argv[1:]
rows = []
for f in files:
    G = json.load(open(RES / f))
    for r in G:
        r = dict(r); r["file"] = f; r["text"] = clean(r["text"]); rows.append(r)
P = json.load(open(RES / "steer_prompts.json"))
if any("main" in f or "base" in f for f in files):
    for i, p in enumerate(P):
        rows.append(dict(file="ref", cond="ref_human", method="ref_human", alpha=0.0, layer=None, pid=i, text=p["human"]))
        rows.append(dict(file="ref", cond="ref_chatgpt", method="ref_chatgpt", alpha=0.0, layer=None, pid=i, text=p["chatgpt"]))
df = pd.DataFrame(rows)
df["text"] = df.text.fillna("").map(lambda t: t if t.strip() else "(empty)")
# truncate references to ~ the same length as generations (160 tokens ~ 120 words) to avoid length artefacts in detectors
df.loc[df.file == "ref", "text"] = df.loc[df.file == "ref", "text"].map(lambda t: " ".join(t.split()[:120]))
S = Scorers()
for n in ["desklib", "fakespot", "formality"]:
    df[n] = S.score(n, df.text.tolist(), bs=64)
del S; torch.cuda.empty_cache()
df = pd.concat([df.reset_index(drop=True), pd.DataFrame([stylometrics(t) for t in df.text])], axis=1)
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
E = emb.encode(df.text.tolist(), batch_size=64, normalize_embeddings=True)
pid_prompts = {}
for f in df.file.unique():
    pf = "pilot_prompts.json" if "pilot" in f else "steer_prompts.json"
    pid_prompts[f] = json.load(open(RES / pf))
Q = {f: emb.encode([p["q"] for p in v], normalize_embeddings=True) for f, v in pid_prompts.items()}
df["sim_question"] = [float(E[i] @ Q[f][p]) for i, (f, p) in enumerate(zip(df.file, df.pid))]
# similarity to the unsteered generation for the same prompt (same file)
base_idx = {(f, p): i for i, (f, p, c) in enumerate(zip(df.file, df.pid, df.cond)) if c in ("none", "base_none")}
df["sim_unsteered"] = [float(E[i] @ E[base_idx[(f, p)]]) if (f, p) in base_idx else np.nan for i, (f, p) in enumerate(zip(df.file, df.pid))]
del emb; torch.cuda.empty_cache()
df["binoculars"], df["falcon_logppl"] = binoculars(df.text.tolist())
tag = "_".join(Path(f).stem for f in files)
df.to_parquet(RES / f"scores_{tag}.parquet")
print(df.groupby("cond")[["desklib", "fakespot", "binoculars", "falcon_logppl", "formality", "sim_unsteered", "n_words", "distinct2"]].mean().round(3).to_string())
