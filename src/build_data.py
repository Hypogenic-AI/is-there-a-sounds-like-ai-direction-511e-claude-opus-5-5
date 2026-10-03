"""Build all text sets used for read-out, direction fitting and steering prompts.
Output: results/texts.jsonl with fields id, text, set, label (1=AI,0=human,-1=n/a), group, meta, split.
Content/length control: each matched pair (same prefix or same question) is truncated to the
same number of Qwen tokens (<= MAX_TOK)."""
import json, random
import pandas as pd
from transformers import AutoTokenizer
from common import *

set_seed(0)
tok = AutoTokenizer.from_pretrained(INSTRUCT)
ntok = lambda t: len(tok(t, add_special_tokens=False).input_ids)
rows = []
MIN = 48


def add_pair(pid, texts: dict, set_, group, split, meta=None):
    """texts: {source_name: text}; truncate all to common token length."""
    n = min(MAX_TOK, *[ntok(t) for t in texts.values()])
    if n < MIN:
        return False
    for src, t in texts.items():
        rows.append(dict(id=f"{pid}|{src}", text=truncate_tokens(tok, t, n), set=set_, label=0 if src.startswith("human") else 1,
                         group=group, src=src, split=split, pair=pid, meta=meta or {}))
    return True

# ---------------- HAP-E: same human prefix -> human vs model continuations
H = str(DATA)+"/hape/text_data/hape-text_"
srcs = {"human": "human-chunk-2", "gpt-4o": "gpt-4o-2024-08-06", "llama70b-inst": "llama-3-70B-Instruct",
        "llama8b-inst": "llama-3-8B-Instruct", "llama8b-base": "llama-3-8B", "llama70b-base": "llama-3-70B"}
dfs = {}
for k, f in srcs.items():
    d = pd.read_parquet(H + f + ".parquet"); d["key"] = d.doc_id.str.split("@").str[0]; dfs[k] = d.set_index("key").text
keys = sorted(set.intersection(*[set(d.index) for d in dfs.values()]))
random.shuffle(keys)
# stratify by genre: 150 train + 60 test per genre (6 genres)
by_g = {}
for k in keys: by_g.setdefault(k.split("_")[0], []).append(k)
for g, ks in by_g.items():
    ntr = nte = 0
    for k in ks:
        if ntr >= 150 and nte >= 60: break
        split = "train" if ntr < 150 else "test"
        texts = {s: dfs[s][k] for s in srcs if isinstance(dfs[s][k], str) and len(dfs[s][k]) > 50}
        if len(texts) < len(srcs): continue
        if add_pair(f"hape:{k}", texts, "hape", g, split):
            ntr += split == "train"; nte += split == "test"
print("hape", sum(r["set"] == "hape" for r in rows))

# ---------------- HC3: same question -> human vs ChatGPT
hc3 = [json.loads(l) for l in open(str(DATA)+"/hc3/all.jsonl")]
random.shuffle(hc3)
quota = {"reddit_eli5": (250, 80), "finance": (150, 60), "medicine": (100, 40), "open_qa": (100, 40), "wiki_csai": (100, 40)}
cnt = {s: [0, 0] for s in quota}
steer_qs, pilot_qs, used = [], [], set()
for r in hc3:
    s = r["source"]
    if not r["human_answers"] or not r["chatgpt_answers"]: continue
    h = normalize_hc3(r["human_answers"][0]); a = r["chatgpt_answers"][0].strip()
    q = r["question"].strip()
    if q in used: continue
    tr, te = quota[s]
    if cnt[s][0] < tr: split = "train"
    elif cnt[s][1] < te: split = "test"
    else: continue
    if add_pair(f"hc3:{s}:{len(used)}", {"human": h, "chatgpt": a}, "hc3", s, split, {"q": q}):
        cnt[s][0 if split == "train" else 1] += 1; used.add(q)
print("hc3", cnt)
# held-out steering prompts: 30 per source, + 6 per source for the pilot; keep human answer as reference
nsteer = {s: 0 for s in quota}; npil = {s: 0 for s in quota}
for r in hc3:
    s, q = r["source"], r["question"].strip()
    if q in used or not r["human_answers"] or len(q) < 15 or len(q) > 400: continue
    h = normalize_hc3(r["human_answers"][0])
    if len(h.split()) < 40: continue
    if npil[s] < 6:
        pilot_qs.append(dict(q=q, source=s, human=h, chatgpt=r["chatgpt_answers"][0] if r["chatgpt_answers"] else "")); npil[s] += 1; used.add(q)
    elif nsteer[s] < 30:
        steer_qs.append(dict(q=q, source=s, human=h, chatgpt=r["chatgpt_answers"][0] if r["chatgpt_answers"] else "")); nsteer[s] += 1; used.add(q)
jdump(steer_qs, RES / "steer_prompts.json"); jdump(pilot_qs, RES / "pilot_prompts.json")
print("steer prompts", len(steer_qs), "pilot", len(pilot_qs))

# ---------------- RAID subset (transfer test, unpaired; truncated to MAX_TOK)
rd = pd.read_parquet(str(DATA)+"/raid_subset/raid_train_subset.parquet")
rd = rd[(rd.decoding.isna() | (rd.decoding == "sampling")) & (rd.repetition_penalty.isna() | (rd.repetition_penalty == "no")) | (rd.model == "human")]
for (m, a), d in rd.groupby(["model", "attack"]):
    if a == "paraphrase" and m != "human": continue
    d = d.sample(min(len(d), 300 if m == "human" else 120), random_state=0)
    for _, r in d.iterrows():
        t = truncate_tokens(tok, r.generation, MAX_TOK)
        if ntok(t) < MIN: continue
        rows.append(dict(id=f"raid:{r.id}", text=t, set="raid", label=0 if m == "human" else 1, group=r.domain,
                         src=m + ("-para" if a == "paraphrase" else ""), split="test", pair=None, meta={}))
print("raid", sum(r["set"] == "raid" for r in rows))

# ---------------- MAGE test + OOD (label 1 = human in MAGE!)
mg = pd.read_csv(str(DATA)+"/mage/test.csv").sample(2000, random_state=0)
for f, nm in [(str(DATA)+"/mage/test_ood_set_gpt.csv", "ood_gpt4"), (str(DATA)+"/mage/test_ood_set_gpt_para.csv", "ood_gpt4para")]:
    o = pd.read_csv(f).sample(600, random_state=0); o["src"] = nm + "_" + o.src.astype(str); mg = pd.concat([mg, o])
for i, r in enumerate(mg.itertuples()):
    if not isinstance(r.text, str): continue
    t = truncate_tokens(tok, r.text, MAX_TOK)
    if ntok(t) < MIN: continue
    sub = "ood" if str(r.src).startswith("ood") else "in"
    rows.append(dict(id=f"mage:{i}", text=t, set="mage_" + sub, label=1 - int(r.label), group=str(r.src).split("_")[0],
                     src=str(r.src), split="test", pair=None, meta={}))
print("mage", sum(r["set"].startswith("mage") for r in rows))

with open(RES / "texts.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
df = pd.DataFrame(rows)
print(df.groupby(["set", "split", "label"]).size())
