"""E1 (read-out, generality), E2 (disentanglement), E4 (base vs instruct) on mean-pooled activations.
All directions are diff-of-means on TRAIN texts; evaluation on TEST texts (disjoint docs/questions)."""
import json
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import *

set_seed(0)
df = pd.read_parquet(RES / "texts_scored.parquet")
lp = {m: np.load(ACTS / f"{m}_lp.npy") for m in ["instruct", "base"]}
df["lp_instruct"], df["lp_base"] = lp["instruct"], lp["base"]
A = {m: np.load(ACTS / f"{m}.npy", mmap_mode="r") for m in ["instruct", "base"]}
G = np.load(ACTS / "confound_gen.npz")
NL = A["instruct"].shape[1]
unit = lambda v: v / (np.linalg.norm(v) + 1e-8)

tr, te = df.split == "train", df.split == "test"
hape, hc3 = df.set == "hape", df.set == "hc3"
human = df.label == 0
AI_EXT = df.src.isin(["gpt-4o", "llama70b-inst", "chatgpt"])  # external instruct generators used for fitting
idx = lambda m: np.where(m.values)[0]


def mean_at(M, l, mask):
    return M[idx(mask), l, :].astype(np.float32).mean(0)


def strat_diff(M, l, mask, score, strata):
    """Within-stratum top-vs-bottom tercile diff-of-means, averaged over strata (controls genre/source)."""
    ds = []
    sub = df[mask]
    for g, d in sub.groupby(strata):
        if len(d) < 30: continue
        lo, hi = d[score].quantile([1/3, 2/3])
        a = M[d.index[d[score] >= hi].values, l, :].astype(np.float32).mean(0)
        b = M[d.index[d[score] <= lo].values, l, :].astype(np.float32).mean(0)
        ds.append(a - b)
    return np.mean(ds, 0)


def directions_at(model, l):
    M = A[model]
    d_hape = mean_at(M, l, tr & hape & AI_EXT) - mean_at(M, l, tr & hape & human)
    d_hc3 = mean_at(M, l, tr & hc3 & ~human) - mean_at(M, l, tr & hc3 & human)
    D = dict(ai=0.5 * (unit(d_hape) + unit(d_hc3)), ai_hape=d_hape, ai_hc3=d_hc3)
    hum_tr = tr & human & (hape | hc3)
    D["formality"] = strat_diff(M, l, hum_tr, "formality", "group")
    D["fluency"] = strat_diff(M, l, hum_tr, f"lp_{model}", "group")
    D["length_wordlen"] = strat_diff(M, l, hum_tr, "mean_word_len", "group")  # lexical sophistication
    D["domain_reddit"] = mean_at(M, l, tr & hc3 & human & (df.group == "reddit_eli5")) - mean_at(M, l, tr & hc3 & human & (df.group != "reddit_eli5"))
    D["posttrain"] = 0.5 * (unit(mean_at(M, l, tr & hape & (df.src == "llama8b-inst")) - mean_at(M, l, tr & hape & (df.src == "llama8b-base")))
                            + unit(mean_at(M, l, tr & hape & (df.src == "llama70b-inst")) - mean_at(M, l, tr & hape & (df.src == "llama70b-base"))))
    if model == "instruct":
        D["assistant_axis"] = G["default"][:, l].astype(np.float32).mean(0) - G["role"][:, l].astype(np.float32).mean(0)
        D["verbosity"] = G["long"][:, l].astype(np.float32).mean(0) - G["brief"][:, l].astype(np.float32).mean(0)
    return {k: unit(v) for k, v in D.items()}


def auc(M, l, d, pos, neg):
    s = M[idx(pos | neg), l, :].astype(np.float32) @ d
    y = pos[pos | neg].values.astype(int)
    return roc_auc_score(y, s)

# --------- test sets (positive = AI)
tests = {
    "hape_gpt4o": (te & hape & (df.src == "gpt-4o"), te & hape & human),
    "hape_llama70b-inst": (te & hape & (df.src == "llama70b-inst"), te & hape & human),
    "hape_llama8b-inst(unseen gen)": (te & hape & (df.src == "llama8b-inst"), te & hape & human),
    "hape_llama8b-BASE": (te & hape & (df.src == "llama8b-base"), te & hape & human),
    "hape_llama70b-BASE": (te & hape & (df.src == "llama70b-base"), te & hape & human),
    "hc3_chatgpt": (te & hc3 & ~human, te & hc3 & human),
    "raid_all": ((df.set == "raid") & ~human, (df.set == "raid") & (df.src == "human")),
    "raid_human-para_vs_human": ((df.set == "raid") & (df.src == "human-para"), (df.set == "raid") & (df.src == "human")),
    "mage_in": ((df.set == "mage_in") & ~human, (df.set == "mage_in") & human),
    "mage_ood_gpt4": ((df.set == "mage_ood") & ~human & df.src.str.startswith("ood_gpt4_"), (df.set == "mage_ood") & human & df.src.str.startswith("ood_gpt4_")),
    "mage_ood_gpt4para": ((df.set == "mage_ood") & ~human & df.src.str.startswith("ood_gpt4para"), (df.set == "mage_ood") & human & df.src.str.startswith("ood_gpt4para")),
}
for g in sorted(df[df.set == "raid"].src.unique()):
    if g not in ("human", "human-para"):
        tests[f"raid_{g}"] = ((df.set == "raid") & (df.src == g), (df.set == "raid") & (df.src == "human"))

out = dict(layers={}, base_layers={})
dirs_all = {}
for model in ["instruct", "base"]:
    dirs_all[model] = []
    for l in range(1, NL):
        D = directions_at(model, l); dirs_all[model].append(D)
        r = {k: auc(A[model], l, D["ai"], *v) for k, v in tests.items()}
        # cross-source transfer
        r["xfer_hape->hc3"] = auc(A[model], l, D["ai_hape"], *tests["hc3_chatgpt"])
        r["xfer_hc3->hape_gpt4o"] = auc(A[model], l, D["ai_hc3"], *tests["hape_gpt4o"])
        r["cos_hape_hc3"] = float(D["ai_hape"] @ D["ai_hc3"])
        # standardized separation (Cohen's d) on pooled matched test pairs
        pos, neg = tests["hape_gpt4o"][0] | tests["hc3_chatgpt"][0], tests["hape_gpt4o"][1] | tests["hc3_chatgpt"][1]
        sp = A[model][idx(pos), l].astype(np.float32) @ D["ai"]; sn = A[model][idx(neg), l].astype(np.float32) @ D["ai"]
        r["cohens_d"] = float((sp.mean() - sn.mean()) / np.sqrt(0.5 * (sp.var() + sn.var())))
        r["cos"] = {k: float(D["ai"] @ v) for k, v in D.items() if k != "ai"}
        out["layers" if model == "instruct" else "base_layers"][l] = r
        print(model, l, {k: round(v, 3) for k, v in r.items() if isinstance(v, float) and k in ("hape_gpt4o", "hc3_chatgpt", "raid_all", "mage_ood_gpt4", "hape_llama8b-BASE", "cohens_d")})
# base vs instruct direction cosine (Qwen2.5-7B-Instruct is finetuned from Qwen2.5-7B)
out["cos_base_instruct"] = {l: float(dirs_all["instruct"][l-1]["ai"] @ dirs_all["base"][l-1]["ai"]) for l in range(1, NL)}
# random-direction baseline AUROC (chance reference)
rng = np.random.default_rng(0)
out["random_auc_hc3_L14"] = [auc(A["instruct"], 14, unit(rng.standard_normal(A["instruct"].shape[2])), *tests["hc3_chatgpt"]) for _ in range(10)]
np.save(RES / "dirs_instruct.npy", np.array([[D[k] for k in ["ai", "formality", "fluency", "length_wordlen", "domain_reddit", "posttrain", "assistant_axis", "verbosity", "ai_hape", "ai_hc3"]] for D in dirs_all["instruct"]]))
np.save(RES / "dirs_base.npy", np.array([[D[k] for k in ["ai", "formality", "fluency", "length_wordlen", "domain_reddit", "posttrain", "ai_hape", "ai_hc3"]] for D in dirs_all["base"]]))
jdump(out, RES / "e1_readout.json")
