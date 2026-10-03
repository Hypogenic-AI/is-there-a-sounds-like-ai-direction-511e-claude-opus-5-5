"""E2 disentanglement + probe replication + base-vs-instruct probes, at the steering layer (and a layer sweep for probes).
Layer convention: index i = hidden_states[i+1] = output of transformer block i (0..27)."""
import json
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import *

set_seed(0)
cfg = json.load(open(RES / "steer_config.json")); LB = cfg["layer"]  # block index
HS = LB + 1  # hidden_states index
NAMES = ["ai", "formality", "fluency", "length_wordlen", "domain_reddit", "posttrain", "assistant_axis", "verbosity", "ai_hape", "ai_hc3"]
D = dict(zip(NAMES, np.load(RES / "dirs_instruct.npy")[LB]))
df = pd.read_parquet(RES / "texts_scored.parquet")
df["lp_instruct"] = np.load(ACTS / "instruct_lp.npy")
A = {m: np.load(ACTS / f"{m}.npy", mmap_mode="r") for m in ["instruct", "base"]}
X = A["instruct"][:, HS].astype(np.float32)
unit = lambda v: v / np.linalg.norm(v)
tr, te = (df.split == "train").values, (df.split == "test").values
human = (df.label == 0).values
matched = df.set.isin(["hape", "hc3"]).values & df.src.isin(["human", "gpt-4o", "llama70b-inst", "chatgpt"]).values
TR, TE = tr & matched, te & matched
y = (~human).astype(int)

# unrelated reference direction: HAP-E human fiction vs academic (genre), and random-split null
g = df.group.values
D["genre_fic_vs_acad"] = unit(X[tr & human & (g == "fic")].mean(0) - X[tr & human & (g == "acad")].mean(0))
rng = np.random.default_rng(0)
hh = np.where(tr & human)[0]; rng.shuffle(hh)
D["null_random_split"] = unit(X[hh[: len(hh)//2]].mean(0) - X[hh[len(hh)//2:]].mean(0))
conf_names = ["formality", "fluency", "length_wordlen", "domain_reddit", "assistant_axis", "verbosity"]
Q, _ = np.linalg.qr(np.stack([D[k] for k in conf_names], 1))
r = D["ai"] - Q @ (Q.T @ D["ai"]); D["ai_resid"] = unit(r)
out = {"layer_block": LB}
keys = list(D)
out["cos"] = {a: {b: float(D[a] @ D[b]) for b in keys} for a in keys}
# whitened cosine (Mahalanobis geometry; removes anisotropy of the residual stream)
mu = X[tr].mean(0); C = np.cov((X[tr] - mu).T) + 1e-2 * np.eye(X.shape[1]) * np.trace(np.cov((X[tr] - mu).T)) / X.shape[1]
ev, V = np.linalg.eigh(C); W = V @ np.diag(ev ** -0.5) @ V.T
Dw = {k: unit(W @ v) for k, v in D.items()}
out["cos_whitened"] = {a: {b: float(Dw[a] @ Dw[b]) for b in keys} for a in keys}


def auc_proj(v, mask):
    return float(roc_auc_score(y[mask], X[mask] @ v))

# AUROC of each direction on matched test pairs (AI vs human)
out["auc_matched_test"] = {k: auc_proj(v, TE) for k, v in D.items()}
# confound subspace removal + re-fit (diff-of-means and logistic probe)
P = np.eye(X.shape[1]) - Q @ Q.T
Xp = X @ P


def probe_auc(Xa, trm, tem, C=0.05):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    clf.fit(Xa[trm], y[trm]); return float(roc_auc_score(y[tem], clf.decision_function(Xa[tem])))

dm = lambda Xa: unit(Xa[TR & ~human].mean(0) - Xa[TR & human].mean(0))
out["after_confound_removal"] = dict(
    diffmean_full=float(roc_auc_score(y[TE], X[TE] @ dm(X))), diffmean_projected=float(roc_auc_score(y[TE], Xp[TE] @ dm(Xp))),
    probe_full=probe_auc(X, TR, TE), probe_projected=probe_auc(Xp, TR, TE))
# transfer of probes to RAID / MAGE OOD
for nm, m in [("raid", (df.set == "raid").values & (df.src != "human-para").values), ("mage_ood", (df.set == "mage_ood").values)]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(X[TR], y[TR])
    out[f"probe_transfer_{nm}"] = float(roc_auc_score(y[m], clf.decision_function(X[m])))
    out[f"diffmean_transfer_{nm}"] = float(roc_auc_score(y[m], X[m] @ D["ai"]))

# covariate regression on matched test texts: does d_AI projection add beyond text covariates?
T = df[TE].copy(); T["proj_ai"] = X[TE] @ D["ai"]
cov = ["formality", "lp_instruct", "mean_word_len", "n_words", "contractions_per100", "ai_isms_per100", "distinct2", "first_person_per100", "markdown_per100"]
dom = pd.get_dummies(T.group, prefix="g").astype(float)
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def cv_auc(F):
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), F, T.label, cv=cv, method="decision_function")
    return float(roc_auc_score(T.label, p))
out["covariate_cv_auc"] = dict(
    covariates=cv_auc(pd.concat([T[cov], dom], axis=1)), proj_only=cv_auc(T[["proj_ai"]]),
    covariates_plus_proj=cv_auc(pd.concat([T[cov + ["proj_ai"]], dom], axis=1)),
    desklib_only=cv_auc(T[["desklib"]]))
out["corr_proj_with_covariates_humanonly"] = T[T.label == 0][cov + ["proj_ai"]].corr()["proj_ai"].drop("proj_ai").round(3).to_dict()
out["corr_proj_with_covariates_all"] = T[cov + ["proj_ai"]].corr()["proj_ai"].drop("proj_ai").round(3).to_dict()

# probe (logistic) AUROC by layer for base vs instruct (replication of 'linearly separable')
out["probe_by_layer"] = {}
for l in [4, 8, 12, 16, 20, 24, 27]:
    out["probe_by_layer"][l] = {m: probe_auc(A[m][:, l + 1].astype(np.float32), TR, TE) for m in ["instruct", "base"]}
    print(l, out["probe_by_layer"][l], flush=True)
jdump(out, RES / "e2_disentangle.json")
print(json.dumps({k: v for k, v in out.items() if k not in ("cos", "cos_whitened")}, indent=1))
print(pd.DataFrame(out["cos"]).loc[["ai", "ai_resid"]].round(2).T.to_string())
print(pd.DataFrame(out["cos_whitened"]).loc[["ai"]].round(2).T.to_string())
