"""(1) Validity of each AI-ness measure: AUROC for separating the HC3 human vs ChatGPT reference answers to the
150 steering questions (truncated to ~120 words). A measure that cannot separate real human from real AI text
cannot be used to claim that steered text 'reads as human'.
(2) Per-text matched coherence: restrict to generations the local judge rated fluency==5 and relevance>=4 and
English, then compare detector scores across conditions (bootstrap CIs; Mann-Whitney vs unsteered).
This removes 'detector shift caused by broken text' as an explanation.
Outputs results/measure_validity.csv, results/matched_fluency_per_text.csv."""
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from analyze_steer import load, boot_ci
from common import *

df = load("gen_main", None)
co = RES / "judge_cohere_gen_main.jsonl"
if co.exists():
    j = pd.read_json(co, lines=True)
    j = j[j["error"].isna()] if "error" in j else j
    j = j.drop_duplicates(["file", "cond", "pid"], keep="last")[["file", "cond", "pid", "ai_likelihood", "fluency", "relevance"]]
    df = df.merge(j.rename(columns={c: f"cohere_{c}" for c in ["ai_likelihood", "fluency", "relevance"]}), on=["file", "cond", "pid"], how="left")
df.to_parquet(RES / "merged_gen_main.parquet")

# ---------- (1) validity on references
ref = df[df.cond.isin(["ref_human", "ref_chatgpt"])]
rows = []
for m, sign in [("desklib", 1), ("fakespot", 1), ("binoculars", -1), ("local_ai_likelihood", 1), ("cohere_ai_likelihood", 1),
                ("openrouter_ai_likelihood", 1), ("proj_ai", 1), ("falcon_logppl", -1)]:
    d = ref.dropna(subset=[m])
    if d.cond.nunique() < 2: continue
    y = (d.cond == "ref_chatgpt").astype(int)
    rows.append(dict(measure=m, auroc_chatgpt_vs_human=roc_auc_score(y, sign * d[m]), n=len(d),
                     mean_human=d[d.cond == "ref_human"][m].mean(), mean_chatgpt=d[d.cond == "ref_chatgpt"][m].mean()))
V = pd.DataFrame(rows); V.to_csv(RES / "measure_validity.csv", index=False)
print(V.round(3).to_string())

# ---------- (2) per-text matched coherence
ok = (df.local_fluency == 5) & (df.local_relevance >= 4) & df.english
conds = ["none", "ai_a-0.1", "ai_a-0.15", "ai_a-0.2", "ai_a-0.25", "ai_resid_a-0.2", "ai_resid_a-0.25", "formality_a-0.25", "formality_a-0.3",
         "assistant_axis_a-0.2", "assistant_axis_a-0.25", "verbosity_a-0.15", "random0_a-0.25", "random1_a-0.25", "random2_a-0.25",
         "random0_a-0.3", "random1_a-0.3", "random2_a-0.3", "L16_ai_a-0.15", "L16_ai_a-0.2", "prompt_human", "prompt_human+ai", "ai_a0.2"]
base = df[(df.cond == "none") & ok]
rows = []
for c in conds:
    d = df[(df.cond == c) & ok]
    if len(d) < 5: continue
    r = dict(cond=c, n_fluent=len(d), frac_fluent=len(d) / (df.cond == c).sum())
    for m in ["desklib", "fakespot", "binoculars", "cohere_ai_likelihood", "proj_ai", "sim_unsteered"]:
        x = d[m].dropna()
        r[m] = x.mean(); r[m + "_ci"] = boot_ci(x)
        if c != "none" and len(x) > 5 and m in ("desklib", "binoculars", "cohere_ai_likelihood"):
            r[m + "_p_vs_none"] = float(mannwhitneyu(x, base[m].dropna()).pvalue)
    r["frac_desklib_human"] = float((d.desklib < 0.5).mean())
    rows.append(r)
F = pd.DataFrame(rows); F.to_csv(RES / "matched_fluency_per_text.csv", index=False)
pd.set_option("display.width", 250)
print(F[["cond", "n_fluent", "frac_fluent", "desklib", "frac_desklib_human", "fakespot", "binoculars", "cohere_ai_likelihood", "proj_ai", "sim_unsteered",
         "desklib_p_vs_none", "binoculars_p_vs_none"]].round(3).to_string())
