"""E3b (mean-clamp) and E4 (base model) analysis. Reuses load/summarize from analyze_steer.
Outputs results/steer_extra_summary.csv, results/steer_base_summary.csv, results/base_vs_instruct_outputs.csv."""
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
from analyze_steer import load, summarize, boot_ci, holm
from common import *

KEY = ["desklib", "fakespot", "binoculars", "local_ai_likelihood", "openrouter_ai_likelihood", "local_fluency", "local_relevance",
       "openrouter_fluency", "sim_unsteered", "english", "n_words", "contractions_per100", "markdown_per100", "proj_ai"]


def table(S):
    return S[["cond", "n"] + [k for k in KEY if k in S]].round(3)


def paired_vs(df, a, b, metrics):
    """Paired per-prompt comparison of condition a vs condition b."""
    A_, B_ = df[df.cond == a].set_index("pid"), df[df.cond == b].set_index("pid")
    out = []
    for m in metrics:
        if m not in A_ or A_[m].isna().all() or B_[m].isna().all(): continue
        x = pd.concat([A_[m], B_[m]], axis=1, keys=["a", "b"]).dropna()
        d = x.a - x.b
        out.append(dict(a=a, b=b, metric=m, mean_a=x.a.mean(), mean_b=x.b.mean(), diff=d.mean(), ci=boot_ci(d),
                        p=float(wilcoxon(x.a, x.b).pvalue) if d.abs().sum() > 0 else 1.0, n=len(x)))
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    M = ["desklib", "binoculars", "local_ai_likelihood", "openrouter_ai_likelihood", "local_fluency", "openrouter_fluency", "sim_unsteered", "proj_ai"]
    # ---------------- E3b clamp
    if (RES / "scores_gen_extra.parquet").exists():
        df = load("gen_extra", None)
        S = summarize(df); S.to_csv(RES / "steer_extra_summary.csv", index=False)
        print(table(S).to_string())
        rows = []
        for c in df.cond.unique():
            if c != "none": rows += paired_vs(df, c, "none", M)
        # d_AI clamp vs the random clamps (mean over the 3 random directions per prompt)
        Mx = [m for m in M if m in df]
        r = df[df.method == "clamp_random"].groupby("pid")[Mx].mean().reset_index().assign(cond="clamp_random_mean")
        rows += paired_vs(pd.concat([df, r]), "clamp_L20_human-1sd", "clamp_random_mean", M)
        P = pd.DataFrame(rows); P["p_holm"] = holm(P.p.values); P.to_csv(RES / "steer_extra_paired.csv", index=False)
        print(P.round(4).to_string())
    # ---------------- E4 base model
    if (RES / "scores_gen_base.parquet").exists():
        db = load("gen_base", None)
        S = summarize(db, ref="base_none"); S.to_csv(RES / "steer_base_summary.csv", index=False)
        print(table(S).to_string())
        rows = []
        for c in db.cond.unique():
            if c.startswith("base_") and c != "base_none": rows += paired_vs(db, c, "base_none", M)
        P = pd.DataFrame(rows); P["p_holm"] = holm(P.p.values); P.to_csv(RES / "steer_base_paired.csv", index=False)
        print(P.round(4).to_string())
        # base answers vs instruct answers vs human / ChatGPT references (same 150 questions)
        dm = load("gen_main", None)
        comp = pd.concat([db[db.cond.isin(["base_none"])], dm[dm.cond.isin(["none", "ref_human", "ref_chatgpt"])]])
        rows = []
        for c, d in comp.groupby("cond"):
            r = dict(cond=c, n=len(d))
            for m in ["desklib", "fakespot", "binoculars", "local_ai_likelihood", "openrouter_ai_likelihood", "proj_ai", "n_words", "markdown_per100", "contractions_per100"]:
                if m in d: r[m] = d[m].mean(); r[m + "_ci"] = boot_ci(d[m])
            rows.append(r)
        C = pd.DataFrame(rows); C.to_csv(RES / "base_vs_instruct_outputs.csv", index=False)
        print(C.round(3).to_string())
        pr = paired_vs(comp, "base_none", "none", ["desklib", "binoculars", "local_ai_likelihood", "openrouter_ai_likelihood", "proj_ai"]) + \
             paired_vs(comp, "base_none", "ref_human", ["desklib", "binoculars", "proj_ai"])
        pd.DataFrame(pr).to_csv(RES / "base_vs_instruct_paired.csv", index=False)
        print(pd.DataFrame(pr).round(4).to_string())
