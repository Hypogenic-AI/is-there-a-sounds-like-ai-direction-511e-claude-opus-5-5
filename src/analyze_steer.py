"""E3/E4 analysis: per-condition means + bootstrap CIs, paired tests vs no-steer and vs random,
matched-coherence comparison, 'humanisation success rate', figures."""
import json, re
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from common import *

rng = np.random.default_rng(0)


def load(tag, files):
    df = pd.read_parquet(RES / f"scores_{tag}.parquet")
    rb = RES / f"readback_{tag}.parquet"
    if rb.exists():
        df = df.merge(pd.read_parquet(rb), on=["file", "cond", "pid"], how="left")
    for be in ["local", "openrouter"]:
        p = RES / f"judge_{be}_{tag}.jsonl"
        if p.exists():
            j = pd.read_json(p, lines=True)
            j = j[j.get("error").isna()] if "error" in j else j
            j = j.drop_duplicates(["file", "cond", "pid"], keep="last")[["file", "cond", "pid", "ai_likelihood", "fluency", "relevance"]]
            j = j.rename(columns={c: f"{be}_{c}" for c in ["ai_likelihood", "fluency", "relevance"]})
            df = df.merge(j, on=["file", "cond", "pid"], how="left")
    df["english"] = df.non_ascii_frac < 0.1
    df["neg_binoculars"] = -df.binoculars  # higher = more AI-like (Binoculars: low score = machine)
    return df


def boot_ci(x, n=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return (np.nan, np.nan)
    b = rng.choice(x, (n, len(x))).mean(1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


METRICS = ["desklib", "fakespot", "binoculars", "local_ai_likelihood", "openrouter_ai_likelihood", "local_fluency", "local_relevance",
           "openrouter_fluency", "openrouter_relevance", "falcon_logppl", "sim_unsteered", "sim_question", "english", "n_words",
           "formality", "contractions_per100", "markdown_per100", "ai_isms_per100", "first_person_per100", "distinct2", "proj_ai", "success"]


def summarize(df, ref="none"):
    rows = []
    base = df[df.cond == ref].set_index("pid")
    for c, d in df.groupby("cond"):
        r = dict(cond=c, method=d.method.iloc[0], alpha=d.alpha.iloc[0], n=len(d))
        for m in METRICS:
            if m not in d: continue
            r[m] = float(d[m].astype(float).mean()); lo, hi = boot_ci(d[m].astype(float)); r[m + "_lo"], r[m + "_hi"] = lo, hi
        # paired Wilcoxon vs no-steer for key AI-ness metrics
        if c != ref:
            dd = d.set_index("pid")
            for m in ["desklib", "binoculars", "local_ai_likelihood", "openrouter_ai_likelihood", "local_fluency"]:
                if m not in dd: continue
                x = pd.concat([dd[m], base[m]], axis=1, keys=["a", "b"]).dropna()
                if len(x) > 10 and (x.a - x.b).abs().sum() > 0:
                    r[m + "_p"] = float(wilcoxon(x.a, x.b).pvalue); r[m + "_diff"] = float((x.a - x.b).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(run, 1)
    return adj


if __name__ == "__main__":
    out = {}
    df = load("gen_main", None)
    # success = coherent (local judge fluency>=4 & relevance>=4), English, and classified human by desklib (<0.5)
    df["success"] = ((df.local_fluency >= 4) & (df.local_relevance >= 4) & df.english & (df.desklib < 0.5)).astype(float)
    S = summarize(df)
    for m in ["desklib", "binoculars", "local_ai_likelihood", "openrouter_ai_likelihood", "local_fluency"]:
        if m + "_p" in S:
            ok = S[m + "_p"].notna(); S.loc[ok, m + "_p_holm"] = holm(S.loc[ok, m + "_p"])
    S.to_csv(RES / "steer_main_summary.csv", index=False)
    # ---- d_AI vs random at equal alpha: per-prompt difference (ai - mean of 3 random dirs)
    vs = []
    for a in sorted(df[df.method == "ai"].alpha.unique()):
        ai = df[(df.method == "ai") & (df.alpha == a)].set_index("pid")
        rn = df[(df.method == "random") & (df.alpha == a)].groupby("pid")
        for m in ["desklib", "binoculars", "local_ai_likelihood", "openrouter_ai_likelihood", "local_fluency", "sim_unsteered", "success"]:
            if m not in ai or ai[m].isna().all(): continue
            r = rn[m].mean()
            x = pd.concat([ai[m], r], axis=1, keys=["ai", "rand"]).dropna()
            if len(x) < 10: continue
            diff = x.ai - x.rand
            vs.append(dict(alpha=a, metric=m, ai=x.ai.mean(), random=x.rand.mean(), diff=diff.mean(), ci=boot_ci(diff),
                           p=float(wilcoxon(x.ai, x.rand).pvalue) if diff.abs().sum() > 0 else 1.0, n=len(x)))
    V = pd.DataFrame(vs); V.to_csv(RES / "steer_ai_vs_random.csv", index=False)
    print(V.round(3).to_string())
    # ---- matched coherence: for each method the strongest |alpha| keeping local fluency >= none-0.5, relevance >= none-0.5, english >= 95%
    none = S[S.cond == "none"].iloc[0]
    mc = []
    for m, d in S[S.method.isin(["ai", "ai_resid", "formality", "assistant_axis", "verbosity", "random", "L16_ai", "L16_random"])].groupby("method"):
        d = d[d.alpha < 0] if m not in ("random",) else d[d.alpha < 0]
        okk = d[(d.local_fluency >= none.local_fluency - 0.5) & (d.local_relevance >= none.local_relevance - 0.5) & (d.english >= 0.95)]
        if m == "random":  # average over the 3 random directions per alpha
            okk = okk.groupby("alpha").mean(numeric_only=True).reset_index()
        if len(okk):
            best = okk.loc[okk.alpha.idxmin()]
            mc.append(dict(method=m, alpha=float(best.alpha), desklib=best.desklib, binoculars=best.binoculars,
                           local_ai=best.local_ai_likelihood, or_ai=best.get("openrouter_ai_likelihood", np.nan),
                           fluency=best.local_fluency, relevance=best.local_relevance, sim=best.sim_unsteered, success=best.success))
    for c in ["none", "prompt_human", "prompt_human+ai", "ai_ablate", "ai_a0.1", "ai_a0.2"]:
        b = S[S.cond == c]
        if len(b):
            b = b.iloc[0]
            mc.append(dict(method=c, alpha=float(b.alpha), desklib=b.desklib, binoculars=b.binoculars, local_ai=b.local_ai_likelihood,
                           or_ai=b.get("openrouter_ai_likelihood", np.nan), fluency=b.local_fluency, relevance=b.local_relevance,
                           sim=b.sim_unsteered, success=b.success))
    MC = pd.DataFrame(mc); MC.to_csv(RES / "steer_matched_coherence.csv", index=False)
    print(MC.round(3).to_string())
    cols = ["cond", "desklib", "fakespot", "binoculars", "local_ai_likelihood", "openrouter_ai_likelihood", "local_fluency", "local_relevance",
            "sim_unsteered", "english", "n_words", "formality", "contractions_per100", "markdown_per100", "proj_ai", "success"]
    print(S[[c for c in cols if c in S]].round(3).to_string())

    # ---------------- figures
    FIG.mkdir(exist_ok=True)
    meths = [("ai", "d_AI (L20)", "C0"), ("ai_resid", "d_AI ⟂ confounds", "C1"), ("formality", "formality", "C2"),
             ("assistant_axis", "assistant axis", "C3"), ("verbosity", "verbosity", "C4"), ("random", "random (3 dirs)", "gray"), ("L16_ai", "d_AI (L16)", "C5")]
    panels = [("desklib", "desklib P(AI)"), ("binoculars", "Binoculars score (higher = more human)"),
              ("local_ai_likelihood", "Llama-3.1-8B judge AI-likelihood"), ("openrouter_ai_likelihood", "Nemotron-3-Ultra judge AI-likelihood"),
              ("local_fluency", "judge fluency (1-5)"), ("sim_unsteered", "content sim. to unsteered"), ("english", "fraction English"), ("proj_ai", "read-back projection on d_AI")]
    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (m, lab) in zip(axs.flat, panels):
        if m not in S: ax.set_visible(False); continue
        for meth, name, col in meths:
            d = S[(S.method == meth)]
            if meth == "random": d = d.groupby("alpha").mean(numeric_only=True).reset_index()
            d = pd.concat([d, S[S.cond == "none"].assign(alpha=0.0)]) if len(d) else d
            d = d.sort_values("alpha")
            if len(d) > 1: ax.plot(d.alpha, d[m], "o-", label=name, color=col, ms=4)
        for c, mk in [("prompt_human", "*"), ("ai_ablate", "X")]:
            b = S[S.cond == c]
            if len(b): ax.scatter([0], b[m], marker=mk, s=120, color="k" if c == "prompt_human" else "red", zorder=5, label=c)
        for c, ls in [("ref_human", ":"), ("ref_chatgpt", "--")]:
            b = S[S.cond == c]
            if len(b) and m in b and not np.isnan(b[m].iloc[0]): ax.axhline(b[m].iloc[0], ls=ls, color="k", lw=1, label=c)
        ax.set_title(lab, fontsize=10); ax.set_xlabel("alpha (fraction of residual norm; <0 = toward human)")
    axs[0, 0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "steer_dose_response.png", dpi=130); plt.close()
    # AI-ness vs coherence frontier
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ym in zip(axs, ["desklib", "openrouter_ai_likelihood" if "openrouter_ai_likelihood" in S else "local_ai_likelihood"]):
        for meth, name, col in meths:
            d = S[S.method == meth]
            if meth == "random": d = d.groupby("alpha").mean(numeric_only=True).reset_index()
            if len(d): ax.plot(d.local_fluency, d[ym], "o-", color=col, label=name)
        for c, mk, col in [("none", "s", "k"), ("prompt_human", "*", "k"), ("ai_ablate", "X", "red"), ("prompt_human+ai", "P", "purple")]:
            b = S[S.cond == c]
            if len(b): ax.scatter(b.local_fluency, b[ym], marker=mk, s=140, color=col, label=c, zorder=5)
        ax.set_xlabel("judge fluency (1-5, Llama-3.1-8B)"); ax.set_ylabel(ym); ax.invert_xaxis()
    axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG / "steer_ainess_vs_coherence.png", dpi=130); plt.close()
    jdump(dict(n_prompts=int(df.pid.nunique())), RES / "steer_analysis_meta.json")
