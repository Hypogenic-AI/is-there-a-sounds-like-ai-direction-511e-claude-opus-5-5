"""Final report figures (static PNG for REPORT.md). Reads the CSV summaries written by analyze_*.py.
Palette: validated 6-slot categorical (dataviz default) + gray for the random-direction null;
every series also gets its own marker shape so identity is never colour-only."""
import ast
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRAY, INK, MUTED = "#8a8a85", "#222222", "#6b6b66"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                     "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2})
S = pd.read_csv(RES / "steer_main_summary.csv")
none = S[S.cond == "none"].iloc[0]
METH = [("ai", "d_AI, block 20", C[0], "o"), ("L16_ai", "d_AI, block 16", C[5], "D"), ("ai_resid", "d_AI ⊥ confounds", C[1], "s"),
        ("formality", "formality (→ informal)", C[2], "^"), ("assistant_axis", "assistant axis (→ away)", C[3], "v"),
        ("verbosity", "verbosity (→ brief)", C[4], "P"), ("random", "random, equal norm (mean of 3)", GRAY, "x")]


def series(meth):
    d = S[S.method == meth]
    if meth == "random":
        d = d.groupby("alpha").mean(numeric_only=True).reset_index()
    return pd.concat([d, S[S.cond == "none"].assign(alpha=0.0)]).sort_values("alpha")


# ---------- Fig 1: dose-response, small multiples
panels = [("desklib", "desklib P(AI)  (↓ = more human)"), ("fakespot", "fakespot P(AI)  (↓ = more human)"),
          ("binoculars", "Binoculars score  (↑ = more human)"), ("local_fluency", "judge fluency, 1–5 (Llama-3.1-8B)"),
          ("sim_unsteered", "content similarity to unsteered answer"), ("proj_ai", "read-back projection on d_AI")]
fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
for ax, (m, lab) in zip(axs.flat, panels):
    for meth, name, col, mk in METH:
        d = series(meth)
        d = d[d.alpha <= 0] if meth != "ai" and meth != "random" else d
        yerr = None
        if meth != "random" and m + "_lo" in d:
            yerr = np.vstack([d[m] - d[m + "_lo"], d[m + "_hi"] - d[m]]); yerr = np.nan_to_num(yerr)
        ax.errorbar(d.alpha, d[m], yerr=yerr, color=col, marker=mk, ms=6, label=name, capsize=2, lw=2 if meth in ("ai", "random") else 1.3,
                    ls="--" if meth == "random" else "-")
    p = S[S.cond == "prompt_human"]
    ax.scatter([0], p[m], marker="*", s=180, color=INK, zorder=6, label="prompt: 'write like a human'")
    for c, ls, lab2 in [("ref_human", ":", "HC3 human answers"), ("ref_chatgpt", "-.", "HC3 ChatGPT answers")]:
        b = S[S.cond == c]
        if len(b) and m in b and not np.isnan(b[m].iloc[0]):
            ax.axhline(b[m].iloc[0], ls=ls, color=MUTED, lw=1.2, label=lab2)
    ax.set_title(lab, loc="left", color=INK); ax.set_xlabel("steering coefficient α (× residual norm); α<0 = toward 'human'")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=9)
fig.suptitle("Qwen2.5-7B-Instruct, 150 held-out HC3 questions: steering with d_AI vs. control directions", x=0.01, ha="left", fontsize=12, color=INK)
plt.tight_layout(rect=(0, 0.08, 1, 0.97)); plt.savefig(FIG / "fig1_steer_dose_response.png", dpi=140); plt.close()

# ---------- Fig 2: per-text matched coherence (only texts judged fluency==5, relevance>=4, English)
F = pd.read_csv(RES / "matched_fluency_per_text.csv")
order = [("none", "unsteered"), ("prompt_human", "prompt 'write like a human'"), ("random0_a-0.25", "random #0, α=−0.25"),
         ("random1_a-0.25", "random #1, α=−0.25"), ("random2_a-0.25", "random #2, α=−0.25"), ("formality_a-0.25", "formality, α=−0.25"),
         ("verbosity_a-0.15", "verbosity, α=−0.15"), ("assistant_axis_a-0.2", "assistant axis, α=−0.2"), ("ai_resid_a-0.2", "d_AI ⊥ confounds, α=−0.2"),
         ("ai_a-0.15", "d_AI (b20), α=−0.15"), ("ai_a-0.2", "d_AI (b20), α=−0.2"), ("L16_ai_a-0.15", "d_AI (b16), α=−0.15"),
         ("prompt_human+ai", "prompt + d_AI, α=−0.15")]
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
for ax, (m, lab) in zip(axs, [("desklib", "desklib P(AI), fluent texts only"), ("binoculars", "Binoculars score, fluent texts only (↑ = more human)")]):
    ys, labs = [], []
    for i, (c, name) in enumerate(order):
        r = F[F.cond == c]
        if not len(r): continue
        r = r.iloc[0]; lo, hi = ast.literal_eval(r[m + "_ci"])
        col = C[0] if "ai" in c and "assistant" not in c and "random" not in c else (GRAY if ("random" in c or c in ("none", "prompt_human")) else C[2])
        ax.plot(r[m], i, "o", ms=9, color=col, zorder=3)
        ax.errorbar(r[m], i, xerr=[[r[m] - lo], [hi - r[m]]], color=INK, capsize=3, lw=1)
        ax.text(hi + 0.005, i, f"{r[m]:.2f}  (n={int(r.n_fluent)})", va="center", fontsize=8, color=INK)
        ys.append(i); labs.append(name)
    ax.set_yticks(ys); ax.set_yticklabels(labs); ax.set_title(lab, loc="left", color=INK)
    ax.set_xlim(0.5 if m == "desklib" else 0.6, 1.08 if m == "desklib" else 0.95)
axs[0].invert_yaxis()
fig.suptitle("Matched coherence per text: detector scores among generations judged perfectly fluent (5/5) and on-topic   "
             "[blue = d_AI, green = confound directions, gray = controls]", x=0.01, ha="left", fontsize=10, color=INK)
plt.tight_layout(rect=(0, 0, 1, 0.94)); plt.savefig(FIG / "fig2_matched_coherence.png", dpi=140); plt.close()

# ---------- Fig 3: base model steering + in-distribution clamp
B = pd.read_csv(RES / "steer_base_summary.csv"); bn = B[B.cond == "base_none"].iloc[0]
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
for ax, m, lab in zip(axs[:2], ["desklib", "binoculars"], ["base model: desklib P(AI)", "base model: Binoculars (↑ = more human)"]):
    for meth, name, col, mk in [("ai", "base model's own d_AI", C[0], "o"), ("ai_from_instruct", "instruct model's d_AI", C[1], "s"),
                                ("random", "random, equal norm", GRAY, "x")]:
        d = pd.concat([B[B.method == meth], B[B.cond == "base_none"].assign(alpha=0.0)]).sort_values("alpha")
        yerr = np.nan_to_num(np.vstack([d[m] - d[m + "_lo"], d[m + "_hi"] - d[m]]))
        ax.errorbar(d.alpha, d[m], yerr=yerr, color=col, marker=mk, ms=6, capsize=2, label=name, ls="--" if meth == "random" else "-")
    ax.set_title(lab, loc="left", color=INK); ax.set_xlabel("α (× residual norm, block 20); α<0 = toward 'human'")
axs[0].legend(fontsize=8)
X = pd.read_csv(RES / "steer_extra_summary.csv")
cl = [("none", "unsteered"), ("clamp_L20_ai", "clamp b20 → AI mean"), ("clamp_L20_human", "clamp b20 → human mean"),
      ("clamp_L20_human-1sd", "clamp b20 → human mean − 1 s.d."), ("clamp_mid_human", "clamp b10–27 → human mean"),
      ("clamp_L20_random0_human-1sd", "random dir #0, same clamp"), ("clamp_L20_random1_human-1sd", "random dir #1, same clamp"),
      ("clamp_L20_random2_human-1sd", "random dir #2, same clamp")]
ax = axs[2]
for i, (c, name) in enumerate(cl):
    r = X[X.cond == c].iloc[0]
    col = GRAY if ("random" in c or c == "none") else C[0]
    ax.plot(r.desklib, i, "o", ms=9, color=col, zorder=3)
    ax.errorbar(r.desklib, i, xerr=[[r.desklib - r.desklib_lo], [r.desklib_hi - r.desklib]], color=INK, capsize=3, lw=1)
    ax.text(r.desklib_hi + 0.002, i, f"{r.desklib:.3f}", va="center", fontsize=8)
ax.set_yticks(range(len(cl))); ax.set_yticklabels([n for _, n in cl]); ax.invert_yaxis(); ax.set_xlim(0.9, 1.01)
ax.set_title("instruct: mean-clamp, desklib P(AI)", loc="left", color=INK)
plt.tight_layout(); plt.savefig(FIG / "fig3_base_and_clamp.png", dpi=140); plt.close()
print("figures written")
