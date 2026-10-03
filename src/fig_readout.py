"""Figures for E1/E2/E4 read-out analyses."""
import json
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

e1 = json.load(open(RES / "e1_readout.json")); e2 = json.load(open(RES / "e2_disentangle.json"))
L = sorted(int(k) for k in e1["layers"])
fig, axs = plt.subplots(1, 3, figsize=(18, 4.5))
ax = axs[0]
for key, col in [("hape_gpt4o", "C0"), ("hc3_chatgpt", "C1"), ("mage_ood_gpt4", "C2"), ("raid_all", "C3"), ("hape_llama8b-BASE", "C4")]:
    ax.plot(L, [e1["layers"][str(l)][key] for l in L], "-", color=col, label=key + " (instruct)")
    ax.plot(L, [e1["base_layers"][str(l)][key] for l in L], "--", color=col, alpha=.7)
ax.axhline(0.5, color="gray", lw=.8); ax.set_xlabel("layer (hidden_states index)"); ax.set_ylabel("AUROC of d_AI projection")
ax.set_title("Diff-of-means d_AI read-out (solid: Instruct, dashed: Base)"); ax.legend(fontsize=7)
ax = axs[1]
pb = e2["probe_by_layer"]; ls = sorted(int(k) for k in pb)
ax.plot([l + 1 for l in ls], [pb[str(l)]["instruct"] for l in ls], "o-", label="logistic probe, Instruct")
ax.plot([l + 1 for l in ls], [pb[str(l)]["base"] for l in ls], "s--", label="logistic probe, Base")
ax.plot(L, [e1["cos_base_instruct"][str(l)] for l in L], "-", color="C3", label="cos(d_AI base, d_AI instruct)")
ax.set_ylim(0.9, 1.005); ax.set_ylabel("AUROC / cosine")
a2 = ax.twinx()
a2.plot(L, [e1["layers"][str(l)]["cohens_d"] for l in L], ":", color="k", label="Cohen's d of d_AI projection (Instruct)")
a2.plot(L, [e1["base_layers"][str(l)]["cohens_d"] for l in L], ":", color="gray", label="Cohen's d (Base)")
a2.set_ylabel("Cohen's d (pooled HAP-E+HC3 test)")
ax.set_xlabel("layer"); ax.set_title("Base vs Instruct: probes, direction cosine, separation"); ax.legend(fontsize=7, loc="lower left"); a2.legend(fontsize=7, loc="lower right")
ax = axs[2]
lb = e2["layer_block"]; r = e1["layers"][str(lb + 1)]
gens = [k for k in r if k.startswith("raid_") and k not in ("raid_all", "raid_human-para_vs_human")]
vals = [r[k] for k in gens] + [r["raid_human-para_vs_human"], r["mage_in"], r["mage_ood_gpt4"], r["mage_ood_gpt4para"], r["hape_llama8b-inst(unseen gen)"], r["hape_llama8b-BASE"], r["hape_llama70b-BASE"]]
names = [k.replace("raid_", "RAID ") for k in gens] + ["RAID paraphrased-human", "MAGE in-dist (old gens)", "MAGE OOD GPT-4", "MAGE OOD GPT-4 para", "HAP-E Llama3-8B-Inst", "HAP-E Llama3-8B BASE", "HAP-E Llama3-70B BASE"]
o = np.argsort(vals)
ax.barh(np.array(names)[o], np.array(vals)[o], color=["C3" if ("BASE" in n or n.endswith(("gpt2", "mpt", "mistral", "cohere", "gpt3")) or "old" in n) else "C0" for n in np.array(names)[o]])
ax.axvline(0.5, color="k", lw=.8); ax.set_xlim(0.4, 1); ax.set_title(f"Transfer of d_AI at block {lb} (AUROC vs human)"); ax.tick_params(labelsize=7)
plt.tight_layout(); plt.savefig(FIG / "readout_layers_transfer.png", dpi=130); plt.close()

keys = ["ai_resid", "posttrain", "length_wordlen", "verbosity", "assistant_axis", "fluency", "domain_reddit", "formality", "genre_fic_vs_acad", "null_random_split"]
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(keys))
ax.bar(x - .2, [e2["cos"]["ai"][k] for k in keys], .4, label="raw cosine with d_AI")
ax.bar(x + .2, [e2["cos_whitened"]["ai"][k] for k in keys], .4, label="whitened cosine")
ax2 = ax.twinx(); ax2.plot(x, [e2["auc_matched_test"][k] for k in keys], "kD", label="AUROC of that direction on human-vs-AI test")
ax2.axhline(.5, color="k", lw=.5, ls=":"); ax2.set_ylim(0, 1); ax2.set_ylabel("AUROC")
ax.set_xticks(x); ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=8); ax.axhline(0, color="k", lw=.5)
ax.set_title(f"d_AI vs confound directions (block {lb})"); ax.legend(fontsize=7, loc="lower left"); ax2.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "disentangle_cosines.png", dpi=130); plt.close()
print("ok")
