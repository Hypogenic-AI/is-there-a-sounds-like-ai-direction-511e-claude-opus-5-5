# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-03T15:15:29.197555Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/is-there-a-sounds-like-ai-direction-511e-claude-opus-5-5`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase:** resource_finder is COMPLETE (marker `.resource_finder_complete`).

**Done**
- Venv `.venv` exists (uv; `pyproject.toml` has `[tool.uv] package=false`).
- 29 papers in `papers/`. The 5 deep-read papers have notes in `papers/notes/`.
- 4 datasets: HC3, HAP-E and MAGE in full, plus a RAID parallel subset of 48k rows. See `datasets/README.md`.
- 12 repos in `code/`.
- Models and detectors pre-fetched to the HF cache (`code/prefetch_models.py`).
- Docs: `literature_review.md` (including the direction ranking) and `resources.md`.

**Key findings**
- Human and AI text are linearly separable at about 50% depth with near-perfect AUROC (2608.24780, 2606.07313, 2605.12890). No prior work steers *generation* with the direction (gap D1).
- Logit-lens and steering analyses point to a formal-vs-casual / instruct-register axis.
- The direction fails on base-model text (SV-Detect on RAID).
- Base Llama-3 writes close to human style (HAP-E), so the signal likely comes from post-training, which points at the assistant persona.

**Direction budget (top 3 kept; full table in literature_review.md §8)**
- D1 **Causal steering.** Diff-of-means human-vs-AI direction (content-matched, from HAP-E/HC3/self-generations) added to or ablated from an instruct model during generation. Scored by independent detectors (Binoculars, Fast-DetectGPT, fakespot/desklib RoBERTa) plus content similarity, perplexity and LLM-judge fluency. Baselines: random direction, prompting, no-steer.
- D2 **Disentanglement.** Formality, length/verbosity, domain, a cheap assistant-axis rebuild (`code/assistant-axis/data/roles`) and a base-vs-instruct direction. Measure per-layer cosines; LEACE/project out; re-test AUROC and steering of the residual direction.
- D3 **Generality.** Transfer across HC3/HAP-E/RAID/MAGE, domains, generators, and base vs. instruct; check whether the projection tracks a continuum (paraphrased human, AI-edited).
- Pruned: SAE decomposition (D4), DPO/RL humanizer (D5), standalone logit lens (D6, folded into D2), human eval (D7).

**Next (experiment_runner)**
- Subject models: gemma-2-2b-it/gemma-2-2b and Qwen2.5-7B-Instruct/Qwen2.5-7B, all cached.
- Layers to sweep: about 40–60% depth. Pick the layer by steering effect.
- Steering scale: α as a fraction of the mean residual norm. Sweep large values, because steering is weak on instruct models (2606.12234).
- Decoding must be fixed. Include a prompting baseline (AxBench).

**Caveats**
- Paper-finder was down (HTTP 500); I searched manually via the arXiv API.
- HC3 human text has tokenization artefacts; normalize them.
- Hello-SimpleAI detector is trained on HC3, so it is not independent of an HC3-derived direction.
- desklib needs a custom model class.
- Binoculars uses about 28GB of GPU memory; run it in a separate pass.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase:** experiment_runner COMPLETE (session 2 resumed after an interruption). All phases (plan, E1–E4, E3b, analysis, docs, validation) are done. Deliverables: REPORT.md, README.md, figures/fig1–3, results/*.csv.

**Key findings**
- **E1.** d_AI at block 20 gives AUROC 0.89–0.91 on matched pairs and transfers to chat generators (0.78–0.90), but not to base-LM text (0.58).
- **E2.** Whitened cos with formality, fluency, domain, verbosity and the assistant axis is ≤ 0.05. Raw cos with the post-training direction is 0.90.
- **E3.** At α = −0.2, desklib drops from 0.996 to 0.84 (random 0.99, formality 0.98, prompt 0.996). The effect holds among perfectly fluent texts (0.85; block 16: 0.67). d_AI⊥ steers equally. The effect is far from human level (0.35).
- **E3b.** The mean-clamp effects are small but specific (−0.034 vs random).
- **E4.** Base/instruct cos is 0.99. Base read-out is ≥ instruct. Steering the base model works both ways.
- **Judges.** The local Llama judge is invalid (AUROC 0.48). The Cohere judge is weak (0.71): d_AI −6.2, but formality −8.3 and the prompt −6.7, so specificity holds only for the trained detectors.

**Deviations**
- The paid OpenRouter key was over its daily limit. Nemotron free covers only 1,035 texts (α = −0.1); Cohere trial covers a 14-condition × 50-prompt subset.
- The HF cache was reset, so models were re-downloaded.
- Zero-ablation broke the text. I replaced it with a mean-clamp (src/steer_extra.py).

**Validation**
- Re-running analyze_steer, analyze_extra and analyze_validity reproduces identical CSVs.
- results/acts (5.8 GB) is gitignored.

**Open:** single model family; Gemma/SAE replication not done; no human eval.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
