# Research State

- Current phase: `None`
- Pipeline completed: `False`

## Previous phases

resource_finder (succeeded), experiment_runner (failed)

## Current phase context

- Phase: `experiment_runner`
- Status: `failed`
- Started: `2026-10-03T13:55:34.536530Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/is-there-a-sounds-like-ai-direction-511e-claude-opus-5-5`
- Directory usable: `True`

## Output validation

- Valid: `False`
- Expected: `REPORT.md`
- Missing: `REPORT.md`
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
Update this section at the end of the `experiment_runner` phase.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
