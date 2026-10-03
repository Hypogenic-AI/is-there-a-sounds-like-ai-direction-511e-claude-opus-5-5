# Cloned Repositories

All are shallow clones (`--depth 1`). No repositories were explicitly specified in the task; these were chosen because they implement the user-specified papers or provide baselines/evaluators. Big repos (fast-detect-gpt 1.2 GB, raid 2.5 GB, CAA 0.5 GB) are big because of bundled data.

| Repo | URL | Purpose | Key entry points |
|---|---|---|---|
| `mgt_probes/` | github.com/gerritq/mgt_probes | Official code of 2608.24780 (user paper): layer-wise (LLP) and concatenated (CLP) linear probes | `src/probes/probe_main.py` (`_train_linear_probe`, `_train_meta_probe`), `src/inference.py` (hidden-state extraction, last-token) |
| `sv-detect/` | github.com/Atmyre/sv-detect | Official code of 2606.07313 (user paper): per-layer mean-diff / logreg / PCA directions + cosine features | `src/extract/compute_steering_vectors.py`, `src/extract/nb_pipeline.py` (SV construction, QR orthonormalization), `src/interpret/` (logit-lens of directions) |
| `assistant-axis/` | github.com/safety-research/assistant-axis | Official code of 2601.10387 (user paper). Roles (279 files in `data/roles`), `data/extraction_questions.jsonl`, trait lists, pipeline to compute the axis, steering & capping utilities | `pipeline/1_generate.py … 5_axis.py`, `assistant_axis/steering.py`, `assistant_axis/axis.py`. Precomputed vectors only for Gemma-2-27B / Qwen3-32B / Llama-3.3-70B (HF `lu-christina/assistant-axis-vectors`) — for 2–9B models we must recompute (cheap variant: ~50 roles × few questions) |
| `steer-to-detect/` | github.com/LuxLiang/steer-to-detect-release | Code of 2605.12890: learned steering vector applied during reading for detection | `train.py`, `evaluate.py`, `evaluate_no_steer.py` |
| `persona_vectors/` | github.com/safety-research/persona_vectors | Persona-vector pipeline (contrastive system prompts → mean-diff vector; steering during generation; projection monitoring) | `generate_vec.py`, `activation_steer.py` (clean steering hook usable for generation), `eval/cal_projection.py` |
| `refusal_direction/` | github.com/andyrdt/refusal_direction | Direction selection over layers/positions, directional ablation & weight orthogonalization | `pipeline/` (select_direction, hooks) |
| `CAA/` | github.com/nrimsky/CAA | Contrastive Activation Addition reference implementation | `generate_vectors.py`, `prompting_with_steering.py` |
| `repeng/` | github.com/vgel/repeng | Lightweight control-vector library (PCA-of-diffs, `ControlModel` wrapper for HF models) — fastest way to try steering | `repeng/extract.py`, `repeng/control.py` |
| `concept-erasure/` | github.com/EleutherAI/concept-erasure | LEACE closed-form linear concept erasure — for "remove the direction" and for erasing confounds (length/formality) before re-fitting | `concept_erasure/leace.py` (`LeaceEraser.fit`) |
| `binoculars/` | github.com/ahans30/Binoculars | Binoculars zero-shot detector (Falcon-7B / Falcon-7B-instruct) — **independent detector** for steered outputs | `binoculars/detector.py` (`Binoculars().compute_score`) |
| `fast-detect-gpt/` | github.com/baoguangsheng/fast-detect-gpt | Fast-DetectGPT zero-shot detector — second independent detector | `scripts/fast_detect_gpt.py`, `scripts/local_infer.py` |
| `raid/` | github.com/liamdugan/raid | RAID benchmark package (`pip install raid-bench`) — data loading + evaluation utilities | `raid/` package, README |

`code/prefetch_models.py` pre-downloads candidate models/detectors to the HF cache (see resources.md).

## Notes / requirements
- All repos are pure PyTorch + HF transformers; no special builds. A single RTX A6000 (48 GB) is available: 7–9B models in bf16 fit for both activation extraction and generation with hooks.
- Binoculars needs two 7B models (~28 GB bf16) — fits alongside a 2B subject model but not with a 7B subject model at once; score steered outputs in a separate pass.
- Not tested end-to-end in this phase (no repo was user-mandated); code inspection only. `persona_vectors/activation_steer.py` and `repeng` are the simplest generation-time steering hooks to adapt.
- `assistant-axis` README notes its capping function clamps from above — check the sign convention before reuse.
