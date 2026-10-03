# Resources Catalog

## Summary
Resources for testing whether an LLM's residual stream contains a causal "sounds like AI" direction, and whether that direction is distinct from formality, domain, length and the assistant persona. The catalog has 29 papers (5 deep-read with notes), 4 datasets (HC3, HAP-E and MAGE in full, plus a parallel RAID subset) and 12 code repositories. Candidate subject models and independent detectors are pre-fetched to the HF cache.

## Papers
I downloaded 29 papers. Full list with relevance notes: `papers/README.md`. Metadata and abstracts: `papers/paper_meta.json`. Deep-read notes: `papers/notes/`.

| Title | Year | File | Key info |
|---|---|---|---|
| Linear Probing Provides Robust and Efficient Detection of MGT [user] | 2026 | papers/2608.24780_*.pdf | Last-token logreg probes on Llama-3-8B, best OOD layer ~50% depth. Proposes a shared "machineness" direction; no causal test. |
| SV-Detect [user] | 2026 | papers/2606.07313_*.pdf | Per-layer human/AI directions. Logit lens shows a formal-vs-casual axis. Fails on base-model text. |
| Feature-Level Insights into ATD with SAEs [user] | 2025 | papers/2503.03601_*.pdf | Gemma-2-2B Gemma Scope layer-16 features (formality 14161, repetition 8264, ...). Steering is evaluated only qualitatively. |
| The Assistant Axis [user] | 2026 | papers/2601.10387_*.pdf | Mean-difference persona axis, the key confound. Released code/roles. |
| Steer-to-Detect | 2026 | papers/2605.12890_*.pdf | Read-time steering for detection; formal-token effects. |
| Do LLMs write like humans? (HAP-E) | 2024 | papers/2410.16107_*.pdf | Instruct models deviate in style while base models stay close to human. |
| Stress-testing MGT detection (DPO) | 2025 | papers/2505.24523_*.pdf | Fine-tuning style shift fools detectors. |
| RAID / MAGE / HC3 | 2023–24 | papers/2405.07940, 2305.13242, 2301.07597 | Benchmarks we use. |
| Binoculars / Fast-DetectGPT / DetectGPT | 2023–24 | papers/2401.12070, 2310.05130, 2301.11305 | Independent zero-shot detectors. |
| ActAdd / CAA / RepE / Refusal dir. / Persona vectors / Style vectors | 2023–25 | papers/2308.10248, 2312.06681, 2310.01405, 2406.11717, 2507.21509, 2402.01618 | Steering methodology. |
| LRH / Geometry of Truth / LEACE | 2023 | papers/2311.03658, 2310.06824, 2306.03819 | Linear-representation theory, mass-mean vs. probe directions, concept erasure. |
| AxBench / Unified steering eval / Effectiveness-fluency / Limits of steering vectors | 2025–26 | papers/2501.17148, 2502.02716, 2606.12234, 2607.01802 | Steering evaluation protocols and expected pitfalls. |
| Idiosyncrasies / AuthorMist / Intrinsic dimension | 2023–25 | papers/2502.12150, 2503.08716, 2306.04723 | Lexical signatures, a humanizer baseline, geometric differences. |

## Datasets
I downloaded 4 datasets. Details and re-download instructions: `datasets/README.md`. Samples: `datasets/samples/`.

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| HC3 | HF Hello-SimpleAI/HC3 | 24.3k Qs; 58.5k human / 26.9k ChatGPT answers | Paired human vs. AI QA | datasets/hc3/ | Content-matched. Normalise tokenisation. 5 domains. |
| HAP-E | HF browndw/human-ai-parallel-corpus | 8,290 docs × (human prompt, human continuation, 6 LLM continuations) | Parallel continuations | datasets/hape/text_data/ | Base and instruct Llama-3 8B/70B plus GPT-4o(-mini). 6 genres. |
| MAGE | HF yaful/MAGE | 319k / 57k / 57k + OOD sets | MGT detection | datasets/mage/ | label 1 = human. 27 generators, 10 domains. |
| RAID subset | HF liamdugan/raid (train.csv streamed) | 48,182 rows (689 human docs + all their generations; attack none/paraphrase) | MGT detection, parallel | datasets/raid_subset/ | 8 domains, base/chat model pairs. Script to re-create included. |

## Code repositories
I cloned 12 repositories. Details: `code/README.md`.

| Name | URL | Purpose | Location |
|---|---|---|---|
| mgt_probes | github.com/gerritq/mgt_probes | Official linear-probe code (2608.24780) | code/mgt_probes |
| sv-detect | github.com/Atmyre/sv-detect | Official SV-Detect code (2606.07313) | code/sv-detect |
| assistant-axis | github.com/safety-research/assistant-axis | Assistant axis pipeline, 279 role prompts, questions, steering/capping | code/assistant-axis |
| steer-to-detect | github.com/LuxLiang/steer-to-detect-release | Read-time steering detector | code/steer-to-detect |
| persona_vectors | github.com/safety-research/persona_vectors | Trait-vector extraction plus generation-time steering hook | code/persona_vectors |
| refusal_direction | github.com/andyrdt/refusal_direction | Direction selection, directional ablation | code/refusal_direction |
| CAA | github.com/nrimsky/CAA | Contrastive activation addition | code/CAA |
| repeng | github.com/vgel/repeng | Lightweight control vectors | code/repeng |
| concept-erasure | github.com/EleutherAI/concept-erasure | LEACE | code/concept-erasure |
| binoculars | github.com/ahans30/Binoculars | Independent zero-shot detector | code/binoculars |
| fast-detect-gpt | github.com/baoguangsheng/fast-detect-gpt | Independent zero-shot detector | code/fast-detect-gpt |
| raid | github.com/liamdugan/raid | RAID loaders and evaluation | code/raid |

## Models pre-fetched to the HF cache
`code/prefetch_models.py` downloads these into `~/.cache/huggingface/hub`; the log is `logs/prefetch_models.log`.
- **Subject models:** google/gemma-2-2b-it, google/gemma-2-2b, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-7B.
- **Gated models:** the HF token has access to meta-llama/Llama-3.1-8B(-Instruct) and gemma-2-9b-it. These were not pre-fetched.
- **Detectors and evaluators:**
  - desklib/ai-text-detector-v1.01 (needs its custom model class; see its model card)
  - fakespot-ai/roberta-base-ai-text-detection-v1
  - Hello-SimpleAI/chatgpt-detector-roberta (trained on HC3, so not independent of an HC3-derived direction)
  - openai-community/roberta-large-openai-detector
  - s-nlp/roberta-base-formality-ranker (formality confound)
  - tiiuae/falcon-7b and falcon-7b-instruct (Binoculars pair)
- **SAEs:** google/gemma-scope-2b-pt-res, layers 12/16/20 at width 16k.
- **Hardware:** 1× RTX A6000 (48 GB); about 380 GB free on disk.

## Resource gathering notes

### Search strategy
The paper-finder service returned HTTP 500 for all queries (three topics, retried). I fell back to about 26 targeted arXiv API queries: MGT plus probes, steering, hidden states and SAEs; persona vectors; style steering; refusal direction; humanizers; steering evaluation. I added the user-specified papers and the foundational steering and detection papers I already knew. Deep reading was done in parallel; notes are in `papers/notes/`.

### Selection criteria
1. The four papers the user specified.
2. Papers that extract human-vs-AI directions from internals.
3. Steering methodology and its evaluation pitfalls.
4. Confound sources: style, persona, base vs. instruct.
5. Independent detectors for evaluation.

### Challenges
- **Paper-finder outage.** Handled by manual arXiv search.
- **RAID size.** train.csv is 11.8 GB, so I streamed it and kept a 1/20 parallel-preserving subsample.
- **uv package build.** It initially failed because there is no package dir; fixed by adding `[tool.uv] package = false`.
- **HC3 human answers are pre-tokenised.** They contain spaces before punctuation, an artefact a direction could latch onto. This is documented.

### Gaps and workarounds
- There are no off-the-shelf formality/AI-ness *contrast pairs* for subject models. The experiment phase should generate them, either with persona-style system prompts or by scoring existing texts with the formality ranker.
- Assistant-axis vectors are released only for models of 27B and up. Recompute a cheap version for 2–9B models from `code/assistant-axis/data/roles`.

## Recommendations for experiment design
1. **Primary data:**
   - HAP-E (genre-matched human vs. instruct continuations, with base-model continuations as the persona control)
   - HC3 (question-matched)
   - Subject-model self-generations vs. human answers
   - RAID and MAGE for transfer tests
2. **Baselines:**
   - no-steer
   - a random direction of matched norm
   - prompting ("write like a human")
   - confound-direction steering (formality, verbosity, assistant axis)
   - optional SAE-feature steering
3. **Metrics:**
   - Detector P(AI) from Binoculars, Fast-DetectGPT and a supervised RoBERTa detector
   - Content similarity
   - Perplexity under a separate LM
   - LLM-judge fluency
   - Output length, formality and AI-ism counts
   - Per-layer cosine matrix with confound directions, and AUROC after LEACE
4. **Code to reuse:**
   - `persona_vectors/activation_steer.py` or `repeng` for steering hooks
   - `sv-detect/src/extract/compute_steering_vectors.py` for direction construction
   - `assistant-axis/pipeline` and roles for the persona axis
   - `concept-erasure` for erasure
   - `binoculars` and `fast-detect-gpt` for scoring
