# Literature Review: Is there a "sounds like AI" direction in the residual stream?

Search method: the paper-finder service returned HTTP 500 for every query, so I searched the arXiv API by keyword (queries in `logs/arxiv_search*.txt`) and downloaded 29 papers. I read 5 papers in full (all chunks): the 4 the user named plus Steer-to-Detect. Per-paper notes with numbers are in `papers/notes/*.md`. I screened the rest by abstract.

## 1. Research area overview

Three lines of work meet here.

1. **Detecting machine-generated text (MGT) from LLM internals.** Since 2025 the main result is that human and AI text are *linearly separable* in mid-layer residual streams of a frozen reader LM, even with very few training samples. Examples: linear probes (2608.24780), steering-vector features (2606.07313), learned read-time steering (2605.12890), SAE features (2503.03601) and intrinsic dimension (2306.04723). These papers are **purely correlational**. None of them steers the *generation* of a model with the direction and checks whether an independent detector's verdict changes.
2. **Linear representations and activation steering.** Directions found by mean difference or probes can be added or ablated to control behaviour: ActAdd (2308.10248), CAA (2312.06681), RepE (2310.01405), the refusal direction (2406.11717), persona vectors (2507.21509), style vectors (2402.01618) and the linear representation hypothesis (2311.03658). Evaluation papers caution that steering costs fluency, works less well on instruct than on base models, and is often beaten by prompting: AxBench (2501.17148), 2606.12234, 2607.01802, 2502.02716.
3. **What makes AI text "AI".** The candidate explanations are stylistic register (formality, nominalisations, participial clauses: 2410.16107), lexical idiosyncrasies (2502.12150), post-training rather than pretraining (base Llama-3 ≈ human; instruct ≠ human: 2410.16107), and the **Assistant persona** (2601.10387). Fine-tuning (DPO 2505.24523) or RL (AuthorMist 2503.08716) toward "human-like" output fools detectors. That is the training-based counterpart of what we want to do with a single direction.

**Gap our project fills:** (a) a causal test, steering *generation* with an MGT direction and scoring the result with *independent* detectors while checking content and coherence; and (b) a disentanglement test against formality, length, domain and the assistant axis. Every detection paper explicitly leaves causality open (2608.24780), and several find hints that the direction is "formal vs casual". The logit lens on SV-Detect directions gives *utilizing/leveraging/endeavors* on the AI side and *basically/maybe/anyway* on the human side; Steer-to-Detect's strongest tokens are *aforementioned/plethora/subsequently*.

## 2. Key papers (deep-read)

### 2608.24780 — Linear Probing Provides Robust and Efficient Detection of MGT (Quaremba et al., 2026, EMNLP) [user-specified]
- **Method:** Reader is base Llama-3-8B. They take the **last-token** hidden state at each layer, apply PCA to 100 dims, then logistic regression (L2, C=1). The direction is the normalised weight vector. LLP averages per-layer probe scores; CLP concatenates layers.
- **Data:** 16 subsets from DetectRL, MultiSocial, RAID and TSM (1.5k train / 500 test, balanced). There are 16 baselines.
- **Results:** in-domain AUC 0.90–1.00; OOD AUC 0.81–0.99, up to +11.9 over the best baseline. Separation appears from about layer 6 of 32 and plateaus by layers 5–10. **Best OOD layer is about 16 (50% depth)**; the last layers degrade. Mean pooling is up to 0.155 AUC *worse* than last-token pooling. Probes reach near-peak with 10–100 samples. Qwen 4–32B readers do slightly better.
- **"Shared MGT direction":** probe vectors have high cosine within a benchmark and moderate cosine across benchmarks (heatmap only). Projection correlates with the degree of AI editing (r up to 0.73).
- **Gaps:** no causal or steering experiment; no formality or persona control. Only a length-truncation check (AUC 0.85 at 25 characters, 0.97 at ≥125 characters). The authors attribute CLP's OOD drop on News to "formal writing cues".
- Code: `code/mgt_probes`.

### 2606.07313 — SV-Detect (Vishnyakov & Gaintseva, 2026, EMNLP) [user-specified]
- **Method:** Reader is frozen GPT-Neo-2.7B (ablations on Qwen3-1.7B, Gemma-3-1B, Llama-2-7B and Llama-3.1-8B). They mean-pool the residual stream per layer and build one direction per layer by diff-of-means, logreg normal (default) or PC1 of paired differences. Each text's L cosine features feed a standardised logistic regression.
- **Results:** DetectRL AUROC 99.8–100. MIRAGE: 0.98 on direct generation and about 0.91 on polish/rewrite. Transfer to PADBen 0.81–0.84. **Under transfer, logreg beats diff-of-means, which beats PCA.** On RAID the method fails for base-model generators (TPR@5%FPR 9–34%), so the direction is an **instruct-register** direction.
- **Interpretation:** a regex baseline of surface cues (em-dashes, "Moreover", markdown, hedges) gets 76–91 AUROC; directions add 9–24 points on top. The logit lens reads as a formal-vs-casual axis.
- **Gaps:** never steers; no confound controls; single seed.
- Code: `code/sv-detect`.

### 2503.03601 — Feature-Level Insights into ATD with SAEs (Kuznetsov et al., 2025) [user-specified]
- **Method:** Gemma-2-2B with Gemma Scope residual SAEs (16k) on even layers 0–24. SAE activations are summed over tokens and fed to XGBoost. SAE features beat raw activations; **layer 16 generalises best** (0.84 macro-F1).
- **Interpretable layer-16 features:**
  - General: 3608 (sentence complexity), 4645 (assertive vs hedged), 6587 (long-winded intros), 14161 (formality).
  - GPT-specific: 8264 (repetition; F1 ≈ 1.0 for GPT-3.5/4/4o), 8689 (synonym swaps, length-sensitive).
  - Domain-specific: 12390 (arXiv), 4560 (Reddit), and others.
  - Length: 1033, 16028.
- **Steering:** adds λ·maxact·decoder for λ ∈ [−4, 4]. Results are judged only qualitatively by GPT-4o, with no detector, content or coherence metric. At ±4 the outputs hallucinate or loop.
- **Caveats:** dataset artefacts (spaces before commas in human text; `\n\n` in about 98% of GPT-4o texts). No code; a feature browser is at mgtsaevis.github.io.

### 2601.10387 — The Assistant Axis (Lu et al., 2026, Anthropic) [user-specified]
- **Models:** Gemma-2-27B-it (layer 22/46), Qwen3-32B (layer 32/64) and Llama-3.3-70B (layer 40/80).
- **Building the axis:** 275 roles × 5 system prompts × 240 questions. A judge filters role adherence (0–3). Role vectors average the residual stream over **response tokens**. **Axis = mean(default-Assistant activations) − mean(fully role-playing role vectors)**. It aligns with PC1 of persona space (cos > 0.71), and PC1 loadings correlate > 0.92 across models. The axis exists already in base models.
- **Steering:** add at one middle layer, scaled as a fraction of the mean residual norm (about ±0.15 for Gemma, ±0.8 for Qwen/Llama). Pushing away from the Assistant makes the model claim to be human and write dramatic, mystical prose. The Assistant end correlates with calm, measured, grounded traits.
- **Activation capping:** at about 25th-percentile projections, capping halves persona-jailbreak success with no capability loss.
- Code and roles are in `code/assistant-axis`. Vectors (27B+ models only) are at HF `lu-christina/assistant-axis-vectors`.
- **Why it matters:** an "AI-text" direction could largely be this axis, so it is the top confound to test.

### 2605.12890 — Steer-to-Detect (Liang & Li, 2026)
- **Method:** learns a vector added at layer 11 of Llama-3.1-8B *while it reads* text, to sharpen human/AI separation. This is a detector and does not steer generation.
- **Results:** a plain logreg on unsteered pooled states already gets 0.96–0.99 AUROC. LLM polishing collapses TPR@1% from 97% to 24%. The tokens the vector moves most are formal words.

### Short notes on other papers
- **2410.16107 Reinhart et al. (HAP-E).** Parallel corpus: a human prefix, then the true human continuation vs. LLM continuations. Instruct models overuse participial clauses (2–5×), nominalisations (about 2×) and long words, plus "tapestry"/"palpable". **Base Llama-3 ≈ human**, so the "AI style" is a product of instruction tuning.
- **2505.24523 Pedrotti et al.** DPO toward human text: a single iteration drops Binoculars F1 from 0.99 to 0.33 on news, and humans can't tell either. This is the fine-tuning baseline analogue of our steering.
- **2502.12150 Idiosyncrasies.** Source LLMs are identifiable mostly from word-level distributions, and this persists through rewriting. Lexical cues alone carry a lot of signal, so steering might only toggle a small vocabulary.
- **2401.12070 Binoculars / 2310.05130 Fast-DetectGPT / 2301.11305 DetectGPT.** Zero-shot, perplexity-based detectors. They are *independent* of our direction, which makes them good evaluators. Caveat: steering also changes perplexity, so a pure perplexity shift could fool them without any change in "style". Use supervised RoBERTa-type detectors as well.
- **2405.07940 RAID, 2305.13242 MAGE, 2301.07597 HC3.** The benchmarks we downloaded. RAID shows detectors are brittle to decoding strategy (sampling, repetition penalty), which itself shifts detector scores. Steering experiments must therefore fix decoding.
- **2308.10248 ActAdd, 2312.06681 CAA, 2310.01405 RepE.** Standard recipes: a mean-difference vector added at one layer (all positions) with a coefficient sweep.
- **2406.11717 Refusal direction.** Directional ablation (project out at every layer) vs. activation addition. Selects the layer/position by causal effect, with KL on harmless prompts as a side-effect guard. This is the methodology for "removing" the AI direction.
- **2507.21509 Persona vectors.** Contrastive system prompts → response-averaged mean-diff vector; steering coefficient sweeps; projection predicts trait shifts. Template for building confound vectors (e.g. "formal", "verbose").
- **2311.03658 LRH, 2310.06824 Geometry of Truth.** Mean-difference ("mass-mean") directions are more *causal* than logreg directions, even though logreg classifies better. Use **diff-of-means for steering**, logreg for detection.
- **2501.17148 AxBench.** Steering evaluation uses three judge scores (concept, instruction relevance, fluency; 0–2) combined by harmonic mean. DiffMean is a strong detector, prompting beats every steering method, and SAEs are weak. **Prompting ("write like a human") is a mandatory baseline.**
- **2606.12234.** Activation steering is far less effective on **instruct** than base models and costs fluency; cheap text metrics correlate with an LLM judge.
- **2607.01802.** Writing-style steering on Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct is trait-dependent and transfers poorly to downstream writing tasks. Expect modest effects.
- **2402.01618 Style vectors.** Activation-mean style vectors steer sentiment, emotion and style smoothly.
- **2306.03819 LEACE.** Closed-form linear erasure, usable both to ablate the AI concept and to erase confounds (length, formality) before re-fitting.
- **2503.08716 AuthorMist.** RL paraphraser that evades detectors: a "humanizer" upper bound.
- **2306.04723 Intrinsic dimension.** AI text has about 1.5 lower intrinsic dimension of its embedding manifold.

## 3. Common methodologies
- **Direction extraction:** diff-of-means (CAA, persona vectors, assistant axis, SV-Detect option); logistic-regression normal (2608.24780, SV-Detect default); PCA of paired diffs (RepE, repeng); SAE features (2503.03601).
- **Pooling:** last token (best for 2608.24780 detection) vs. mean over tokens (SV-Detect, assistant axis over response tokens). For steering, response-token means are the norm.
- **Layer:** about 40–60% depth is consistently best: L16/32 in Llama-3-8B, L16/26 in Gemma-2-2B, middle layers in the assistant axis.
- **Intervention:** add α·v at one layer at all positions, with α scaled relative to the mean residual norm; or ablate h − (h·v̂)v̂ at all layers.

## 4. Standard baselines
- **For causal steering:** a random direction of equal norm; prompting ("Write in a natural, human style", or "You are a human writer"); confound directions (formality, length/verbosity, assistant axis) steered at the same norm; SAE-feature steering (Gemma Scope 8264/6587/14161); and no-steer.
- **For detection, as a sanity check that the direction is real:** RoBERTa detectors, Binoculars, Fast-DetectGPT, and a regex/lexical cue baseline.

## 5. Evaluation metrics
- **AI-ness of steered outputs:** at least two *independent* detectors with different principles.
  - Zero-shot: Binoculars (Falcon-7B pair), Fast-DetectGPT.
  - Supervised: `desklib/ai-text-detector-v1.01`, `fakespot-ai/roberta-base-ai-text-detection-v1`, `Hello-SimpleAI/chatgpt-detector-roberta` (trained on HC3, so not independent if the direction came from HC3), `openai-community/roberta-large-openai-detector` (GPT-2 era).
  - Report mean P(AI) and AUROC vs. human references.
  - Optionally an LLM judge: "which text was written by a human?"
- **Content preservation:** embedding cosine (e.g. all-MiniLM / e5) between steered and unsteered answers to the same prompt; an LLM-judge relevance score (AxBench-style instruction score).
- **Coherence/fluency:** perplexity under a *separate* LM; distinct-n / repetition rate; an LLM-judge fluency score (0–2).
- **Confounds, measured on outputs:** length (tokens); formality (`s-nlp/roberta-base-formality-ranker`); lexical "AI-isms" counts (delve, tapestry, moreover, em-dash, markdown bullets); readability (Flesch).
- **Geometry:** cosine between directions per layer; AUROC of each direction before and after projecting out the others (LEACE); variance explained.

## 6. Datasets in the literature
RAID (2608.24780, 2606.07313, 2503.03601), DetectRL (2608.24780, 2606.07313, 2605.12890), MAGE, HC3, MIRAGE, HAP-E (2410.16107), COLING-25 GenAI (2503.03601). We have HC3, HAP-E, MAGE and a parallel RAID subset locally.

## 7. Gaps and opportunities
1. No paper causally steers *generation* with a human-vs-AI direction and evaluates the result with independent detectors and quality controls.
2. No paper quantifies overlap with formality, length, domain or **assistant-axis** directions. Several papers' own interpretation analyses point to formality.
3. The base-vs-instruct asymmetry (direction fails on base-model text; base LMs ≈ human style) suggests the direction is a *post-training* signature, possibly the assistant persona. This is untested.
4. Prior steering is evaluated only qualitatively (2503.03601).

## 8. Direction ranking (direction budget: keep top 3)

Score is 1–5 on each of evidence, relevance, information gain and feasibility.

| # | Direction | Evid. | Relev. | Info gain | Feasib. | Total | Decision |
|---|---|---|---|---|---|---|---|
| D1 | **Causal steering test.** Extract a content-matched diff-of-means human-vs-AI direction in an instruct model (HAP-E + HC3), add/subtract it during generation, and score with independent detectors plus content/coherence metrics. Baselines: random, prompting, no-steer. | 5 | 5 | 5 | 4 | 19 | **KEEP** |
| D2 | **Disentanglement.** Build formality, length/verbosity, domain and assistant-axis (cheap rebuild) directions plus a base-vs-instruct direction; compute per-layer cosines; LEACE/project them out and re-test both detection AUROC and steering efficacy of the residual. | 5 | 5 | 5 | 4 | 19 | **KEEP** |
| D3 | **Generality of the direction.** Transfer across datasets (HC3↔HAP-E↔RAID↔MAGE), domains and generators; base vs instruct reader/generator; does the projection track a continuum (paraphrased-human, AI-edited)? | 4 | 4 | 4 | 5 | 17 | **KEEP** (cheap; supports D1/D2 interpretation) |
| D4 | SAE decomposition (Gemma Scope): which features make up the direction; SAE steering baseline | 3 | 3 | 3 | 4 | 13 | Pruned. AxBench shows SAE steering is weak. Optionally use feature 14161/8264 steering as an extra baseline within D1 if time allows. |
| D5 | DPO/RL "humanizer" fine-tuning comparison | 4 | 3 | 2 | 2 | 11 | Pruned. Expensive and answers a different question (training vs. a direction). Cite 2505.24523 as the reference effect size. |
| D6 | Logit-lens / lexical interpretation of the direction | 4 | 3 | 2 | 5 | 14 | Pruned as a standalone direction; fold in as a cheap diagnostic inside D2 (is it just a "delve" knob?). |
| D7 | Human evaluation of steered text | 3 | 4 | 3 | 1 | 11 | Pruned (infeasible). Use an LLM judge as a proxy. |

## 9. Recommendations for the experiment

**Subject models.** A 2B and a 7B instruct model, each with its base model:
- **Gemma-2-2b-it / gemma-2-2b.** Fast, and Gemma Scope SAEs exist; reuse the 2503.03601 features as confound/interpretation aids.
- **Qwen2.5-7B-Instruct / Qwen2.5-7B.** Strong, ungated, and used in 2607.01802.

Llama-3.1-8B-Instruct is also accessible (gated access works). All are pre-fetched to the HF cache.

**Direction data.** Content-matched pairs are essential:
- **HAP-E:** human chunk-2 vs. the Llama-3-8B-Instruct / GPT-4o continuation of the same chunk-1 (genre fixed).
- **HC3:** human vs. ChatGPT answer to the same question; normalise HC3 tokenisation (`" ,"`, `"n't"`).
- **Self-generated pairs:** the subject model's own answers vs. human answers, so the direction is "in-distribution" for steering.
- **Hold-out:** RAID (8 domains) and MAGE for transfer.
- Length-match or truncate to equal token counts (e.g. 128–256 tokens).

**Extraction.** Diff-of-means over response/text tokens (mean pooling) at about 40–60% depth. Sweep layers, and select the layer by *steering effect*, not by AUROC alone. Also fit a logreg direction and compare the two.

**Intervention.**
- Add α·v̂·‖h‖_mean at the chosen layer, with α ∈ {−8…+8}·(fraction of norm), for example ±{0.05, 0.1, 0.2, 0.4}.
- Directional ablation (project out at all layers) as the "remove" arm.
- Fixed greedy or fixed-seed sampling decoding; 100–300 held-out prompts, e.g. HC3/ELI5 questions plus writing prompts.

**Evaluation.**
- **Detectors:** Binoculars + Fast-DetectGPT + one supervised RoBERTa detector that was *not* trained on the direction's source data (desklib or fakespot).
- **Quality:** content similarity to the unsteered output, perplexity under a separate LM, and an LLM-judge fluency/relevance score.
- **Confounds on outputs:** length, formality score, AI-ism counts.
- **Statistics:** bootstrap CIs; paired tests (Wilcoxon) per prompt.

**Confound directions for D2.**
- **Formality:** formal vs. informal text pairs scored by `s-nlp/roberta-base-formality-ranker`, or a persona-vector-style "formal" vs "casual" system prompt.
- **Length/verbosity:** long vs. short answers.
- **Domain:** HC3 source labels.
- **Assistant axis:** about 50 roles from `code/assistant-axis/data/roles`, default prompt vs. role prompts over response tokens.
- **Base-vs-instruct:** same text read by the base model vs. the instruct model, or base-model continuation vs. instruct continuation in HAP-E.

Report the cosine matrix and steering with the residualised direction (AI direction ⟂ confounds).

**Pitfalls.**
- Separability is expected (AUROC ≈ 1), so it is not evidence of anything.
- Perplexity-based detectors can be fooled by fluency loss. Always pair them with quality metrics and a supervised detector.
- Steering is weaker on instruct models (2606.12234), so sweep larger α.
- Prompting may beat steering (AxBench), so report it honestly.
