# Research Plan: Is there a "sounds like AI" direction in the residual stream?

## Motivation & Novelty Assessment

### Why This Research Matters
Several papers show that human-written and AI-written text can be read out of LLM activations with a linear probe. A direction that can also *write* (add it and text sounds more machine-like; remove it and text sounds more human) would matter in three places:
- **Detection and evasion.** It would be a one-vector humanizer, and it would show whether detectors key on a single internal feature.
- **Interpretability.** It would tell us whether "sounding like AI" is a distinct concept or a relabelling of formality, domain, verbosity or the Assistant persona.
- **Post-training.** It would show whether chat tuning creates or amplifies the signature.

### Gap in Existing Work
From `literature_review.md`:
- 2608.24780, SV-Detect (2606.07313) and Steer-to-Detect (2605.12890) only *read* the direction, for detection.
- 2503.03601 steers SAE features but judges the result only qualitatively.
- The Assistant Axis (2601.10387) is a persona direction and was never related to text-authorship style.
- No prior work has:
  - steered generation with a human-vs-AI direction and scored the result with independent detectors under coherence and content controls;
  - quantified the direction's overlap with formality, domain, perplexity, verbosity and the assistant axis;
  - compared the direction between a base model and its instruct model.

### Our Novel Contribution
A controlled causal test, plus a disentanglement and a base-vs-instruct comparison, all on Qwen2.5-7B-Instruct and Qwen2.5-7B:
1. Extract a content- and length-matched diff-of-means "AI-ness" direction (HAP-E + HC3, external generators).
2. Steer or ablate it during generation. Score the outputs with three independent AI-ness measures: a supervised DeBERTa detector, zero-shot Binoculars, and an LLM judge. Hold content (embedding similarity, judge relevance) and coherence (judge fluency, Falcon perplexity) fixed.
3. Compare against equal-norm random, formality, assistant-axis and residualised directions, a prompting baseline, and matched-coherence comparisons.

### Experiment Justification
- **E1 Read-out and generality.** Is a linear direction present at all? Which layers? Does it transfer across datasets and generators (RAID, MAGE)? These are prerequisites for steering. They are also a sanity check that our direction is not a tokenisation or length artefact.
- **E2 Disentanglement.** Cosines with the confound directions; AUROC after projecting confounds out; covariate-controlled regression. This tells us whether "AI-ness" is a new axis or a relabelling.
- **E3 Causal steering (main).** The only way to show the direction *controls* writing. Baselines make the claim specific:
  - random direction: rules out a generic perturbation effect;
  - formality and assistant-axis directions: is it just a confound?
  - prompting: is steering better than asking?
  - matched coherence: does it just break the text?
- **E4 Base vs instruct.** Does the direction exist in the base model? Is it amplified by chat tuning? Does it also steer the base model?

## Research Question
Is there a residual-stream direction in an instruct LLM that (a) separates human from AI text and (b) causally controls whether the model's own output reads as AI-written to independent detectors, with content and coherence held? Is it distinct from formality, domain, fluency/perplexity, verbosity and the assistant persona? Is it present in the base model?

## Hypothesis Decomposition
- **H1 (read-out).** A diff-of-means direction from matched pairs gives test AUROC > 0.9 at mid layers and transfers to RAID/MAGE above chance.
- **H2 (causal).** Subtracting the direction lowers detector P(AI) and judge AI-likeness on ≥2 of 3 measures. The decrease is significantly larger than for a random direction of equal norm, at matched coherence (judge fluency within 0.5 of unsteered). Adding it raises AI-ness.
- **H3 (distinct).** |cos| with each confound direction < 0.5. Projecting out the confounds leaves AUROC > 0.85. The residualised direction still steers.
- **H4 (base).** The direction is linearly readable in the base model (AUROC > 0.9). Its separation is larger in the instruct model. The base model's own outputs project closer to human than the instruct model's outputs do.

Each hypothesis can fail independently, and every failure is reportable.

## Proposed Methodology

### Models
- **Subject model:** Qwen2.5-7B-Instruct, with Qwen2.5-7B as the base model (ungated, cached, 28 layers, d=3584). bf16 on 1× A6000.
- **Optional replication:** gemma-2-2b-it, if time permits.

### Experimental Steps
1. **Data.** Build content- and length-matched pairs:
   - HAP-E: human chunk-2 vs GPT-4o and Llama-3-70B-Instruct continuations of the same prefix.
   - HC3: human vs ChatGPT answers to the same question. Normalise HC3 tokenisation artefacts.
   - Truncate each pair to the same token count (≤128 Qwen tokens).
   - Split by document/question (train/test). Hold out 150 HC3 questions, never used for directions, as steering prompts.
   - Transfer sets: RAID subset (11 generators, 8 domains, paraphrased human); MAGE test and OOD (GPT-4, paraphrased).
2. **Activations.** Read raw text (no chat template). Mean-pool the residual stream over text tokens at every layer, for the instruct and base models.
3. **Directions** (diff-of-means, per layer):
   - d_AI: AI − human, pooled over HAP-E and HC3 train.
   - Per-source variants (d_hape, d_hc3) and a logistic-regression direction.
   - Confounds (from the same model):
     - formality: human-only, top vs bottom tercile of s-nlp formality ranker within stratum;
     - domain: HC3 human, reddit_eli5 vs the other sources;
     - fluency: human-only, high vs low mean log-prob within stratum;
     - verbosity: Qwen responses under "be extremely brief" vs "be detailed" system prompts;
     - assistant axis: Qwen responses as the default assistant vs 30 role-play system prompts × 10 questions, response tokens (cheap rebuild of Lu et al.);
     - post-training style: HAP-E Llama-3-8B-Instruct vs Llama-3-8B base continuations.
4. **E1/E2 analyses.**
   - Per-layer AUROC.
   - Cross-dataset transfer (fit on X, test on Y).
   - Cosine matrix.
   - AUROC after projecting out the confound subspace, with re-fit.
   - Logistic regression of the label on text covariates (formality score, log-ppl, length, domain) with and without the d_AI projection.
5. **Layer and scale pilot.** On 30 separate pilot prompts, steer at layers {10, 14, 18} with α·‖h‖. Pick the layer and α grid where coherence degrades gracefully. Chosen before running the main test prompts.
6. **E3 main steering.** 150 held-out HC3 questions, chat template, sampling T=0.7, top_p=0.95, 160 new tokens. Same seed per condition. Conditions:
   - no-steer;
   - d_AI at −4 levels / +2 levels;
   - random at the same 4 negative norms;
   - formality(−, toward informal) at 4 levels;
   - assistant axis (−, away from Assistant) at 4 levels;
   - residualised d_AI (⟂ confounds) at 4 levels;
   - directional ablation of d_AI at all layers;
   - prompting: "write like a human" system prompt.
7. **E4 base vs instruct.**
   - Per-layer AUROC in the base model.
   - Cosine between base and instruct d_AI (the residual bases are related because Instruct was finetuned from base).
   - Base-model answers (completion format) vs instruct answers vs human answers: projection and detectors.
   - Steer the base model with its own d_AI at ±.

### Baselines
No-steer; equal-norm random direction; formality direction; assistant-axis direction; prompting ("write like a human"); human reference answers (HC3 human answers to the same questions).

### Evaluation Metrics
- **AI-ness**, three independent measures, none trained on our activations:
  1. desklib/ai-text-detector-v1.01 P(AI). Supervised DeBERTa-v3-large, not trained on HC3/HAP-E as far as we know.
  2. Binoculars score (Falcon-7B / Falcon-7B-instruct). Zero-shot, different model family. Lower = more AI.
  3. LLM judge (Cohere `command-a-plus-05-2026`): 0–100 likelihood that the text is AI-written.
  4. Also fakespot RoBERTa as a secondary detector.
- **Content:** cosine similarity of all-mpnet-base-v2 embeddings to the unsteered answer and to the question; judge relevance 1–5.
- **Coherence:** judge fluency 1–5; Falcon-7B perplexity; distinct-2 / repetition.
- **Confounds on outputs:** word count, formality score, AI-ism lexicon rate, markdown rate, contraction rate.

### Statistical Analysis Plan
- Per-prompt paired differences vs no-steer. Bootstrap 95% CIs (2000 resamples). Wilcoxon signed-rank with Holm correction across conditions.
- Steering vs random at equal α: paired test.
- **Matched coherence.** For each method, the strongest α whose mean judge fluency is ≥ (no-steer − 0.5) and whose mean Falcon log-ppl is not more than 0.5 nats above no-steer. Compare detector shifts at that point. Also plot AI-ness vs coherence curves.
- α = 0.05.

## Expected Outcomes
- **Supports the hypothesis:** subtracting d_AI drops desklib P(AI) and judge AI-likeness well beyond random at matched coherence; residualised d_AI keeps most of the effect.
- **Refutes it:**
  - effects no larger than random, or only once coherence collapses (a negative "reads ≠ writes" result);
  - or formality/assistant-axis directions do as well, and residualised d_AI loses the effect (the "relabelling" result).

## Timeline
- Data + activations: 1 h
- Directions/E1/E2: 1 h
- Pilot + steering generation: 1.5 h
- Scoring (detectors, Binoculars, judge): 1.5 h
- E4: 0.5 h
- Analysis + report: 1.5 h

## Potential Challenges
- **OpenRouter daily key limit already exhausted** (403 at start). OPENAI_API_KEY is invalid. Fallback: Cohere API for the judge (documented deviation).
- Steering may be weak on instruct models (2606.12234). Sweep α up to the point where coherence breaks.
- Binoculars/perplexity detectors are confounded by fluency loss. Mitigation: a supervised detector plus the judge, and matched coherence.
- HC3 ChatGPT answers vs Qwen style differ. That is fine: the generator transfer is part of the claim.

## Success Criteria
All four experiments are run with controls, and REPORT.md answers H1–H4 with CIs, including negative results.

## Direction budget (from literature_review.md §8)
- **Kept:** D1 causal steering (E3), D2 disentanglement (E2), D3 generality (E1) + base-vs-instruct (E4, part of D2/D3).
- **Pruned:**
  - D4 SAE decomposition: AxBench shows SAE steering is weak, and our model has no SAEs.
  - D5 DPO humanizer: expensive, and it answers a different question.
  - D6 standalone logit lens: folded in as a token-level diagnostic only.
  - D7 human eval: infeasible.
