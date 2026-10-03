# Is there a "sounds like AI" direction in the residual stream?

We looked for a human-vs-AI text direction (d_AI) in **Qwen2.5-7B-Instruct** and its base model **Qwen2.5-7B**, and asked three questions:
- Is it *causal* for generation? Does steering it make the model's own output read as more or less AI-written to independent detectors, with content and coherence held?
- Is it distinct from formality, verbosity, domain, fluency and the Assistant persona?
- Does it exist in the base model?

Full report: **[REPORT.md](REPORT.md)**.

## Key findings
- **Read-out.** One diff-of-means direction separates content-matched human/AI pairs (AUROC 0.89–0.91 at block 20) and transfers to unseen *chat* generators (0.78–0.90). It barely separates *base*-LM text (0.58), so it behaves like a "chat-tuned style" direction.
- **Causal and specific.** Steering toward "human" (α = −0.2 × residual norm, block 20):
  - lowers the supervised detectors: desklib P(AI) 0.996 → 0.84, fakespot 0.994 → 0.85;
  - raises Binoculars from 0.70 to 0.83 (more human);
  - equal-norm random directions do nothing on the supervised detectors (0.99);
  - the effect holds among outputs judged perfectly fluent and on-topic: 0.85 vs. 0.99–1.00 for random and 0.98 for formality; block 16 reaches 0.67;
  - a "write like a human" prompt does not move desklib at all.
- **Distinct.**
  - Whitened cosine with formality, fluency, domain, verbosity and the Assistant axis is ≤ 0.05; with the Assistant axis it is 0.005.
  - The version with those confounds projected out steers just as well.
  - Steering the Assistant axis moves detectors only by wrecking the text.
  - The only large overlap is with an instruct-minus-base "post-training" direction (raw cos 0.90).
- **Base vs. instruct.**
  - The base-model direction is nearly identical (cos 0.99), reads out at least as well, and steers the base model in **both** directions (desklib 0.94 → 0.87 at −0.2; fakespot 0.77 → 0.91 at +0.3).
  - Chat tuning does not create or amplify the direction. It moves the model's own outputs further along it.
- **Weak lever (partial negative result).**
  - At preserved coherence, detectors move only partway: human references score 0.35.
  - The unsteered model reads steered outputs as human-level on d_AI, yet detectors still flag them.
  - In-distribution clamping to the human mean has small (though specific) effects.
  - So a single direction does not make the model's output read as human.
- **Measurement split.** A blind Cohere LLM judge (weakly valid: AUROC 0.71 on real human vs. ChatGPT answers) also rates d_AI-steered text as less AI-like (−6 points vs. unsteered, −4 vs. random). On it, though, formality steering (−8) and the "write like a human" prompt (−7) work as well. Specificity holds only for the trained detectors.
- **Measurement note.** A local Llama-3.1-8B judge is at chance (AUROC 0.48) on real human vs. ChatGPT answers. It is used for fluency only.

## Reproduce
```bash
uv venv && source .venv/bin/activate && uv sync      # or: uv pip install -r pyproject deps
cd src
python build_data.py && python extract_acts.py && python score_corpus.py && python confound_gen.py
python directions.py && python disentangle.py && python fig_readout.py           # E1, E2
python steer.py pilot && python steer.py main && python steer.py base && python steer_extra.py   # E3, E4, E3b
for f in gen_main gen_base gen_extra; do python score_gens.py $f.json; python readback.py scores_$f.parquet; python judge.py scores_$f.parquet local; done
python judge.py scores_gen_main.parquet cohere '<cond regex>' 50                 # needs COHERE_API_KEY
python analyze_steer.py && python analyze_extra.py && python analyze_validity.py && python fig_final.py
```
Requirements:
- 1 GPU with ≥ 40 GB (Binoculars loads two Falcon-7B models).
- HF access to `meta-llama/Llama-3.1-8B-Instruct` for the local judge.
- Datasets in `datasets/` (see `datasets/README.md`).

## Layout
- `src/`: all code (see the table in REPORT §8).
- `results/`: generations (`gen_*.json`), per-text scores (`scores_*.parquet`, `merged_gen_main.parquet`), judge outputs (`judge_*.jsonl`), summary CSVs/JSONs.
- `figures/`: `fig1_steer_dose_response.png`, `fig2_matched_coherence.png`, `fig3_base_and_clamp.png`, `readout_layers_transfer.png`, `disentangle_cosines.png`.
- `planning.md`: plan and direction ranking. `literature_review.md`, `resources.md`: background.
