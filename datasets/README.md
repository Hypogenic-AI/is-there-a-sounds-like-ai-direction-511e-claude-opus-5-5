# Datasets

Data files are NOT committed to git (see `.gitignore`). Re-create with the instructions below.
Small example records are in `datasets/samples/`.

| Name | Local path | Size | Rows | Why we use it |
|---|---|---|---|---|
| HC3 | `datasets/hc3/` | 74 MB | 24,322 questions (58,546 human / 26,903 ChatGPT answers) | Same question answered by human & ChatGPT → **content-matched** pairs; 5 domains |
| HAP-E (Human-AI Parallel corpus, English) | `datasets/hape/text_data/` | 110 MB | 8,290 docs × 8 files | Human 500-word prefix + true human continuation vs. continuations by **base and instruct** Llama-3 8B/70B and GPT-4o(-mini) → best set for content/genre-matched direction and **base-vs-instruct (assistant-persona) control** |
| MAGE | `datasets/mage/` | 550 MB | train 319k / valid 57k / test 57k / OOD 1.5k + 2.4k | Large multi-domain, 27 generators incl. old base LMs; OOD test for direction generality |
| RAID (subset) | `datasets/raid_subset/raid_train_subset.parquet` | 36 MB | 48,182 | 8 domains × 11 generators (base + chat pairs: llama/mistral/mpt/cohere) + human, attack ∈ {none, paraphrase}; **parallel** (all generations of a sampled human doc kept) |

## HC3 — `Hello-SimpleAI/HC3`
- Format: JSONL; fields `question, human_answers[list], chatgpt_answers[list], source`. Sources: reddit_eli5 17,112; finance 3,933; medicine 1,248; open_qa 1,187; wiki_csai 842. Per-domain files also present.
- Download: `python datasets/download_small.py` (uses `huggingface_hub.snapshot_download("Hello-SimpleAI/HC3", repo_type="dataset", local_dir="datasets/hc3")`).
- Load: `rows = [json.loads(l) for l in open("datasets/hc3/all.jsonl")]`
- Notes: human answers (esp. ELI5) are tokenized with spaces before punctuation (" , ", "n't" split) — **normalise whitespace/punctuation before computing directions**, else the direction learns tokenisation artefacts. ChatGPT answers are longer: length-match or control.

## HAP-E — `browndw/human-ai-parallel-corpus`
- Files: `hape-text_human-chunk-1.parquet` (human prompt chunk, ~500 words), `hape-text_human-chunk-2.parquet` (true human continuation), and model continuations of chunk-1: `gpt-4o-2024-08-06`, `gpt-4o-mini-2024-07-18`, `llama-3-8B`, `llama-3-8B-Instruct`, `llama-3-70B`, `llama-3-70B-Instruct`. Columns `doc_id, text`; `doc_id` = `<genre>_<nnnn>@<source>` → join on the prefix before `@`.
- Genres (n): spok 1721, blog 1526, fic 1395, news 1322, acad 1227, tvm 1099.
- Download: `snapshot_download("browndw/human-ai-parallel-corpus", repo_type="dataset", local_dir="datasets/hape")` (in `download_small.py`).
- Load: `pd.read_parquet("datasets/hape/text_data/hape-text_llama-3-8B-Instruct.parquet")`
- Key use: human chunk-2 vs instruct-model continuation = AI-ness contrast with genre & topic held fixed; base vs instruct continuation = isolates post-training "assistant" style (Reinhart et al. 2410.16107 find base Llama-3 ≈ human, instruct ≠ human).

## MAGE — `yaful/MAGE`
- CSV columns `text, label, src`. **Label convention: 1 = human, 0 = machine** (verified: test has 28,741 label-1 rows = 28,741 `human` src rows). `src` encodes domain + generator (e.g. `cmv_machine_continuation_gpt-3.5-trubo`).
- Splits: train 319,071 / valid 56,792 / test 56,819 / `test_ood_set_gpt` 1,562 / `test_ood_set_gpt_para` 2,362 (GPT-4 + paraphrased).
- Download: `snapshot_download("yaful/MAGE", repo_type="dataset", local_dir="datasets/mage")` (in `download_small.py`).

## RAID subset — `liamdugan/raid` (full train.csv = 11.8 GB; we did not keep it)
- Re-create: `python datasets/raid_subset/download_raid_subset.py` (streams train.csv in 200k-row chunks, ~6 min). Keeps attack ∈ {none, paraphrase} and every row whose `source_id` (own `id` for human rows) has `md5 % 20 == 0` → 689 human docs + all their generations.
- Columns: `id, adv_source_id, source_id, model, decoding (greedy/sampling), repetition_penalty (yes/no), attack, domain, title, prompt, generation`.
- Counts (attack=none): human 689; chatgpt/gpt3/gpt4/cohere/cohere-chat 1,378 each; gpt2/llama-chat/mistral/mistral-chat/mpt/mpt-chat 2,756 each. Domains: abstracts, books, news, poetry, recipes, reddit, reviews, wiki.
- Parallel join: AI rows' `source_id` == human row `id`.
- Median words: human 231; chatgpt 264; gpt4 282; mistral-chat 200; mpt-chat 142 → length differs by generator, control it.
- Note: human rows with attack=paraphrase exist (641) — paraphrased human text is a useful "AI-edited human" control.

## Sanity checks done
All files load; row counts above verified; no empty-text issues noticed in samples. Full EDA left to the experiment phase.
