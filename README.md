# NoLBERT-Calls

NoLBERT-Calls is [NoLBERT](https://huggingface.co/alikLab/NoLBERT) adapted to US earnings-call transcripts up to a fixed date. NoLBERT is a DeBERTa-v3-base model pretrained only on text from 1976–1995. Neither its pretraining text nor its adaptation text goes past the cutoff, so the model reads later calls without hindsight, which makes it usable for backtests that start after the cutoff.

The released model, **NoLBERT-Calls-2023**, has read no text dated after 2023-12-31.

## Results

Except for the time test, every result is on US calls from January 2025 to October 2026, all after the cutoff. All model choices were made on 300 US calls from 2024, never on these calls.

| Task | Metric | NoLBERT-Calls-2023 |
|---|---|---|
| **Within-call answer retrieval** (20,323 calls, 254,575 analyst questions; chance 8.0%) | top-1 [95% CI] | **78.0** [77.8, 78.2] |
| | top-1, blended with word overlap | **79.0** [78.8, 79.2] |
| | top-3 / MRR | 94.0 / 86.4 |
| Passage → call matching (1,000 calls) | top-1 / top-10 | 7.6 / 21.9 |
| Speaker role, analyst question vs executive answer (frozen probe) | balanced accuracy | 96.2 |
| Masked finance terms (100 terms in 3,000 passages) | top-1 / top-5 | 25.6 / 47.1 |
| SubjECTive-QA, frozen probe (6 properties × 3 author splits) | weighted F1 (majority class 55.6) | 64.8 |
| SubjECTive-QA, fine-tuned heads (6 properties × 3 author splits) | weighted F1, mean (sd over splits) | 65.8 (1.5) |
| GLUE dev | CoLA (MCC) / SST-2 / QNLI | 31.9 / 86.8 / 88.2 |
| **Time test** (500 US calls a year 2013–24, plus the 2025–26 test) | top-1 in 2013 / 2023 / 2026 | 73.7 / 78.0 / 78.3 |
| | lead over stage 1 (calls up to 2012): on 2013–23 calls, which it trained on / on 2025–26 calls, which it never saw | +5.3 / +6.7 |

How to read the table:
- **Answer retrieval.** For each analyst question, the model ranks every management answer in the same call. Top-1 is the share of questions whose own answer comes first.
- **Word overlap.** The blend adds z-scored TF-IDF similarity, with its weight chosen on the 2024 validation calls.
- **Time test.** The released model gains nothing on the years it trained on. Its lead over its own stage 1, which stops at 2012, is larger on 2025–26 calls (+6.7) than on 2013–23 calls (+5.3), so its test score is not inflated by look-ahead.
- **For reference:** the base NoLBERT, before adaptation, scores 26.3% top-1.
- **Full tables:** `results/tables/tables.md`.

## How it was trained

1. **Stage 1** (`pipeline/c04_finetune.py`): NoLBERT adapted on earnings calls from all countries up to 2012-12-31, for 480 minutes on one RTX 3060.
   - The objective is InfoNCE (temperature 0.05) on analyst question → management answer pairs and on pairs of passages from the same call, plus masked-LM loss.
   - The kept checkpoint is the one with the best answer retrieval on 300 calls from 2013–15.
2. **Stage 2** (`training/part_b_adaptation.py`): two 150-minute runs on one T4 on US calls up to 2023-12-31, the second continuing from the first.
   - The objective is the same.
   - Batches are packed with up to 32 exchanges from the same call, so the call's other answers are hard negatives.
   - Answers are read up to 256 tokens and questions up to 128.
   - Checkpoints are chosen on the 300 validation calls of 2024. The released checkpoint scored 80.7% validation top-1.

Details are in `model/README.md` and `training/README.md`.

## Use the model

The weights are split into 45 MB parts, because GitHub limits files to 100 MB. They are merged on first use, or by hand with `python model/merge_model.py` (or `cat model.safetensors.part-* > model.safetensors`).

```python
from nolbert_calls import NoLBERTCalls
m = NoLBERTCalls("model/nolbert_calls_2023")
q = m.embed(["How should we think about gross margin next year?"], side="question")
a = m.embed(["We expect gross margin to improve as input costs ease ..."], side="answer")
print(q @ a.T)                                   # cosine similarity
m.rank_answers(question, answers)                # best-matching answer first
m.fill_mask("Revenue grew on higher<mask> in the quarter.")
```

`python -m nolbert_calls.model --demo` runs a small invented example. The model needs `torch` and `transformers` (5.x).

## Repository

| Path | What it holds |
|---|---|
| `pipeline/` | Data and stage 1. `c01_download.py` downloads from WRDS Capital IQ, `c02_clean.py` cleans, `c03_pairs.py` builds question→answer exchanges and passages, `c04_finetune.py` is stage 1, and `c05_bundle.py` builds the US-only tables that stage 2 and the evaluation read. |
| `training/` | The stage-2 trainer and the benchmark-head trainer, with the released recipe in `part_b_config.json` |
| `nolbert_calls/` | `model.py` (use the released model), `run.py` (stage 2 and heads), `evaluate.py`, `report.py`, `common.py` |
| `model/` | The released NoLBERT-Calls-2023, with `merge_model.py` and `pack_model.py` |
| `results/tables/` | The released model's results (aggregate numbers only) |

## Reproduce

```bash
pip install -r requirements.txt
# 1. data (needs a WRDS account; writes data/, which is git-ignored)
python pipeline/c01_download.py --username YOUR_WRDS_ID
python pipeline/c02_clean.py && python pipeline/c03_pairs.py
for step in "meta" "qa 2004 2026" "passages 2004 2026" "turns 2004 2026" "role"; do
  python pipeline/c05_bundle.py $step; done
# 2. stage 1: calls up to 2012 (480 min on one 12 GB GPU)
python pipeline/c04_finetune.py --grad-ckpt --budget-min 480 --out data/encoders/nolbert_calls_2012
# 3. stage 2: US calls up to 2023, two 150-minute runs (one T4)
python -m nolbert_calls.run adapt --name nolbert_calls_2023_run1 --init nolbert_calls_2012
python -m nolbert_calls.run adapt --name nolbert_calls_2023 --init adapted:nolbert_calls_2023_run1
# 4. evaluation, heads and tables
python -m nolbert_calls.evaluate --readers nolbert_calls_2023 --out results/eval
python -m nolbert_calls.evaluate --readers nolbert_calls_2012 --tasks retrieval time --out results/eval_stage1  # time-test reference
for t in subjective cola sst2 qnli; do python -m nolbert_calls.run heads --task $t --models nolbert_calls_2023; done
python -m nolbert_calls.report --eval results/eval results/eval_stage1 --heads partb_outputs --out results/tables
```

Notes:
- **Without step 3,** `evaluate` and `heads` use the released model in `model/`.
- **SubjECTive-QA** comes as the authors' splits, saved as `sqa_<seed>_<split>.parquet` next to the bundle. The heads can also download them from Hugging Face; the dataset is gated, so accept access and log in first.

## Authors

Shivesh Prakash, Swapnil Vatsal, Hino Wirachapong Suwanphibun, Kevin Foley and Jonathan Hartadi.

## Data and licence

- **Code:** MIT licence (`LICENSE`).
- **Transcripts:** S&P Capital IQ transcripts from WRDS are licensed. This repository contains no transcript text and no derived data.
- **Model weights** are trained on that text and are for research use.
- **SubjECTive-QA** (CC BY 4.0) and **GLUE** are downloaded from Hugging Face.
- **Base model:** NoLBERT (Kakhbod and Li).
