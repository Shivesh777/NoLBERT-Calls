# NoLBERT-Calls-2023 (release)

A DeBERTa-v3-base masked language model (109M parameters) with its masked-LM head. The weights are stored in fp16 and split into 45 MB parts.

## Start and training

- **Base model:** [alikLab/NoLBERT](https://huggingface.co/alikLab/NoLBERT), pretrained on text from 1976–1995.
- **Stage 1:** adapted on earnings calls up to 2012-12-31, from all countries, for 480 minutes (`pipeline/c04_finetune.py`).
- **Stage 2:** adapted on US calls up to 2023-12-31 in two 150-minute runs on one T4, the second continuing the first (`training/part_b_adaptation.py`).
- **Objective:** InfoNCE with temperature 0.05, plus masked-LM loss.
- **Stage-2 batches:** packed with up to 32 question→answer exchanges from the same call. Answers are cut at 256 tokens and questions at 128.
- **Checkpoint selection:** on 300 US calls from 2024. The released checkpoint scored 80.7% validation top-1, at step 3,722, the last step of the second run.

## Using it

- **Hindsight.** The model has read no text dated after 2023-12-31, so it can be used on calls from 2024 onwards without look-ahead from its training data.
- **Embedding recipe.** Take the mean of the last hidden layer over all non-padding tokens, then L2-normalise (see `embedding_config.json`). `nolbert_calls/model.py` applies this for you.
- **Test score.** On all US calls from January 2025 to October 2026, within-call answer retrieval reaches 78.0% top-1, or 79.0% when blended with word overlap. The main README has the other benchmarks.

```bash
python model/merge_model.py          # -> model/nolbert_calls_2023/model.safetensors (checksum-verified)
python -m nolbert_calls.model --demo
```

## Licence and data

Research use only. The training text is licensed (S&P Capital IQ via WRDS) and is not distributed here.
