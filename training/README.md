# Training code

Stage 2 of NoLBERT-Calls was trained with `part_b_adaptation.py`, and the benchmark heads with `part_b_pipeline.train_head`. Run both through `python -m nolbert_calls.run`, which reads the bundle from `pipeline/c05_bundle.py`. The settings in `part_b_config.json` are the released model's recipe:

- **Objective:** in-batch InfoNCE (temperature 0.05) plus 1.0 × masked-LM loss, with 15% of tokens masked.
- **Batches:** 32 examples. 75% of batches are question→answer batches, packed with up to 32 exchanges from the same call so that the call's other answers are hard negatives. The rest pair two passages of the same call.
- **Input length:** questions are cut at 128 tokens and answers at 256.
- **Optimiser:** AdamW at 2e-5, weight decay 0.01, betas (0.9, 0.98), gradients clipped at 1.
- **Schedule:** 5% warm-up, then cosine decay over a 150-minute wall-clock budget, on one T4 in fp16.
- **Selection:** within-call answer retrieval on the 300 validation calls of 2024 (`pipeline/val_calls_main.json`), every 15 minutes. The best checkpoint is kept, and step 0 is eligible.

Stage 2 is two such runs. The second starts from the first run's selected checkpoint (`--init adapted:<run>`).

**Heads.**
- A fresh sequence-classification head is fitted and the whole encoder is updated.
- Optimisation: AdamW at 2e-5, weight decay 0.01, 10% warm-up, 3 epochs. The checkpoint with the lowest validation loss is kept.
- SubjECTive-QA uses the authors' three splits at 256 tokens.
- GLUE (CoLA, SST-2, QNLI) uses 128 tokens. It tunes on a grouped 10% split of the official training data and is reported on the official dev sets.
- Head weights are not kept; their metrics and predictions are.

Changes to the original code are marked `A (10 Nov)` in the source. Runs are registered by `nolbert_calls.run`, head options can be set per task, head checkpoints go to a scratch folder, and evaluation keeps only the logits.
