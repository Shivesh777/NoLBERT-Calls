"""The stage-2 trainer of NoLBERT-Calls: joint contrastive (InfoNCE) + masked-LM adaptation with the
model's native masked-LM head. Choices the recipe leaves open are recorded in training/part_b_config.json
(recipe.implementation_choices)."""
from __future__ import annotations

import contextlib
import gc
import json
import math
from pathlib import Path
import random
import sqlite3
import time

import numpy as np
import pandas as pd

from part_b_pipeline import (ADAPTATIONS, config_hash, file_hash, model_reference, load_tokenizer,
                             read_table, require, run_record, validate_splits, write_json)


BUNDLE_VERSION = "part-b-audit-v5"


def source_files(path):
    path = Path(path)
    require(path.exists(), f"Missing data source: {path}")
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    require(bool(files), f"No Parquet files in {path}")
    return files


def source_hashes(path):
    return {str(p.resolve()): file_hash(p) for p in source_files(path)}


def prepare_data(config):
    """Stream local QA/passages into an indexed SQLite corpus. Never create splits."""
    import pyarrow.dataset as ds
    calls_path = Path(config["calls_root"]) / "calls.parquet"
    calls = pd.read_parquet(calls_path, columns=["transcriptid", "call_date", "country"])
    splits = validate_splits(calls, read_table(config["splits_path"]))
    # The final shared pairs file can replace the local QA folder without changing code.
    qa_path = config.get("pairs_path") or str(Path(config["calls_root"]) / "qa")
    passages_path = config.get("passages_path") or str(Path(config["calls_root"]) / "passages")
    source_files(qa_path)
    source_files(passages_path)
    allowed = splits.loc[splits.split.isin(["train", "validation"]), "transcriptid"].unique().tolist()
    output = Path(config["prepared_root"])
    require(not output.exists(), f"Prepared folder exists: {output}; reload it or use a new version")
    output.mkdir(parents=True)
    temporary = output / "corpus.incomplete.sqlite"
    db = sqlite3.connect(temporary)
    try:
        db.executescript("""
            CREATE TABLE splits (protocol TEXT, transcriptid INTEGER, split TEXT,
                                 PRIMARY KEY(protocol, transcriptid));
            CREATE TABLE qa (id INTEGER PRIMARY KEY, transcriptid INTEGER NOT NULL,
                             exchange_id TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL,
                             UNIQUE(transcriptid, exchange_id));
            CREATE TABLE passages (id INTEGER PRIMARY KEY, passage_id TEXT NOT NULL UNIQUE,
                                   transcriptid INTEGER NOT NULL, text TEXT NOT NULL);
        """)
        db.executemany("INSERT INTO splits VALUES (?,?,?)",
                       splits[["protocol", "transcriptid", "split"]].itertuples(index=False, name=None))
        for kind, path in [("qa", qa_path), ("passages", passages_path)]:
            dataset = ds.dataset([str(p) for p in source_files(path)], format="parquet")
            columns = dataset.schema.names
            if kind == "qa":
                key = "pair_id" if "pair_id" in columns else "ex"
                needed = ["transcriptid", key, "question", "answer"]
            else:
                needed = ["passage_id", "transcriptid", "text"]
            require(set(needed) <= set(columns), f"{kind} source requires columns {needed}")
            scanner = dataset.scanner(columns=needed, filter=ds.field("transcriptid").isin(allowed), batch_size=4096)
            for batch in scanner.to_batches():
                frame = batch.to_pandas()
                require(not frame[needed].isna().any().any(), f"Null {kind} fields; reconcile source data")
                for text_column in (["question", "answer"] if kind == "qa" else ["text"]):
                    require(frame[text_column].map(lambda x: isinstance(x, str) and bool(x.strip())).all(),
                            f"Empty/non-string {kind}/{text_column}")
                if kind == "qa":
                    rows = ((int(row.transcriptid), str(getattr(row, key)), row.question, row.answer)
                            for row in frame.itertuples(index=False))
                    db.executemany("INSERT INTO qa(transcriptid,exchange_id,question,answer) VALUES (?,?,?,?)", rows)
                else:
                    rows = ((str(row.passage_id), int(row.transcriptid), row.text) for row in frame.itertuples(index=False))
                    db.executemany("INSERT INTO passages(passage_id,transcriptid,text) VALUES (?,?,?)", rows)
                db.commit()
        db.executescript("CREATE INDEX qa_call ON qa(transcriptid); CREATE INDEX passages_call ON passages(transcriptid);")
        report = []
        for protocol in ["main", "time"]:
            for split in ["train", "validation"]:
                condition = (protocol, split)
                count = db.execute("SELECT COUNT(*) FROM splits WHERE protocol=? AND split=?", condition).fetchone()[0]
                qa = db.execute("SELECT COUNT(*), COUNT(DISTINCT q.transcriptid) FROM qa q JOIN splits s USING(transcriptid) WHERE s.protocol=? AND s.split=?", condition).fetchone()
                passages = db.execute("SELECT COUNT(*) FROM passages p JOIN splits s USING(transcriptid) WHERE s.protocol=? AND s.split=?", condition).fetchone()[0]
                require(qa[0] > 0, f"No QA pairs in {protocol}/{split}")
                report.append({"protocol": protocol, "split": split, "manifest_calls": count,
                               "calls_with_pairs": qa[1], "pairs": qa[0], "passages": passages})
        db.commit()
    finally:
        db.close()
    temporary.rename(output / "corpus.sqlite")
    provenance = {"format": "sqlite-v2", "calls": {str(calls_path.resolve()): file_hash(calls_path)},
                  "splits": {str(Path(config["splits_path"]).resolve()): file_hash(config["splits_path"])},
                  "qa": source_hashes(qa_path), "passages": source_hashes(passages_path),
                  "database_sha256": file_hash(output / "corpus.sqlite"), "counts": report}
    write_json(output / "provenance.json", provenance)
    return {"database": str((output / "corpus.sqlite").resolve()),
            "provenance": str((output / "provenance.json").resolve())}, pd.DataFrame(report)


def load_data(config, verify_sources=True):
    root = Path(config["prepared_root"])
    require((root / "provenance.json").is_file() and (root / "corpus.sqlite").is_file(),
            f"Corpus preparation is incomplete at {root}. Set prepared_root to a new unused folder and rerun Section 5; existing partial data is not treated as complete.")
    provenance = json.loads((root / "provenance.json").read_text())
    require(provenance.get("format") == "sqlite-v2", "Reprepare with a new prepared_root for the local trainer")
    require(file_hash(root / "corpus.sqlite") == provenance["database_sha256"], "Prepared database changed")
    if verify_sources:
        qa_path = config.get("pairs_path") or str(Path(config["calls_root"]) / "qa")
        passages_path = config.get("passages_path") or str(Path(config["calls_root"]) / "passages")
        current = {"calls": source_hashes(Path(config["calls_root"]) / "calls.parquet"),
                   "splits": source_hashes(config["splits_path"]),
                   "qa": source_hashes(qa_path), "passages": source_hashes(passages_path)}
        for key, value in current.items():
            require(value == provenance[key], f"Shared {key} source changed; prepare a new version")
    return {"database": str((root / "corpus.sqlite").resolve()),
            "provenance": str((root / "provenance.json").resolve())}, pd.DataFrame(provenance["counts"])


class Corpus:
    """Index call IDs in RAM, fetching selected text from SQLite per batch."""
    def __init__(self, path, protocol):
        self.db = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
        self.protocol = protocol
        self.qa_calls = self.call_counts("qa", "train", 1)
        self.passage_calls = self.call_counts("passages", "train", 2)

    def close(self):
        self.db.close()

    def call_counts(self, table, split, minimum):
        require(table in ["qa", "passages"], "Unknown corpus table")
        sql = f"""SELECT p.transcriptid, COUNT(*) FROM {table} p JOIN splits s USING(transcriptid)
                  WHERE s.protocol=? AND s.split=? GROUP BY p.transcriptid HAVING COUNT(*)>=? ORDER BY p.transcriptid"""
        return dict(self.db.execute(sql, (self.protocol, split, minimum)).fetchall())

    def qa_rows(self, call):
        return self.db.execute("SELECT id, question, answer FROM qa WHERE transcriptid=? ORDER BY id", (int(call),)).fetchall()

    def sample(self, kind, size, rng, max_per_call=8):
        left, right, ids, call_ids = [], [], [], []
        if kind == "qa":
            candidates = rng.sample(list(self.qa_calls), min(size, len(self.qa_calls)))
            for call in candidates:
                rows = self.qa_rows(call)
                chosen = rng.sample(rows, min(max_per_call, len(rows), size - len(left)))
                for identity, question, answer in chosen:
                    left.append(question); right.append(answer); ids.append(identity); call_ids.append(call)
                if len(left) == size:
                    break
        else:
            require(kind == "passages", "Unknown batch type")
            require(len(self.passage_calls) >= size, f"Need {size} distinct training calls with >=2 passages")
            for call in rng.sample(list(self.passage_calls), size):
                candidates = self.db.execute("SELECT id FROM passages WHERE transcriptid=? ORDER BY id", (int(call),)).fetchall()
                pair = rng.sample(candidates, 2)
                texts = [self.db.execute("SELECT text FROM passages WHERE id=?", (row[0],)).fetchone()[0] for row in pair]
                left.append(texts[0]); right.append(texts[1]); ids.append([pair[0][0], pair[1][0]]); call_ids.append(call)
        require(len(left) == size, f"Not enough distinct QA pairs to fill batch {size}")
        return {"kind": kind, "left": left, "right": right, "ids": ids, "calls": call_ids}


def mean_pool(hidden, attention_mask):
    import torch.nn.functional as F
    weights = attention_mask.unsqueeze(-1).float()
    # Include all non-padding tokens, including special tokens, as specified.
    return F.normalize((hidden.float() * weights).sum(1) / weights.sum(1).clamp_min(1), dim=-1)


def contrastive_loss(left, right, temperature):
    import torch
    import torch.nn.functional as F
    scores = left.float() @ right.float().T / temperature
    labels = torch.arange(len(left), device=scores.device)
    return (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2


def mask_tokens(encoded, tokenizer, probability=0.15):
    import torch
    ids = encoded["input_ids"]
    special = torch.zeros_like(ids, dtype=torch.bool)
    for token in tokenizer.all_special_ids:
        special |= ids.eq(token)
    eligible = encoded["attention_mask"].bool() & ~special
    selected = (torch.rand(ids.shape, device=ids.device) < probability) & eligible
    labels = ids.clone()
    labels[~selected] = -100
    draws = torch.rand(ids.shape, device=ids.device)
    masked_ids = ids.clone()
    require(tokenizer.mask_token_id is not None, "Tokenizer must have a mask token")
    masked_ids[selected & (draws < 0.8)] = tokenizer.mask_token_id
    random_positions = selected & (draws >= 0.8) & (draws < 0.9)
    random_ids = torch.randint(len(tokenizer), ids.shape, device=ids.device)
    masked_ids[random_positions] = random_ids[random_positions]
    # The last 10% remain unchanged. Special/padding tokens are never targets.
    return {**encoded, "input_ids": masked_ids}, labels


def precision_mode(requested, device):
    import torch
    # Storage/emulation support is not native tensor-core BF16 support (notably T4).
    native_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    if requested == "auto":
        return "bf16" if native_bf16 else "fp16" if device.type == "cuda" else "fp32"
    require(requested in ["bf16", "fp16", "fp32"], "precision must be auto/bf16/fp16/fp32")
    require(requested != "fp16" or device.type == "cuda", "fp16 autocast requires CUDA in this trainer")
    require(requested != "bf16" or native_bf16, "bf16 requested without native hardware support; use auto/fp16/fp32")
    return requested


def autocast_context(precision, device):
    import torch
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def tokenize(texts, tokenizer, length, device):
    # Explicit right truncation: keep the start of the question/answer/passage.
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=length, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def joint_loss(model, tokenizer, batch, recipe, precision, device):
    import torch
    import torch.nn.functional as F
    left = tokenize(batch["left"], tokenizer, recipe["question_length"], device)
    right = tokenize(batch["right"], tokenizer, recipe["answer_length"], device)
    masked, labels = mask_tokens(right, tokenizer, recipe["mlm_probability"])
    # Separate clean-text contrastive forwards and a corrupted right-side MLM forward.
    with autocast_context(precision, device):
        q_hidden = model.base_model(**left, return_dict=True).last_hidden_state
        a_hidden = model.base_model(**right, return_dict=True).last_hidden_state
        q, a = mean_pool(q_hidden, left["attention_mask"]), mean_pool(a_hidden, right["attention_mask"])
        outputs = model(**masked, return_dict=True)
    nce = contrastive_loss(q, a, recipe["temperature"])
    selected = labels.ne(-100)
    # Selecting before fp32 conversion avoids expanding the whole vocabulary tensor in fp32.
    mlm = F.cross_entropy(outputs.logits[selected].float(), labels[selected]) if selected.any() else outputs.logits.reshape(-1)[:1].float().sum() * 0
    total = nce + recipe["mlm_weight"] * mlm
    require(bool(torch.isfinite(total).item()), "Nonfinite loss; use a new run with fp32 or review the checkpoint")
    return total, {"loss": float(total.detach()), "infonce": float(nce.detach()), "mlm": float(mlm.detach())}


def time_lr_scale(progress, warmup_fraction):
    progress = min(1.0, max(0.0, progress))
    if progress < warmup_fraction:
        return progress / warmup_fraction
    return 0.5 * (1 + math.cos(math.pi * (progress - warmup_fraction) / (1 - warmup_fraction)))


def validation_calls(corpus, recipe):
    candidates = sorted(corpus.call_counts("qa", "validation", recipe["validation_min_exchanges"]))
    require(bool(candidates), "No validation calls with enough exchanges")
    rng = random.Random(recipe["seed"])
    return sorted(rng.sample(candidates, min(recipe["validation_calls"], len(candidates))))


def evaluate_retrieval(model, tokenizer, corpus, calls, recipe, precision, device, batch_size):
    import torch
    model.eval()
    scores, details = [], []
    def embed(texts, length):
        embeddings = []
        for offset in range(0, len(texts), batch_size):
            inputs = tokenize(texts[offset:offset + batch_size], tokenizer, length, device)
            with autocast_context(precision, device):
                hidden = model.base_model(**inputs, return_dict=True).last_hidden_state
            embeddings.append(mean_pool(hidden, inputs["attention_mask"]).cpu())
        return torch.cat(embeddings)
    with torch.no_grad():
        for call in calls:
            rows = corpus.qa_rows(call)
            q = embed([row[1] for row in rows], recipe["question_length"])
            a = embed([row[2] for row in rows], recipe["answer_length"])
            correct = (q @ a.T).argmax(1).eq(torch.arange(len(rows)))
            score = float(correct.float().mean())
            scores.append(score)
            details.append({"transcriptid": int(call), "exchanges": len(rows), "top1": score})
    return {"top1": float(np.mean(scores)), "calls": len(scores), "aggregation": "equal-weight mean of within-call accuracies", "by_call": details}


def save_checkpoint(model, tokenizer, path, recipe, score, step):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    write_json(path / "embedding_config.json", {
        "pooling": "mean_last_hidden_nonpadding", "include_special_tokens": True,
        "normalize": True, "projection": None, "question_length": recipe["question_length"],
        "answer_length": recipe["answer_length"], "truncation_side": "right",
        "step": step, "validation_top1": score})


def validate_recipe(recipe):
    require(recipe.get("implementation") == "local", "Set recipe.implementation=local")
    for name in ["question_length", "answer_length", "batch_size", "min_batch_size", "max_exchanges_per_call",
                 "validation_calls", "validation_min_exchanges"]:
        require(isinstance(recipe.get(name), int) and recipe[name] > 0, f"Invalid recipe.{name}")
    require(isinstance(recipe.get("seed"), int), "Set an integer sampling seed")
    for name in ["temperature", "learning_rate", "budget_minutes", "validation_every_minutes"]:
        require(recipe[name] > 0, f"recipe.{name} must be positive")
    require(0 < recipe["warmup_fraction"] < 1, "Invalid warmup fraction")
    require(0 <= recipe["qa_batch_probability"] <= 1, "Invalid QA batch probability")
    require(0 < recipe["mlm_probability"] < 1 and recipe["mlm_weight"] >= 0, "Invalid MLM settings")
    require(recipe["batch_size"] >= recipe["min_batch_size"], "Invalid minimum batch size")


def validate_loaded_weights(model, loading, allow_new_mlm_head):
    """An explicit new-head exception must never permit a missing encoder."""
    missing = loading.get("missing_keys", [])
    prefix = model.base_model_prefix + "."
    backbone_keys = set(model.base_model.state_dict())
    backbone_missing = [key for key in missing if key.startswith(prefix) or key in backbone_keys]
    require(not backbone_missing, f"Checkpoint is missing encoder weights: {backbone_missing}")
    require(not missing or allow_new_mlm_head,
            f"Checkpoint is missing pretrained MLM parameters: {missing}. Supply a native MLM checkpoint; only allow_new_mlm_head as an explicit documented deviation.")
    require(not loading.get("mismatched_keys"), "Checkpoint has mismatched tensor shapes")


def plan(config, name, prepared, smoke=False):
    require(name in ADAPTATIONS, f"Unknown adaptation {name}")
    recipe = dict(config["recipe"])
    validate_recipe(recipe)
    base, protocol = ADAPTATIONS[name]
    reference = model_reference(config["models"][base])
    provenance = json.loads(Path(prepared["provenance"]).read_text())
    require(provenance.get("format") == "sqlite-v2", "Use the new local data-preparation section first")
    output = Path(config["output_root"]) / ("smoke_adapted" if smoke else "adapted") / name
    return output, {"kind": "adaptation", "name": name, "base": reference, "protocol": protocol,
                    "cutoff": 2023 if protocol == "main" else 2012, "recipe": recipe, "smoke": smoke,
                    "corpus_sha256": provenance["database_sha256"],
                    "provenance_sha256": file_hash(prepared["provenance"]),
                    "implementation_sha256": file_hash(__file__)}


def train(config, name, prepared, execute=False, smoke=False):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, set_seed
    output, snapshot = plan(config, name, prepared, smoke)
    if not execute:
        return {"output": str(output), "configuration": snapshot, "config_sha256": config_hash(snapshot)}
    checked, _ = load_data(config)
    require(checked == prepared, "Use the verified current prepared corpus")
    recipe = snapshot["recipe"]
    base, protocol = ADAPTATIONS[name]
    spec = config["models"][base]
    corpus = None
    with run_record(output, snapshot) as record:
        set_seed(recipe["seed"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        require(torch.cuda.device_count() <= 1, "This recipe is single-GPU; expose one device with CUDA_VISIBLE_DEVICES before starting the kernel")
        precision = precision_mode(recipe["precision"], device)
        record["effective_precision"] = precision
        record["precision_differs_from_bf16_recipe"] = precision != "bf16"
        kwargs = {"revision": spec["revision"]} if spec.get("revision") else {"local_files_only": True}
        tokenizer = load_tokenizer(spec)
        record["tokenizer"] = getattr(tokenizer, "_afp_source", {})
        tokenizer.truncation_side = "right"
        model_kwargs = dict(kwargs)
        require(not ({"revision", "local_files_only", "dtype", "torch_dtype"} & set(spec.get("load_kwargs", {}))),
                "load_kwargs cannot override the pinned reference or fp32 model weights")
        model_kwargs.update(spec.get("load_kwargs", {}))
        model, loading = AutoModelForMaskedLM.from_pretrained(
            spec["path"], output_loading_info=True, dtype=torch.float32, **model_kwargs)
        missing = loading.get("missing_keys", [])
        record["missing_pretrained_keys"] = missing
        validate_loaded_weights(model, loading, recipe["allow_new_mlm_head"])
        model.to(device)
        if recipe["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"],
                                      betas=tuple(recipe["betas"]), weight_decay=recipe["weight_decay"], foreach=False)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        corpus = Corpus(prepared["database"], protocol)
        try:
            rng = random.Random(recipe["seed"])
            calls = validation_calls(corpus, recipe)
            if smoke:
                calls = calls[:2]
            write_json(output / "validation_calls.json", calls)
            size = min(recipe["batch_size"], recipe["smoke_batch_size"]) if smoke else recipe["batch_size"]
            minimum = min(recipe["min_batch_size"], size)
            require(len(corpus.passage_calls) >= size, "Insufficient distinct passage calls for the configured batch size")
            def evaluate():
                nonlocal size
                while True:
                    try:
                        return evaluate_retrieval(model, tokenizer, corpus, calls, recipe, precision, device, size)
                    except torch.OutOfMemoryError:
                        if device.type != "cuda" or size <= minimum:
                            raise
                        size = max(minimum, size // 2)
                        gc.collect(); torch.cuda.empty_cache()
            baseline = evaluate()
            best, best_step = baseline["top1"], 0
            save_checkpoint(model, tokenizer, output / "best", recipe, best, 0)
            history = [{"step": 0, "elapsed_minutes": 0, **baseline}]
            write_json(output / "validation_history.json", history)
            started = time.monotonic()
            next_validation = recipe["validation_every_minutes"] * 60
            budget = recipe["budget_minutes"] * 60
            step, attempts = 0, 0
            with (output / "steps.jsonl").open("w") as log:
                while time.monotonic() - started < budget:
                    if smoke and step >= recipe["smoke_steps"]:
                        break
                    kind = "qa" if rng.random() < recipe["qa_batch_probability"] else "passages"
                    batch = corpus.sample(kind, size, rng, recipe["max_exchanges_per_call"])
                    while True:
                        optimizer.zero_grad(set_to_none=True)
                        model.train()
                        cpu_rng = torch.get_rng_state()
                        cuda_rng = torch.cuda.get_rng_state() if device.type == "cuda" else None
                        try:
                            loss, values = joint_loss(model, tokenizer, batch, recipe, precision, device)
                            scaler.scale(loss).backward()
                            del loss
                            break
                        except torch.OutOfMemoryError:
                            if device.type != "cuda" or size <= minimum:
                                raise
                            optimizer.zero_grad(set_to_none=True)
                            if "loss" in locals():
                                del loss
                            torch.set_rng_state(cpu_rng)
                            torch.cuda.set_rng_state(cuda_rng)
                            size = max(minimum, size // 2)
                            batch = {key: value[:size] if isinstance(value, list) else value for key, value in batch.items()}
                            log.write(json.dumps({"event": "oom_halved_batch", "step": step, "new_batch_size": size}) + "\n")
                            log.flush()
                            gc.collect(); torch.cuda.empty_cache()
                    elapsed = time.monotonic() - started
                    scale = time_lr_scale(elapsed / budget, recipe["warmup_fraction"])
                    for group in optimizer.param_groups:
                        group["lr"] = recipe["learning_rate"] * scale
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"])
                    require(precision == "fp16" or bool(torch.isfinite(grad_norm)), "Nonfinite gradient norm")
                    old_scale = scaler.get_scale()
                    # An OOM in optimizer.step may partially update weights. Abort safely
                    # rather than treating it as a retryable forward/backward OOM.
                    scaler.step(optimizer)
                    scaler.update()
                    skipped = scaler.get_scale() < old_scale
                    attempts += 1
                    step += int(not skipped)
                    log.write(json.dumps({"step": step, "attempt": attempts, "elapsed_minutes": elapsed / 60,
                                          "kind": kind, "batch_size": size, "lr": optimizer.param_groups[0]["lr"],
                                          "optimizer_skipped": skipped, **values}) + "\n")
                    log.flush()
                    if attempts == 1 or attempts % recipe["print_every_steps"] == 0:
                        print(f"{name}: step={step}, minutes={elapsed/60:.1f}, batch={size}, loss={values['loss']:.4f}", flush=True)
                    if time.monotonic() - started >= next_validation:
                        result = evaluate()
                        history.append({"step": step, "elapsed_minutes": (time.monotonic() - started) / 60, **result})
                        if result["top1"] > best:
                            best, best_step = result["top1"], step
                            save_checkpoint(model, tokenizer, output / "best", recipe, best, step)
                        write_json(output / "validation_history.json", history)
                        next_validation = (int((time.monotonic() - started) // (recipe["validation_every_minutes"] * 60)) + 1) * recipe["validation_every_minutes"] * 60
                # Also evaluate at the end. Baseline remains eligible when ties occur.
                result = evaluate()
                history.append({"step": step, "elapsed_minutes": (time.monotonic() - started) / 60, **result})
                if result["top1"] > best:
                    best, best_step = result["top1"], step
                    save_checkpoint(model, tokenizer, output / "best", recipe, best, step)
                write_json(output / "validation_history.json", history)
            require(step > 0, "No successful optimizer updates; do not count this run as trained")
            # Handoff remains a standard encoder directory; native MLM weights stay in best/.
            del optimizer, scaler, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            selected = AutoModelForMaskedLM.from_pretrained(output / "best", local_files_only=True, dtype=torch.float32,
                                                           **spec.get("load_kwargs", {}))
            encoder = selected.base_model
            encoder.save_pretrained(output / "model")
            tokenizer.save_pretrained(output / "model")
            embedding_config = json.loads((output / "best" / "embedding_config.json").read_text())
            write_json(output / "model" / "embedding_config.json", embedding_config)
            record.update(checkpoint=str((output / "model").resolve()), best_step=best_step,
                          best_validation_top1=best, optimizer_steps=step, optimizer_attempts=attempts,
                          final_batch_size=size, validation_calls=len(calls))
            (output / "README.md").write_text(
                f"{name}: calls through {snapshot['cutoff']}; selected step {best_step}; validation top-1 {best:.6f}.\n"
                "Load encoder/tokenizer from model/. Use mean last-layer nonpadding pooling, including special tokens, and L2 normalization. Native MLM checkpoint: best/.\n")
        finally:
            if corpus is not None:
                corpus.close()
    return str((output / "model").resolve())
