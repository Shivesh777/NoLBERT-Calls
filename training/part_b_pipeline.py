"""B's trainer code: corpus preparation, stage-2 runs and benchmark heads (run them through
nolbert_calls/run.py). No downloads or training at import time."""
from __future__ import annotations

import contextlib
import hashlib
from functools import lru_cache
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BUNDLE_VERSION = "part-b-audit-v5"

FEATURES = ["CLEAR", "ASSERTIVE", "CAUTIOUS", "OPTIMISTIC", "SPECIFIC", "RELEVANT"]
SUBJECTIVE_SEEDS = [5768, 78516, 944601]
# run name -> (key of its starting checkpoint in config["models"], protocol); nolbert_calls/run.py registers each run
ADAPTATIONS = {}
GLUE_TASKS = {
    "cola": (["sentence"], 2), "sst2": (["sentence"], 2),
    "qnli": (["question", "sentence"], 2),
    "qqp": (["question1", "question2"], 2),
    "mnli": (["premise", "hypothesis"], 3),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_table(path):
    path = Path(path)
    require(path.exists(), f"Missing input: {path}")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def file_hash(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _cached_file_hash(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=1024)
def _cached_file_hash(path, size, mtime_ns, ctime_ns):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = Path(path).stat()
    require((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == (size, mtime_ns, ctime_ns),
            f"Input changed while hashing: {path}")
    return digest.hexdigest()


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def inventory(root):
    import pyarrow.parquet as pq
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data folder does not exist on this machine: {root}. "
            f"Working directory: {Path.cwd()}. Update calls_root in part_b_config.json "
            "to the folder containing calls.parquet, qa/, and passages/."
        )
    paths = sorted(root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files found under {root}. Check the uploaded/mounted data path.")
    rows = []
    for path in paths:
        with pq.ParquetFile(path) as parquet:
            rows.append({"file": str(path.relative_to(root)),
                         "rows": parquet.metadata.num_rows,
                         "columns": len(parquet.schema_arrow)})
    return pd.DataFrame(rows, columns=["file", "rows", "columns"])


def validate_calls_root(root, qa_path=None, passages_path=None):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"calls_root does not exist here: {root}. Working directory: {Path.cwd()}. "
            "Set calls_root to this machine's folder containing calls.parquet, qa/, and passages/."
        )
    missing = []
    if not (root / "calls.parquet").is_file():
        missing.append("calls.parquet")
    for folder, override in [("qa", qa_path), ("passages", passages_path)]:
        source = Path(override).expanduser().resolve() if override else root / folder
        present = source.is_file() and source.suffix == ".parquet"
        if not present:
            present = source.is_dir() and any(source.rglob("*.parquet"))
        if not present:
            missing.append(f"{folder} Parquet source at {source}")
    require(not missing, f"Incorrect/incomplete calls_root {root}; missing: {', '.join(missing)}")
    return root


def training_inventory(config):
    """Inspect only the configured inputs, even when text folders live elsewhere."""
    import pyarrow.parquet as pq
    root = validate_calls_root(config["calls_root"], config.get("pairs_path"), config.get("passages_path"))
    sources = {"calls.parquet": root / "calls.parquet",
               "qa": Path(config.get("pairs_path") or root / "qa").expanduser().resolve(),
               "passages": Path(config.get("passages_path") or root / "passages").expanduser().resolve()}
    rows = []
    for group, source in sources.items():
        paths = [source] if source.is_file() else sorted(source.rglob("*.parquet"))
        for path in paths:
            with pq.ParquetFile(path) as parquet:
                rows.append({"group": group, "file": str(path), "rows": parquet.metadata.num_rows,
                             "columns": len(parquet.schema_arrow)})
    return pd.DataFrame(rows, columns=["group", "file", "rows", "columns"])


def canonical_call_metadata(calls):
    """One transcript can have multiple company rows; train on its text only once."""
    require({"transcriptid", "call_date", "country"} <= set(calls), "Missing call metadata columns")
    data = calls[["transcriptid", "call_date", "country"]].copy()
    require(data.transcriptid.notna().all(), "Missing transcript identifiers")
    data["call_date"] = pd.to_datetime(data.call_date, errors="raise")
    require(data.call_date.notna().all(), "Missing call dates")
    disagreements = data.groupby("transcriptid", sort=False).call_date.nunique()
    require(not disagreements.gt(1).any(), "A transcript has conflicting call dates; resolve before splitting")
    # Include a shared transcript in the US scope if any linked company row is US.
    data["_us"] = data.country.eq("United States")
    data = data.sort_values("_us", ascending=False, kind="stable").drop_duplicates("transcriptid")
    return data.drop(columns="_us").reset_index(drop=True)


def validate_splits(calls, splits):
    """Canonical shared manifest: protocol, transcriptid, split. Never generate splits."""
    require({"transcriptid", "call_date", "country"} <= set(calls), "Missing call metadata columns")
    require({"protocol", "transcriptid", "split"} <= set(splits), "Missing manifest columns")
    calls = canonical_call_metadata(calls)
    require(not splits[["protocol", "transcriptid", "split"]].isna().any().any(), "Null split identifiers")
    require(not splits.duplicated(["protocol", "transcriptid"]).any(), "A call has multiple assignments within a protocol")
    require(set(splits.protocol) <= {"main", "time"}, "Use protocol main or time")
    require(set(splits.split) <= {"train", "validation", "test"}, "Use split train, validation, or test")
    data = splits.merge(calls[["transcriptid", "call_date", "country"]],
                        on="transcriptid", how="left", validate="many_to_one", indicator=True)
    require(data._merge.eq("both").all(), "Manifest contains unknown transcript IDs")
    data["call_date"] = pd.to_datetime(data.call_date, errors="raise")
    require(data.call_date.notna().all(), "Missing call dates")
    require(data.country.eq("United States").all(),
            "The paper specifies US calls. Ask A to reconcile non-US manifest entries.")
    for protocol, cutoff, start, end in [
        ("main", "2024-01-01", "2024-01-01", "2025-01-01"),
        ("time", "2013-01-01", "2013-01-01", "2016-01-01"),
    ]:
        part = data[data.protocol.eq(protocol)]
        for split in ["train", "validation"]:
            require(part.split.eq(split).any(), f"Missing {protocol}/{split}")
        require((part.loc[part.split.eq("train"), "call_date"] < cutoff).all(), f"{protocol}: training crosses cutoff")
        dates = part.loc[part.split.eq("validation"), "call_date"]
        require(((dates >= start) & (dates < end)).all(), f"{protocol}: invalid validation dates")
        dates = part.loc[part.split.eq("test"), "call_date"]
        lower, upper = ("2025-01-01", "2026-11-01") if protocol == "main" else ("2013-01-01", "2027-01-01")
        require(((dates >= lower) & (dates < upper)).all(), f"{protocol}: invalid test dates")
    return data.drop(columns="_merge")


def build_date_splits(calls_path, output):
    """Create a proposed US call manifest using the supplied date rules, without randomness."""
    output = Path(output)
    require(not output.exists(), f"Refusing to overwrite split file: {output}")
    calls = pd.read_parquet(calls_path, columns=["transcriptid", "call_date", "country"])
    source_rows = len(calls)
    calls = canonical_call_metadata(calls)
    us = calls.loc[calls.country.eq("United States")].copy()
    us["call_date"] = pd.to_datetime(us.call_date, errors="raise")
    require(us.call_date.notna().all(), "US calls have missing dates; resolve them before splitting")
    pieces = []
    rules = {
        "main": [("train", None, "2024-01-01"),
                 ("validation", "2024-01-01", "2025-01-01"),
                 ("test", "2025-01-01", "2026-11-01")],
        "time": [("train", None, "2013-01-01"),
                 ("validation", "2013-01-01", "2016-01-01"),
                 ("test", "2016-01-01", "2026-11-01")],
    }
    for protocol, ranges in rules.items():
        for split, start, end in ranges:
            mask = us.call_date.lt(end)
            if start is not None:
                mask &= us.call_date.ge(start)
            part = us.loc[mask, ["transcriptid"]].copy()
            part["protocol"], part["split"] = protocol, split
            pieces.append(part[["protocol", "transcriptid", "split"]])
    manifest = pd.concat(pieces, ignore_index=True).sort_values(["protocol", "transcriptid"]).reset_index(drop=True)
    validate_splits(calls, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output, index=False)
    counts = manifest.groupby(["protocol", "split"]).size().rename("calls").reset_index()
    write_json(output.with_suffix(".metadata.json"), {
        "status": "proposed_date_based_manifest", "scope": "country == United States",
        "source": str(Path(calls_path).resolve()), "source_sha256": file_hash(calls_path),
        "manifest_sha256": file_hash(output), "rules": rules, "counts": counts.to_dict("records"),
        "source_company_rows": source_rows, "unique_transcripts": len(calls),
        "duplicate_company_rows_collapsed": source_rows - len(calls),
        "multi_company_rule": "One text unit per transcriptid; US scope when any linked company is US; dates must agree",
        "note": "Generated from the supplied plan, not supplied/approved by A. Share this version with the team. Time validation years 2013-15 may also be scored separately; they are not untouched test data."
    })
    return manifest


def ensure_splits(config):
    """Use the configured existing manifest, or create the explicitly enabled date proposal."""
    path = Path(config["splits_path"])
    if not path.is_file():
        require(config.get("create_date_splits_if_missing", False),
                f"Missing {path}. Supply A's manifest or enable create_date_splits_if_missing for a date-based proposal.")
        build_date_splits(Path(config["calls_root"]) / "calls.parquet", path)
        print(f"Created proposed date-based splits: {path}")
    return read_table(path)


def prepare_pairs(calls_root, pairs_path, splits_path, output):
    """Export all shared train/validation pairs; no resampling or deduplication."""
    calls = pd.read_parquet(Path(calls_root) / "calls.parquet")
    splits = validate_splits(calls, read_table(splits_path))
    pairs = read_table(pairs_path)
    require({"transcriptid", "question", "answer"} <= set(pairs), "Pairs need transcriptid, question, answer")
    key = ["pair_id"] if "pair_id" in pairs else ["transcriptid", "ex"]
    require(set(key) <= set(pairs), "Pairs need pair_id, or transcriptid + ex; preserve A/Swapnil's identifiers")
    require(not pairs[key].isna().any().any(), "Null pair identifiers")
    require(not pairs.duplicated(key).any(), "Duplicate pair identifiers; reconcile shared file")
    require(pairs.transcriptid.notna().all() and pairs.transcriptid.isin(calls.transcriptid).all(), "Unknown/null pair transcript IDs")
    for column in ["question", "answer"]:
        require(pairs[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all(), f"Empty/non-text {column}")
    output = Path(output)
    require(not output.exists(), f"Prepared directory already exists: {output}; use a new versioned directory")
    paths, report = {}, []
    # Check coverage before writing any files.
    for protocol in ["main", "time"]:
        paths[protocol] = {}
        for split in ["train", "validation"]:
            ids = splits.loc[splits.protocol.eq(protocol) & splits.split.eq(split), "transcriptid"]
            selected = pairs[pairs.transcriptid.isin(ids)]
            require(len(selected) > 0, f"No pairs for {protocol}/{split}")
            report.append({"protocol": protocol, "split": split, "manifest_calls": len(ids),
                           "calls_with_pairs": selected.transcriptid.nunique(), "pairs": len(selected)})
    output.mkdir(parents=True)
    for row in report:
        protocol, split = row["protocol"], row["split"]
        ids = splits.loc[splits.protocol.eq(protocol) & splits.split.eq(split), "transcriptid"]
        destination = output / f"{protocol}_{split}.parquet"
        pairs[pairs.transcriptid.isin(ids)].to_parquet(destination, index=False)
        paths[protocol][split] = str(destination.resolve())
    write_json(output / "provenance.json", {
        "pairs_sha256": file_hash(pairs_path), "splits_sha256": file_hash(splits_path),
        "calls_sha256": file_hash(Path(calls_root) / "calls.parquet"),
        "exports": {p: {s: file_hash(v) for s, v in files.items()} for p, files in paths.items()},
        "counts": report, "note": "All shared pairs retained; sampling belongs to the original recipe."})
    return paths, pd.DataFrame(report)


def load_prepared(output, *, pairs_path=None, splits_path=None, calls_root=None):
    output = Path(output)
    provenance = json.loads((output / "provenance.json").read_text())
    sources = {"pairs_sha256": pairs_path, "splits_sha256": splits_path,
               "calls_sha256": Path(calls_root) / "calls.parquet" if calls_root is not None else None}
    for key, path in sources.items():
        if path is not None:
            require(Path(path).is_file(), f"Missing original source: {path}")
            require(file_hash(path) == provenance[key],
                    f"Shared source changed: {path}; prepare a new version instead of reusing old exports")
    paths = {p: {s: str((output / f"{p}_{s}.parquet").resolve())
                 for s in ["train", "validation"]} for p in ["main", "time"]}
    for protocol, files in paths.items():
        for split, path in files.items():
            require(file_hash(path) == provenance["exports"][protocol][split], "Prepared pairs changed; create a new version")
    return paths, pd.DataFrame(provenance["counts"])


def environment():
    import torch
    packages = {}
    for name in ["torch", "transformers", "datasets", "accelerate", "pandas", "scikit-learn"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "packages": packages,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


@contextlib.contextmanager
def run_record(output, run_config):
    output = Path(output)
    require(not output.exists(), f"Refusing to overwrite run: {output}. Use a new run name.")
    output.mkdir(parents=True)
    record = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "config": run_config, "config_sha256": config_hash(run_config), "environment": environment()}
    write_json(output / "run.json", record)
    started = time.monotonic()
    try:
        yield record
        record["status"] = "complete"
    except BaseException:
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
        raise
    finally:
        record["wall_hours"] = (time.monotonic() - started) / 3600
        # Allocated-device hours; not utilization or energy consumption.
        record["allocated_gpu_hours"] = record["wall_hours"] * record["environment"]["gpu_count"]
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", record)


def model_reference(spec):
    require(spec and spec.get("path"), "Fill the exact model checkpoint/path in the configuration")
    path = Path(spec["path"])
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
        require(any(p.name == "config.json" for p in files), f"No model config in {path}")
        return {"path": str(path.resolve()), "files": {str(p.relative_to(path)): file_hash(p) for p in sorted(files)},
                "load_kwargs": spec.get("load_kwargs", {})}
    require(spec.get("revision"), f"Pin a Hugging Face commit revision for {spec['path']}")
    return {"path": spec["path"], "revision": spec["revision"], "load_kwargs": spec.get("load_kwargs", {})}


def load_tokenizer(spec):
    """Load current tokenizer assets, or explicitly build a legacy BERT WordPiece vocabulary."""
    from transformers import AutoTokenizer, BertTokenizerFast
    reference = spec["path"]
    path = Path(reference)
    vocab_path = None
    lowercase = True
    if path.is_dir() and (path / "vocab.txt").is_file() and not (path / "tokenizer.json").is_file():
        model_config = json.loads((path / "config.json").read_text()) if (path / "config.json").is_file() else {}
        is_bert = model_config.get("model_type") == "bert" or any(
            name.startswith("Bert") for name in model_config.get("architectures", []))
        if is_bert:
            vocab_path = path / "vocab.txt"
            if (path / "tokenizer_config.json").is_file():
                lowercase = json.loads((path / "tokenizer_config.json").read_text()).get("do_lower_case", True)
    if vocab_path is not None:
        tokens = vocab_path.read_text(encoding="utf-8").splitlines()
        require(len(tokens) == len(set(tokens)), "BERT vocabulary contains duplicate entries")
        vocabulary = {token: index for index, token in enumerate(tokens)}
        tokenizer = BertTokenizerFast(vocab=vocabulary, do_lower_case=lowercase)
        require(len(tokenizer) == len(tokens), "Tokenizer added special tokens absent from the original BERT vocabulary")
        tokenizer._afp_source = {"loader": "explicit_bert_wordpiece", "vocabulary_sha256": file_hash(vocab_path),
                                 "vocabulary_size": len(tokens), "do_lower_case": lowercase,
                                 "original_reference": reference, "revision": spec.get("revision")}
    else:
        kwargs = {"revision": spec["revision"]} if spec.get("revision") else {"local_files_only": True}
        tokenizer = AutoTokenizer.from_pretrained(reference, **kwargs)
        tokenizer._afp_source = {"loader": "AutoTokenizer", "reference": reference, "revision": spec.get("revision")}
    tokenizer.truncation_side = "right"
    return tokenizer


def archive_failed_adaptation(config, name):
    """Explicit retry helper: preserve a failed run before clearing only its original path."""
    import shutil
    require(name in ADAPTATIONS, f"Unknown adaptation: {name}")
    source = Path(config["output_root"]) / "adapted" / name
    if not source.exists():
        return None
    require((source / "run.json").is_file(), "Run folder has no status log; inspect it before resetting")
    record = json.loads((source / "run.json").read_text())
    require(record.get("status") == "failed", "Only a failed run can be archived; completed or running jobs are preserved")
    destination = source.with_name(f"{name}_failed_{time.time_ns()}")
    shutil.copytree(source, destination)
    shutil.rmtree(source)
    return str(destination)


def prepare_adaptation_data(config):
    from part_b_adaptation import prepare_data
    ensure_splits(config)
    return prepare_data(config)


def load_adaptation_data(config, verify_sources=True):
    from part_b_adaptation import load_data
    return load_data(config, verify_sources=verify_sources)


def run_adaptation(config, name, prepared, execute=False, smoke=False):
    """Run the locally implemented joint contrastive + MLM trainer."""
    from part_b_adaptation import train
    return train(config, name, prepared, execute=execute, smoke=smoke)


def metric_values(task, labels, predictions):
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    result = {"accuracy": float(accuracy_score(labels, predictions))}
    if task == "subjective":
        result.update(f1_weighted=float(f1_score(labels, predictions, labels=[0, 1, 2], average="weighted", zero_division=0)),
                      f1_macro=float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)))
    elif task == "cola":
        result["matthews_correlation"] = float(matthews_corrcoef(labels, predictions))
    elif task == "qqp":
        result["f1"] = float(f1_score(labels, predictions, zero_division=0))
    return result


def validate_classification_data(frames, text_columns, label_column, num_labels):
    """Reject missing labels, invalid classes and text duplicated across partitions."""
    seen = set()
    for split, frame in frames.items():
        require(len(frame) > 0, f"Empty {split}")
        require(set(text_columns + [label_column]) <= set(frame), f"Missing {split} columns")
        for col in text_columns:
            require(frame[col].map(lambda x: isinstance(x, str) and bool(x.strip())).all(), f"Invalid {split}/{col} text")
        labels = pd.to_numeric(frame[label_column], errors="raise")
        require(labels.notna().all() and labels.isin(range(num_labels)).all(), f"Invalid/unavailable {split} labels")
        identities = set(map(tuple, frame[text_columns].to_numpy()))
        require(not seen.intersection(identities), f"Duplicate input text across {split} and another split; reconcile with A")
        seen.update(identities)


def load_subjective(config, seed):
    """Read A's author-provided split exports; do not create new random splits."""
    root = Path(config["subjective"]["data_root"]) / str(seed)
    frames, provenance = {}, {}
    for split in ["train", "validation", "test"]:
        candidates = [root / f"{split}.parquet", root / f"{split}.csv"]
        if split == "validation":
            candidates += [root / "val.parquet", root / "val.csv"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None and config["subjective"].get("download_if_missing", False):
            from huggingface_hub import hf_hub_download
            cfg = config["subjective"]
            require(cfg.get("revision"), "Pin the SubjECTive-QA dataset revision")
            published = "val" if split == "validation" else split
            try:
                downloaded = hf_hub_download(
                    repo_id=cfg.get("dataset_id", "gtfintechlab/SubjECTive-QA"), repo_type="dataset",
                    revision=cfg["revision"], filename=f"{seed}/{published}-00000-of-00001.parquet",
                    cache_dir=str(root.parent / "hf_cache"))
            except Exception as exc:
                raise RuntimeError(
                    "SubjECTive-QA download failed. Its author repository is access-gated: "
                    "accept access at https://huggingface.co/datasets/gtfintechlab/SubjECTive-QA "
                    "and sign in on the training machine using huggingface_hub.login(), "
                    "or upload the author's split files. Do not paste tokens into this notebook/config."
                ) from exc
            import shutil
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"{split}.parquet"
            shutil.copyfile(downloaded, path)
        require(path is not None, f"Ask A for author-provided {seed}/{split}.parquet or .csv")
        frames[split] = read_table(path)
        provenance[split] = {"path": str(path.resolve()), "sha256": file_hash(path)}
    for feature in FEATURES:
        validate_classification_data(frames, ["QUESTION", "ANSWER"], feature, 3)
    return frames, provenance


def split_benchmark_training(frame, text_columns, fraction, seed):
    """Keep identical inputs together in the proposed internal tuning split."""
    from sklearn.model_selection import train_test_split
    require(0 < fraction < 1, "GLUE tuning_fraction must be between zero and one")
    identities = frame[text_columns].apply(tuple, axis=1)
    groups = pd.DataFrame({"identity": identities, "label": frame["label"]})
    # Official GLUE can assign different labels to repeated inputs. Preserve all
    # examples; majority label is used only for group stratification, never relabelling.
    unique = groups.groupby("identity", sort=False).label.agg(lambda values: values.mode().min()).reset_index()
    train_groups, validation_groups = train_test_split(
        np.arange(len(unique)), test_size=fraction, random_state=seed, stratify=unique.label)
    validation_ids = set(unique.iloc[validation_groups].identity)
    mask = identities.isin(validation_ids)
    return frame.loc[~mask].copy(), frame.loc[mask].copy()


def prepare_glue(config, task):
    """Use an agreed holdout inside official train for tuning; reserve official dev for reporting."""
    from datasets import load_dataset
    require(task in GLUE_TASKS, f"Unsupported GLUE task {task}")
    cfg = config["glue"]
    require(cfg["revision"], "Pin the nyu-mll/glue dataset commit revision")
    raw = load_dataset("nyu-mll/glue", task, revision=cfg["revision"])
    # Official GLUE test labels are hidden. Never evaluate label=-1 as ground truth.
    columns, classes = GLUE_TASKS[task]
    original_train = raw["train"].to_pandas()
    score_split = "validation_matched" if task == "mnli" else "validation"
    report = raw[score_split].to_pandas()
    # Proposed held-out protocol: exclude report-overlapping training inputs,
    # preserving the entire official report set and every remaining training row.
    policy = cfg.get("report_overlap_policy", "error")
    require(policy in ["error", "exclude_from_training"], "Unknown GLUE report_overlap_policy")
    report_ids = set(map(tuple, report[columns].to_numpy()))
    overlaps = original_train[columns].apply(tuple, axis=1).isin(report_ids)
    require(policy != "error" or not overlaps.any(),
            "Official GLUE train/report inputs overlap; choose and document a shared overlap policy")
    eligible = original_train.loc[~overlaps] if policy == "exclude_from_training" else original_train
    training, tuning = split_benchmark_training(eligible, columns,
                                               cfg["tuning_fraction"], cfg["split_seed"])
    frames = {"train": training,
              "validation": tuning,
              "report": report}
    validate_classification_data(frames, columns, "label", classes)
    provenance = {"dataset": "nyu-mll/glue", "revision": cfg["revision"],
                  "task": task, "tuning_fraction": cfg["tuning_fraction"],
                  "split_seed": cfg["split_seed"], "report_split": score_split,
                  "report_overlap_policy": policy, "training_rows_overlapping_report": int(overlaps.sum()),
                  "tuning_method": "stratified groups of identical input text; original rows preserved",
                  "fingerprints": {"source_train": raw["train"]._fingerprint,
                                   "report": raw[score_split]._fingerprint},
                  "partition_rows_sha256": {name: config_hash(frame.index.tolist())
                                            for name, frame in [("train", training), ("validation", tuning)]}}
    return frames, provenance


def restore_best_classifier(model, checkpoint, load_kwargs=None):
    """Use native key conversion (e.g. LayerNorm gamma/beta), then copy all weights strictly."""
    import torch
    from transformers import AutoModelForSequenceClassification
    selected, info = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.float32, output_loading_info=True,
        **(load_kwargs or {}))
    require(not info.get("missing_keys") and not info.get("mismatched_keys"),
            f"Selected classifier checkpoint failed to restore: {info}")
    model.load_state_dict(selected.state_dict(), strict=True)
    del selected


def train_head(config, model_key, task, seed, frames, provenance, feature=None, smoke=False):
    """Fresh standard sequence classifier per task. Full encoder updates unless frozen in config."""
    import torch
    from datasets import Dataset, DatasetDict
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments, set_seed)
    options = {**config["heads"], **config["heads"].get("per_task", {}).get(task, {})}
    options.pop("per_task", None)
    require(options["protocol_note"], "Document the benchmark protocol agreed with A before training")
    require(task != "subjective" or feature in FEATURES, "Select one SubjECTive-QA feature")
    needed = {"train", "validation", "test" if task == "subjective" else "report"}
    require(needed <= set(frames), f"Benchmark data needs partitions: {sorted(needed)}")
    spec = config["models"][model_key]
    reference = model_reference(spec)
    text_columns, classes = (["QUESTION", "ANSWER"], 3) if task == "subjective" else GLUE_TASKS[task]
    target = feature if task == "subjective" else "label"
    validate_classification_data(frames, text_columns, target, classes)
    name = f"{model_key}__{feature or task}__seed{seed}" + ("__SMOKE" if smoke else "")
    output = Path(config["output_root"]) / ("smoke" if smoke else "heads") / task / name
    snapshot = {"kind": "head", "model": model_key, "model_reference": reference,
                "task": task, "feature": feature, "seed": seed, "smoke": smoke,
                "options": options, "data": provenance,
                "implementation_sha256": file_hash(__file__),
                "input_format": "question + space + answer" if task == "subjective" else text_columns,
                "head": "AutoModelForSequenceClassification; native architecture head"}
    with run_record(output, snapshot) as record:
        set_seed(seed)
        kwargs = {"revision": spec["revision"]} if spec.get("revision") else {"local_files_only": True}
        tokenizer = load_tokenizer(spec)
        record["tokenizer"] = getattr(tokenizer, "_afp_source", {})
        model_kwargs = {**kwargs, **spec.get("load_kwargs", {})}
        require(not ({"revision", "local_files_only", "dtype", "torch_dtype"} & set(spec.get("load_kwargs", {}))),
                "load_kwargs is for architecture options; use the pinned reference and configured precision")
        model = AutoModelForSequenceClassification.from_pretrained(
            spec["path"], num_labels=classes, dtype=torch.float32, **model_kwargs)
        tokenizer.truncation_side = "right"
        # Do not ignore mismatched weights silently. Supply an encoder-only checkpoint if needed.
        if options["freeze_encoder"]:
            for parameter in model.base_model.parameters():
                parameter.requires_grad = False
        datasets = DatasetDict({})
        for split, frame in frames.items():
            if smoke and split not in {"train", "validation"}:
                continue
            subset = frame.head(16).copy() if smoke else frame
            def tokenize(batch):
                if task == "subjective":
                    texts = [q + " " + a for q, a in zip(batch["QUESTION"], batch["ANSWER"])]
                    result = tokenizer(texts, truncation=True, max_length=options["max_length"])
                else:
                    args = [batch[column] for column in text_columns]
                    result = tokenizer(*args, truncation=True, max_length=options["max_length"])
                result["labels"] = [int(x) for x in batch[target]]
                return result
            data = Dataset.from_pandas(subset, preserve_index=False)
            datasets[split] = data.map(tokenize, batched=True, remove_columns=data.column_names)
        def metrics(prediction):
            logits = prediction.predictions
            logits = logits[0] if isinstance(logits, tuple) else logits
            return metric_values(task, prediction.label_ids, np.argmax(logits, axis=-1))
        class FiniteLossCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                for key, value in (logs or {}).items():
                    if "loss" in key and isinstance(value, (float, int)):
                        require(np.isfinite(value), f"Nonfinite {key}; retain failed log and review precision/config")
        precision = options["precision"]
        require(precision in ["fp32", "fp16"], "Use fp32 or fp16 autocast; no pure-half model cast")
        require(precision != "fp16" or torch.cuda.is_available(), "fp16 training requires a CUDA GPU here")
        import math
        updates_per_epoch = math.ceil(math.ceil(len(datasets["train"]) / options["batch_size"]) /
                                      options["gradient_accumulation_steps"])
        planned_steps = 2 if smoke else math.ceil(updates_per_epoch * options["epochs"])
        args = TrainingArguments(
            output_dir=str(Path(os.environ.get("PARTB_HEADS_TMP", str(output))) / name / "training"),
            learning_rate=options["learning_rate"], save_only_model=True,
            num_train_epochs=options["epochs"], max_steps=2 if smoke else -1,
            per_device_train_batch_size=options["batch_size"],
            per_device_eval_batch_size=options["eval_batch_size"],
            gradient_accumulation_steps=options["gradient_accumulation_steps"],
            weight_decay=options["weight_decay"], warmup_steps=math.ceil(planned_steps * options["warmup_ratio"]),
            eval_strategy="steps" if smoke else "epoch", eval_steps=1 if smoke else None,
            save_strategy="steps" if smoke else "epoch", save_steps=1 if smoke else 500,
            save_total_limit=2, load_best_model_at_end=False,
            metric_for_best_model="eval_loss", greater_is_better=False,
            fp16=precision == "fp16", bf16=False, seed=seed, data_seed=seed,
            use_cpu=options.get("use_cpu", False),
            logging_steps=1 if smoke else 50, logging_first_step=True,
            logging_nan_inf_filter=False, report_to="none", push_to_hub=False,
            eval_accumulation_steps=8,   # A (10 Nov): move accumulated predictions to CPU
            dataloader_num_workers=0, dataloader_pin_memory=torch.cuda.is_available())
        trainer = Trainer(model=model, args=args, train_dataset=datasets["train"],
                          eval_dataset=datasets["validation"], processing_class=tokenizer,
                          data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics,
                          # A (10 Nov): keep only the logits; full outputs can exhaust GPU memory in evaluation
                          preprocess_logits_for_metrics=lambda logits, labels: logits[0] if isinstance(logits, tuple) else logits,
                          callbacks=[FiniteLossCallback()])
        trainer.train()
        best_checkpoint = trainer.state.best_model_checkpoint
        require(best_checkpoint is not None, "Training produced no selected validation checkpoint")
        restore_best_classifier(trainer.model, best_checkpoint, spec.get("load_kwargs"))
        record["selected_training_checkpoint"] = best_checkpoint
        if os.environ.get("PARTB_KEEP_HEADS") == "1":
            trainer.save_model(str(output / "model"))
            tokenizer.save_pretrained(output / "model")
        validation = trainer.evaluate(metric_key_prefix="validation")
        write_json(output / "validation_metrics.json", validation)
        if smoke:
            record["note"] = "Pipeline check only; no benchmark report/test evaluation."
        else:
            split = "test" if task == "subjective" else "report"
            prediction = trainer.predict(datasets[split], metric_key_prefix=split)
            write_json(output / "metrics.json", prediction.metrics)
            logits = prediction.predictions
            logits = logits[0] if isinstance(logits, tuple) else logits
            pd.DataFrame({"row": np.arange(len(logits)), "label": prediction.label_ids,
                          "prediction": np.argmax(logits, axis=-1)}).to_csv(output / "predictions.csv", index=False)
            record["metrics"] = prediction.metrics
        record["checkpoint"] = str((output / "model").resolve()) if os.environ.get("PARTB_KEEP_HEADS") == "1" else None
    import shutil
    shutil.rmtree(Path(args.output_dir), ignore_errors=True)
    del trainer, model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(output)


def readiness(config):
    """Non-mutating prerequisite report; does not download or fingerprint large models."""
    rows = []
    def add(stage, item, ready, detail):
        rows.append({"stage": stage, "requirement": item, "ready": bool(ready), "detail": detail})
    for name in ["torch", "transformers", "datasets", "accelerate", "pyarrow", "sklearn"]:
        import importlib.util
        add("environment", name, importlib.util.find_spec(name) is not None, "Install requirements-part-b.txt if missing")
    qa_source = config.get("pairs_path") or str(Path(config["calls_root"]) / "qa")
    passage_source = config.get("passages_path") or str(Path(config["calls_root"]) / "passages")
    for label, path in [("Call metadata", str(Path(config["calls_root"]) / "calls.parquet")),
                        ("QA pairs source", qa_source), ("Passage source", passage_source),
                        ("Shared splits", config["splits_path"])]:
        add("adaptation", label, Path(path).exists(), str(path))
    recipe = config["recipe"]
    try:
        from part_b_adaptation import validate_recipe
        validate_recipe(recipe)
        recipe_ready = True
        detail = "Local trainer; no external training script required"
    except (ValueError, KeyError) as exc:
        recipe_ready, detail = False, str(exc)
    add("adaptation", "Local recipe configured", recipe_ready, detail)
    model_keys = {base for base, _ in ADAPTATIONS.values()} | set(config["glue"]["models"]) | set(config["subjective"]["models"])
    derived = set(ADAPTATIONS)
    for key in sorted(model_keys):
        spec = config["models"].get(key) or {}
        value = spec.get("path")
        ready = bool(value) and (Path(value).is_dir() or bool(spec.get("revision")))
        add("checkpoint", key, ready,
            "Produced by an earlier adaptation run" if key in derived else "Provide exact local checkpoint or remote ID + commit revision")
    add("subjective", "Main-model list", bool(config["subjective"]["models"]), "Confirm the list with A; currently empty" if not config["subjective"]["models"] else str(config["subjective"]["models"]))
    add("subjective", "Three required seeds", config["subjective"]["seeds"] == SUBJECTIVE_SEEDS, str(SUBJECTIVE_SEEDS))
    for seed in SUBJECTIVE_SEEDS:
        root = Path(config["subjective"]["data_root"]) / str(seed)
        for split in ["train", "validation", "test"]:
            present = any((root / f"{split}{ext}").is_file() for ext in [".csv", ".parquet"])
            if split == "validation":
                present = present or any((root / f"val{ext}").is_file() for ext in [".csv", ".parquet"])
            downloadable = config["subjective"].get("download_if_missing") and config["subjective"].get("revision")
            add("subjective", f"{seed}/{split}", present or downloadable,
                str(root) if present else "Author split will be downloaded; requires accepted Hugging Face dataset access/login")
    add("glue", "Dataset revision pinned", bool(config["glue"]["revision"]), "nyu-mll/glue commit revision")
    add("heads", "Benchmark protocol recorded", bool(config["heads"]["protocol_note"]),
        "Agree on splits, model selection, encoder updates, token length and native-head differences")
    return pd.DataFrame(rows)


def batch_preflight(config, full=False):
    """Fail before a long sequential run when known prerequisites are missing."""
    if config.get("create_date_splits_if_missing", False):
        ensure_splits(config)
    report = readiness(config)
    bases = {base for base, _ in ADAPTATIONS.values()}
    if full:
        required = ~((report.stage == "checkpoint") & report.requirement.isin(ADAPTATIONS))
    else:
        required = ((report.stage == "adaptation") |
                    ((report.stage == "checkpoint") & report.requirement.isin(bases)) |
                    ((report.stage == "environment") & report.requirement.isin(["torch", "transformers", "pyarrow"])))
    missing = report[required & ~report.ready]
    require(missing.empty, "Complete these prerequisites before starting:\n" + missing[["stage", "requirement"]].to_string(index=False))
    for key in bases:
        spec = config["models"][key]
        if Path(spec["path"]).is_dir():
            require((Path(spec["path"]) / "config.json").is_file(), f"Missing model config for {key}")
    calls = pd.read_parquet(Path(config["calls_root"]) / "calls.parquet",
                            columns=["transcriptid", "call_date", "country"])
    validate_splits(calls, read_table(config["splits_path"]))
    for name in ADAPTATIONS:
        require(not (Path(config["output_root"]) / "adapted" / name).exists(),
                f"Existing adaptation run: {name}. Select unfinished jobs individually or use a new versioned output root.")
    if full:
        require({"cola", "sst2", "qnli"} <= set(config["glue"]["tasks"]), "Full run must include the three required GLUE tasks")
        for key in set(config["subjective"]["models"]) | set(config["glue"]["models"]):
            require(key in config["models"], f"Unknown benchmark model: {key}")
            if key in ADAPTATIONS:
                expected = (Path(config["output_root"]) / "adapted" / key / "model").resolve()
                require(Path(config["models"][key]["path"]).resolve() == expected,
                        f"Update {key} to the checkpoint produced under the selected output_root")
        # Validate the small public exports before spending hours adapting encoders.
        for seed in config["subjective"]["seeds"]:
            load_subjective(config, seed)
        for model in config["subjective"]["models"]:
            for feature in FEATURES:
                for seed in config["subjective"]["seeds"]:
                    path = Path(config["output_root"]) / "heads" / "subjective" / f"{model}__{feature}__seed{seed}"
                    require(not path.exists(), f"Existing benchmark run: {path}")
        for model in config["glue"]["models"]:
            for task in config["glue"]["tasks"]:
                for seed in config["glue"]["seeds"]:
                    path = Path(config["output_root"]) / "heads" / task / f"{model}__{task}__seed{seed}"
                    require(not path.exists(), f"Existing benchmark run: {path}")
    return report


def summarize(output_root):
    rows = []
    for path in sorted(Path(output_root).rglob("run.json")):
        run = json.loads(path.read_text())
        cfg = run["config"]
        rows.append({"run": str(path.parent), "status": run["status"], "kind": cfg["kind"],
                     "model": cfg.get("model", cfg.get("name")), "task": cfg.get("task"),
                     "feature": cfg.get("feature"), "seed": cfg.get("seed", cfg.get("recipe", {}).get("seed")),
                     "smoke": cfg.get("smoke", False), "config_sha256": run["config_sha256"],
                     "wall_hours": run.get("wall_hours"), "allocated_gpu_hours": run.get("allocated_gpu_hours"),
                     "gpu": "; ".join(run["environment"]["gpus"]), **run.get("metrics", {})})
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.to_csv(Path(output_root) / "all_runs.csv", index=False)
        completed = frame[frame.status.eq("complete") & ~frame.smoke & frame.kind.eq("head")]
        numeric = [c for c in completed if c.endswith(("f1_weighted", "f1_macro", "accuracy", "matthews_correlation"))]
        if numeric:
            completed.groupby(["model", "task", "feature"], dropna=False)[numeric].agg(["mean", "std", "count"]).to_csv(Path(output_root) / "benchmark_summary.csv")
    return frame
