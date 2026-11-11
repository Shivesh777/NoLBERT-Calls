"""Stage 2 and the benchmark heads of NoLBERT-Calls: runs B's trainer (training/part_b_*.py) on the bundle.

  # stage 2: two 150-minute runs on one T4, the second continuing from the first's selected checkpoint
  python -m nolbert_calls.run adapt --name nolbert_calls_2023_run1 --init nolbert_calls_2012
  python -m nolbert_calls.run adapt --name nolbert_calls_2023 --init adapted:nolbert_calls_2023_run1
  # fine-tuned heads (SubjECTive-QA: the authors' three splits; GLUE CoLA, SST-2, QNLI)
  python -m nolbert_calls.run heads --task subjective --models nolbert_calls_2023
  python -m nolbert_calls.run heads --task cola --models nolbert_calls_2023
  python -m nolbert_calls.run summarize

--init is a folder in NCALLS_MODELS (e.g. nolbert_calls_2012, the stage-1 output of
pipeline/c04_finetune.py; its best/ checkpoint is used) or adapted:<run>, the selected checkpoint
of an earlier run. The recipe is "recipe" in training/part_b_config.json; --recipe '{"key": value}'
overrides it (logged in the run's run.json). Heads start from a run's encoder export
(adapted/<run>/model), or from the released model in model/ for nolbert_calls_2023.

Environment: NCALLS_DATA (the bundle of pipeline/c05_bundle.py, default data/bundle), NCALLS_MODELS
(starting checkpoints, default data/encoders), NCALLS_TMP (scratch, default tmp/), PARTB_OUT (outputs,
default partb_outputs/)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from . import common as C

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "training"))
TMP = Path(os.environ.get("NCALLS_TMP", "tmp")) / "partb"
OUT = Path(os.environ.get("PARTB_OUT", "partb_outputs"))


def config(smoke: bool = False) -> dict:
    cfg = json.load(open(PKG / "training" / "part_b_config.json"))
    tag = "_smoke" if smoke else ""
    cfg.update(calls_root=str(TMP / "calls"), pairs_path=str(TMP / f"qa{tag}"), passages_path=str(TMP / f"passages{tag}"),
               splits_path=str(TMP / "shared" / "proposed_splits.parquet"), prepared_root=str(TMP / f"prepared{tag}"),
               output_root=str(OUT), create_date_splits_if_missing=True)
    # SubjECTive-QA: sqa_<seed>_<split>.parquet files next to the bundle (copied by layout), else the
    # authors' splits are downloaded from Hugging Face (gated: accept access and log in first)
    cfg["subjective"].update(data_root=str(TMP / "shared" / "subjectiveqa"))
    cfg["heads"].update(precision=os.environ.get("PARTB_HEADS_PRECISION", "fp16"))
    return cfg


def _link(src: Path, dst: Path):
    try:
        os.symlink(src, dst)
    except FileExistsError:                   # another job of the same session made it
        pass


def layout(smoke: bool = False, corpus: bool = True):
    """B's folder layout from the bundle; calls are all US (country set accordingly). Safe when two
    jobs of one session call it at once (links tolerate existing ones, files are written atomically)."""
    d = C.data_dir()
    (TMP / "calls").mkdir(parents=True, exist_ok=True)
    if not (TMP / "calls" / "calls.parquet").exists():
        m = pd.read_parquet(d / "calls_us.parquet", columns=["transcriptid", "call_date"])
        m["country"] = "United States"
        tmp = TMP / "calls" / f".calls.{os.getpid()}.parquet"
        m.to_parquet(tmp, index=False)
        os.replace(tmp, TMP / "calls" / "calls.parquet")
    if corpus:
        tag = "_smoke" if smoke else ""
        qa_dir, p_dir = TMP / f"qa{tag}", TMP / f"passages{tag}"
        qa_dir.mkdir(parents=True, exist_ok=True); p_dir.mkdir(parents=True, exist_ok=True)
        years = [2011, 2012, 2013, 2024] if smoke else list(range(2000, 2027))
        for f in sorted(d.glob("qa_us_*.parquet")):
            if int(f.stem.split("_")[-1]) in years and not (qa_dir / f.name).exists():
                _link(f, qa_dir / f.name)
        for f in sorted(d.glob("passages_us_*.parquet")):
            if int(f.stem.split("_")[-1]) in years and not (p_dir / f.name).exists():
                p = pd.read_parquet(f, columns=["passage_id", "transcriptid", "text", "uniform"])
                tmp = p_dir / f".{f.stem}.{os.getpid()}.tmp"
                p[p["uniform"].fillna(False)].drop(columns="uniform").to_parquet(tmp, index=False)
                os.replace(tmp, p_dir / f.name)
    sq = C._find("sqa_5768_train.parquet", [str(d)])
    if sq is not None:
        for s in [5768, 78516, 944601]:
            r = TMP / "shared" / "subjectiveqa" / str(s)
            r.mkdir(parents=True, exist_ok=True)
            for split in ["train", "val", "test"]:
                if not (r / f"{split}.parquet").exists():
                    tmp = r / f".{split}.{os.getpid()}.tmp"
                    shutil.copyfile(sq / f"sqa_{s}_{split}.parquet", tmp)
                    os.replace(tmp, r / f"{split}.parquet")


def start_checkpoint(init: str) -> Path:
    """The checkpoint a stage-2 run starts from (see --init)."""
    if init.startswith("adapted:"):
        return OUT / "adapted" / init.split(":", 1)[1] / "best"
    path = C.models_dir() / init
    return path / "best" if (path / "best" / "config.json").exists() else path


def head_start(m: str) -> Path:
    """The encoder a head starts from: a run's export (adapted/<run>/model), else the released model
    (nolbert_calls_2023)."""
    local = OUT / "adapted" / m / "model"
    if (local / "config.json").exists():
        return local
    if m == "nolbert_calls_2023" and C.RELEASED.exists():
        return C.released()
    raise FileNotFoundError(f"no encoder for {m}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["adapt", "heads", "summarize"])
    ap.add_argument("--name", default="nolbert_calls_2023", help="name of the stage-2 run")
    ap.add_argument("--init", default="nolbert_calls_2012", help="starting checkpoint: a folder key or adapted:<run>")
    ap.add_argument("--recipe", default="{}", help="JSON overrides of the recipe in training/part_b_config.json")
    ap.add_argument("--task", default="subjective", help="subjective, cola, sst2 or qnli")
    ap.add_argument("--models", nargs="*", default=["nolbert_calls_2023"])
    ap.add_argument("--features", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    import part_b_pipeline as P
    cfg = config(a.smoke)
    if a.cmd == "adapt":
        path = start_checkpoint(a.init)
        base = f"start_{a.name}"
        cfg["models"][base] = {"path": str(path), "revision": None}
        C.log(f"{a.name}: starts from {path}")
        P.ADAPTATIONS[a.name] = (base, "main")
        cfg["recipe"].update(json.loads(a.recipe))
        # one corpus and split file per run, so that two jobs of one session never write the same files
        cfg["splits_path"] = str(TMP / "shared" / a.name / "proposed_splits.parquet")
        cfg["prepared_root"] = str(TMP / f"prepared_{a.name}{'_smoke' if a.smoke else ''}")
        layout(a.smoke)
        C.log("corpus layout ready; preparing the trainer's corpus")
        if Path(cfg["prepared_root"]).exists():
            prepared, counts = P.load_adaptation_data(cfg)
        else:
            prepared, counts = P.prepare_adaptation_data(cfg)
        print(counts.to_string(index=False), flush=True)
        if not a.smoke:
            # the validation calls must be the 300 calls of pipeline/val_calls_main.json
            import part_b_adaptation as A
            corpus = A.Corpus(prepared["database"], "main")
            mine = A.validation_calls(corpus, cfg["recipe"])
            corpus.close()
            ref = json.load(open(C.data_dir() / C.PROTOCOLS["main"]["val_calls"]))
            C.log(f"validation calls identical to val_calls_main.json: {sorted(mine) == sorted(ref)} ({len(mine)})")
        out = P.run_adaptation(cfg, a.name, prepared, execute=True, smoke=a.smoke)
        C.log(f"adaptation exported: {out}")
    elif a.cmd == "heads":
        layout(a.smoke, corpus=False)
        for m in a.models:
            if m not in cfg["models"]:
                cfg["models"][m] = {"path": str(head_start(m)), "revision": None}
                C.log(f"{m}: {cfg['models'][m]['path']}")
        if a.task == "subjective":
            for seed in (a.seeds or cfg["subjective"]["seeds"]):
                frames, prov = P.load_subjective(cfg, seed)
                for m in a.models:
                    for feat in (a.features or P.FEATURES):
                        tag = f"{m}__{feat}__seed{seed}"
                        if (OUT / "heads" / "subjective" / tag / "metrics.json").exists():
                            continue
                        try:
                            P.train_head(cfg, m, "subjective", seed, frames, prov, feature=feat, smoke=a.smoke)
                            C.log(f"head done: {tag}")
                        except Exception as e:
                            C.log(f"head FAILED: {tag}: {str(e)[:300]}")
        else:
            frames, prov = P.prepare_glue(cfg, a.task)
            C.log(f"GLUE {a.task}: " + ", ".join(f"{k} {len(v):,}" for k, v in frames.items()))
            for m in a.models:
                for seed in (a.seeds or cfg["glue"]["seeds"]):
                    try:
                        P.train_head(cfg, m, a.task, seed, frames, prov, smoke=a.smoke)
                        C.log(f"head done: {m} {a.task} {seed}")
                    except Exception as e:
                        C.log(f"head FAILED: {m} {a.task} {seed}: {str(e)[:300]}")
        P.summarize(cfg["output_root"])
    else:
        P.summarize(cfg["output_root"])


if __name__ == "__main__":
    main()
