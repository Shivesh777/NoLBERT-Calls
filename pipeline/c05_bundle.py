"""c05: the compact US-only bundle that training and evaluation read (licensed data: keep private).

python pipeline/c05_bundle.py meta
python pipeline/c05_bundle.py qa 2004 2026          # years, inclusive
python pipeline/c05_bundle.py passages 2004 2026
python pipeline/c05_bundle.py turns 2004 2026
python pipeline/c05_bundle.py role                   # after turns

Reads data/calls/ (c02, c03), writes data/bundle/:
  calls_us.parquet            one row per US call (a transcript is US when any linked company is)
  qa_us_<year>.parquet        every question -> answer exchange of US calls
  passages_us_<year>.parquet  four passages per call drawn uniformly ("uniform"; the training pairs)
                              plus up to two per source (prepared remarks / answers; tests 2 and 4)
  role_turns.parquet          analyst questions and executive answers for the role probe
  val_calls_{main,time}.json  the fixed validation calls (copied from pipeline/)
Prints counts only, never text."""
import json, os, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NCALLS_DATA_ROOT", ROOT / "data"))
SRC = DATA / "calls"
OUT = Path(os.environ.get("NCALLS_DATA", DATA / "bundle"))
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
US = "United States"


def us_meta():
    f = OUT / "calls_us.parquet"
    if f.exists():
        return pd.read_parquet(f)
    m = pd.read_parquet(SRC / "calls.parquet", columns=["transcriptid", "companyid", "gvkey", "country",
                                                        "call_date", "n_questions", "has_qa"])
    us = m.groupby("transcriptid")["country"].apply(lambda s: (s == US).any())
    m = m[m["transcriptid"].isin(us.index[us])].drop_duplicates("transcriptid")
    m["call_date"] = pd.to_datetime(m["call_date"])
    m["year"] = m["call_date"].dt.year.astype("int16")
    m = m.drop(columns=["country"]).sort_values("transcriptid").reset_index(drop=True)
    m.to_parquet(f, compression="zstd")
    return m


def key(ids, salt):
    """deterministic pseudo-random key per id (uniform sampling without holding RNG state)"""
    x = (np.asarray(ids, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(salt)) & np.uint64(0xFFFFFFFFFFFF)
    x ^= x >> np.uint64(17)
    x = (x * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFF)
    return x


def do_qa(years):
    ids = pa.array(us_meta()["transcriptid"].to_numpy())
    for y in years:
        f = SRC / "qa" / f"{y}.parquet"
        o = OUT / f"qa_us_{y}.parquet"
        if not f.exists() or o.exists():
            continue
        t = pq.read_table(f, columns=["transcriptid", "ex", "question", "answer", "q_words", "a_words"])
        n0 = t.num_rows
        t = t.filter(pa.compute.is_in(t["transcriptid"], value_set=ids))
        pq.write_table(t, o, compression="zstd")
        print(y, "qa rows", n0, "->", t.num_rows, f"{o.stat().st_size/1e6:.0f} MB", flush=True)


def do_passages(years):
    meta = us_meta().set_index("transcriptid")
    ids = pa.array(meta.index.to_numpy())
    for y in years:
        f = SRC / "passages" / f"{y}.parquet"
        o = OUT / f"passages_us_{y}.parquet"
        if not f.exists() or o.exists():
            continue
        k = pq.read_table(f, columns=["passage_id", "transcriptid", "source"])
        k = k.filter(pa.compute.is_in(k["transcriptid"], value_set=ids)).to_pandas()
        n_us = len(k)
        k["r"] = key(pd.factorize(k["passage_id"])[0] if k["passage_id"].dtype == object else k["passage_id"], SEED)
        k = k.sort_values(["transcriptid", "r"])
        k["rank_all"] = k.groupby("transcriptid").cumcount()
        k["rank_src"] = k.groupby(["transcriptid", "source"]).cumcount()
        # uniform 4 per call (training pairs: two distinct passages of a call) + up to 2 per source (eval tasks)
        keep = k[(k["rank_all"] < 4) | (k["rank_src"] < 2)].copy()
        keep["uniform"] = keep["rank_all"] < 4
        sel = pa.array(keep["passage_id"].to_numpy())
        frames = []
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=100_000, columns=["passage_id", "transcriptid", "source", "text", "n_words"]):
            t = pa.Table.from_batches([b])
            t = t.filter(pa.compute.is_in(t["passage_id"], value_set=sel))
            if t.num_rows:
                frames.append(t)
        if not frames:
            print(y, "no US passages"); continue
        t = pa.concat_tables(frames).to_pandas()
        t = t.merge(keep[["passage_id", "uniform"]], on="passage_id", how="left")
        t.to_parquet(o, compression="zstd", index=False)
        print(y, "passages US", n_us, "kept", len(t), "uniform", int(t["uniform"].sum()),
              "sources", t["source"].value_counts().to_dict(), f"{o.stat().st_size/1e6:.0f} MB", flush=True)


PERIODS = {"main_train": ("1900-01-01", "2023-12-31"), "main_val": ("2024-01-01", "2024-12-31"),
           "main_test": ("2025-01-01", "2026-10-31"), "time_train": ("1900-01-01", "2012-12-31"),
           "time_val": ("2013-01-01", "2015-12-31"), "time_test": ("2016-01-01", "2026-10-31")}


def role_calls():
    """1,500 random US calls per period (seeded); a call may serve two periods (e.g. main_train and time_train)."""
    m = us_meta()
    rng = np.random.default_rng(SEED)
    out = {}
    for p, (lo, hi) in PERIODS.items():
        ids = m[(m["call_date"] >= lo) & (m["call_date"] <= hi) & m["has_qa"]]["transcriptid"].to_numpy()
        out[p] = sorted(rng.permutation(ids)[:1500].tolist())
    return out


def do_turns(years):
    calls = role_calls()
    allids = pa.array(sorted({i for v in calls.values() for i in v}))
    for y in years:
        f = SRC / "turns" / f"{y}.parquet"
        o = OUT / f"_turns_role_{y}.parquet"
        if not f.exists() or o.exists():
            continue
        frames = []
        for b in pq.ParquetFile(f).iter_batches(batch_size=200_000, columns=["transcriptid", "componentorder", "section", "ctype", "speaker", "text", "n_words"]):
            t = pa.Table.from_batches([b])
            t = t.filter(pa.compute.is_in(t["transcriptid"], value_set=allids))
            if t.num_rows:
                d = t.to_pandas()
                d = d[d["section"].eq("qa") & (d["n_words"] >= 12)]
                d = d[(d["ctype"].eq("Question") & d["speaker"].eq("Analysts")) | (d["ctype"].eq("Answer") & d["speaker"].eq("Executives"))]
                frames.append(d)
        d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        d.to_parquet(o, compression="zstd", index=False)
        print(y, "role turns", len(d), flush=True)


def do_role():
    calls = role_calls()
    t = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("_turns_role_*.parquet"))], ignore_index=True)
    t["y"] = t["speaker"].eq("Analysts").astype("int8")
    t["text"] = t["text"].str.replace("?", " ", regex=False)
    out = []
    for p, ids in calls.items():
        d = t[t["transcriptid"].isin(set(ids))]
        d = pd.concat([g.sample(min(len(g), 3000), random_state=SEED) for _, g in d.groupby("y")])
        out.append(d.assign(period=p)[["period", "transcriptid", "y", "text"]])
        print(p, len(d), "analyst share", round(d["y"].mean(), 3))
    pd.concat(out, ignore_index=True).to_parquet(OUT / "role_turns.parquet", compression="zstd", index=False)


if __name__ == "__main__":
    t0 = time.time()
    cmd = sys.argv[1]
    yrs = list(range(int(sys.argv[2]), int(sys.argv[3]) + 1)) if len(sys.argv) > 3 else []
    if cmd == "meta":
        m = us_meta(); print("US calls", len(m), m["call_date"].min(), m["call_date"].max())
        # the 300 validation calls of each protocol, as drawn by training/part_b_adaptation.py (seed 0)
        for p in ["main", "time"]:
            shutil.copyfile(Path(__file__).resolve().parent / f"val_calls_{p}.json", OUT / f"val_calls_{p}.json")
    elif cmd == "qa":
        do_qa(yrs)
    elif cmd == "passages":
        do_passages(yrs)
    elif cmd == "turns":
        do_turns(yrs)
    elif cmd == "role":
        do_role()
    print(f"[{time.time() - t0:.0f}s]")
