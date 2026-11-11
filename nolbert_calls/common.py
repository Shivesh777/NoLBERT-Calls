"""Shared data access, the reader (encoder), word overlap and retrieval metrics.

A reader embeds text the way NoLBERT-Calls was trained: mean of the last layer over all
non-padding tokens (special tokens included), L2-normalised. Questions are cut at 128 tokens
and answers at the length in the checkpoint's embedding_config.json (256 for the released model).

Reader keys:
  nolbert_calls_2023  a stage-2 run of that name (partb_outputs/adapted/), else the released model in model/
  nolbert_calls_2012  stage 1 (pipeline/c04_finetune.py output in NCALLS_MODELS)
  nolbert             the base model, NoLBERT (Hugging Face, pinned revision)
  <run name>          any other run in partb_outputs/adapted/"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0
T0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
RELEASED = ROOT / "model" / "nolbert_calls_2023"


def log(*a):
    print(f"[{(time.time() - T0) / 60:7.1f} min]", *a, flush=True)


# --- locations ---------------------------------------------------------------------------
def _find(name: str, roots) -> Path | None:
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        if (r / name).exists():
            return r
        for p in r.rglob(name):
            return p.parent
    return None


def data_dir() -> Path:
    """The US-only bundle of pipeline/c05_bundle.py (NCALLS_DATA, default data/bundle)."""
    d = Path(os.environ.get("NCALLS_DATA", ROOT / "data" / "bundle"))
    if not (d / "calls_us.parquet").exists():
        raise SystemExit(f"data bundle not found in {d} (pipeline/c05_bundle.py writes it; or set NCALLS_DATA)")
    return d


def models_dir() -> Path:
    """Starting checkpoints, one folder per key (NCALLS_MODELS, default data/encoders: stage 1 writes there)."""
    return Path(os.environ.get("NCALLS_MODELS", ROOT / "data" / "encoders"))


def trained_dir() -> Path:
    """Stage-2 runs: <PARTB_OUT>/adapted/<run>/ (NCALLS_TRAINED overrides)."""
    d = os.environ.get("NCALLS_TRAINED")
    if d:
        return Path(d)
    return Path(os.environ.get("PARTB_OUT", "partb_outputs")) / "adapted"


def released() -> Path:
    """The released model folder, its split weights merged (and checksum-verified) on first use."""
    if not (RELEASED / "model.safetensors").exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("merge_model", ROOT / "model" / "merge_model.py")
        merge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(merge)
        merge.merge(RELEASED)
    return RELEASED


# --- data -----------------------------------------------------------------------------------
PROTOCOLS = {
    "main": {"train_end": "2023-12-31", "val": ("2024-01-01", "2024-12-31"), "test": ("2025-01-01", "2026-10-31"),
             "val_calls": "val_calls_main.json"},
}


def meta() -> pd.DataFrame:
    m = pd.read_parquet(data_dir() / "calls_us.parquet")
    m["call_date"] = pd.to_datetime(m["call_date"])
    return m


def read_qa(ids, years=None, columns=("transcriptid", "ex", "question", "answer")) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    ids = pa.array(np.asarray(sorted(set(int(i) for i in ids)), dtype=np.int64))
    frames = []
    for f in sorted(data_dir().glob("qa_us_*.parquet")):
        y = int(f.stem.split("_")[-1])
        if years is not None and y not in years:
            continue
        t = pq.read_table(f, columns=list(columns))
        t = t.filter(pc.is_in(t["transcriptid"], value_set=ids))
        if t.num_rows:
            frames.append(t.to_pandas())
    qa = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(columns))
    qa = qa.dropna(subset=[c for c in ("question", "answer") if c in qa])
    return qa.sort_values(["transcriptid", "ex"]).reset_index(drop=True)


def read_passages(ids=None, years=None, uniform_only=False) -> pd.DataFrame:
    frames = []
    for f in sorted(data_dir().glob("passages_us_*.parquet")):
        y = int(f.stem.split("_")[-1])
        if years is not None and y not in years:
            continue
        p = pd.read_parquet(f)
        if ids is not None:
            p = p[p["transcriptid"].isin(set(ids))]
        if uniform_only:
            p = p[p["uniform"].fillna(False)]
        frames.append(p)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def years_between(lo: str, hi: str):
    return set(range(pd.Timestamp(lo).year, pd.Timestamp(hi).year + 2))


# --- readers ---------------------------------------------------------------------------------
HF = {"nolbert": ("alikLab/NoLBERT", "0b0359c62b3f7031c8f0ffa9d323881e829a147b")}
LABEL = {
    "nolbert": "NoLBERT (base model, 1976-95 text)",
    "nolbert_calls_2012": "NoLBERT-Calls-2012 (stage 1, calls up to 2012)",
    "nolbert_calls_2023": "NoLBERT-Calls-2023 (released)",
}


def source(key: str):
    """(path or Hugging Face repo, revision) of a reader key."""
    if key in HF:
        return HF[key]
    for root in [models_dir(), trained_dir()]:
        for cand in [root / key / "best", root / key]:
            if (cand / "config.json").exists():
                return str(cand), None
    if key == "nolbert_calls_2023" and RELEASED.exists():
        return str(released()), None
    raise FileNotFoundError(key)


def load_tokenizer(key: str, path: str, revision):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, revision=revision)


def load_model(path: str, revision=None, mlm: bool = False, **kw):
    from transformers import AutoModel, AutoModelForMaskedLM
    return (AutoModelForMaskedLM if mlm else AutoModel).from_pretrained(path, revision=revision, **kw)


class Encoder:
    def __init__(self, key: str, device: str | None = None, max_len: int = 128):
        import torch
        self.torch = torch
        self.key = key
        path, revision = source(key)
        self.path, self.revision = path, revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer(key, path, revision)
        self.model = load_model(path, revision).to(self.device).eval()
        self.pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        # the lengths a checkpoint was trained with (embedding_config.json); 128/128 otherwise
        self.q_len = self.a_len = max_len
        try:
            ec = json.load(open(Path(path) / "embedding_config.json"))
            self.q_len, self.a_len = int(ec.get("question_length", max_len)), int(ec.get("answer_length", max_len))
        except Exception:
            pass
        self.dim = self.model.config.hidden_size
        self.fp32_rows = 0

    def info(self) -> dict:
        return {"reader": self.key, "path": self.path, "revision": self.revision,
                "question_length": self.q_len, "answer_length": self.a_len}

    def _forward(self, x, mask, half: bool):
        torch = self.torch
        with torch.autocast("cuda", torch.float16, enabled=half):
            hidden = self.model(input_ids=x, attention_mask=mask).last_hidden_state
        hidden = hidden.float()
        w = mask.unsqueeze(-1).to(hidden.dtype)
        z = (hidden * w).sum(1) / w.sum(1).clamp(min=1e-6)
        return torch.nn.functional.normalize(z, dim=-1)

    def embed(self, texts, side: str = "answer", batch: int = 128) -> np.ndarray:
        """side: 'question' (cut at q_len) or 'answer' (cut at a_len)"""
        torch = self.torch
        texts = ["" if t is None else str(t) for t in texts]
        length = self.q_len if side == "question" else self.a_len
        out = np.empty((len(texts), self.dim), dtype=np.float16)
        cuda = self.device.startswith("cuda")
        for lo0 in range(0, len(texts), 65536):           # tokenise in slabs to bound memory
            ids = self.tokenizer(texts[lo0:lo0 + 65536], truncation=True, max_length=length)["input_ids"]
            order = np.argsort([len(x) for x in ids], kind="stable")
            with torch.inference_mode():
                for lo in range(0, len(order), batch):
                    rows = order[lo:lo + batch]
                    width = max(1, max(len(ids[i]) for i in rows))
                    x = np.full((len(rows), width), self.pad, dtype=np.int64)
                    m = np.zeros((len(rows), width), dtype=np.int64)
                    for j, i in enumerate(rows):
                        x[j, :len(ids[i])] = ids[i]
                        m[j, :len(ids[i])] = 1
                    xt = torch.as_tensor(x, device=self.device)
                    mt = torch.as_tensor(m, device=self.device)
                    z = self._forward(xt, mt, cuda)
                    bad = ~torch.isfinite(z).all(1)
                    if bad.any():                      # fp16 overflow: redo those rows in fp32
                        self.fp32_rows += int(bad.sum())
                        z[bad] = self._forward(xt[bad], mt[bad], False)
                    out[lo0 + rows] = z.cpu().numpy().astype(np.float16)
        return out

    def close(self):
        del self.model
        gc.collect()
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass


# --- word overlap ----------------------------------------------------------------------------
EXTRA_STOP = {
    "said", "says", "mr", "ms", "inc", "corp", "co", "ltd", "llc", "company", "companies",
    "share", "shares", "stock", "stocks", "percent", "pct", "quarter", "year", "week", "month",
    "reuters", "nasdaq", "nyse", "views", "opinions", "author", "herein", "com", "www", "http",
    "https", "zacks", "motley", "fool", "benzinga", "rttnews", "symbol",
}
STOP_CALLS = {
    "thank", "thanks", "question", "questions", "quarter", "quarters", "year", "years",
    "think", "just", "going", "really", "know", "yes", "okay", "sure", "great", "good",
    "operator", "call", "guys", "lot", "bit", "maybe", "kind", "sort", "look", "looking",
    "right", "said", "say", "mean", "actually", "basically", "obviously", "certainly",
    "little", "want", "talk", "talked", "talking", "color", "line", "morning", "afternoon",
    "evening", "today", "everybody", "everyone", "hi", "hey", "congratulations",
}


def clean_for_words(texts) -> pd.Series:
    s = pd.Series(list(texts)).fillna("").astype(str)
    s = s.str.replace(r"[^A-Za-z\s]", " ", regex=True).str.lower()
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


class WordOverlap:
    """TF-IDF cosine, vocabulary and IDF fitted on training-period text only."""

    def __init__(self, train_texts):
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfTransformer
        stop = sorted(set(ENGLISH_STOP_WORDS) | EXTRA_STOP | STOP_CALLS)
        self.cv = CountVectorizer(stop_words=stop, token_pattern=r"[a-z]{2,}", min_df=3, max_df=0.5,
                                  max_features=100_000, dtype=np.float32)
        counts = self.cv.fit_transform(clean_for_words(train_texts))
        self.tfidf = TfidfTransformer(sublinear_tf=True).fit(counts)

    def tfidf_vectors(self, texts):
        return self.tfidf.transform(self.cv.transform(clean_for_words(texts)))


# --- within-call retrieval -------------------------------------------------------------------
def call_bounds(calls: np.ndarray):
    calls = np.asarray(calls)
    starts = np.flatnonzero(np.r_[True, calls[1:] != calls[:-1]])
    ends = np.r_[starts[1:], len(calls)]
    return list(zip(starts, ends))


def ranks_from_blocks(blocks) -> np.ndarray:
    """Rank (1 = best) of each question's own answer among its call's answers; ties count against."""
    out = []
    for s in blocks:
        s = np.asarray(s, dtype=np.float64)
        own = np.diag(s)[:, None]
        out.append(1 + (s >= own).sum(axis=1) - 1)
    return np.concatenate(out).astype(np.int32) if out else np.array([], dtype=np.int32)


def emb_blocks(q: np.ndarray, a: np.ndarray, bounds):
    for lo, hi in bounds:
        yield q[lo:hi].astype(np.float32) @ a[lo:hi].astype(np.float32).T


def zrows(s: np.ndarray) -> np.ndarray:
    sd = s.std(axis=1, keepdims=True)
    return (s - s.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)


def summary(ranks: np.ndarray, calls: np.ndarray, sizes: np.ndarray, n_boot: int = 1000, seed: int = SEED) -> dict:
    ranks, calls, sizes = map(np.asarray, (ranks, calls, sizes))
    hit = (ranks == 1).astype(float)
    codes, uniq = pd.factorize(calls)
    ph = np.bincount(codes, hit)
    pn = np.bincount(codes)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = ph[draws].sum(1) / pn[draws].sum(1)
    return {"questions": int(len(ranks)), "calls": int(len(uniq)), "top1": float(hit.mean()),
            "top1_lo": float(np.quantile(boot, 0.025)), "top1_hi": float(np.quantile(boot, 0.975)),
            "top1_call_weighted": float((ph / pn).mean()), "top3": float((ranks <= 3).mean()),
            "mrr": float((1.0 / ranks).mean()), "chance_top1": float((1.0 / sizes).mean()),
            "median_candidates": float(np.median(sizes))}
