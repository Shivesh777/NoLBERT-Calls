"""Shared settings and helpers of the data pipeline and the stage-1 fine-tune (c01-c04).

    c01 download   every CIQ earnings call (event type 48) on WRDS, every firm
    c02 clean      the professor's cleaning rules, text stored once per speaker turn
    c03 pairs      analyst question -> management answer exchanges, call passages
    c04 finetune   NoLBERT-Calls: masked-LM on call text + contrastive pairs
                   (question -> answer, two passages of one call), calls up to FT_END only
    c05 bundle     the compact US-only tables that training and evaluation read

Time discipline: the encoder sees calls up to FT_END; choices (checkpoint, probe
regularisation) use VALIDATION calls; TEST calls are read once at the end.

WRDS / S&P licence: transcript text and anything derived from it at the call level stays
under data/calls/ (git-ignored), on our machines. Never commit it, never send it to an
API-based model.
"""
from __future__ import annotations

from pathlib import Path
import os
import re
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NCALLS_DATA_ROOT", ROOT / "data"))   # licensed: git-ignored

# --- paths ---------------------------------------------------------------------------
CALLS = DATA / "calls"
RAW = CALLS / "raw"
COMPONENTS = RAW / "components"           # <year>/part_XXXX.parquet, one row per speaker turn
DETAIL = RAW / "detail_all_versions.parquet"
TRANSCRIPT_LIST = RAW / "transcript_list.parquet"
COMPANY = RAW / "company.parquet"
META = CALLS / "calls.parquet"            # one row per kept call (c02)
DROPPED = CALLS / "dropped.parquet"
TURNS = CALLS / "turns"                   # <year>.parquet, cleaned speaker turns (c02)
QA = CALLS / "qa"                         # <year>.parquet, question -> answer exchanges (c03)
PASSAGES = CALLS / "passages"             # <year>.parquet, passages of management text (c03)
ENCODERS = DATA / "encoders"
RESULTS = ROOT / "results"

# --- periods -------------------------------------------------------------------------
FT_END = "2012-12-31"                     # the fine-tune reads calls up to this day only
SPLITS = {                                # by call date
    "train": ("2000-01-01", FT_END),
    "validation": ("2013-01-01", "2015-12-31"),
    "test": ("2016-01-01", "2026-12-31"),
}

NOLBERT = "alikLab/NoLBERT"
NOLBERT_REVISION = "0b0359c62b3f7031c8f0ffa9d323881e829a147b"
SEED = 0

# Speaker-turn types in wrds_transcript_person.transcriptcomponenttypename
QUESTION, ANSWER = "Question", "Answer"


def log(message: str, start=[time.time()]) -> None:
    print(f"[{(time.time() - start[0]) / 60:6.1f} min] {message}", flush=True)


def years_on_disk(folder: Path) -> list[int]:
    return sorted(int(p.stem) for p in folder.glob("*.parquet") if p.stem.isdigit())


def n_words(s: pd.Series) -> np.ndarray:
    return s.fillna("").str.count(r"\S+").to_numpy()


def word_chunks(text: str, size: int, min_words: int) -> list[str]:
    """text cut into consecutive windows of `size` words; a tail shorter than min_words
    is merged into the window before it."""
    words = text.split()
    if len(words) < min_words:
        return []
    out = [words[i:i + size] for i in range(0, len(words), size)]
    if len(out) > 1 and len(out[-1]) < min_words:
        out[-2].extend(out.pop())
    return [" ".join(w) for w in out]


# --- encoders ------------------------------------------------------------------------
def resolve_encoder(name: str) -> tuple[str, str | None]:
    """(path or repo, revision): 'nolbert' is the pinned base model; a name without a slash
    that is a folder of ENCODERS is that folder; anything else is passed through."""
    if name.lower() in ("nolbert", "base", NOLBERT.lower()):
        return NOLBERT, NOLBERT_REVISION
    if (ENCODERS / name).is_dir():
        return str(ENCODERS / name), None
    return name, None


def load_tokenizer(path: str, revision: str | None = None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, revision=revision)


def load_model(path: str, revision: str | None = None, mlm: bool = False, **kw):
    from transformers import AutoModel, AutoModelForMaskedLM
    return (AutoModelForMaskedLM if mlm else AutoModel).from_pretrained(path, revision=revision, **kw)


# --- within-call answer retrieval ----------------------------------------------------
def own_answer_ranks(score_blocks) -> np.ndarray:
    """Rank (1 = best) of each question's own answer among the answers of its call.

    score_blocks: iterable of square matrices S (questions x answers of one call, row i's
    own answer in column i). Ties count against the own answer (rank = 1 + number of
    answers scoring at least as high, other than itself)."""
    out = []
    for s in score_blocks:
        s = np.asarray(s, dtype=np.float64)
        own = np.diag(s)[:, None]
        beat = (s >= own).sum(axis=1) - 1
        out.append(1 + beat)
    return np.concatenate(out) if out else np.array([], dtype=int)


def blocks_from_embeddings(q: np.ndarray, a: np.ndarray, calls: np.ndarray):
    """Cosine-score blocks per call for row-aligned unit-length embeddings."""
    calls = np.asarray(calls)
    bounds = np.flatnonzero(np.r_[True, calls[1:] != calls[:-1], True])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        yield q[lo:hi] @ a[lo:hi].T


# --- markup cleaning (the professor's rule 5) ---------------------------------------
BLOCK_TAGS = re.compile(r"(?i)</?(?:p|br|div|li|ul|ol|h[1-6]|table|tr|td)\b[^>]*>")
INLINE_TAGS = re.compile(r"(?i)</?(?:strong|b|i|em|u|span|font|a|sup|sub)\b[^>]*>")
ENTITY = re.compile(r"&(?:[A-Za-z]+|#[0-9]+|#x[0-9A-Fa-f]+);")
TAG_MARKERS = re.compile(r"</?TAG>")
AUTHOR_NOTES = re.compile(r"\[Author ID[0-9]*:[^\]]*\]")
SA_BOILERPLATE = re.compile(
    r"(?m)^[^\n]*(?:Copyright policy:|TEXTUAL REPRESENTATION OF THE APPLICABLE COMPANY|IN NO WAY DOES SEEKING ALPHA"
    r"|additional questions about our online transcripts|To sponsor a Seeking Alpha transcript|Did you know\? Seeking Alpha"
    r"|transcripts available on the Seeking Alpha|attribute the transcript to Seeking Alpha)[^\n]*\n?")


def clean_markup(text: pd.Series) -> pd.Series:
    import html

    s = text.astype(object)
    s = s.str.replace(TAG_MARKERS, "", regex=True)
    s = s.str.replace(AUTHOR_NOTES, "", regex=True)
    s = s.str.replace(SA_BOILERPLATE, "", regex=True)
    s = s.str.replace(BLOCK_TAGS, " ", regex=True)
    s = s.str.replace(INLINE_TAGS, "", regex=True)
    has = s.str.contains(ENTITY, regex=True, na=False)
    if has.any():
        s.loc[has] = s.loc[has].map(html.unescape)
    return s
