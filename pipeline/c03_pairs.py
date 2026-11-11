"""Question -> answer exchanges and passages of every cleaned call.

    python pipeline/c03_pairs.py [--passage-words 200] [--min-passage-words 40]

Exchanges (data/calls/qa/<year>.parquet, one row per exchange): in the Q&A section a run
of consecutive "Question" turns (an analyst's question and any follow-up before the
answer) is the question; the "Answer" turns that follow, up to the next question, are
the answer (operator messages are skipped). Kept when the question has at least
--min-q-words and the answer at least --min-a-words words. Columns: transcriptid, ex
(1, 2, ... in call order), analyst, answerers, question, answer, q_words, a_words.

Passages (data/calls/passages/<year>.parquet): management's words cut into windows of
--passage-words words: every non-operator turn of the prepared remarks (source "pres")
and every answer of an exchange (source "answer"). A turn or answer shorter than
--min-passage-words is skipped; a short tail joins the window before it. Columns:
passage_id (transcriptid * 10000 + position), transcriptid, source, text, n_words.

These are the units of stage 1 (c04: question -> answer pairs, and two passages of one call)
and, through the US-only bundle (c05), of stage 2 and the evaluation (nolbert_calls/).
"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np
import pandas as pd


def exchanges(turns: pd.DataFrame, min_q: int, min_a: int) -> pd.DataFrame:
    t = turns[turns["section"].eq("qa")].sort_values(["transcriptid", "componentorder"]).reset_index(drop=True)
    if t.empty:
        return pd.DataFrame(columns=["transcriptid", "ex", "analyst", "answerers", "question", "answer",
                                     "q_words", "a_words"])
    is_q = t["ctype"].eq(C.QUESTION).to_numpy()
    is_a = t["ctype"].eq(C.ANSWER).to_numpy()
    same = np.r_[False, t["transcriptid"].to_numpy()[1:] == t["transcriptid"].to_numpy()[:-1]]
    prev_q = np.r_[False, is_q[:-1]] & same
    start = is_q & ~prev_q
    t["ex"] = pd.Series(start.astype(int)).groupby(t["transcriptid"]).cumsum().to_numpy()
    t = t[t["ex"] > 0]
    is_q, is_a = t["ctype"].eq(C.QUESTION), t["ctype"].eq(C.ANSWER)
    key = ["transcriptid", "ex"]
    q = t[is_q].groupby(key).agg(question=("text", "\n".join), q_words=("n_words", "sum"),
                                 analyst=("name", "first"))
    a = t[is_a].groupby(key).agg(answer=("text", "\n".join), a_words=("n_words", "sum"),
                                 answerers=("name", lambda x: "; ".join(dict.fromkeys(x))))
    out = q.join(a, how="inner").reset_index()
    out = out[(out["q_words"] >= min_q) & (out["a_words"] >= min_a)]
    out["ex"] = out.groupby("transcriptid").cumcount() + 1
    return out[["transcriptid", "ex", "analyst", "answerers", "question", "answer", "q_words", "a_words"]]


def passages(turns: pd.DataFrame, qa: pd.DataFrame, size: int, min_words: int) -> pd.DataFrame:
    pres = turns[turns["section"].eq("pres") & ~turns["speaker"].str.contains("Operator", na=False)
                 & ~turns["ctype"].str.contains("Operator", na=False)
                 & (turns["n_words"] >= min_words)].sort_values(["transcriptid", "componentorder"])
    units = pd.concat([
        pd.DataFrame({"transcriptid": pres["transcriptid"].to_numpy(), "source": "pres", "text": pres["text"].to_numpy()}),
        pd.DataFrame({"transcriptid": qa["transcriptid"].to_numpy(), "source": "answer", "text": qa["answer"].to_numpy()}),
    ], ignore_index=True)
    tid, src, txt = [], [], []
    for i, s, text in zip(units["transcriptid"].to_numpy(), units["source"].to_numpy(), units["text"].to_numpy()):
        for chunk in C.word_chunks(text, size, min_words):
            tid.append(i); src.append(s); txt.append(chunk)
    out = pd.DataFrame({"transcriptid": np.asarray(tid, dtype=np.int64), "source": src, "text": txt})
    pos = out.groupby("transcriptid").cumcount().to_numpy()
    if len(pos) and pos.max() >= 10_000:
        raise ValueError("more than 9,999 passages in one call")
    out.insert(0, "passage_id", out["transcriptid"].to_numpy() * 10_000 + pos)
    out["n_words"] = C.n_words(out["text"]).astype("int32")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-q-words", type=int, default=8)
    ap.add_argument("--min-a-words", type=int, default=20)
    ap.add_argument("--passage-words", type=int, default=200)
    ap.add_argument("--min-passage-words", type=int, default=40)
    a = ap.parse_args()
    C.QA.mkdir(parents=True, exist_ok=True)
    C.PASSAGES.mkdir(parents=True, exist_ok=True)
    n_ex = n_pass = 0
    for year in C.years_on_disk(C.TURNS):
        turns = pd.read_parquet(C.TURNS / f"{year}.parquet")
        qa = exchanges(turns, a.min_q_words, a.min_a_words)
        ps = passages(turns, qa, a.passage_words, a.min_passage_words)
        qa.to_parquet(C.QA / f"{year}.parquet", index=False, compression="zstd")
        ps.to_parquet(C.PASSAGES / f"{year}.parquet", index=False, compression="zstd")
        n_ex += len(qa); n_pass += len(ps)
        C.log(f"{year}: {len(qa):,} exchanges in {qa['transcriptid'].nunique():,} calls, "
              f"{len(ps):,} passages ({(ps['source'] == 'pres').mean():.0%} prepared remarks)")
    C.log(f"total {n_ex:,} exchanges, {n_pass:,} passages")


if __name__ == "__main__":
    main()
