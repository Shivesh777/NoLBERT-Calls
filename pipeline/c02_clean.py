"""Clean the downloaded calls with the professor's rules, in a lean speaker-turn layout.

    python pipeline/c02_clean.py [--min-words 500] [--min-en-ratio 0.25] [--years 2010 2011]

Rules (as in 2_clean_ciq_transcripts_local.ipynb, same thresholds; on the professor's
download they kept 413,882 of 414,246 calls), plus an English filter:

1. no text: calls of the list with no speaker turn on WRDS are dropped;
2. missing turns: gaps in the turn numbering (more than --max-missing-turns) are dropped;
3. too short: fewer than --min-words words after markup removal are dropped;
4. exact duplicates: the same text under two events of one company: keep one copy if the
   dates agree, drop every copy if they do not (the call date is then unknown);
5. markup and Seeking Alpha boilerplate are removed from every turn before counting;
6. English only: CIQ has calls in other languages (and English calls with untranslated
   stretches). A call is dropped as not_english when English function words make up less
   than --min-en-ratio of its words (English calls: about 0.45; other languages: under
   0.1), or when turns of 40+ words that are themselves not English (ratio under 0.15)
   hold more than --max-non-en-share of its words. Both numbers are stored per call
   (en_ratio, non_en_share) so the thresholds can be checked; the drop log lists them.

Kept and flagged: pre-recorded calls, several calls on one day, one transcript shared by
several companies, calls without Q&A, Preliminary-only copies.

The text is stored ONCE, one row per speaker turn (data/calls/turns/<year>.parquet:
transcriptid, componentorder, section pres|qa, ctype, speaker, name, text, n_words), next
to one row per call (data/calls/calls.parquet). The notebook's layout (full text +
presentation + Q&A per call) is the same text three times; join the turns of a call to
rebuild any of the three.

Event time: CIQ's mostimportantdateutc + mostimportanttimeutc converted to New York time.
A call that starts at or after 16:00 New York time, or whose time is unknown, can only
move prices the next trading day (after_close = True).
"""
from pathlib import Path
import argparse
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

KEY = ["companyid", "keydevid"]
# Frequent English function words: about 40% of the words of an English call and rare in
# other languages. Words that are also common Portuguese/Spanish/French function words
# (a, as, do, i) are left out; German "in" is the main remaining overlap.
EN_WORDS = ("the of and to in that is we for it on this with our are be have you at was will by so "
            "not but from they an or which there about has more can been would what if all were these some "
            "its also very their just think going see that's we're it's there's we've i'm you're they're "
            "would could should because than when how where who other into over than then those well").split()
EN_RE = r"\b(?:" + "|".join(sorted(set(w.replace("'", "['’]") for w in EN_WORDS), key=len, reverse=True)) + r")\b"
NON_EN_TURN_RATIO = 0.15


def clean_part(comp: pd.DataFrame) -> pd.DataFrame:
    comp = comp.dropna(subset=["componenttext"]).sort_values(["transcriptid", "componentorder"])
    ctype = comp.get("transcriptcomponenttypename", pd.Series("", index=comp.index)).fillna("")
    out = pd.DataFrame({
        "transcriptid": comp["transcriptid"].astype("int64").to_numpy(),
        "componentorder": comp["componentorder"].astype("int32").to_numpy(),
        "section": np.where(ctype.str.contains("Question|Answer", regex=True), "qa", "pres"),
        "ctype": ctype.to_numpy(),
        "speaker": comp.get("speakertypename", pd.Series("", index=comp.index)).fillna("").to_numpy(),
        "name": comp.get("transcriptpersonname", pd.Series("", index=comp.index)).fillna("").to_numpy(),
        "text": C.clean_markup(comp["componenttext"]).to_numpy(),
    })
    arr = pa.array(out["text"].tolist(), type=pa.large_string())
    out["n_words"] = pc.fill_null(pc.count_substring_regex(arr, pattern=r"\S+"), 0).to_numpy().astype("int32")
    out["n_en"] = pc.fill_null(pc.count_substring_regex(arr, pattern=EN_RE, ignore_case=True), 0).to_numpy().astype("int32")
    return out


def turn_stats(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby("transcriptid", sort=False)
    s = pd.DataFrame({
        "first": g["componentorder"].min(), "last": g["componentorder"].max(),
        "n_unique": g["componentorder"].nunique(), "n_components": g.size(),
        "n_words": g["n_words"].sum(),
    })
    s["missing_turns"] = s["last"] - s["first"] + 1 - s["n_unique"]
    s["en_ratio"] = g["n_en"].sum() / s["n_words"].clip(lower=1)
    foreign = (t["n_words"] >= 40) & (t["n_en"] < NON_EN_TURN_RATIO * t["n_words"])
    s["non_en_share"] = (t["n_words"].where(foreign, 0).groupby(t["transcriptid"]).sum()
                         .reindex(s.index) / s["n_words"].clip(lower=1))
    by_sec = t.pivot_table(index="transcriptid", columns="section", values="n_words", aggfunc="sum")
    s["n_words_presentation"] = by_sec.get("pres", pd.Series(dtype=float)).reindex(s.index).fillna(0)
    s["n_words_qa"] = by_sec.get("qa", pd.Series(dtype=float)).reindex(s.index).fillna(0)
    q = t[t["ctype"].eq(C.QUESTION)]
    s["n_questions"] = q.groupby("transcriptid").size().reindex(s.index).fillna(0)
    s["n_analysts"] = q.groupby("transcriptid")["name"].nunique().reindex(s.index).fillna(0)
    s["text_md5"] = g["text"].agg(lambda x: hashlib.md5("\n".join(x).encode("utf-8")).hexdigest())
    return s.drop(columns=["first", "last", "n_unique"]).reset_index()


def event_time(calls: pd.DataFrame) -> pd.DataFrame:
    date = pd.to_datetime(calls["mostimportantdateutc"]).dt.normalize()
    raw = calls.get("mostimportanttimeutc", pd.Series(index=calls.index, dtype=object))
    hhmm = raw.astype(str).str.extract(r"(\d{1,2}:\d{2}(?::\d{2})?)")[0]
    hhmm = hhmm.where(hhmm.str.count(":") == 2, hhmm + ":00")
    clock = pd.to_timedelta(hhmm, errors="coerce")
    known = clock.notna()
    ts = (date + clock.fillna(pd.Timedelta(0))).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    out = pd.DataFrame(index=calls.index)
    out["call_time_known"] = known.to_numpy()
    out["call_ts_ny"] = ts.dt.tz_localize(None).where(known)
    out["call_date"] = ts.dt.tz_localize(None).dt.normalize().where(known, date)
    minutes = ts.dt.hour * 60 + ts.dt.minute
    out["after_close"] = (~known | (minutes >= 16 * 60)).to_numpy()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-words", type=int, default=500)
    ap.add_argument("--max-missing-turns", type=int, default=0)
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--min-en-ratio", type=float, default=0.25)
    ap.add_argument("--max-non-en-share", type=float, default=0.10)
    ap.add_argument("--all-languages", action="store_true", help="skip the English filter")
    ap.add_argument("--years", type=int, nargs="*")
    a = ap.parse_args()

    lst = pd.read_parquet(C.TRANSCRIPT_LIST)
    for c in ["transcriptid", "keydevid", "companyid"]:
        lst[c] = lst[c].astype("int64")
    if a.years:
        lst = lst[lst["year"].isin(a.years)].reset_index(drop=True)
    if C.COMPANY.exists() and "industry" not in lst:
        comp_info = pd.read_parquet(C.COMPANY)
        lst = lst.merge(comp_info[[c for c in ["companyid", "country", "industry"] if c in comp_info
                                   and (c == "companyid" or c not in lst)]], on="companyid", how="left")
    years = sorted(lst["year"].unique())
    C.log(f"{len(lst):,} calls in the list, years {years[0]}-{years[-1]}")

    tmp = C.TURNS / "_all"
    tmp.mkdir(parents=True, exist_ok=True)
    wanted = set(lst["transcriptid"])
    stats = []
    for year in years:
        parts = sorted((C.COMPONENTS / str(year)).glob("part_*.parquet"))
        frames = []
        for f in parts:
            comp = pd.read_parquet(f)
            comp = comp[comp["transcriptid"].isin(wanted)]
            if len(comp):
                frames.append(clean_part(comp))
        if not frames:
            continue
        turns = pd.concat(frames, ignore_index=True).drop_duplicates(["transcriptid", "componentorder"])
        stats.append(turn_stats(turns))
        turns.to_parquet(tmp / f"{year}.parquet", index=False, compression="zstd")
        C.log(f"{year}: {turns['transcriptid'].nunique():,} transcripts, {len(turns):,} turns")
    stats = pd.concat(stats, ignore_index=True)

    ev = lst.merge(stats, on="transcriptid", how="left", indicator="has_text")
    ev["drop_reason"] = None
    ev.loc[ev["has_text"].eq("left_only"), "drop_reason"] = "no_text"
    ev.loc[ev["drop_reason"].isna() & (ev["missing_turns"] > a.max_missing_turns), "drop_reason"] = "missing_turns"
    ev.loc[ev["drop_reason"].isna() & (ev["n_words"] < a.min_words), "drop_reason"] = "too_short"
    if not a.all_languages:          # after the professor's rules, so their counts are unchanged
        foreign = (ev["en_ratio"] < a.min_en_ratio) | (ev["non_en_share"] > a.max_non_en_share)
        ev.loc[ev["drop_reason"].isna() & foreign, "drop_reason"] = "not_english"
    if not a.keep_duplicates:
        live = ev[ev["drop_reason"].isna()]
        dup = live[live.groupby(["companyid", "text_md5"])["keydevid"].transform("size") > 1]
        same_day = dup.groupby(["companyid", "text_md5"])["mostimportantdateutc"].transform("nunique") == 1
        keep_one = (dup[same_day].sort_values(["version_rank", "created_utc", "keydevid"], ascending=[False, False, True])
                    .drop_duplicates(["companyid", "text_md5"]).index)
        ev.loc[dup[same_day].index.difference(keep_one), "drop_reason"] = "duplicate_same_day"
        ev.loc[dup[~same_day].index, "drop_reason"] = "duplicate_different_dates"
    dropped = ev[ev["drop_reason"].notna()]
    dropped[KEY + ["transcriptid", "companyname", "headline", "mostimportantdateutc",
                   "transcriptcollectiontypename", "n_words", "missing_turns", "en_ratio", "non_en_share",
                   "drop_reason"] + (["country"] if "country" in dropped else [])].to_parquet(C.DROPPED, index=False)
    print(ev["drop_reason"].fillna("kept").value_counts().to_string())
    if "country" in ev and not a.all_languages:
        ne = ev[ev["drop_reason"].eq("not_english")]
        if len(ne):
            print("not_english by country (top 10):")
            print(ne["country"].value_counts().head(10).to_string())

    keep = ev[ev["drop_reason"].isna()].copy()
    keep = pd.concat([keep, event_time(keep)], axis=1)
    keep["is_prerecorded"] = keep["headline"].str.contains(r"pre[\s-]?recorded", case=False, regex=True, na=False)
    keep["n_calls_same_day"] = keep.groupby(["companyid", "call_date"])["keydevid"].transform("size")
    keep["n_companies_on_transcript"] = keep.groupby("transcriptid")["companyid"].transform("nunique")
    keep["has_qa"] = keep["n_words_qa"] > 0
    cols = ["transcriptid", "keydevid", "companyid", "gvkey", "companyname", "country", "industry", "headline",
            "mostimportantdateutc", "call_date", "call_ts_ny", "call_time_known", "after_close", "year",
            "transcriptcollectiontypename", "transcriptpresentationtypename", "n_versions", "created_utc",
            "audiolengthsec", "is_prerecorded", "n_calls_same_day", "n_companies_on_transcript", "has_qa",
            "n_components", "n_words", "n_words_presentation", "n_words_qa", "n_questions", "n_analysts",
            "en_ratio", "non_en_share"]
    keep = keep[[c for c in cols if c in keep]].reset_index(drop=True)
    for c in ["n_components", "n_words", "n_words_presentation", "n_words_qa", "n_questions", "n_analysts"]:
        keep[c] = keep[c].astype("int64")
    keep.to_parquet(C.META, index=False)

    ids = set(keep["transcriptid"])
    total = 0
    for f in sorted(tmp.glob("*.parquet")):
        t = pd.read_parquet(f)
        t = t[t["transcriptid"].isin(ids)]
        t.to_parquet(C.TURNS / f.name, index=False, compression="zstd")
        total += len(t)
        f.unlink()
    tmp.rmdir()
    us = keep["country"].eq("United States").sum() if "country" in keep else float("nan")
    nc = keep["country"].nunique() if "country" in keep else float("nan")
    C.log(f"{len(keep):,} English calls kept ({us:,} US, {nc} countries), {total:,} turns -> {C.TURNS}; "
          f"GVKEY {keep['gvkey'].notna().mean():.1%}, after the close {keep['after_close'].mean():.0%}")


if __name__ == "__main__":
    main()
