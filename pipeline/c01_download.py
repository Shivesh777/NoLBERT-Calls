"""Download EVERY Capital IQ earnings-call transcript on WRDS (all firms, all countries).

    pip install wrds pyarrow            (wrds 3.5 wants pandas < 2.3: use its own venv)
    python pipeline/c01_download.py --username YOUR_WRDS_ID [--start-year 2000] [--end-year 2026]
                                 [--countries "United States"]   # default: every country

Same queries and the same version choice as the professor's notebook
(1_download_ciq_transcripts_local.ipynb), with three changes:

* no firm or sample filter: every call of every company (event type 48) in the window;
* the speaker-turn files keep only the columns the pipeline uses (turn text, turn type,
  speaker type, speaker name) and are zstd-compressed; the full-text files of the
  notebook's step 9 are not built (they store the text a second and third time: about
  30 GB for all countries, against about 8 GB here);
* company country and industry are saved (raw/company.parquet), so the US subset that
  links to CRSP can be chosen later without another WRDS session.

Writes data/calls/raw/: detail_all_versions.parquet, transcript_list.parquet (one transcript
per call, GVKEY attached), company.parquet and components/<year>/part_XXXX.parquet.
Resumable: re-run after a dropped connection and finished parts are skipped. The password
can come from the WRDS_PASSWORD environment variable (set it in the shell for this run
only). Each new connection asks for a Duo push; answer y to the .pgpass question to avoid retyping the
password (it is then stored in plain text in %APPDATA%\\postgresql\\pgpass.conf).
"""
from pathlib import Path
import argparse
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np
import pandas as pd

VERSION_RANK = {"Audited Copy": 4, "SA Audited Copy": 4, "Proofed Copy": 3, "SA Proofed Copy": 3,
                "Edited Copy": 2, "SA Edited Copy": 2, "EV Copy": 1,
                "Spellchecked Copy": 0, "SA Spellchecked Copy": 0, "AI Preliminary Copy": 0}
PERSON_COLS = ["transcriptcomponenttypename", "speakertypename", "transcriptpersonname"]


class Wrds:
    def __init__(self, username: str):
        self.username = username
        self.db = None
        self.connect()

    def connect(self):
        import wrds
        if self.db is not None:
            try:
                self.db.close()
            except Exception:
                pass
        # password from the environment (never written to disk); empty -> .pgpass or a prompt
        self.db = wrds.Connection(wrds_username=self.username or None,
                                  wrds_password=os.environ.get("WRDS_PASSWORD", ""))

    def sql(self, query: str, retries: int = 3, **kw) -> pd.DataFrame:
        for attempt in range(retries):
            try:
                return self.db.raw_sql(query, **kw)
            except Exception as e:                 # WRDS drops idle connections
                if attempt == retries - 1:
                    raise
                C.log(f"query failed ({str(e)[:120]}); reconnecting in 30 s")
                time.sleep(30)
                self.connect()

    def has_table(self, lib: str, table: str) -> bool:
        try:
            return table in self.db.list_tables(library=lib)
        except Exception:
            return False

    def columns(self, lib: str, table: str) -> list[str]:
        return list(self.db.describe_table(lib, table)["name"])


def pick_versions(detail: pd.DataFrame) -> pd.DataFrame:
    """One transcript per (company, event): Final over Preliminary, then the highest
    version rank (Audited > Proofed > Edited > EV > Spellchecked), then the newest."""
    d = detail.copy()
    d["created_utc"] = pd.to_datetime(d["transcriptcreationdate_utc"].astype(str) + " " +
                                      d["transcriptcreationtime_utc"].astype(str), errors="coerce")
    d["created_utc"] = d["created_utc"].fillna(pd.to_datetime(d["transcriptcreationdate_utc"], errors="coerce"))
    d["is_final"] = d["transcriptpresentationtypename"].eq("Final")
    d["version_rank"] = d["transcriptcollectiontypename"].map(VERSION_RANK)
    unknown = d.loc[d["version_rank"].isna(), "transcriptcollectiontypename"].dropna().unique()
    if len(unknown):
        print("collection types not in VERSION_RANK (ranked lowest):", list(unknown))
    d["version_rank"] = d["version_rank"].fillna(-1)
    d["n_versions"] = d.groupby(["companyid", "keydevid"])["transcriptid"].transform("size")
    return (d.sort_values(["companyid", "keydevid", "is_final", "version_rank", "created_utc", "transcriptid"])
             .drop_duplicates(["companyid", "keydevid"], keep="last").reset_index(drop=True))


def company_info(w: Wrds, ids: np.ndarray) -> pd.DataFrame:
    """Country and industry of each company (best effort: the schema varies by subscription)."""
    frames = []
    for lo in range(0, len(ids), 5000):
        idl = ",".join(str(int(i)) for i in ids[lo:lo + 5000])
        try:
            frames.append(w.sql(f"""
                select c.companyid, c.companyname, g.country, i.simpleindustrydescription as industry
                from ciq.ciqcompany c
                left join ciq.ciqcountrygeo g on c.countryid = g.countryid
                left join ciq.ciqsimpleindustry i on c.simpleindustryid = i.simpleindustryid
                where c.companyid in ({idl})""", retries=1))
        except Exception:
            try:
                frames.append(w.sql(f"""
                    select c.companyid, c.companyname, g.country
                    from ciq.ciqcompany c left join ciq.ciqcountrygeo g on c.countryid = g.countryid
                    where c.companyid in ({idl})""", retries=1))
            except Exception as e2:
                print(f"company info unavailable ({str(e2)[:150]}); country filter disabled")
                return pd.DataFrame({"companyid": ids})
    out = pd.concat(frames, ignore_index=True)
    out["companyid"] = out["companyid"].astype("int64")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--username", default="", help="WRDS username (prompted if empty)")
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--countries", nargs="*", help='keep only these countries, e.g. "United States"')
    ap.add_argument("--chunk", type=int, default=1000, help="transcripts per query / part (keep fixed once started)")
    ap.add_argument("--meta-only", action="store_true", help="refresh the lists, skip the text")
    a = ap.parse_args()
    C.RAW.mkdir(parents=True, exist_ok=True)
    w = Wrds(a.username)

    lib = next((lib for lib in ["ciq_transcripts", "ciq"] if w.has_table(lib, "wrds_transcript_detail")), None)
    if lib is None:
        raise SystemExit("wrds_transcript_detail not found in ciq_transcripts or ciq: check the CIQ Transcripts subscription")
    C.log(f"library {lib}")

    detail = w.sql(f"""
        select * from {lib}.wrds_transcript_detail
        where keydeveventtypeid = 48
          and mostimportantdateutc between '{a.start_year}-01-01' and '{a.end_year}-12-31'""")
    detail["mostimportantdateutc"] = pd.to_datetime(detail["mostimportantdateutc"])
    detail["transcriptcreationdate_utc"] = pd.to_datetime(detail["transcriptcreationdate_utc"])
    detail["year"] = detail["mostimportantdateutc"].dt.year
    for c in ["transcriptid", "keydevid", "companyid"]:
        detail[c] = detail[c].astype("int64")
    detail.to_parquet(C.DETAIL, index=False)
    C.log(f"{len(detail):,} transcript versions, {detail['keydevid'].nunique():,} events, "
          f"{detail['companyid'].nunique():,} companies")

    calls = pick_versions(detail)
    gv = w.sql("select companyid, gvkey, startdate, enddate from ciq.wrds_gvkey")
    gv["companyid"] = gv["companyid"].astype("int64")
    gv["startdate"] = pd.to_datetime(gv["startdate"], errors="coerce")
    gv["enddate"] = pd.to_datetime(gv["enddate"], errors="coerce")
    m = calls[["transcriptid", "companyid", "mostimportantdateutc"]].merge(gv, on="companyid")
    ok = ((m["startdate"].isna() | (m["mostimportantdateutc"] >= m["startdate"])) &
          (m["enddate"].isna() | (m["mostimportantdateutc"] <= m["enddate"])))
    m = m[ok].drop_duplicates(["transcriptid", "companyid"])
    calls = calls.drop(columns="gvkey", errors="ignore").merge(
        m[["transcriptid", "companyid", "gvkey"]], on=["transcriptid", "companyid"], how="left")

    company = company_info(w, np.sort(calls["companyid"].unique()))
    company.to_parquet(C.COMPANY, index=False)
    if "country" in company:
        calls = calls.merge(company[["companyid", "country"]], on="companyid", how="left")
    if a.countries:
        if "country" not in calls:
            raise SystemExit("no country information: run without --countries")
        calls = calls[calls["country"].isin(a.countries)].reset_index(drop=True)
    calls.to_parquet(C.TRANSCRIPT_LIST, index=False)
    C.log(f"{len(calls):,} calls kept, GVKEY for {calls['gvkey'].notna().mean():.1%}"
          + (f", {calls['country'].eq('United States').mean():.0%} US" if "country" in calls else ""))
    if a.meta_only:
        return

    comp_cols = set(w.columns(lib, "ciqtranscriptcomponent"))
    have = set(w.columns(lib, "wrds_transcript_person"))
    person = [c for c in PERSON_COLS if c in have and c not in comp_cols]
    print("person columns:", person)

    def download(ids) -> pd.DataFrame:
        idl = ",".join(str(int(i)) for i in ids)
        sel = "".join(f", p.{c}" for c in person)
        return w.sql(f"""
            select c.transcriptid, c.componentorder, c.componenttext{sel}
            from {lib}.ciqtranscriptcomponent as c
            left join {lib}.wrds_transcript_person as p
              on c.transcriptcomponentid = p.transcriptcomponentid and p.transcriptid in ({idl})
            where c.transcriptid in ({idl})
            order by c.transcriptid, c.componentorder""")

    def on_disk(year: int) -> set:
        ids = set()
        for f in (C.COMPONENTS / str(year)).glob("part_*.parquet"):
            ids.update(pd.read_parquet(f, columns=["transcriptid"])["transcriptid"].unique())
        return ids

    t0 = time.time()
    for year in sorted(calls["year"].unique()):
        folder = C.COMPONENTS / str(year)
        folder.mkdir(parents=True, exist_ok=True)
        k = max([int(f.stem[5:9]) for f in folder.glob("part_*.parquet")], default=-1) + 1
        todo = sorted(set(calls.loc[calls["year"] == year, "transcriptid"]) - on_disk(year))
        chunks = [todo[i:i + a.chunk] for i in range(0, len(todo), a.chunk)]
        for j, ids in enumerate(chunks):
            comp = download(ids)
            if comp.empty:
                print(f"{year} {j + 1}/{len(chunks)}: no text for {len(ids)} transcripts")
                continue
            comp["transcriptid"] = comp["transcriptid"].astype("int64")
            comp["componentorder"] = comp["componentorder"].astype("int32")
            out = folder / f"part_{k:04d}.parquet"
            k += 1
            comp.to_parquet(str(out) + ".tmp", index=False, compression="zstd")
            Path(str(out) + ".tmp").replace(out)
            print(f"{year} {j + 1}/{len(chunks)}: {comp['transcriptid'].nunique()} transcripts, "
                  f"{len(comp):,} turns ({(time.time() - t0) / 60:.1f} min)", flush=True)

    done = set()
    for f in C.COMPONENTS.glob("*/part_*.parquet"):
        done.update(pd.read_parquet(f, columns=["transcriptid"])["transcriptid"].unique())
    missing = set(calls["transcriptid"]) - done
    C.log(f"{len(done):,} transcripts on disk; {len(missing):,} of the list have no text")
    w.db.close()


if __name__ == "__main__":
    main()
