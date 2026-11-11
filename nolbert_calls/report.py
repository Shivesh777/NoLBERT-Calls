"""Results tables of NoLBERT-Calls from evaluation and head folders (aggregate numbers only).

  python -m nolbert_calls.report --eval results/eval --heads partb_outputs --out results/tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import LABEL


def lab(r: str) -> str:
    if r.endswith("+word_overlap"):
        base = r[: -len("+word_overlap")]
        return f"{LABEL.get(base, base)} + word overlap"
    return LABEL.get(r, r)


def fmt(x, d=1):
    return "" if pd.isna(x) else f"{100 * x:.{d}f}"


def load(evals):
    names = ["retrieval", "retrieval_by_year", "blend_weights", "call_match", "speaker_role", "masked_words",
             "subjective_probe"]
    acc = {n: [] for n in names}
    for d in map(Path, evals):
        for n in names:
            if (d / f"{n}.csv").exists():
                acc[n].append(pd.read_csv(d / f"{n}.csv"))
    return {n: (pd.concat(v, ignore_index=True).drop_duplicates() if v else pd.DataFrame()) for n, v in acc.items()}


def head_runs(folders) -> pd.DataFrame:
    runs = []
    for h in map(Path, folders):
        for f in h.rglob("run.json"):
            r = json.loads(f.read_text())
            c = r["config"]
            if c.get("kind") != "head" or c.get("smoke") or r.get("status") != "complete":
                continue
            m = r.get("metrics", {})
            runs.append({"model": c["model"], "task": c["task"], "feature": c.get("feature"), "seed": c["seed"],
                         **{k.split("_", 1)[1]: v for k, v in m.items() if k.startswith(("test_", "report_"))}})
    return pd.DataFrame(runs)


def time_table(y: pd.DataFrame, ref: str, out: Path) -> list[str]:
    """Time test: top-1 by call year (500 calls a year 2013-24, the whole test for 2025-26), and the
    in-sample premium: a model's lead over `ref` in 2013-23 minus its lead in 2025-26."""
    t = y[y["set"].str.startswith("time_")]
    if not len(t):
        return []
    keep = set(t["reader"])
    tt = pd.concat([t, y[y["set"].eq("main_test") & y["reader"].isin(keep)]])
    piv = tt.pivot_table(index="reader", columns="year", values="top1", aggfunc="first")
    piv.to_csv(out / "table1c_time_test.csv")
    cols = list(piv.columns)
    lines = ["## Time test: top-1 (%) by call year", "",
             "500 random US calls a year for 2013-24 (seed 0), every test call for 2025-26.", "",
             "| Model | " + " | ".join(str(c) for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for k, row in piv.iterrows():
        lines.append(f"| {lab(k)} | " + " | ".join(fmt(v) for v in row) + " |")
    lines.append("")
    if ref in piv.index and len(piv) > 1:
        rows = []
        for k in [i for i in piv.index if i != ref]:
            lead = piv.loc[k] - piv.loc[ref]
            seen = lead[[c for c in cols if 2013 <= c <= 2023]].mean()
            unseen = lead[[c for c in cols if c >= 2025]].mean()
            if pd.notna(seen) and pd.notna(unseen):
                rows.append({"reader": k, "reference": ref, "lead_2013_23": seen, "lead_2025_26": unseen, "premium": seen - unseen})
        if not rows:
            return lines
        pr = pd.DataFrame(rows)
        pr.to_csv(out / "table1d_in_sample_premium.csv", index=False)
        lines += [f"In-sample premium: lead over {lab(ref)} in 2013-23 (calls a 2023 model trained on) minus the lead "
                  "in 2025-26 (calls no model saw), in points of top-1.", "",
                  "| Model | Lead 2013-23 | Lead 2025-26 | Premium |", "|---|---|---|---|"]
        for _, r in pr.iterrows():
            lines.append(f"| {lab(r['reader'])} | {100 * r['lead_2013_23']:+.1f} | {100 * r['lead_2025_26']:+.1f} | "
                         f"{100 * r['premium']:+.1f} |")
        lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--heads", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rename", nargs="*", default=[], help="old=new reader names, e.g. my_run=nolbert_calls_2023")
    ap.add_argument("--reference", default="nolbert_calls_2012", help="the time test's reference model (calls up to 2012)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = load(a.eval)
    rename = dict(x.split("=", 1) for x in a.rename)
    rename.update({f"{k}+word_overlap": f"{v}+word_overlap" for k, v in list(rename.items())})
    for df in t.values():
        if "reader" in df:
            df["reader"] = df["reader"].replace(rename)
    lines = ["# NoLBERT-Calls: results", ""]

    # ---- Table 1: within-call answer retrieval -----------------------------------------------
    res = t["retrieval"]
    if len(res):
        r = res[res["set"].eq("main_test")].copy()
        r["label"] = r["reader"].map(lab)
        r.to_csv(out / "table1_retrieval.csv", index=False)
        x = r.iloc[0]
        lines += [f"## Table 1. Within-call answer retrieval, US calls Jan 2025 - Oct 2026 ({int(x['calls']):,} calls, "
                  f"{int(x['questions']):,} questions; chance top-1 {fmt(x['chance_top1'])}%)", "",
                  "Each analyst question ranks every management answer of its call (ties count against it). % of "
                  "questions; 95% interval resampling calls. '+ word overlap' blends z-scored TF-IDF with the model, "
                  "weight chosen on the 300 validation calls of 2024.", "",
                  "| Model | Top-1 | 95% CI | Top-1 (call-weighted) | Top-3 | MRR |", "|---|---|---|---|---|---|"]
        for _, x in r.iterrows():
            lines.append(f"| {x['label']} | {fmt(x['top1'])} | {fmt(x['top1_lo'])}–{fmt(x['top1_hi'])} | "
                         f"{fmt(x['top1_call_weighted'])} | {fmt(x['top3'])} | {fmt(x['mrr'])} |")
        lines.append("")
        y = t["retrieval_by_year"]
        if len(y):
            piv = y[y["set"].eq("main_test") & y["reader"].isin(set(r["reader"]))].pivot_table(
                index="reader", columns="year", values="top1", aggfunc="first")
            piv.to_csv(out / "table1b_by_year.csv")
            lines += ["Top-1 (%) by call year:", "", "| Model | " + " | ".join(str(c) for c in piv.columns) + " |",
                      "|---|" + "---|" * len(piv.columns)]
            for k, row in piv.iterrows():
                lines.append(f"| {lab(k)} | " + " | ".join(fmt(v) for v in row) + " |")
            lines.append("")
            lines += time_table(y, a.reference, out)
    if len(t["blend_weights"]):
        t["blend_weights"].to_csv(out / "blend_weights.csv", index=False)

    # ---- Table 2: other tasks on frozen representations ----------------------------------------
    parts = []
    if len(t["call_match"]):
        parts.append(t["call_match"].set_index("reader")[["top1", "top10", "mrr"]].add_prefix("match_"))
    if len(t["speaker_role"]):
        parts.append(t["speaker_role"].set_index("reader")[["test_bal_acc"]].rename(columns={"test_bal_acc": "role_bal_acc"}))
    if len(t["masked_words"]):
        parts.append(t["masked_words"].set_index("reader")[["top1_macro", "top5_macro"]].add_prefix("masked_"))
    if len(t["subjective_probe"]):
        parts.append(t["subjective_probe"].groupby("reader")["test_f1_weighted"].mean().rename("sqa_probe_f1w").to_frame())
    if parts:
        t2 = pd.concat(parts, axis=1)
        t2.to_csv(out / "table2_other_tasks.csv")
        lines += ["## Table 2. Other tasks on 2025-26 calls (frozen representations)", "",
                  "Match: a prepared-remarks passage finds its call's answer passage among 1,000 calls (top-1, top-10). "
                  "Role: analyst question vs executive answer, logistic probe (balanced accuracy). Masked: 100 finance "
                  "terms masked in 3,000 passages (macro top-1, top-5). SubjECTive-QA probe: logistic regression on "
                  "frozen embeddings, weighted F1 averaged over six properties and the authors' three splits. All in %.", "",
                  "| Model | Match top-1 | Match top-10 | Role | Masked top-1 | Masked top-5 | SubjECTive-QA probe F1 |",
                  "|---|---|---|---|---|---|---|"]
        for k in [i for i in t2.index if i != "majority class"] + [i for i in t2.index if i == "majority class"]:
            x = t2.loc[k]
            g = lambda c: fmt(x[c]) if c in x else ""
            lines.append(f"| {lab(k)} | {g('match_top1')} | {g('match_top10')} | {g('role_bal_acc')} | "
                         f"{g('masked_top1_macro')} | {g('masked_top5_macro')} | {g('sqa_probe_f1w')} |")
        lines.append("")

    # ---- Tables 3-4: fine-tuned heads (SubjECTive-QA, GLUE) -------------------------------------
    hr = head_runs(a.heads)
    if len(hr):
        hr["model"] = hr["model"].replace(rename)
        hr.to_csv(out / "heads_runs.csv", index=False)
        sq = hr[hr["task"].eq("subjective")]
        if len(sq):
            pv = sq.groupby(["model", "feature"])["f1_weighted"].agg(["mean", "std", "count"]).reset_index()
            pv.to_csv(out / "table3_subjective_heads.csv", index=False)
            per_split = sq.groupby(["model", "seed"])["f1_weighted"].mean().unstack()
            lines += ["## Table 3. SubjECTive-QA, fine-tuned heads: test weighted F1 (%), mean (sd) over the authors' "
                      "three splits", ""]
            feats = sorted(pv["feature"].unique())
            lines += ["| Model | " + " | ".join(feats) + " | Mean |", "|---|" + "---|" * (len(feats) + 1)]
            for m, g in pv.groupby("model"):
                cell = {f: f"{100 * v:.1f} ({100 * s:.1f})" for f, v, s in zip(g["feature"], g["mean"], g["std"])}
                ms = per_split.loc[m]
                lines.append(f"| {lab(m)} | " + " | ".join(cell.get(f, "") for f in feats) +
                             f" | {100 * ms.mean():.1f} ({100 * ms.std(ddof=1):.1f}) |")
            lines.append("")
        gl = hr[hr["task"].isin(["cola", "sst2", "qnli"])]
        if len(gl):
            gl = gl.assign(score=np.where(gl["task"].eq("cola"), gl.get("matthews_correlation"), gl.get("accuracy")))
            wide = gl.pivot_table(index="model", columns="task", values="score", aggfunc="mean")
            wide.to_csv(out / "table4_glue.csv")
            lines += ["## Table 4. GLUE dev (CoLA: Matthews correlation; QNLI, SST-2: accuracy; %)", "",
                      "| Model | " + " | ".join(wide.columns) + " |", "|---|" + "---|" * len(wide.columns)]
            for m, row in wide.iterrows():
                lines.append(f"| {lab(m)} | " + " | ".join(fmt(v) for v in row) + " |")
            lines.append("")
    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
