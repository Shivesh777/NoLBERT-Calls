"""Evaluation of NoLBERT-Calls on US earnings calls (aggregate numbers only).

  python -m nolbert_calls.evaluate --out results/eval                         # the released model, every task
  python -m nolbert_calls.evaluate --readers nolbert_calls_2023 nolbert --tasks retrieval --out results/eval

Readers: see nolbert_calls/common.py (nolbert_calls_2023, nolbert_calls_2012, nolbert or a run name).

Sets (fixed by seed 0, written to <out>/sets.npz):
  main_test   every US call from Jan 2025 to Oct 2026 with >= 3 exchanges
  val_main    the 300 validation calls of 2024 (pipeline/val_calls_main.json): the blend weight
  time_YYYY   500 random US calls of each year 2013-2024 with >= 3 exchanges (the time test)
Tasks:
  retrieval   each analyst question ranks the answers of its own call (ties count against it); also
              blended with word overlap, w * z(TF-IDF) + (1 - w) * z(reader), w chosen on validation
  time        the same retrieval, by call year 2013-2024 (with retrieval: 2013-2026)
  match       a prepared-remarks passage finds its call's answer passage among 1,000 calls
  role        logistic probe, analyst question vs executive answer (fit <= 2023, C on 2024, test 2025-26)
  masked      finance terms masked in 2025-26 passages, predicted by the masked-LM head
  sqa_probe   SubjECTive-QA: logistic regression on frozen embeddings, the authors' three splits
              (files sqa_<seed>_<split>.parquet next to the bundle)
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C

TIME_YEARS = list(range(2013, 2025))
FINANCE_TERMS = """revenue revenues sales margin margins profit profits earnings income loss losses
guidance outlook forecast growth demand pricing price prices cost costs expenses inflation
interest rates rate debt leverage cash dividend dividends buyback capital spending investment
investments acquisition acquisitions inventory inventories backlog orders customers customer
volume volumes market share competition supply shortage capacity production utilization
headwinds tailwinds recession economy currency tariffs tax taxes regulation regulatory
pipeline launch products product segment segments subscribers deposits loans credit
reserves impairment restructuring layoffs hiring wages labor productivity efficiency
liquidity refinancing equity bonds yield yields commodity oil gas steel freight shipping
retail consumer consumers advertising software cloud data digital online mobile network
quarter fiscal annual organic operating gross net adjusted expectations visibility""".split()


def build_sets(meta: pd.DataFrame, smoke: bool, time_test: bool = False):
    rng = np.random.default_rng(C.SEED)

    def take(ids, lo, hi, n=None):
        qa = C.read_qa(ids, years=C.years_between(lo, hi))
        k = qa.groupby("transcriptid").size()
        pool = np.sort(k[k >= 3].index.to_numpy())
        if n is not None:
            pool = np.sort(rng.permutation(pool)[:n])
        return qa[qa["transcriptid"].isin(set(pool))].reset_index(drop=True)

    p = C.PROTOCOLS["main"]
    lo, hi = p["test"]
    ids = meta.loc[(meta["call_date"] >= lo) & (meta["call_date"] <= hi), "transcriptid"]
    v = json.load(open(C.data_dir() / p["val_calls"]))
    sets = {"main_test": take(ids, lo, hi, 300 if smoke else None),
            "val_main": C.read_qa(v[:20] if smoke else v, years=C.years_between(*p["val"]))}
    if time_test:
        for y in (TIME_YEARS[:2] if smoke else TIME_YEARS):
            ids = meta.loc[meta["call_date"].dt.year == y, "transcriptid"]
            sets[f"time_{y}"] = take(ids, f"{y}-01-01", f"{y}-12-31", 30 if smoke else 500)
    sets = {k: v for k, v in sets.items() if len(v)}
    years = meta.set_index("transcriptid")["call_date"].dt.year
    for df in sets.values():
        df["year"] = years.reindex(df["transcriptid"]).to_numpy()
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readers", nargs="*", default=["nolbert_calls_2023"])
    ap.add_argument("--tasks", nargs="*", default=["retrieval", "time", "match", "role", "masked", "sqa_probe"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--match-calls", type=int, default=1000)
    ap.add_argument("--per-term", type=int, default=30)
    ap.add_argument("--idf-calls", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = C.meta()
    rng = np.random.default_rng(C.SEED)
    t0 = time.time()
    sets = build_sets(meta, a.smoke, "time" in a.tasks)
    bounds = {k: C.call_bounds(v["transcriptid"].to_numpy()) for k, v in sets.items()}
    sizes = {k: np.concatenate([[hi - lo] * (hi - lo) for lo, hi in b]) for k, b in bounds.items()}
    np.savez_compressed(out / "sets.npz", **{f"{k}__calls": v["transcriptid"].to_numpy() for k, v in sets.items()},
                        **{f"{k}__years": v["year"].to_numpy() for k, v in sets.items()},
                        **{f"{k}__sizes": sizes[k] for k in sets})
    C.log("sets: " + ", ".join(f"{k} {v['transcriptid'].nunique():,} calls/{len(v):,} q" for k, v in sets.items()))

    results, by_year, blends, match_rows, role_rows, sqa_rows, infos = [], [], [], [], [], [], []
    ranks_store = {k: {} for k in sets}

    def record(reader, setname, ranks, kind="reader"):
        ranks_store[setname][reader] = ranks.astype(np.int16 if ranks.max() < 32000 else np.int32)
        df = sets[setname]
        results.append({"reader": reader, "set": setname, "kind": kind,
                        **C.summary(ranks, df["transcriptid"].to_numpy(), sizes[setname])})
        if setname == "main_test":
            for y, rows in df.groupby("year").indices.items():
                by_year.append({"reader": reader, "set": setname, "year": int(y), "questions": len(rows),
                                "top1": float((ranks[rows] == 1).mean()), "mrr": float((1 / ranks[rows]).mean())})
        elif setname.startswith("time_"):
            by_year.append({"reader": reader, "set": setname, "year": int(setname[5:]), "questions": len(ranks),
                            "top1": float((ranks == 1).mean()), "mrr": float((1 / ranks).mean())})

    # --- word overlap: vocabulary and IDF from calls up to the training cutoff only ----------------
    end = C.PROTOCOLS["main"]["train_end"]
    ids = meta.loc[meta["call_date"] <= end, "transcriptid"].to_numpy()
    ids = rng.permutation(ids)[: (500 if a.smoke else a.idf_calls)]
    # the paper's runs drew a second sample here (calls up to 2012); drawing it again keeps the
    # passage, match and masked-word samples below identical to the paper's
    rng.permutation(int((meta["call_date"] <= "2012-12-31").sum()))
    tf_blocks = {}
    if "retrieval" in a.tasks:
        tr = C.read_qa(ids, years=set(range(2000, pd.Timestamp(end).year + 2)))
        words = C.WordOverlap(pd.concat([tr["question"], tr["answer"]], ignore_index=True))
        C.log(f"word overlap: {len(words.cv.vocabulary_):,} terms from {len(ids):,} calls <= {end}")
        for s in ["main_test", "val_main"]:
            df = sets[s]
            tq, ta = words.tfidf_vectors(df["question"]), words.tfidf_vectors(df["answer"])
            tf_blocks[s] = [(tq[lo:hi] @ ta[lo:hi].T).toarray() for lo, hi in bounds[s]]

    # passages for the match task and contexts for the masked words (main test calls)
    test_ids = sets["main_test"]["transcriptid"].unique()
    ps = C.read_passages(ids=test_ids, years={2025, 2026, 2027})
    ps = ps.iloc[rng.permutation(len(ps))].reset_index(drop=True)
    pres = ps[ps["source"].eq("pres")].drop_duplicates("transcriptid")
    ans = ps[ps["source"].eq("answer")].drop_duplicates("transcriptid")
    match = pres.merge(ans, on="transcriptid", suffixes=("_q", "_t"))
    match = match.iloc[rng.permutation(len(match))[: (100 if a.smoke else a.match_calls)]].reset_index(drop=True)
    role = {}
    if "role" in a.tasks:
        rt = pd.read_parquet(C.data_dir() / "role_turns.parquet")
        role = {s: rt[rt["period"].eq(f"main_{s}")].reset_index(drop=True) for s in ["train", "val", "test"]}
        if a.smoke:
            role = {k: v.groupby("y").head(100).reset_index(drop=True) for k, v in role.items()}
    sqa = load_sqa() if "sqa_probe" in a.tasks else None

    def score_match(name, zq, zt):
        s = zq.astype(np.float32) @ zt.astype(np.float32).T
        own = np.diag(s)[:, None]
        r = (s >= own).sum(1)
        match_rows.append({"reader": name, "calls": len(s), "top1": float((r == 1).mean()),
                           "top10": float((r <= 10).mean()), "mrr": float((1 / r).mean()), "chance_top1": 1 / max(1, len(s))})

    def score_role(name, featurise):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        X = {k: featurise(v["text"]) for k, v in role.items()}
        best = None
        for c in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=c, max_iter=3000).fit(X["train"], role["train"]["y"])
            acc = balanced_accuracy_score(role["val"]["y"], clf.predict(X["val"]))
            if best is None or acc > best[1] + 1e-9:
                best = (c, acc, clf)
        test = balanced_accuracy_score(role["test"]["y"], best[2].predict(X["test"]))
        role_rows.append({"reader": name, "C": best[0], "val_bal_acc": best[1], "test_bal_acc": float(test),
                          "test_turns": len(role["test"])})

    def blend(reader, emb_val):
        """word-overlap weight with the best call-weighted top-1 on the validation calls"""
        grid = np.round(np.linspace(0, 1, 11), 2)
        scores = []
        for w in grid:
            r = C.ranks_from_blocks([w * C.zrows(t) + (1 - w) * C.zrows(e) for t, e in zip(tf_blocks["val_main"], emb_val)])
            scores.append(float(np.mean([(r[lo:hi] == 1).mean() for lo, hi in bounds["val_main"]])))
        best_s = max(scores)
        w = min((g for g, sc in zip(grid, scores) if sc >= best_s - 1e-12), key=lambda g: abs(g - 0.5))
        blends.append({"reader": reader, "w_word_overlap": float(w), "val_top1": best_s,
                       "val_top1_model_only": scores[0], "val_top1_word_overlap_only": scores[-1]})
        return float(w)

    for key in a.readers:
        t1 = time.time()
        try:
            enc = C.Encoder(key)
        except Exception as e:
            C.log(f"skip {key}: {str(e)[:300]}")
            infos.append({"reader": key, "error": str(e)[:500]})
            continue
        try:
            info = enc.info()
            if "retrieval" in a.tasks:
                blocks = {}
                for s in ["main_test", "val_main"]:
                    df = sets[s]
                    zq = enc.embed(df["question"].tolist(), "question", a.batch)
                    za = enc.embed(df["answer"].tolist(), "answer", a.batch)
                    blocks[s] = list(C.emb_blocks(zq, za, bounds[s]))
                    record(key, s, C.ranks_from_blocks(blocks[s]))
                    del zq, za
                w = blend(key, blocks["val_main"])
                r = C.ranks_from_blocks([w * C.zrows(t) + (1 - w) * C.zrows(e)
                                         for t, e in zip(tf_blocks["main_test"], blocks["main_test"])])
                record(f"{key}+word_overlap", "main_test", r, "blend")
                del blocks
            for s in [k for k in sets if k.startswith("time_")]:
                df = sets[s]
                zq = enc.embed(df["question"].tolist(), "question", a.batch)
                za = enc.embed(df["answer"].tolist(), "answer", a.batch)
                record(key, s, C.ranks_from_blocks(C.emb_blocks(zq, za, bounds[s])))
                del zq, za
            if "match" in a.tasks and len(match):
                score_match(key, enc.embed(match["text_q"].tolist(), "question", a.batch),
                            enc.embed(match["text_t"].tolist(), "answer", a.batch))
            if role:
                score_role(key, lambda s: enc.embed(s.tolist(), "answer", a.batch).astype(np.float32))
            if sqa is not None:
                keep = enc.a_len
                enc.a_len = 256                            # the SubjECTive-QA authors' input length
                sqa_probe(key, lambda t: enc.embed(list(t), "answer", 64).astype(np.float32), sqa, sqa_rows,
                          majority=not sqa_rows)
                enc.a_len = keep
            info.update(seconds=round(time.time() - t1, 1), fp32_rows=enc.fp32_rows)
            infos.append(info)
            C.log(f"{key}: done in {(time.time() - t1) / 60:.1f} min (fp32 fallback rows {enc.fp32_rows})")
        except Exception as e:
            traceback.print_exc()
            infos.append({"reader": key, "error": str(e)[:500]})
        finally:
            enc.close()
            gc.collect()
        save(out, results, by_year, blends, match_rows, role_rows, sqa_rows, infos, ranks_store)

    if "masked" in a.tasks:
        try:
            mw, mt = masked_words(a.readers, ps, a, rng)
            if len(mw):
                mw.to_csv(out / "masked_words.csv", index=False)
                mt.to_csv(out / "masked_words_by_term.csv", index=False)
        except Exception:
            traceback.print_exc()
    save(out, results, by_year, blends, match_rows, role_rows, sqa_rows, infos, ranks_store)
    C.log(f"evaluation finished in {(time.time() - t0) / 60:.1f} min")


def masked_words(keys, ps: pd.DataFrame, a, rng):
    """Finance terms (single tokens for every model) masked in 2025-26 passages: macro top-1/top-5."""
    import torch
    loaded = []
    for key in keys:
        try:
            path, revision = C.source(key)
            tok = C.load_tokenizer(key, path, revision)
            loaded.append((key, path, revision, tok))
        except Exception as e:
            C.log(f"masked words: skip {key} ({str(e)[:150]})")

    def target_id(tok, term):
        ids = tok(" " + term, add_special_tokens=False)["input_ids"]
        return ids[0] if len(ids) == 1 and ids[0] != tok.unk_token_id else None

    def space_before_mask(tok):
        ids = tok(f"the {tok.mask_token} rose", add_special_tokens=False)["input_ids"]
        toks = tok.convert_ids_to_tokens(ids)
        i = ids.index(tok.mask_token_id) if tok.mask_token_id in ids else -1
        return not (i > 0 and toks[i - 1] in ("Ġ", "▁", " "))

    terms = [t for t in dict.fromkeys(FINANCE_TERMS) if all(target_id(tok, t) is not None for _, _, _, tok in loaded)]
    contexts, need = [], {t: a.per_term for t in terms}
    texts = ps["text"].tolist()
    for i in rng.permutation(len(texts)):
        words = str(texts[i]).split()
        for j, w in enumerate(words[1:], start=1):
            core = w.rstrip(".,;:!?)\"'")
            if need.get(core, 0) > 0:
                left = " ".join(words[max(0, j - 40):j])
                right = (w[len(core):] + " " + " ".join(words[j + 1:j + 40])).strip()
                contexts.append((core, left, right)); need[core] -= 1
                break
        if not any(need.values()):
            break
    C.log(f"masked words: {len(terms)} terms single-token for all {len(loaded)} models, {len(contexts):,} contexts")
    rows, per_term = [], []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for key, path, revision, tok in loaded:
        try:
            model = C.load_model(path, revision, mlm=True).to(dev).eval()
            sp = space_before_mask(tok)
            h1, h5, term_of = [], [], []
            for lo in range(0, len(contexts), 32):
                chunk = contexts[lo:lo + 32]
                sep = lambda r: "" if r[:1] in ".,;:!?)\"'" else " "
                masked = [f"{l}{' ' if sp else ''}{tok.mask_token}{sep(r)}{r}" for _, l, r in chunk]
                enc = tok(masked, return_tensors="pt", padding=True, truncation=True, max_length=160).to(dev)
                with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=dev == "cuda"):
                    logits = model(**enc).logits
                for k, (term, _, _) in enumerate(chunk):
                    pos = (enc["input_ids"][k] == tok.mask_token_id).nonzero()
                    if len(pos) != 1:
                        continue
                    top = logits[k, pos[0, 0]].float().topk(5).indices.tolist()
                    t_id = target_id(tok, term)
                    h1.append(top[0] == t_id); h5.append(t_id in top); term_of.append(term)
            df = pd.DataFrame({"term": term_of, "top1": h1, "top5": h5})
            pt = df.groupby("term")[["top1", "top5"]].mean()
            rows.append({"reader": key, "terms": int(df["term"].nunique()), "contexts": len(df),
                         "top1_macro": float(pt["top1"].mean()), "top5_macro": float(pt["top5"].mean()),
                         "top1_micro": float(df["top1"].mean()), "top5_micro": float(df["top5"].mean())})
            per_term.append(pt.assign(reader=key).reset_index())
            C.log(f"masked words {key}: top-1 {rows[-1]['top1_macro']:.3f}")
            del model
        except Exception as e:
            C.log(f"masked words {key} failed: {str(e)[:200]}")
    return pd.DataFrame(rows), (pd.concat(per_term, ignore_index=True) if per_term else pd.DataFrame())


SQA_FEATURES = ["CLEAR", "ASSERTIVE", "CAUTIOUS", "OPTIMISTIC", "SPECIFIC", "RELEVANT"]


def load_sqa():
    """SubjECTive-QA, the authors' three published splits (question + space + answer)."""
    root = C._find("sqa_5768_train.parquet", [str(C.data_dir())])
    if root is None:
        C.log("SubjECTive-QA files not found: sqa_probe skipped")
        return None
    return {s: {sp: pd.read_parquet(root / f"sqa_{s}_{sp}.parquet") for sp in ["train", "val", "test"]}
            for s in [5768, 78516, 944601]}


def sqa_probe(name, featurise, sqa, rows, majority=False):
    """Frozen features + multinomial logistic regression per property and split; C chosen on the
    validation split by weighted F1; test accuracy, weighted and macro F1 (and the majority class)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    texts = sorted({q + " " + a for d in sqa.values() for f in d.values() for q, a in zip(f["QUESTION"], f["ANSWER"])})
    feats = featurise(texts)
    index = {t: i for i, t in enumerate(texts)}
    f1 = lambda y, p, avg: float(f1_score(y, p, average=avg, labels=[0, 1, 2], zero_division=0))
    for seed, d in sqa.items():
        X = {sp: feats[[index[q + " " + a] for q, a in zip(f["QUESTION"], f["ANSWER"])]] for sp, f in d.items()}
        for feat in SQA_FEATURES:
            y = {sp: f[feat].astype(int).to_numpy() for sp, f in d.items()}
            best = None
            for c in [0.01, 0.1, 1.0, 10.0]:
                clf = LogisticRegression(C=c, max_iter=3000).fit(X["train"], y["train"])
                v = f1(y["val"], clf.predict(X["val"]), "weighted")
                if best is None or v > best[1] + 1e-12:
                    best = (c, v, clf)
            pred = best[2].predict(X["test"])
            rows.append({"reader": name, "seed": seed, "feature": feat, "C": best[0], "val_f1_weighted": best[1],
                         "test_accuracy": float(accuracy_score(y["test"], pred)),
                         "test_f1_weighted": f1(y["test"], pred, "weighted"), "test_f1_macro": f1(y["test"], pred, "macro")})
            if majority:
                maj = np.full_like(y["test"], np.bincount(y["train"]).argmax())
                rows.append({"reader": "majority class", "seed": seed, "feature": feat,
                             "test_accuracy": float(accuracy_score(y["test"], maj)),
                             "test_f1_weighted": f1(y["test"], maj, "weighted"), "test_f1_macro": f1(y["test"], maj, "macro")})


def save(out, results, by_year, blends, match_rows, role_rows, sqa_rows, infos, ranks_store):
    for name, rows in [("retrieval", results), ("retrieval_by_year", by_year), ("blend_weights", blends),
                       ("call_match", match_rows), ("speaker_role", role_rows), ("subjective_probe", sqa_rows)]:
        if rows:
            pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    json.dump(infos, open(out / "readers.json", "w"), indent=1)
    for s, d in ranks_store.items():
        if d:
            np.savez_compressed(out / f"ranks_{s}.npz", **d)


if __name__ == "__main__":
    main()
