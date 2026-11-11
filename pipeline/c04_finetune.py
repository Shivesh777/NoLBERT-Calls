"""Stage 1 of NoLBERT-Calls: adapt NoLBERT to earnings calls without letting later text in.

    python pipeline/c04_finetune.py --grad-ckpt --budget-min 480 --out data/encoders/nolbert_calls_2012
    # ablations (same budget): --mlm-weight 0 (contrastive only), --no-contrastive (masked-LM only)

NoLBERT (alikLab/NoLBERT, pinned commit) is a DeBERTa-v3-base masked-language model
pretrained on 1976-1995 text. Only calls dated up to --train-end are read (default
2012-12-31; validation calls are 2013-2015). Two objectives, summed:

* contrastive: symmetric InfoNCE with in-batch negatives, temperature 0.05, mean pooling of
  the last layer, L2 normalisation. Two kinds of positive pairs, in separate batches:
    - analyst question -> management answer (data/calls/qa). A batch is packed call by
      call, up to --per-call exchanges of each call, so the other answers of the same call
      are in-batch negatives: the hard negatives of the retrieval test;
    - two passages of the same call (data/calls/passages), one pair per call per batch
      (--same-share of the batches);
* masked-LM on call text: the passage / answer side of every batch is masked (15% of
  tokens; 80% <mask>, 10% random, 10% kept) and NoLBERT's own LM head predicts it, which
  keeps the model a language model while it learns the vocabulary of calls.

Optimisation: every parameter trained, AdamW lr 2e-5, weight decay 0.01, betas (0.9, 0.98),
5% linear warm-up then cosine decay to 0 over the wall-clock budget (or one pass, if that
comes first), clipping at 1, bf16 autocast (fp16 overflows in DeBERTa's attention), batches
of 32 pairs, 128 tokens a side, checkpoints every --ckpt-min minutes (loadable and resumable
with --resume); on CUDA out-of-memory the batch is halved and the pass restarted.

Monitoring: within-call answer retrieval (top-1) on --val-calls validation calls, before
training, at every checkpoint and at the end; the checkpoint with the best validation
top-1 is kept in <out>/best/ (the choice is made on validation, never on test).

Output: <out>/ holds a DebertaV2ForMaskedLM with its tokenizer, and train_log.json the
settings, the pairs used and the loss / retrieval history. <out>/best/ is where stage 2
starts (python -m nolbert_calls.run adapt --init nolbert_calls_2012, with NCALLS_MODELS=data/encoders).

Cost: the released model's stage 1 ran 480 minutes on one RTX 3060 (12 GB), about 19,000 steps.
"""
from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import pandas as pd


# --- data ---------------------------------------------------------------------------
def call_dates() -> pd.DataFrame:
    meta = pd.read_parquet(C.META, columns=["transcriptid", "call_date", "country"])
    meta["call_date"] = pd.to_datetime(meta["call_date"])
    return meta.drop_duplicates("transcriptid")


def load_qa(ids: set, years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        f = C.QA / f"{y}.parquet"
        if f.exists():
            q = pd.read_parquet(f, columns=["transcriptid", "ex", "question", "answer"])
            frames.append(q[q["transcriptid"].isin(ids)])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def qa_pairs(meta: pd.DataFrame, train_end: str, max_pairs: int, rng) -> pd.DataFrame:
    train = meta[meta["call_date"] <= train_end]
    years = sorted(set(train["call_date"].dt.year) | set(train["call_date"].dt.year + 1))
    qa = load_qa(set(train["transcriptid"]), years)
    qa = qa.drop_duplicates("answer")
    calls = rng.permutation(qa["transcriptid"].unique())
    size = qa.groupby("transcriptid").size().reindex(calls).to_numpy()
    take = calls[: int(np.searchsorted(np.cumsum(size), max_pairs, side="right")) + 1]
    qa = qa[qa["transcriptid"].isin(set(take))]
    return qa.sort_values(["transcriptid", "ex"]).head(max_pairs).reset_index(drop=True)


def same_call_pairs(meta: pd.DataFrame, train_end: str, max_pairs: int, rng) -> pd.DataFrame:
    train = meta[meta["call_date"] <= train_end]
    ids = set(train["transcriptid"])
    years = sorted(set(train["call_date"].dt.year) | set(train["call_date"].dt.year + 1))
    first, second, tids = [], [], []
    for y in years:
        f = C.PASSAGES / f"{y}.parquet"
        if not f.exists():
            continue
        p = pd.read_parquet(f, columns=["transcriptid", "text"])
        p = p[p["transcriptid"].isin(ids)].drop_duplicates("text")
        p = p.iloc[rng.permutation(len(p))]
        p["k"] = p.groupby("transcriptid").cumcount()
        two = p[p["k"] < 2].pivot(index="transcriptid", columns="k", values="text")
        if two.shape[1] < 2:
            continue
        two = two.dropna()
        first += two[0].tolist(); second += two[1].tolist(); tids += two.index.tolist()
    order = rng.permutation(len(first))[:max_pairs]
    return pd.DataFrame({"transcriptid": np.asarray(tids)[order],
                         "a": [first[i] for i in order], "b": [second[i] for i in order]})


def validation_calls(meta: pd.DataFrame, n_calls: int, rng, min_ex: int = 4) -> pd.DataFrame:
    lo, hi = C.SPLITS["validation"]
    val = meta[(meta["call_date"] >= lo) & (meta["call_date"] <= hi)]
    years = sorted(set(val["call_date"].dt.year) | set(val["call_date"].dt.year + 1))
    qa = load_qa(set(val["transcriptid"]), years)
    n = qa.groupby("transcriptid").size()
    pick = rng.permutation(n[n >= min_ex].index.to_numpy())[:n_calls]
    return qa[qa["transcriptid"].isin(set(pick))].sort_values(["transcriptid", "ex"]).reset_index(drop=True)


def tokenize(texts, tokenizer, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids = np.full((len(texts), max_len), pad, dtype=np.int32)
    lengths = np.zeros(len(texts), dtype=np.int32)
    texts = list(texts)
    for lo in range(0, len(texts), 4096):
        enc = tokenizer(texts[lo:lo + 4096], truncation=True, max_length=max_len)["input_ids"]
        for j, seq in enumerate(enc):
            ids[lo + j, :len(seq)] = seq
            lengths[lo + j] = len(seq)
    return ids, lengths


def qa_batches(call_of_pair: np.ndarray, batch: int, per_call: int, rng) -> list[np.ndarray]:
    """Batches of pair indices packed call by call (at most per_call pairs of a call)."""
    bounds = np.flatnonzero(np.r_[True, call_of_pair[1:] != call_of_pair[:-1], True])
    groups = [np.arange(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:])]
    out, current = [], []
    for g in (groups[i] for i in rng.permutation(len(groups))):
        g = rng.permutation(g)
        for lo in range(0, len(g), per_call):
            piece = list(g[lo:lo + per_call])
            while piece:
                room = batch - len(current)
                current += piece[:room]; piece = piece[room:]
                if len(current) == batch:
                    out.append(np.array(current)); current = []
    return out


def plain_batches(n: int, batch: int, rng) -> list[np.ndarray]:
    order = rng.permutation(n)
    return [order[i:i + batch] for i in range(0, n - batch + 1, batch)]


# --- training -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="nolbert", help="nolbert (pinned base) or a folder")
    ap.add_argument("--out", default=str(C.ENCODERS / "nolbert_calls"))
    ap.add_argument("--train-end", default=C.FT_END)
    ap.add_argument("--max-qa-pairs", type=int, default=1_500_000)
    ap.add_argument("--max-same-pairs", type=int, default=300_000)
    ap.add_argument("--same-share", type=float, default=0.25,
                    help="share of batches that are same-call passage pairs")
    ap.add_argument("--per-call", type=int, default=8, help="max exchanges of one call in a batch")
    ap.add_argument("--q-len", type=int, default=128)
    ap.add_argument("--a-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--mlm-weight", type=float, default=1.0, help="0 = contrastive only")
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--no-contrastive", action="store_true", help="masked-LM only (ablation)")
    ap.add_argument("--budget-min", type=float, default=150.0)
    ap.add_argument("--ckpt-min", type=float, default=15.0)
    ap.add_argument("--val-calls", type=int, default=300)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    if a.no_contrastive and a.mlm_weight <= 0:
        raise SystemExit("--no-contrastive needs --mlm-weight > 0")
    import torch
    import torch.nn.functional as F

    torch.manual_seed(a.seed)
    cuda = a.device.startswith("cuda")
    if cuda and not torch.cuda.is_available():
        raise SystemExit("no CUDA device (use --device cpu only for a smoke test)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # pairs: only calls dated up to train_end
    meta = call_dates()
    if meta["call_date"].min() > pd.Timestamp(a.train_end):
        raise SystemExit("no call is dated up to --train-end")
    qa = qa_pairs(meta, a.train_end, a.max_qa_pairs, rng)
    same = same_call_pairs(meta, a.train_end, a.max_same_pairs, rng) if a.same_share > 0 else pd.DataFrame()
    val = validation_calls(meta, a.val_calls, rng)
    assert meta.set_index("transcriptid").loc[qa["transcriptid"].unique(), "call_date"].max() <= pd.Timestamp(a.train_end)
    C.log(f"{len(qa):,} question->answer pairs from {qa['transcriptid'].nunique():,} calls, "
          f"{len(same):,} same-call passage pairs, all dated up to {a.train_end}; "
          f"{val['transcriptid'].nunique()} validation calls ({len(val):,} questions)")

    path, revision = C.resolve_encoder(a.init)
    tokenizer = C.load_tokenizer(path, revision)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    q_ids, q_len = tokenize(qa["question"], tokenizer, a.q_len)
    a_ids, a_len = tokenize(qa["answer"], tokenizer, a.a_len)
    s1_ids = s1_len = s2_ids = s2_len = None
    if len(same):
        s1_ids, s1_len = tokenize(same["a"], tokenizer, a.a_len)
        s2_ids, s2_len = tokenize(same["b"], tokenizer, a.a_len)
    vq_ids, _ = tokenize(val["question"], tokenizer, a.q_len)
    va_ids, _ = tokenize(val["answer"], tokenizer, a.a_len)
    val_calls = val["transcriptid"].to_numpy()
    call_of_pair = qa["transcriptid"].to_numpy()
    C.log(f"mean tokens: question {q_len.mean():.0f} (cut at {a.q_len}: {(q_len >= a.q_len).mean():.0%}), "
          f"answer {a_len.mean():.0f} (cut at {a.a_len}: {(a_len >= a.a_len).mean():.0%})")

    model, info = C.load_model(path, revision, mlm=True, output_loading_info=True)
    head_missing = [k for k in info["missing_keys"] if not k.startswith("deberta.")]
    if a.mlm_weight > 0 and head_missing:
        print(f"WARNING: {a.init} has no masked-LM head ({len(head_missing)} weights initialised at random); "
              "the MLM loss starts from a random head")
    model.to(a.device)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
    encoder = model.base_model
    special = torch.as_tensor(sorted(set(tokenizer.all_special_ids)), device=a.device)
    mask_id, vocab = tokenizer.mask_token_id, model.config.vocab_size

    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    n_qa_batches = len(qa) // a.batch
    planned = int(n_qa_batches / max(1e-9, 1 - a.same_share)) if len(same) else n_qa_batches

    def lr_factor(progress: float) -> float:
        """5% linear warm-up, then cosine to 0. progress in [0, 1] is the larger of the share
        of the budget used and the share of one pass done, so the schedule completes within
        the budget whatever the speed of the GPU (one pass over 1.3 million pairs is far
        longer than any budget we run)."""
        p = min(1.0, max(0.0, progress))
        return p / 0.05 if p < 0.05 else 0.5 * (1 + math.cos(math.pi * (p - 0.05) / 0.95))

    state_file, step, history, first_before, elapsed0 = out / "state.pt", 0, [], None, 0.0
    best = {"top1": -1.0, "step": None}
    if a.resume and state_file.exists():
        state = torch.load(state_file, map_location=a.device, weights_only=False)
        model.load_state_dict(state["model"]); optimiser.load_state_dict(state["opt"])
        step, elapsed0 = state["step"], state.get("elapsed_min", 0.0)
        history, best = state.get("history", []), state.get("best", best)
        first_before = state.get("before")
        rng = np.random.default_rng(a.seed + step)
        C.log(f"resumed at step {step}")

    def tensor(x: np.ndarray):
        width = max(1, int((x != pad).sum(1).max()))
        return torch.as_tensor(x[:, :width].astype(np.int64), device=a.device)

    def embed(x):
        m = (x != pad).long()
        h = encoder(input_ids=x, attention_mask=m).last_hidden_state
        w = m.unsqueeze(-1).to(h.dtype)
        return (h * w).sum(1) / w.sum(1).clamp(min=1e-6)

    def info_nce(xa, xb):
        with torch.autocast("cuda", torch.bfloat16, enabled=cuda):
            za, zb = embed(xa), embed(xb)
        za, zb = F.normalize(za.float(), dim=-1), F.normalize(zb.float(), dim=-1)
        logits = za @ zb.T / a.temperature
        target = torch.arange(len(za), device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
        return loss, (logits.argmax(1) == target).float().mean()

    def mlm_loss(x):
        m = (x != pad)
        candidates = m & ~torch.isin(x, special)
        chosen = candidates & (torch.rand(x.shape, device=x.device) < a.mlm_prob)
        if not chosen.any():
            return None
        labels = torch.where(chosen, x, torch.full_like(x, -100))
        roll = torch.rand(x.shape, device=x.device)
        masked = torch.where(chosen & (roll < 0.8), torch.full_like(x, mask_id), x)
        rand_tok = torch.randint(5, vocab, x.shape, device=x.device)
        masked = torch.where(chosen & (roll >= 0.8) & (roll < 0.9), rand_tok, masked)
        with torch.autocast("cuda", torch.bfloat16, enabled=cuda):
            logits = model(input_ids=masked, attention_mask=m.long()).logits
        return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)

    def val_top1() -> float:
        model.eval()
        qz, az = [], []
        with torch.inference_mode():
            for lo in range(0, len(vq_ids), 64):
                with torch.autocast("cuda", torch.bfloat16, enabled=cuda):
                    qz.append(F.normalize(embed(tensor(vq_ids[lo:lo + 64])).float(), dim=-1).cpu().numpy())
                    az.append(F.normalize(embed(tensor(va_ids[lo:lo + 64])).float(), dim=-1).cpu().numpy())
        model.train()
        ranks = C.own_answer_ranks(C.blocks_from_embeddings(np.vstack(qz), np.vstack(az), val_calls))
        return float((ranks == 1).mean())

    def save(final: bool = False) -> None:
        torch.save({"model": model.state_dict(), "opt": optimiser.state_dict(), "step": step,
                    "elapsed_min": elapsed(), "history": history, "best": best, "before": before}, state_file)
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)

    def maybe_best(score: float) -> None:
        if score > best["top1"]:
            best.update(top1=score, step=step)
            model.save_pretrained(out / "best"); tokenizer.save_pretrained(out / "best")

    before = val_top1() if first_before is None else first_before
    C.log(f"validation within-call answer top-1 {'before' if first_before is None else 'of the base model'}: {before:.3f} "
          f"(chance {np.mean(1 / val.groupby('transcriptid').size()):.3f})")
    if step == 0:
        maybe_best(before)
        history.append({"step": 0, "val_top1": round(before, 4)})

    started = saved = time.time()
    elapsed = lambda: elapsed0 + (time.time() - started) / 60       # training minutes, resumes included
    done = False
    model.train()
    while not done:
        plan = [("qa", b) for b in qa_batches(call_of_pair, a.batch, a.per_call, rng)]
        if len(same):
            n_same = int(round(len(plan) * a.same_share / max(1e-9, 1 - a.same_share)))
            sb = plain_batches(len(same), a.batch, rng)
            plan += [("same", sb[i % len(sb)]) for i in range(n_same)]
        plan = [plan[i] for i in rng.permutation(len(plan))]
        restart = False
        for kind, rows in plan:
            if kind == "qa":
                xa, xb = tensor(q_ids[rows]), tensor(a_ids[rows])
            else:
                xa, xb = tensor(s1_ids[rows]), tensor(s2_ids[rows])
            try:
                record = {"step": step + 1, "kind": kind}
                if not a.no_contrastive:
                    loss, acc = info_nce(xa, xb)
                    loss.backward()
                    record.update(nce=round(loss.item(), 4), acc=round(acc.item(), 4))
                if a.mlm_weight > 0:
                    ml = mlm_loss(xb)
                    if ml is not None:
                        (a.mlm_weight * ml).backward()
                        record["mlm"] = round(ml.item(), 4)
            except torch.cuda.OutOfMemoryError:
                optimiser.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if a.batch <= 8:
                    raise SystemExit("out of memory at 8 pairs a batch: lower --a-len or use --grad-ckpt")
                a.batch //= 2
                C.log(f"out of memory: batch halved to {a.batch}, pass restarted")
                restart = True
                break
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            lr = a.lr * lr_factor(max(elapsed() / a.budget_min, step / max(1, planned)))
            for group in optimiser.param_groups:
                group["lr"] = lr
            optimiser.step(); optimiser.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                record["lr"] = lr
                history.append(record)
                C.log(" ".join(f"{k} {v}" for k, v in record.items() if k != "lr"))
            if (time.time() - saved) / 60 >= a.ckpt_min:
                score = val_top1()
                history.append({"step": step, "val_top1": round(score, 4)})
                maybe_best(score)
                save(); saved = time.time()
                C.log(f"checkpoint at step {step}: validation top-1 {score:.3f} (best {best['top1']:.3f})")
            if elapsed() >= a.budget_min:
                C.log(f"budget of {a.budget_min:g} minutes reached"); done = True
                break
        else:
            done = True
        if restart:
            continue

    after = val_top1()
    history.append({"step": step, "val_top1": round(after, 4)})
    maybe_best(after)
    save(final=True)
    C.log(f"validation top-1: before {before:.3f}, after {after:.3f}, best {best['top1']:.3f} at step {best['step']}")
    (out / "train_log.json").write_text(json.dumps({
        "config": vars(a), "steps": step, "pairs_seen": step * a.batch,
        "qa_pairs": len(qa), "qa_calls": int(qa["transcriptid"].nunique()), "same_pairs": len(same),
        "val_calls": int(val["transcriptid"].nunique()), "val_top1_before": before,
        "val_top1_after": after, "best": best, "history": history}, indent=2, default=str))
    print(f"NoLBERT-Calls -> {out} (best validation checkpoint in {out / 'best'})")


if __name__ == "__main__":
    main()
