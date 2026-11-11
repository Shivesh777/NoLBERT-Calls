"""Use the released NoLBERT-Calls model: sentence embeddings, answer ranking and the masked-LM head.

    from nolbert_calls import NoLBERTCalls
    m = NoLBERTCalls("model/nolbert_calls_2023")          # merges the split weights on first use
    z = m.embed(["How should we think about gross margin next year?"])      # (n, 768), unit length
    m.rank_answers(question, [answer_1, answer_2, ...])                      # best match first
    m.fill_mask("Revenue grew on higher<mask> in the quarter.")             # masked-LM head

    python -m nolbert_calls.model --demo

Embeddings use the pooling the model was trained with: mean of the last layer over all
non-padding tokens, L2-normalised. Questions are cut at 128 tokens and answers at the length
in the model's embedding_config.json (256 for NoLBERT-Calls-2023). Cosine similarity is a dot
product. Needs only torch and transformers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT = Path(__file__).resolve().parents[1] / "model" / "nolbert_calls_2023"


class NoLBERTCalls:
    def __init__(self, path: str | Path = DEFAULT, device: str | None = None):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        path = Path(path)
        if not (path / "model.safetensors").exists() and list(path.glob("model.safetensors.part-*")):
            from importlib import util
            spec = util.spec_from_file_location("merge_model", path.parent / "merge_model.py")
            merge = util.module_from_spec(spec); spec.loader.exec_module(merge)
            merge.merge(path)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForMaskedLM.from_pretrained(path, dtype=torch.float32).to(self.device).eval()
        self.encoder = self.model.base_model
        self.pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        cfg = json.loads((path / "embedding_config.json").read_text()) if (path / "embedding_config.json").exists() else {}
        self.q_len, self.a_len = int(cfg.get("question_length", 128)), int(cfg.get("answer_length", 128))
        # bf16 on GPUs that support it; fp32 otherwise (fp16 can overflow in DeBERTa's attention)
        self.amp = torch.bfloat16 if self.device.startswith("cuda") and torch.cuda.is_bf16_supported() else None
        self.path = path

    def embed(self, texts, side: str = "question", batch: int = 64) -> np.ndarray:
        """Unit-length embeddings (n, 768) float32. side: 'question' or 'answer' (sets the length)."""
        torch = self.torch
        texts = [texts] if isinstance(texts, str) else ["" if t is None else str(t) for t in texts]
        length = self.q_len if side == "question" else self.a_len
        ids = self.tokenizer(texts, truncation=True, max_length=length)["input_ids"]
        order = np.argsort([len(x) for x in ids], kind="stable")
        out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for lo in range(0, len(order), batch):
                rows = order[lo:lo + batch]
                width = max(1, max(len(ids[i]) for i in rows))
                x = np.full((len(rows), width), self.pad, dtype=np.int64)
                m = np.zeros((len(rows), width), dtype=np.int64)
                for j, i in enumerate(rows):
                    x[j, :len(ids[i])] = ids[i]
                    m[j, :len(ids[i])] = 1
                x, m = torch.as_tensor(x, device=self.device), torch.as_tensor(m, device=self.device)
                with torch.autocast("cuda", self.amp or torch.float32, enabled=self.amp is not None):
                    h = self.encoder(input_ids=x, attention_mask=m).last_hidden_state.float()
                z = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
                out[rows] = torch.nn.functional.normalize(z, dim=-1).cpu().numpy()
        return out

    def rank_answers(self, question: str, answers) -> list[tuple[int, float]]:
        """(index, cosine) of each answer, best match first."""
        s = self.embed([question], "question")[0] @ self.embed(list(answers), "answer").T
        return [(int(i), float(s[i])) for i in np.argsort(-s)]

    def fill_mask(self, text: str, top: int = 5) -> list[tuple[str, float]]:
        """Top predictions for the single mask token in text. NoLBERT's <mask> carries its own
        leading space: write 'higher<mask> in', not 'higher <mask> in'."""
        torch = self.torch
        enc = self.tokenizer([text], return_tensors="pt").to(self.device)
        pos = (enc["input_ids"][0] == self.tokenizer.mask_token_id).nonzero()
        if len(pos) != 1:
            raise ValueError(f"text must contain exactly one {self.tokenizer.mask_token}")
        with torch.inference_mode():
            p = self.model(**enc).logits[0, int(pos[0, 0])].float().softmax(-1)
        values, idx = p.topk(top)
        return [(self.tokenizer.decode([int(i)]).strip(), float(v)) for v, i in zip(values, idx)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT))
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    m = NoLBERTCalls(a.model)
    # an invented example (no transcript text)
    question = "Can you talk about gross margin and input costs? How should we think about pricing next year?"
    answers = ["We expect gross margin to improve as input costs ease and the price increases we took in the "
               "spring flow through. Freight is still a headwind but smaller than last year.",
               "On capital allocation, we bought back 2 million shares and raised the dividend by 5%.",
               "Demand in Asia was soft, orders in China were down, and we are watching tariffs closely."]
    print(f"model: {m.path} ({m.device}); question {m.q_len} tokens, answers {m.a_len} tokens")
    for i, s in m.rank_answers(question, answers):
        print(f"  {s:+.3f}  answer {i + 1}: {answers[i][:70]}...")
    text = f"Revenue grew on higher{m.tokenizer.mask_token} in the quarter."
    print("fill-mask:", text, "->", m.fill_mask(text))


if __name__ == "__main__":
    main()
