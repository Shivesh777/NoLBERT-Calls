"""Maintainers: turn a trained checkpoint (the best/ folder of a run) into the release folder.

    python model/pack_model.py <run>/best model/nolbert_calls_2023 [--fp32]

Copies config, tokenizer and embedding_config; stores the weights in fp16 (half the size;
the evaluation ran in fp16 autocast) and splits them into 45 MB parts with SHA256SUMS."""
import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

PART = 45 * 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "embedding_config.json"]:
        if (src / name).exists():
            shutil.copyfile(src / name, dst / name)
    from safetensors.numpy import load_file, save_file
    import numpy as np
    tensors = load_file(src / "model.safetensors")
    if not a.fp32:
        tensors = {k: (v.astype(np.float16) if v.dtype == np.float32 else v) for k, v in tensors.items()}
    full = Path(tempfile.mkdtemp()) / "model.safetensors"      # only the parts go into the release folder
    save_file(tensors, full, metadata={"format": "pt"})
    lines = [f"{sha256(full)}  model.safetensors"]
    data = full.read_bytes()
    for i in range(0, len(data), PART):
        p = dst / f"model.safetensors.part-{i // PART:02d}"
        p.write_bytes(data[i:i + PART])
        lines.append(f"{sha256(p)}  {p.name}")
    (dst / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    full.unlink()
    print(f"{dst}: {len(lines) - 1} parts, {len(data) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
