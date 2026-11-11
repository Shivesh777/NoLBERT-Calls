"""Rebuild model.safetensors from its parts and check it (GitHub keeps files under 100 MB).

    python model/merge_model.py                 # every model folder under model/
    python model/merge_model.py model/nolbert_calls_2023

Same as:  cat model.safetensors.part-* > model.safetensors   (then compare with SHA256SUMS)."""
import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def merge(folder: Path, out: Path | None = None) -> Path:
    folder = Path(folder)
    sums = dict(reversed(line.split()) for line in (folder / "SHA256SUMS").read_text().split("\n") if line.strip())
    parts = sorted(folder.glob("model.safetensors.part-*"))
    if not parts:
        raise SystemExit(f"no parts in {folder}")
    for p in parts:
        if sha256(p) != sums[p.name]:
            raise SystemExit(f"{p.name}: checksum mismatch (re-download it)")
    out = Path(out) if out else folder / "model.safetensors"
    with open(out, "wb") as w:
        for p in parts:
            w.write(p.read_bytes())
    if sha256(out) != sums["model.safetensors"]:
        out.unlink()
        raise SystemExit("merged file: checksum mismatch")
    print(f"{out} ({out.stat().st_size / 1e6:.0f} MB, checksum ok)")
    return out


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for d in ([Path(a) for a in sys.argv[1:]] or sorted(p.parent for p in here.glob("*/SHA256SUMS"))):
        merge(d)
