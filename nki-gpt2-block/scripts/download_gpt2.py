#!/usr/bin/env python3
"""Download the GPT-2 (124M) checkpoint from Hugging Face into weights/gpt2/."""

import argparse
import shutil
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/openai-community/gpt2/resolve/main"
FILES = ("config.json", "model.safetensors")  # about 550 MB


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=Path(__file__).resolve().parents[1] / "weights" / "gpt2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = out / name
        if dst.exists():
            print(f"have     {dst}")
            continue
        print(f"fetching {name} ...")
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(f"{REPO}/{name}") as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.rename(dst)
        print(f"saved    {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
