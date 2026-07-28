#!/usr/bin/env python3
"""Maintainer-only: copy weights + release files into one folder for HF upload."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Stage HF upload folder for ROSE-1H")
    p.add_argument("checkpoint", type=Path, help="path to best_model.pt")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("hf_bundle"),
        help="output directory to upload with huggingface-cli",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = p.parse_args()

    ckpt = args.checkpoint.expanduser().resolve()
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint: {ckpt}")

    root = args.repo_root
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(ckpt, out / "best_model.pt")
    shutil.copy2(root / "configs" / "rose.yaml", out / "rose.yaml")
    shutil.copy2(root / "hub" / "config.json", out / "config.json")
    shutil.copy2(root / "README.md", out / "README.md")

    print(f"staged: {out}")
    print("upload when ready:")
    print(f"  huggingface-cli upload romboai/ROSE-1H {out} --repo-type model")


if __name__ == "__main__":
    main()
