#!/usr/bin/env python3
"""Smoke: download romboai/rose-1h-nmr and load RoseModel."""
from rose import DEFAULT_REPO_ID, load


def main() -> None:
    print("repo", DEFAULT_REPO_ID)
    model = load()
    n = sum(p.numel() for p in model.parameters())
    print("ok", type(model).__name__, n)
    if not (7_000_000 <= n <= 9_000_000):
        raise SystemExit(f"unexpected param count: {n}")


if __name__ == "__main__":
    main()
