#!/usr/bin/env python
"""Visualize mask types and damaged MNIST digits (Phase 1–2 sanity check).

Usage:
    python scripts/visualize_masks.py --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize inpainting masks")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"Phase 1–2: grid of center / rectangle / brush / holes masks "
        f"(out_dir={args.out_dir})"
    )


if __name__ == "__main__":
    main()
