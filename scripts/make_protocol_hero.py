"""Build a memorable protocol hero figure from existing sample grids."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
PAPER_FIG = ROOT / "docs" / "paper_overleaf" / "figures"

UNSAFE = Path(r"C:\Users\Se7s\Downloads\inpaint_center_epoch_030_cr040_j10r5.png")
SAFE = ASSETS / "celeba_conditioned_cr040_fullT.png"
OUT_NAME = "celeba_protocol_hero.png"

METRICS = {
    "unsafe": {"psnr": 19.75, "lpips": 0.102},
    "safe": {"psnr": 27.36, "lpips": 0.031},
}


def _segments(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Return (start, end) inclusive ranges where mask is True and length >= min_len."""
    segs: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            in_seg = True
            start = i
        elif not v and in_seg:
            in_seg = False
            if i - start >= min_len:
                segs.append((start, i - 1))
    if in_seg and len(mask) - start >= min_len:
        segs.append((start, len(mask) - 1))
    return segs


def _crop_grid_panels(path: Path, row: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop original / damaged / inpainted from an 8x3 comparison grid PNG."""
    arr = np.asarray(Image.open(path).convert("RGB"))
    gray = arr.mean(axis=2)
    # Face tiles are darker than white page background.
    row_content = gray.mean(axis=1) < 245
    col_content = gray.mean(axis=0) < 245
    # Title / labels are thin; keep only tall tile rows and wide tile columns.
    row_segs = _segments(row_content, min_len=120)
    col_segs = _segments(col_content, min_len=120)
    if len(row_segs) < 8 or len(col_segs) < 3:
        raise RuntimeError(
            f"{path.name}: expected >=8 row tiles and 3 col tiles, "
            f"got {len(row_segs)} x {len(col_segs)}"
        )
    # Drop thin label bands: keep the 8 tallest rows and 3 widest cols.
    row_segs = sorted(row_segs, key=lambda s: s[1] - s[0], reverse=True)[:8]
    col_segs = sorted(col_segs, key=lambda s: s[1] - s[0], reverse=True)[:3]
    row_segs = sorted(row_segs, key=lambda s: s[0])
    col_segs = sorted(col_segs, key=lambda s: s[0])

    def cell(r: int, c: int) -> np.ndarray:
        y0, y1 = row_segs[r]
        x0, x1 = col_segs[c]
        return arr[y0 : y1 + 1, x0 : x1 + 1]

    return cell(row, 0), cell(row, 1), cell(row, 2)


def _resize(img: np.ndarray, size: int = 160) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((size, size), Image.Resampling.BICUBIC))


def _absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    d = np.mean(np.abs(a - b), axis=2)
    return d / (d.max() + 1e-8)


def main() -> None:
    # Row with a clear sunglasses failure under truncated T.
    row = 1
    gt_u, inp_u, out_u = _crop_grid_panels(UNSAFE, row=row)
    gt_s, inp_s, out_s = _crop_grid_panels(SAFE, row=row)

    # Prefer GT/input from full-T grid (same seed/faces as paper conditioned figure).
    gt, inp = _resize(gt_s), _resize(inp_s)
    unsafe, safe = _resize(out_u), _resize(out_s)
    heat = _absdiff(unsafe, safe)

    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.55), constrained_layout=True)
    panels = [
        (gt, "Ground truth", None),
        (inp, "Input (masked)", None),
        (
            unsafe,
            r"Unsafe $T'{=}250$",
            f"PSNR {METRICS['unsafe']['psnr']:.1f}  |  LPIPS {METRICS['unsafe']['lpips']:.3f}",
        ),
        (
            safe,
            r"Full $T{=}1000$",
            f"PSNR {METRICS['safe']['psnr']:.1f}  |  LPIPS {METRICS['safe']['lpips']:.3f}",
        ),
    ]
    for ax, (img, title, sub) in zip(axes[:4], panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        if sub:
            ax.set_xlabel(sub, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    im = axes[4].imshow(heat, cmap="magma", vmin=0, vmax=1)
    axes[4].set_title(r"$|$unsafe $-$ full$T|$", fontsize=10)
    axes[4].set_xlabel("difference heatmap", fontsize=8)
    axes[4].set_xticks([])
    axes[4].set_yticks([])
    cbar = fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    for out_dir in (ASSETS, PAPER_FIG):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / OUT_NAME
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print("wrote", out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
