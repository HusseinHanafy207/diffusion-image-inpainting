# README figures — download checklist

Copy these into this folder (`docs/assets/`) with the **exact filenames** below.
All CelebA grids must be **full T=1000** (no `--timesteps`), unless noted.

## Already in repo (keep)

| File | Used for |
|------|----------|
| `mask_types_mnist.png` | optional / legacy |
| `mask_types_fashion_mnist.png` | optional / legacy |
| `inpaint_center_repaint_j10r5.png` | MNIST result |
| `fashion_inpaint_center_j10r5.png` | Fashion result |
| `eval_center_gt_input_output.png` | MNIST eval (optional) |
| `fashion_eval_center_gt_input_output.png` | Fashion eval (optional) |
| `inpaint_center_noise_matched.png` | method illustration (optional) |

## Download from Drive (required for CelebA README)

| Save as | Source on Drive (typical) | Notes |
|---------|---------------------------|--------|
| `celeba_t250_vs_t1000.png` | Make a **side-by-side** of truncated vs full-T | Left: any old `*_j10r5.png` with sunglasses (`T=250`). Right: `samples_fullT` / conditioned e030 full T. **Most important figure.** |
| `celeba_conditioned_cr040_fullT.png` | `inpainting_runs/samples_fullT_cr040/baseline_e030/` or `celeba/samples_fullT/` | Conditioned `epoch_030`, `cr=0.4`, full T |
| `celeba_uncond_cr040_fullT.png` | `inpainting_runs/celeba_uncond/samples_fullT/` | Uncond e020 (or e040 when ready), `cr=0.4`, full T |

## Nice to have (optional)

| Save as | Source | Notes |
|---------|--------|--------|
| `mask_types_celeba.png` | Run `python scripts/visualize_masks.py --config configs/celeba.yaml` | Or screenshot from an old mask viz |
| `celeba_phases_grid.png` | Montage of baseline / phase1–4b full-T grids | Shows “ablations look alike” |
| `celeba_hole_size_cr020_cr040.png` | Hole-size sweep **re-run at full T** if you redo Phase A | Don’t use truncated-T metrics grids |

## Colab helper (copy files to a Drive folder you can download)

```python
ROOT = "/content/drive/MyDrive/inpainting_runs"
PACK = f"{ROOT}/readme_assets"
!mkdir -p {PACK}

# Adjust paths to wherever your full-T grids landed
!cp -v {ROOT}/samples_fullT_cr040/baseline_e030/*.png {PACK}/ 2>/dev/null
!cp -v {ROOT}/celeba_uncond/samples_fullT/*.png {PACK}/ 2>/dev/null
!cp -v {ROOT}/celeba/samples_fullT/*.png {PACK}/ 2>/dev/null
# Old truncated (sunglasses) example for the comparison figure:
!cp -v {ROOT}/celeba/hole_size_sweep/*.png {PACK}/truncated_t250/ 2>/dev/null
!ls -lhR {PACK}
```

Then download `readme_assets` from Drive, rename to the filenames above, and place them in `docs/assets/`.
