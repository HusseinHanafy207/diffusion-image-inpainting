# diffusion-image-inpainting

Mask-conditioned **and** unconditional **DDPM inpainting** on top of my
[`generative-models`](https://github.com/HusseinHanafy207/generative-models) package
(RePaint sampling; noise schedule / U-Net reused, not copied).

```text
image_inpainting  →  imports  →  generative_models.ddpm
```

| Stage | Dataset | Status |
|-------|---------|--------|
| 1 | MNIST | Done |
| 2 | Fashion-MNIST | Done |
| 3 | CelebA (64×64) | Done |
| — | Places365 / medical | Future |

---

## Method

**Train (conditioned).** Denoise with mask context:

```text
concat(x_t, masked_image, mask)  →  ε̂   # MSE vs true noise
```

**Train (unconditional).** Standard face DDPM on clean images only (`in_channels=3`).

**Infer (both).** RePaint: reverse step → reinsert known pixels at matching noise level
`q(x₀, t−1)` → optional forward jumps (`j`, `r`). Use **full trained T** (default 1000).
Truncating `--timesteps` below `T` while starting from pure noise is unsafe and refused
unless `--allow-unsafe-timesteps` (see [Experiments](#experiments)).

---

## Results (short)

### MNIST / Fashion-MNIST

Center mask, epoch 40, RePaint `j=10 r=5`, full protocol on those runs:

| Dataset | Output PSNR | Output SSIM |
|---------|-------------|-------------|
| MNIST | ~19.3 | ~0.89 |
| Fashion-MNIST | ~24.1 | ~0.87 |

<p align="center">
  <img src="docs/assets/inpaint_center_repaint_j10r5.png" alt="MNIST center RePaint" width="380" />
  <img src="docs/assets/fashion_inpaint_center_j10r5.png" alt="Fashion center RePaint" width="380" />
</p>

### CelebA — main findings

1. **Sampling bug.** Previewing with `--timesteps 250` on a `T=1000` model starts from pure noise at the wrong noise level → systematic dark “sunglasses” fills. **Full T=1000** restores coherent faces on the same checkpoints.
2. **Training ablations** (mask reweight, LR drop, hole-weighted loss, hole-size curriculum) look **visually similar** under full T — no clear win over baseline `epoch_030`.
3. **Unconditional + RePaint** also fills large centers; identity can drift more than conditioned (expected).

<p align="center">
  <img src="docs/assets/celeba_t250_vs_t1000.png" alt="CelebA truncated T vs full T" width="720" />
</p>

<p align="center">
  <img src="docs/assets/celeba_conditioned_cr040_fullT.png" alt="CelebA conditioned full T" width="420" />
  <img src="docs/assets/celeba_uncond_cr040_fullT.png" alt="CelebA unconditional full T" width="420" />
</p>

---

## Experiments

| ID | What | Outcome (full T) |
|----|------|------------------|
| Baseline | Conditioned CelebA → e40 | Works at `cr=0.4` |
| Phase A | Metrics by `cr` / mask (also ran under truncated T) | Truncated numbers invalid; redo with full T for paper tables |
| Phase 1 | Upweight center + jitter | ≈ baseline |
| Phase 2 | Finetune `lr=2e-5` | ≈ baseline |
| Phase 3 | Hole-weighted noise loss | ≈ baseline |
| Phase 4 / 4b | Center-ratio curriculum | ≈ baseline (4b = fixed `persistent_workers`) |
| Uncond | Clean DDPM + RePaint at test | Works; different identity tradeoff |

Configs live under `configs/celeba*.yaml`. Lab notes: `docs/challenges-log.md`.

---

## Setup

```bash
pip install -e /path/to/generative-models
pip install -e ".[dev]"
pytest -q
```

---

## Usage

```bash
# Conditioned train / inpaint / eval — omit --timesteps (full T)
python scripts/train.py --config configs/celeba.yaml --epochs 40
python scripts/inpaint.py --config configs/celeba.yaml \
  --checkpoint outputs/celeba/checkpoints/epoch_030.pt \
  --mask-type center --center-ratio 0.4 \
  --jump-length 10 --jump-n-sample 5

python scripts/evaluate.py --config configs/celeba.yaml \
  --checkpoint outputs/celeba/checkpoints/epoch_030.pt \
  --mask-type center --center-ratio 0.4 --max-samples 32 \
  --jump-length 10 --jump-n-sample 5

# Unconditional
python scripts/train_unconditional.py --config configs/celeba_unconditional.yaml --epochs 40
python scripts/inpaint.py --config configs/celeba_unconditional.yaml \
  --checkpoint outputs/celeba_uncond/checkpoints/epoch_040.pt \
  --unconditional --mask-type center --center-ratio 0.4 \
  --jump-length 10 --jump-n-sample 5
```

---

## Layout

```text
configs/   experiment YAMLs
scripts/   train, train_unconditional, inpaint, evaluate, visualize_masks
src/image_inpainting/
  masks/ datasets/ models/ diffusion/ trainers/ evaluation/ utils/
tests/   docs/assets/
```

---

## Papers

- [DDPM](https://arxiv.org/abs/2006.11239) — base model (sibling package)
- [RePaint](https://arxiv.org/abs/2201.09865) — inference constraints + resampling

## License

See [LICENSE](LICENSE).
