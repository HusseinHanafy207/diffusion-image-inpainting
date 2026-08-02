# Diffusion Image Inpainting

Mask-conditioned and unconditional **DDPM inpainting** with **RePaint**, built on
[`generative-models`](https://github.com/HusseinHanafy207/generative-models)
(noise schedule and U-Net reused, not copied).

**Main finding:** inference protocol is part of the result. Truncating reverse sampling
below the trained $T$ while starting from pure noise produces systematic face artifacts
and understates metrics; full $T{=}1000$ restores coherent fills on the same checkpoints.

```text
image_inpainting  →  imports  →  generative_models.ddpm
```

| Dataset | Status |
|---------|--------|
| MNIST / Fashion-MNIST | Done |
| CelebA $64\times64$ | Done |
| Places365 / medical | Future |

Technical report: [`docs/final_paper.pdf`](docs/final_paper.pdf) · Overleaf sources: [`docs/paper_overleaf.zip`](docs/paper_overleaf.zip)

---

## Method

- **Conditioned train:** `concat(x_t, masked_image, mask) → ε̂` (noise MSE)
- **Unconditional train:** standard RGB face DDPM (`in_channels=3`)
- **Inference:** RePaint with noise-matched known-pixel stitch and jumps `(j, r)`  
  Default reverse length = trained $T$. Truncation from pure noise is refused unless
  `--allow-unsafe-timesteps`.

<p align="center">
  <img src="docs/assets/inpaint_center_noise_matched.png" alt="Noise-matched stitch" width="400" />
</p>

---

## Results

### Protocol sensitivity (CelebA)

Conditioned `epoch_030`, center `cr=0.4`, $n{=}100$, RePaint $j{=}10$, $r{=}5$.
Paired tests: ΔPSNR $= 7.61$ dB ($p = 3.06\times10^{-23}$), ΔLPIPS $= -0.071$ ($p = 2.98\times10^{-19}$).

| Reverse length | PSNR ↑ | SSIM ↑ | Hole PSNR ↑ | LPIPS ↓ |
|----------------|--------|--------|-------------|---------|
| $T'{=}250$ (mismatched) | $19.75 \pm 5.40$ | $0.852 \pm 0.060$ | $11.93$ | $0.102 \pm 0.063$ |
| Full $T{=}1000$ | $27.36 \pm 3.45$ | $0.939 \pm 0.025$ | $19.53$ | $0.031 \pm 0.020$ |

<p align="center">
  <img src="docs/assets/celeba_protocol_hero.png" alt="Protocol hero figure" width="780" />
</p>

<p align="center">
  <img src="docs/assets/celeba_t250_vs_t1000.png" alt="Additional protocol examples" width="700" />
</p>

### Digits / clothing

Center mask, epoch 40, full protocol: MNIST ~$19.3$ dB / $0.89$ SSIM; Fashion-MNIST ~$24.1$ dB / $0.87$ SSIM.

<p align="center">
  <img src="docs/assets/inpaint_center_repaint_j10r5.png" alt="MNIST" width="360" />
  <img src="docs/assets/fashion_inpaint_center_j10r5.png" alt="Fashion-MNIST" width="360" />
</p>

### Mask difficulty & ablations (full $T$)

Sparse brush/holes masks are near-solved; large centers remain hardest
(`cr=0.2/0.3/0.4` PSNR $37.7$ / $32.2$ / $27.4$ at full $T$).
Short finetunes (LR drop, hole-weighted loss, curriculum) give only modest gains (~$+2$ dB vs epoch 30).
Unconditional RePaint matches conditioned PSNR; FaceNet identity
($0.485 \pm 0.187$ vs $0.462 \pm 0.172$, $n{=}32$) is not statistically distinguishable
($p{=}0.381$).

<p align="center">
  <img src="docs/assets/celeba_conditioned_cr040_fullT.png" alt="Conditioned" width="400" />
  <img src="docs/assets/celeba_uncond_cr040_fullT.png" alt="Unconditional" width="400" />
</p>

Configs: `configs/celeba*.yaml`.

---

## Setup

```bash
pip install -e /path/to/generative-models
pip install -e ".[dev]"
# optional LPIPS metrics:
pip install -e ".[perceptual]"
pytest -q
```

## Usage

```bash
# Conditioned train / inpaint / eval (omit --timesteps → full T)
python scripts/train.py --config configs/celeba.yaml --epochs 40
python scripts/inpaint.py --config configs/celeba.yaml \
  --checkpoint outputs/celeba/checkpoints/epoch_030.pt \
  --mask-type center --center-ratio 0.4 --jump-length 10 --jump-n-sample 5
python scripts/evaluate.py --config configs/celeba.yaml \
  --checkpoint outputs/celeba/checkpoints/epoch_030.pt \
  --mask-type center --center-ratio 0.4 --max-samples 100 \
  --jump-length 10 --jump-n-sample 5 --lpips

# Unconditional
python scripts/train_unconditional.py --config configs/celeba_unconditional.yaml --epochs 40
python scripts/inpaint.py --config configs/celeba_unconditional.yaml \
  --checkpoint outputs/celeba_uncond/checkpoints/epoch_040.pt \
  --unconditional --mask-type center --center-ratio 0.4 \
  --jump-length 10 --jump-n-sample 5
```

```text
configs/  scripts/  src/image_inpainting/  tests/  docs/
```

## References

- [DDPM](https://arxiv.org/abs/2006.11239) — base diffusion model
- [RePaint](https://arxiv.org/abs/2201.09865) — inference constraints + resampling
- [LPIPS](https://arxiv.org/abs/1801.03924) — perceptual distance

## License

See [LICENSE](LICENSE).
