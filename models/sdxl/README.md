# SDXL Text-to-Image with IREE ONNX Runtime EP

Generates 1024x1024 images from text prompts using Stable Diffusion XL
via the IREE Execution Provider for ONNX Runtime.

Supports three precision modes via a single `--dtype` flag:

| `--dtype` | Text Encoders | UNet | VAE | Notes |
|-----------|--------------|------|-----|-------|
| `fp32` | fp32 | fp32 | fp32 | Baseline |
| `fp16` | fp16 | fp16 | fp16 | 2x faster |
| `int8` | fp16 | W8A8 int8 | fp16 | 1.73x faster than fp16 |

## Setup

### 1. Install dependencies

```bash
pip install -r models/requirements.txt
```

### 2. Run

Models are exported (or downloaded for int8 UNet) automatically on first run
and cached under `./sdxl_models/`.

```bash
# fp16 (recommended default)
python models/sdxl/run.py --target gfx942 --driver hip --dtype fp16

# int8 UNet (downloads quantized UNet from Azure, uses fp16 for TE/VAE)
python models/sdxl/run.py --target gfx942 --driver hip --dtype int8

# fp32 baseline
python models/sdxl/run.py --target gfx942 --driver hip --dtype fp32
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dtype` | `fp16` | Precision: `fp32`, `fp16`, or `int8` |
| `--target` | *(required)* | Target arch (e.g. `gfx942`, `gfx1100`) |
| `--driver` | *(required)* | IREE HAL driver (`hip`, `vulkan`, `local-task`) |
| `--prompt` | `"a photograph of an astronaut riding a horse"` | Text prompt |
| `--steps` | `20` | Denoising steps |
| `--guidance-scale` | `7.5` | Classifier-free guidance scale |
| `--seed` | `42` | Random seed |
| `--models-dir` | `./sdxl_models` | Model cache directory |
| `--output` | `sdxl_output.png` | Output image path |
| `-v` | off | Verbose logging |

## Model details

- **Text Encoders**: CLIP-L (123M) + OpenCLIP-bigG (695M), exported from
  `stabilityai/stable-diffusion-xl-base-1.0` via `torch.onnx.export`
- **UNet** (fp32/fp16): exported from diffusers; (int8): pre-quantized W8A8
  ONNX with native `MatMulInteger(i8,i8)→i32` ops, downloaded from
  [sharkpublic](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/ONNX/unet/int8/)
- **VAE**: `madebyollin/sdxl-vae-fp16-fix` decoder, exported via
  `torch.onnx.export`
