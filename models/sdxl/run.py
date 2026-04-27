"""SDXL text-to-image with ONNX Runtime and the IREE Execution Provider.

Supports three precision modes via a single ``--dtype`` flag:
  fp32  — all components in fp32
  fp16  — all components in fp16
  int8  — W8A8 quantized UNet (native i8 MFMA) + fp16 text encoders and VAE

Models are exported from HuggingFace diffusers on first run (or downloaded
from Azure for int8 UNet) and cached under ``--models-dir``.

Usage:
    python models/sdxl/run.py --target gfx942 --driver hip --dtype fp16
    python models/sdxl/run.py --target gfx942 --driver hip --dtype int8
    python models/sdxl/run.py --target gfx942 --driver hip --dtype fp32
"""

import argparse
import logging
import pathlib
import time

import numpy as np
import onnxruntime as ort
import onnxruntime_ep_iree
from transformers import CLIPTokenizer

from .export import MODEL_ID, ensure_models

logger = logging.getLogger(__name__)

INT8_SEQ_LEN = 64


# ---------------------------------------------------------------------------
# IREE EP session helpers
# ---------------------------------------------------------------------------


def get_iree_device(driver):
    """Find an IREE EP device for the given driver."""
    ep_devices = ort.get_ep_devices()
    for dev in ep_devices:
        if dev.device.metadata.get("iree.driver") == driver:
            return dev
    available = [d.device.metadata.get("iree.driver") for d in ep_devices]
    raise RuntimeError(f"No IREE device with driver '{driver}'. Available: {available}")


def create_session(model_path, iree_device, provider_options, name=None):
    """Create an ORT InferenceSession with the IREE EP."""
    p = pathlib.Path(model_path)
    label = name or f"{p.parent.parent.name}/{p.parent.name}"
    logger.info("  Compiling %s...", label)
    t = time.time()
    so = ort.SessionOptions()
    so.add_provider_for_devices([iree_device], provider_options)
    sess = ort.InferenceSession(str(model_path), so)
    logger.info("  %s compiled in %.1fs", label, time.time() - t)
    return sess


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(args, models_dir, iree_device, provider_options):
    """Run the SDXL txt2img pipeline."""
    te_dtype = "fp16" if args.dtype == "int8" else args.dtype
    vae_dtype = "fp16" if args.dtype == "int8" else args.dtype
    use_int8 = args.dtype == "int8"
    io_np_dtype = np.float16 if args.dtype in ("fp16", "int8") else np.float32

    def model_path(component, dt):
        return models_dir / component / dt / "model.onnx"

    # --- 1. Tokenize ---
    logger.info("[1/4] Tokenizing...")
    tok1 = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2")

    def tokenize(tokenizer, text):
        return tokenizer(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        ).input_ids.astype(np.int64)

    tokens_1 = tokenize(tok1, args.prompt)
    tokens_2 = tokenize(tok2, args.prompt)
    neg_tokens_1 = tokenize(tok1, args.negative_prompt)
    neg_tokens_2 = tokenize(tok2, args.negative_prompt)

    # --- 2. Text Encoding ---
    logger.info("[2/4] Text encoding (%s)...", te_dtype)
    te1 = create_session(
        model_path("text_encoder_1", te_dtype), iree_device, provider_options
    )
    te2 = create_session(
        model_path("text_encoder_2", te_dtype), iree_device, provider_options
    )

    t0 = time.time()
    hs1 = te1.run(None, {"input_ids": tokens_1})[0]
    hs2, text_embeds = te2.run(None, {"input_ids": tokens_2})
    neg_hs1 = te1.run(None, {"input_ids": neg_tokens_1})[0]
    neg_hs2, neg_text_embeds = te2.run(None, {"input_ids": neg_tokens_2})

    encoder_hs = np.concatenate([hs1, hs2], axis=-1).astype(io_np_dtype)
    neg_encoder_hs = np.concatenate([neg_hs1, neg_hs2], axis=-1).astype(io_np_dtype)
    text_embeds = text_embeds.astype(io_np_dtype)
    neg_text_embeds = neg_text_embeds.astype(io_np_dtype)
    logger.info("  Text encoding: %.1fs", time.time() - t0)

    if use_int8:
        # SDXL supports up to 77 tokens, but 64 is chosen deliberately: it
        # is the nearest power-of-two-aligned length that fits within the
        # 77-token limit, giving the hardware a nicely aligned shape to work
        # with. Truncate longer sequences to 64; pad shorter ones with zeros.
        if encoder_hs.shape[1] > INT8_SEQ_LEN:
            encoder_hs = encoder_hs[:, :INT8_SEQ_LEN, :]
            neg_encoder_hs = neg_encoder_hs[:, :INT8_SEQ_LEN, :]
        elif encoder_hs.shape[1] < INT8_SEQ_LEN:
            pad = INT8_SEQ_LEN - encoder_hs.shape[1]
            encoder_hs = np.pad(encoder_hs, ((0, 0), (0, pad), (0, 0)))
            neg_encoder_hs = np.pad(neg_encoder_hs, ((0, 0), (0, pad), (0, 0)))

    # --- 3. UNet Denoise ---
    logger.info("[3/4] Denoising (%d steps, %s UNet)...", args.steps, args.dtype)

    unet = create_session(model_path("unet", args.dtype), iree_device, provider_options)

    time_ids = np.array([[1024, 1024, 0, 0, 1024, 1024]], dtype=io_np_dtype)

    # Euler Discrete scheduler.
    rng = np.random.default_rng(args.seed)
    betas = np.linspace(0.00085**0.5, 0.012**0.5, 1000) ** 2
    alphas_cumprod = np.cumprod(1.0 - betas)
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    step_ratio = 1000 // args.steps
    timesteps = (np.arange(args.steps) * step_ratio)[::-1].copy()
    schedule_sigmas = np.append([sigmas[t] for t in timesteps], 0.0)

    latents = rng.standard_normal((1, 4, 128, 128)).astype(np.float32)
    latents *= schedule_sigmas[0]

    t_start = time.time()
    for i, t in enumerate(timesteps):
        sigma = schedule_sigmas[i]
        sigma_next = schedule_sigmas[i + 1]
        scaled = (latents / ((sigma**2 + 1) ** 0.5)).astype(io_np_dtype)

        if use_int8:
            ts = np.array([t], dtype=np.int32)
            gs = np.array([args.guidance_scale], dtype=np.float16)
            noise_pred = unet.run(
                None,
                {
                    "varg0": scaled,
                    "varg1": ts,
                    "varg2": np.concatenate([neg_encoder_hs, encoder_hs], axis=0),
                    "varg3": np.concatenate([neg_text_embeds, text_embeds], axis=0),
                    "varg4": np.concatenate([time_ids, time_ids], axis=0),
                    "varg5": gs,
                },
            )[0].astype(np.float32)
        else:
            ts = np.array([t], dtype=io_np_dtype)
            uncond = unet.run(
                None,
                {
                    "latent_sample": scaled,
                    "timestep": ts,
                    "encoder_hidden_states": neg_encoder_hs,
                    "text_embeds": neg_text_embeds,
                    "time_ids": time_ids,
                },
            )[0]
            cond = unet.run(
                None,
                {
                    "latent_sample": scaled,
                    "timestep": ts,
                    "encoder_hidden_states": encoder_hs,
                    "text_embeds": text_embeds,
                    "time_ids": time_ids,
                },
            )[0]
            noise_pred = uncond.astype(np.float32) + args.guidance_scale * (
                cond.astype(np.float32) - uncond.astype(np.float32)
            )

        latents = latents + noise_pred * (sigma_next - sigma)

        if (i + 1) % 5 == 0 or i == 0:
            logger.info(
                "  Step %d/%d (t=%d) — %.1fs",
                i + 1,
                args.steps,
                t,
                time.time() - t_start,
            )

    denoise_time = time.time() - t_start
    logger.info(
        "  Denoising: %.1fs (%.0fms/step)",
        denoise_time,
        denoise_time / args.steps * 1000,
    )

    # --- 4. VAE Decode ---
    logger.info("[4/4] Decoding (%s VAE)...", vae_dtype)
    vae = create_session(
        model_path("vae_decoder", vae_dtype), iree_device, provider_options
    )
    vae_io = np.float16 if vae_dtype == "fp16" else np.float32
    latents_scaled = (latents / 0.13025).astype(vae_io)

    t0 = time.time()
    image = vae.run(None, {"latent_sample": latents_scaled})[0]
    logger.info("  VAE decode: %.0fms", (time.time() - t0) * 1000)

    # --- Save ---
    image = np.clip((image.astype(np.float32) + 1.0) / 2.0, 0, 1)
    image = (image * 255).astype(np.uint8)[0].transpose(1, 2, 0)

    try:
        from PIL import Image

        Image.fromarray(image).save(args.output)
    except ImportError:
        np.save(args.output.replace(".png", ".npy"), image)
    logger.info("Image saved: %s (%dx%d)", args.output, image.shape[1], image.shape[0])

    return denoise_time


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="SDXL text-to-image with IREE ONNX Runtime EP"
    )
    p.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="Precision mode (default: fp16)",
    )
    p.add_argument(
        "--target", required=True, help="Compilation target arch (e.g. gfx942, gfx1100)"
    )
    p.add_argument(
        "--driver", required=True, help="IREE HAL driver (hip, vulkan, local-task)"
    )
    p.add_argument("--prompt", default="a photograph of an astronaut riding a horse")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--models-dir",
        default=None,
        help="Directory for cached models (default: ./sdxl_models)",
    )
    p.add_argument("--output", default="sdxl_output.png")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    models_dir = pathlib.Path(args.models_dir or "sdxl_models")

    # Register IREE EP.
    ep_lib = onnxruntime_ep_iree.get_library_path()
    ort.set_default_logger_severity(0 if args.verbose else 2)
    ort.register_execution_provider_library("IREE", str(ep_lib))

    iree_device = get_iree_device(args.driver)
    provider_options = {"target_arch": args.target}

    logger.info("Device: %s", iree_device.device.metadata)
    logger.info('Prompt: "%s"', args.prompt)
    logger.info(
        "Steps: %d, CFG: %.1f, Seed: %d, dtype: %s",
        args.steps,
        args.guidance_scale,
        args.seed,
        args.dtype,
    )

    # Prepare models (export or download).
    ensure_models(models_dir, args.dtype)

    # Run pipeline.
    t_total = time.time()
    run_pipeline(args, models_dir, iree_device, provider_options)
    logger.info("=== Pipeline complete (%.1fs wall-clock) ===", time.time() - t_total)


if __name__ == "__main__":
    main()
