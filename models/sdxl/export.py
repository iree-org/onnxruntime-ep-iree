"""Download pre-patched SDXL component ONNX models from Azure.

Provides a single entry point :func:`ensure_models` that downloads all ONNX
models required for a given precision mode.  Individual component functions
are also usable standalone:

- :func:`export_text_encoders` — CLIP-L + OpenCLIP-bigG
- :func:`export_unet` — fp32/fp16/int8 (W8A8 native i8 MFMA)
- :func:`export_vae` — VAE decoder

All models are pre-exported and pre-patched (Resize, Slice, Cast fixes for
IREE compatibility) and hosted on Azure sharkpublic.
"""

import hashlib
import logging
import pathlib
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

_AZURE_BASE = "https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/ONNX"
_DOWNLOAD_TIMEOUT_S = 600
_DOWNLOAD_RETRIES = 3

# {component: {dtype: [(filename, sha256), ...]}}
_MODEL_FILES = {
    "text_encoder_1": {
        "fp32": [
            (
                "model.onnx",
                "89020f346c1da1220ee8a24a38f2bf560871d6abf699ed7a2eafdf543a308cca",
            ),
        ],
        "fp16": [
            (
                "model.onnx",
                "f3d3f10f65052f68c65e6bc9112507c5e530d5a06c378dc78e4231021cc10f21",
            ),
        ],
    },
    "text_encoder_2": {
        "fp32": [
            (
                "model.onnx",
                "5f636234384e2444eeb62bccde99ef64470535eaec3225e3e9689a5ca98cda75",
            ),
            (
                "model.onnx.data",
                "1e520ee76280ba8b917d580fe144a97682ba96dd78efa715a5668053cc142db1",
            ),
        ],
        "fp16": [
            (
                "model.onnx",
                "445928de35fb8a9b5dcce112623771ae1f8daccdbffc6b27e6225f67499cb3f0",
            ),
            (
                "model.onnx.data",
                "4a8b0c99f2c6dcaefca5084af84e060377a5634acda2def7353f34cfc1b610a0",
            ),
        ],
    },
    "unet": {
        "fp32": [
            (
                "model.onnx",
                "3cd9369275b663a4e7db7be30569d48b3eb0225199d8960d1c27f34525857fec",
            ),
            (
                "model.onnx.data",
                "0a05914ac0e3c8bad44715c0b2910ce8f738a3a82ad886cc6003f199cb06b5e0",
            ),
        ],
        "fp16": [
            (
                "model.onnx",
                "eaff30b7adf2085fcc3c89f685bd740c7848cfdab31a62b13901358298ca3b0b",
            ),
            (
                "model.onnx.data",
                "2a4bb27c03469801c54c0bce6d5582bcfe8b8d4d0d27567165aee00b1d0a97de",
            ),
        ],
        "int8": [
            (
                "model.onnx",
                "a1abd212eb8648b39697ed0fb8bf1cd95686a2526bc51c3ea2bd20c1187c2de0",
            ),
            (
                "model.onnx.data",
                "371065cf0a54a9c0c52df75706e587d2ca02b3da73296efcf8f8e9d456f95c1b",
            ),
        ],
    },
    "vae_decoder": {
        "fp32": [
            (
                "model.onnx",
                "47bbddefe59f93ddec8fc6d45887356fea736115677696d92365074fe5dc54aa",
            ),
        ],
        "fp16": [
            (
                "model.onnx",
                "117b404461158184563ba53df782b17d8a273c69752309b66e6af13f2f150466",
            ),
        ],
    },
}


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def _sha256(path: pathlib.Path) -> str:
    """Compute SHA-256 hex digest of a file, reading in 8 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: pathlib.Path, expected_sha256: str) -> None:
    """Download a single file with retries, timeout, and hash verification.

    Uses a per-read timeout so that stalled connections are detected even on
    multi-GB files (the timeout applies to each socket read, not the total
    transfer time).
    """
    chunk_size = 8 * 1024 * 1024  # 8 MB
    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        try:
            resp = urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S)
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
        except (urllib.error.URLError, OSError) as exc:
            dest.unlink(missing_ok=True)
            if attempt < _DOWNLOAD_RETRIES:
                wait = 2**attempt
                logger.warning(
                    "    Attempt %d/%d failed (%s), retrying in %ds...",
                    attempt,
                    _DOWNLOAD_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Failed to download {url} after {_DOWNLOAD_RETRIES} attempts"
            ) from exc

        actual = _sha256(dest)
        if actual == expected_sha256:
            return
        logger.warning(
            "    Hash mismatch (expected %s…, got %s…), attempt %d/%d",
            expected_sha256[:12],
            actual[:12],
            attempt,
            _DOWNLOAD_RETRIES,
        )
        dest.unlink(missing_ok=True)
        if attempt == _DOWNLOAD_RETRIES:
            raise RuntimeError(
                f"Hash verification failed for {dest.name}: "
                f"expected {expected_sha256}, got {actual}"
            )


def _download_component(models_dir: pathlib.Path, component: str, dtype: str) -> None:
    """Download a single component's ONNX files from Azure if not cached."""
    out_dir = models_dir / component / dtype
    file_entries = _MODEL_FILES[component][dtype]

    if all((out_dir / fname).exists() for fname, _ in file_entries):
        logger.info("  %s/%s already cached", component, dtype)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, expected_hash in file_entries:
        dest = out_dir / fname
        if dest.exists():
            if _sha256(dest) == expected_hash:
                continue
            logger.warning("  %s exists but hash mismatch, re-downloading", dest)
            dest.unlink()
        url = f"{_AZURE_BASE}/{component}/{dtype}/{fname}"
        logger.info("  Downloading %s/%s/%s ...", component, dtype, fname)
        _download_file(url, dest, expected_hash)
        logger.info("  Saved: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
# Per-component entry points
# ---------------------------------------------------------------------------


def export_text_encoders(models_dir: pathlib.Path, dtype: str) -> None:
    """Download both CLIP text encoders."""
    _download_component(models_dir, "text_encoder_1", dtype)
    _download_component(models_dir, "text_encoder_2", dtype)


def export_unet(models_dir: pathlib.Path, dtype: str) -> None:
    """Download SDXL UNet (fp32, fp16, or int8 W8A8)."""
    _download_component(models_dir, "unet", dtype)


def export_vae(models_dir: pathlib.Path, dtype: str) -> None:
    """Download SDXL VAE decoder."""
    _download_component(models_dir, "vae_decoder", dtype)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def ensure_models(models_dir: pathlib.Path, dtype: str) -> None:
    """Download all models needed for the given dtype.

    Args:
        models_dir: Root directory for cached models.
        dtype: ``"fp32"``, ``"fp16"``, or ``"int8"``.
              ``int8`` uses fp16 text encoders and VAE with a downloaded
              W8A8 quantized UNet.
    """
    te_dtype = "fp16" if dtype == "int8" else dtype
    vae_dtype = "fp16" if dtype == "int8" else dtype

    logger.info("=== Preparing models (dtype=%s) ===", dtype)
    export_text_encoders(models_dir, te_dtype)
    export_unet(models_dir, dtype)
    export_vae(models_dir, vae_dtype)
    logger.info("=== All models ready ===\n")
