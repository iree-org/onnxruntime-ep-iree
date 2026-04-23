"""ResNet-50 image classification with ONNX Runtime and the IREE EP."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import onnxruntime_ep_iree as iree_ep
from PIL import Image

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "examples" / "model.onnx"
DEFAULT_LABELS_PATH = REPO_ROOT / "examples" / "imagenet-simple-labels.json"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STDDEV = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ResNet-50 image classification through the IREE ONNX Runtime EP."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the ONNX model (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help=f"Path to the ImageNet labels JSON (default: {DEFAULT_LABELS_PATH}).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        required=True,
        help="Input image to classify. Repeat for multiple images.",
    )
    parser.add_argument(
        "--driver",
        default="local-task",
        help="IREE driver to use, for example local-task or hip.",
    )
    parser.add_argument(
        "--target",
        default="none",
        help="IREE target arch, for example none on CPU or gfx1201 on RDNA4.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of predictions to print per image.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose ONNX Runtime and script logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    ort.set_default_logger_severity(0 if verbose else 2)


def validate_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def load_labels(path: Path) -> list[str]:
    with path.open() as f:
        labels = json.load(f)
    if not isinstance(labels, list):
        raise ValueError(f"Expected a JSON list of labels in {path}")
    return labels


def register_iree_ep() -> None:
    ep_name = iree_ep.get_ep_name()
    ep_library = iree_ep.get_library_path()
    LOGGER.debug("Registering execution provider %s from %s", ep_name, ep_library)
    ort.register_execution_provider_library(ep_name, ep_library)


def get_iree_device(driver: str):
    ep_devices = ort.get_ep_devices()
    for dev in ep_devices:
        if dev.device.metadata.get("iree.driver") == driver:
            LOGGER.debug("Selected IREE device metadata: %s", dev.device.metadata)
            return dev

    available = sorted(
        {
            dev.device.metadata.get("iree.driver")
            for dev in ep_devices
            if dev.device.metadata.get("iree.driver")
        }
    )
    raise RuntimeError(
        f"IREE device with driver '{driver}' not found. Available drivers: {available}"
    )


def create_session(model_path: Path, target: str, driver: str):
    register_iree_ep()
    iree_device = get_iree_device(driver)

    sess_options = ort.SessionOptions()
    sess_options.add_provider_for_devices(
        [iree_device],
        {
            "target_arch": target,
            "opt_level": "O3",
        },
    )
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        enable_fallback=False,
    )
    return session, iree_device


def get_model_io(session: ort.InferenceSession) -> tuple[str, str, int, int]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise ValueError(f"Expected a single model input, found {len(inputs)}")
    if len(outputs) != 1:
        raise ValueError(f"Expected a single model output, found {len(outputs)}")

    model_input = inputs[0]
    if len(model_input.shape) != 4:
        raise ValueError(
            f"Expected a 4D NCHW input tensor, got shape {model_input.shape}"
        )

    _, channels, height, width = model_input.shape
    if channels != 3:
        raise ValueError(f"Expected 3 input channels, got {channels}")
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError(f"Expected static image size, got shape {model_input.shape}")

    return model_input.name, outputs[0].name, height, width


def preprocess_image(image_path: Path, height: int, width: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize((width, height), resample=resampling.BILINEAR)

    image_data = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    image_data = image_data / 255.0
    image_data = (image_data - MEAN[:, None, None]) / STDDEV[:, None, None]
    return image_data.reshape(1, 3, height, width).astype(np.float32)


def softmax(values: np.ndarray) -> np.ndarray:
    values = values.reshape(-1)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def run_inference(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    image_tensor: np.ndarray,
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    output = session.run([output_name], {input_name: image_tensor})[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return softmax(np.asarray(output)), elapsed_ms


def print_predictions(
    image_path: Path,
    probabilities: np.ndarray,
    labels: list[str],
    elapsed_ms: float,
    top_k: int,
) -> None:
    top_k = min(top_k, len(labels))
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    best_index = int(top_indices[0])

    print(f"Image: {image_path}")
    print(f"Inference time: {elapsed_ms:.2f} ms")
    print(
        "Top prediction: "
        f"{labels[best_index]} ({probabilities[best_index] * 100.0:.2f}%)"
    )
    print(f"Top {top_k} predictions:")
    for rank, index in enumerate(top_indices, start=1):
        print(f"  {rank}. {labels[index]} ({probabilities[index] * 100.0:.2f}%)")
    print()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    model_path = validate_path(args.model, "Model")
    labels_path = validate_path(args.labels, "Labels")
    image_paths = [validate_path(path, "Image") for path in args.image]

    labels = load_labels(labels_path)
    session, iree_device = create_session(model_path, args.target, args.driver)
    input_name, output_name, height, width = get_model_io(session)

    LOGGER.info(
        "Running ResNet-50 on IREE driver=%s target=%s input=%s output=%s size=%dx%d",
        iree_device.device.metadata.get("iree.driver"),
        args.target,
        input_name,
        output_name,
        width,
        height,
    )

    for image_path in image_paths:
        image_tensor = preprocess_image(image_path, height, width)
        probabilities, elapsed_ms = run_inference(
            session, input_name, output_name, image_tensor
        )
        print_predictions(image_path, probabilities, labels, elapsed_ms, args.top_k)


if __name__ == "__main__":
    main()
