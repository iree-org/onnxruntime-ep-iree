"""Llama 3.1 text generation with ONNX Runtime and the IREE Execution Provider."""

import argparse
import logging
import time

import numpy as np
import onnxruntime as ort
import onnxruntime_ep_iree
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# IREE vendor ID for device allocation
IREE_VENDOR_ID = 0x1EEE


def _ort_type_to_numpy(ort_type: str):
    """Map an ORT type string like 'tensor(float16)' to a numpy dtype."""
    t = ort_type.lower()
    if "float16" in t or "half" in t:
        return np.float16
    if "float64" in t or "double" in t:
        return np.float64
    return np.float32


def get_iree_session(model_path: str, target: str, driver: str, verbose: bool = False):
    """Create an ONNX Runtime session with the IREE EP."""
    ep_lib_path = onnxruntime_ep_iree.get_library_path()

    ort.set_default_logger_severity(0 if verbose else 2)

    logger.debug("EP library path: %s", ep_lib_path)
    ort.register_execution_provider_library("IREE", str(ep_lib_path))
    logger.debug("EP plugin registered")

    ep_devices = ort.get_ep_devices()
    iree_device = None
    for dev in ep_devices:
        if dev.device.metadata.get("iree.driver") == driver:
            iree_device = dev
            break

    if not iree_device:
        available = [d.device.metadata.get("iree.driver") for d in ep_devices]
        raise RuntimeError(
            f"IREE device with driver '{driver}' not found. Available: {available}"
        )

    logger.debug(
        "IREE EP: driver=%s, device_id=%s", driver, iree_device.device.device_id
    )

    sess_options = ort.SessionOptions()
    provider_options = {
        "target_arch": target,
        "save_intermediates": "1",
        "opt_level": "O3",
        "dim_specs": (
            "sequence_length(1, 1),batch_size(1, 1);"
            "batch_size(1, 1),sequence_length(1, 1000000, 32)"
        ),
    }
    sess_options.add_provider_for_devices([iree_device], provider_options)

    session = ort.InferenceSession(model_path, sess_options=sess_options)
    return session, iree_device


def log_model_info(session):
    """Log model input/output information."""
    logger.debug("=== Model Inputs ===")
    for inp in session.get_inputs():
        logger.debug("  %s: shape=%s, type=%s", inp.name, inp.shape, inp.type)

    logger.debug("=== Model Outputs (first 5) ===")
    for out in session.get_outputs()[:5]:
        logger.debug("  %s: shape=%s, type=%s", out.name, out.shape, out.type)
    logger.debug("  ... (%d total outputs)", len(session.get_outputs()))


def get_model_config(session):
    """Extract model configuration from session inputs."""
    input_names = set()
    num_layers = 0
    num_kv_heads = 8
    head_dim = 128
    kv_dtype = np.float32

    for inp in session.get_inputs():
        input_names.add(inp.name)
        if "past_key_values" in inp.name and ".key" in inp.name:
            num_layers += 1
            kv_dtype = _ort_type_to_numpy(inp.type)
            if len(inp.shape) == 4:
                if isinstance(inp.shape[1], int):
                    num_kv_heads = inp.shape[1]
                if isinstance(inp.shape[3], int):
                    head_dim = inp.shape[3]

    logger.debug(
        "Model config: %d layers, %d KV heads, %d head dim, dtype=%s",
        num_layers,
        num_kv_heads,
        head_dim,
        kv_dtype,
    )

    return {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "kv_dtype": kv_dtype,
        "input_names": input_names,
    }


def generate_text_iobinding(
    session,
    iree_device,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
):
    """Generate text using ONNX Runtime IO binding with the IREE EP.

    IREE doesn't support 0-size tensors, so the KV cache is initialized with
    1 dummy position and attention masking is used to ignore it.
    """
    config = get_model_config(session)
    num_layers = config["num_layers"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    kv_dtype = config["kv_dtype"]
    model_input_names = config["input_names"]

    has_attention_mask = "attention_mask" in model_input_names
    has_position_ids = "position_ids" in model_input_names

    device_id = iree_device.device.device_id

    def alloc_kv(shape):
        """Allocate a KV cache tensor on the IREE device."""
        return ort.OrtValue.ortvalue_from_shape_and_type(
            list(shape),
            kv_dtype,
            device_type="gpu",
            device_id=device_id,
            vendor_id=IREE_VENDOR_ID,
        )

    output_names = [out.name for out in session.get_outputs()]

    try:
        logits_index = output_names.index("logits")
    except ValueError:
        raise RuntimeError(
            f"Could not find 'logits' output. Available: {output_names[:5]}..."
        )

    # Map output names (present.*) -> input names (past_key_values.*).
    kv_output_to_input = {}
    for name in output_names:
        past_name = name.replace("present.", "past_key_values.")
        if past_name in model_input_names and "past_key_values" in past_name:
            kv_output_to_input[name] = past_name

    # Tokenize and pad to a multiple of 32 for dim spec matching.
    input_ids = tokenizer.encode(prompt, return_tensors="np").astype(np.int64)
    prompt_len = input_ids.shape[1]
    pad_multiple = 32
    padded_len = ((prompt_len + pad_multiple - 1) // pad_multiple) * pad_multiple
    pad_amount = padded_len - prompt_len
    if pad_amount > 0:
        input_ids = np.concatenate(
            [input_ids, np.zeros((1, pad_amount), dtype=np.int64)], axis=1
        )
    logger.debug(
        "Prompt: '%s' -> %d tokens (padded to %d)", prompt, prompt_len, padded_len
    )

    # Initialize KV cache with 1 dummy position (IREE workaround for 0-size tensors).
    num_dummy = 1
    generated_tokens = list(input_ids[0, :prompt_len])

    # === PREFILL PHASE ===
    logger.info("Prefill phase (%d tokens)...", padded_len)

    inputs = {"input_ids": input_ids}

    if has_attention_mask:
        inputs["attention_mask"] = np.ones((1, num_dummy + padded_len), dtype=np.int64)

    if has_position_ids:
        inputs["position_ids"] = np.arange(padded_len, dtype=np.int64).reshape(1, -1)

    for i in range(num_layers):
        for suffix in ("key", "value"):
            name = f"past_key_values.{i}.{suffix}"
            if name in model_input_names:
                inputs[name] = np.zeros(
                    (1, num_kv_heads, num_dummy, head_dim), dtype=kv_dtype
                )

    t0 = time.perf_counter()
    outputs = session.run(None, inputs)
    prefill_ms = (time.perf_counter() - t0) * 1000
    logits = outputs[logits_index]
    logger.debug("Prefill: %.1f ms, logits shape: %s", prefill_ms, logits.shape)

    next_token = int(np.argmax(logits[0, prompt_len - 1, :]))
    generated_tokens.append(next_token)
    logger.debug("First token: %d = '%s'", next_token, tokenizer.decode([next_token]))

    # Strip padding from KV cache and move to device.
    keep_len = num_dummy + prompt_len
    logger.debug(
        "Moving KV cache to device (keeping %d of %d positions, stripping %d padding)...",
        keep_len,
        num_dummy + padded_len,
        pad_amount,
    )
    kv_cache_device = {}
    for i, name in enumerate(output_names):
        if name in kv_output_to_input:
            kv_data = outputs[i][:, :, :keep_len, :].copy()
            tensor = alloc_kv(kv_data.shape)
            tensor.update_inplace(kv_data)
            kv_cache_device[kv_output_to_input[name]] = tensor

    del outputs, logits, inputs

    # After stripping, KV cache contains: [dummy (1)] + [real tokens (prompt_len)]
    past_seq_len = keep_len

    # === DECODE PHASE ===
    logger.info("Decode phase...")
    decode_times = []

    for step in range(max_new_tokens - 1):
        io_binding = session.io_binding()

        input_ids_tensor = ort.OrtValue.ortvalue_from_numpy(
            np.array([[next_token]], dtype=np.int64)
        )
        io_binding.bind_ortvalue_input("input_ids", input_ids_tensor)

        if has_attention_mask:
            attention_mask = np.ones((1, past_seq_len + 1), dtype=np.int64)
            attn_tensor = ort.OrtValue.ortvalue_from_numpy(attention_mask)
            io_binding.bind_ortvalue_input("attention_mask", attn_tensor)

        if has_position_ids:
            real_pos = prompt_len + step
            position_ids = np.array([[real_pos]], dtype=np.int64)
            pos_tensor = ort.OrtValue.ortvalue_from_numpy(position_ids)
            io_binding.bind_ortvalue_input("position_ids", pos_tensor)

        for name, tensor in kv_cache_device.items():
            io_binding.bind_ortvalue_input(name, tensor)

        # Bind outputs: KV cache to pre-allocated GPU tensors, logits to CPU.
        new_kv_seq_len = past_seq_len + 1
        kv_output_tensors = {}
        for name in output_names:
            if name in kv_output_to_input:
                out_tensor = alloc_kv((1, num_kv_heads, new_kv_seq_len, head_dim))
                io_binding.bind_ortvalue_output(name, out_tensor)
                kv_output_tensors[kv_output_to_input[name]] = out_tensor
            else:
                io_binding.bind_output(name, device_type="cpu")

        t0 = time.perf_counter()
        session.run_with_iobinding(io_binding)
        step_ms = (time.perf_counter() - t0) * 1000
        decode_times.append(step_ms)
        logger.debug("  Decode step %d: %.1f ms", step, step_ms)

        # Get logits from CPU, KV cache stays on device.
        ort_outputs = io_binding.get_outputs()
        logits = ort_outputs[logits_index].numpy()
        next_token = int(np.argmax(logits[0, -1, :]))
        generated_tokens.append(next_token)

        # Free old KV input tensors and swap in the output tensors.
        del ort_outputs
        del io_binding
        while kv_cache_device:
            _, tensor = kv_cache_device.popitem()
            del tensor
        kv_cache_device = kv_output_tensors

        past_seq_len += 1

        if next_token == tokenizer.eos_token_id:
            logger.info("EOS at step %d", step + 1)
            break

    if decode_times:
        avg_ms = sum(decode_times) / len(decode_times)
        print(
            f"Prefill: {prefill_ms:.1f} ms | "
            f"Decode avg: {avg_ms:.1f} ms/token ({len(decode_times)} steps)"
        )

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Llama 3.1 text generation with IREE ONNX Runtime EP"
    )
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer")
    parser.add_argument(
        "--target",
        required=True,
        help="Compilation target arch (e.g. gfx1100, host, vulkan-spirv)",
    )
    parser.add_argument(
        "--driver",
        required=True,
        help="IREE HAL driver (hip, vulkan, local-task)",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of the United States is",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=20, help="Max new tokens to generate"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    return parser.parse_args()


def setup_logging(verbose: bool):
    """Configure logging level."""
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )


def main():
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("Loading tokenizer from %s...", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info("Loading model from %s...", args.model)
    logger.info("Using IREE EP: target=%s, driver=%s", args.target, args.driver)
    session, iree_device = get_iree_session(
        args.model, args.target, args.driver, verbose=args.verbose
    )

    log_model_info(session)

    generated = generate_text_iobinding(
        session, iree_device, tokenizer, args.prompt, max_new_tokens=args.max_tokens
    )

    print(generated)


if __name__ == "__main__":
    main()
