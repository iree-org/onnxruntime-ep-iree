"""Shared test utilities for IREE ONNX EP tests."""

import tempfile

import onnx
import onnxruntime as ort
import onnxruntime_ep_iree

_ep_registered = False


def register_ep():
    """Register the IREE EP plugin. Safe to call multiple times."""
    global _ep_registered
    if not _ep_registered:
        ep_lib_path = onnxruntime_ep_iree.get_library_path()
        ort.register_execution_provider_library(
            onnxruntime_ep_iree.get_ep_name(), ep_lib_path
        )
        _ep_registered = True


def get_iree_device(driver="local-task"):
    """Find an IREE EP device by driver name. Returns None if not found."""
    for dev in ort.get_ep_devices():
        if dev.device.metadata.get("iree.driver") == driver:
            return dev
    return None


def save_model(model):
    """Save ONNX model to a temp file. Returns path (caller must delete)."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        return f.name


def create_session(model_path, device, provider_options=None):
    """Create an ORT InferenceSession with the given IREE device."""
    opts = ort.SessionOptions()
    opts.add_provider_for_devices([device], provider_options or {})
    return ort.InferenceSession(model_path, sess_options=opts)
