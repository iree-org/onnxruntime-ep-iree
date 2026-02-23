"""Tests for compiler library discovery (C++ runtime paths and env vars).

Covers:
  - Python get_library_path() for EP shared library
  - C++ runtime env var override (IREE_EP_COMPILER_LIB / IREE_EP_COMPILER_LIB_DIR)
  - Clear error on invalid compiler path
"""

import os
import pathlib
import subprocess
import sys

import pytest

import onnxruntime_ep_iree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMPILER_LIB_NAMES = ["libIREECompiler.so", "libIREECompiler.dylib", "IREECompiler.dll"]


def _find_compiler_lib_or_skip():
    """Return path to the pip-installed compiler library, or skip."""
    try:
        import iree.compiler._mlir_libs as libs
    except ImportError:
        pytest.skip("iree-base-compiler not installed")

    lib_dir = pathlib.Path(libs.__file__).parent
    for name in _COMPILER_LIB_NAMES:
        candidate = lib_dir / name
        if candidate.exists():
            return str(candidate.resolve())
    pytest.skip("iree-base-compiler installed but compiler library not found")


def _run_ep_subprocess(env_overrides):
    """Run a subprocess that registers the EP and lists devices."""
    env = os.environ.copy()
    env.update(env_overrides)

    script = """\
import sys
import onnxruntime as ort
import onnxruntime_ep_iree

ep_lib_path = onnxruntime_ep_iree.get_library_path()
ort.register_execution_provider_library(
    onnxruntime_ep_iree.get_ep_name(), ep_lib_path
)

devices = ort.get_ep_devices()
iree_devices = [d for d in devices if d.device.metadata.get("iree.driver")]
if iree_devices:
    print("EP loaded successfully")
    sys.exit(0)
else:
    print("No IREE devices found")
    sys.exit(1)
"""

    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_library_path():
    """get_library_path() finds the EP shared library."""
    path = onnxruntime_ep_iree.get_library_path()
    assert pathlib.Path(path).exists(), f"EP library not found at {path}"


def test_env_var_compiler_lib_dir():
    """IREE_EP_COMPILER_LIB_DIR env var directs compiler discovery."""
    compiler_path = _find_compiler_lib_or_skip()
    compiler_dir = str(pathlib.Path(compiler_path).parent)

    result = _run_ep_subprocess({"IREE_EP_COMPILER_LIB_DIR": compiler_dir})
    assert result.returncode == 0, (
        f"EP failed with IREE_EP_COMPILER_LIB_DIR={compiler_dir}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_env_var_compiler_lib():
    """IREE_EP_COMPILER_LIB env var provides full path to compiler library."""
    compiler_path = _find_compiler_lib_or_skip()

    result = _run_ep_subprocess({"IREE_EP_COMPILER_LIB": compiler_path})
    assert result.returncode == 0, (
        f"EP failed with IREE_EP_COMPILER_LIB={compiler_path}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_bad_compiler_path_error_message():
    """Invalid compiler path produces a clear, diagnostic error message.

    ORT falls back to CPU when the EP fails — it doesn't raise an exception.
    We check that the error output contains the bad path and search order.
    """
    script = """\
import sys
import onnxruntime as ort
import onnxruntime_ep_iree

ep_lib_path = onnxruntime_ep_iree.get_library_path()
ort.register_execution_provider_library(
    onnxruntime_ep_iree.get_ep_name(), ep_lib_path
)

opts = ort.SessionOptions()
opts.add_session_config_entry(
    "ep.iree.compiler_lib_path", "/nonexistent/libIREECompiler.so"
)

from onnx import TensorProto, helper
input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 2])
output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 2])
relu_node = helper.make_node("Relu", ["A"], ["C"])
graph = helper.make_graph([relu_node], "test", [input_a], [output])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8

import tempfile, os
f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
from onnx import save
save(model, f.name)
f.close()

for dev in ort.get_ep_devices():
    if dev.device.metadata.get("iree.driver") == "local-task":
        opts.add_provider_for_devices([dev], {"target_arch": "host"})
        break

try:
    session = ort.InferenceSession(f.name, sess_options=opts)
finally:
    os.unlink(f.name)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = result.stdout + result.stderr
    assert "/nonexistent/libIREECompiler.so" in combined, (
        f"Error output should contain bad path.\nstdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}"
    )
    assert "Search order" in combined, (
        f"Error output should contain search order listing.\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )
    assert "session option" in combined, (
        f"Error output should mention session option step.\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )
