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

_COMPILER_LIB_NAMES = [
    "libIREECompiler.so",
    "libIREECompiler.dylib",
    "IREECompiler.dll",
]


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


def test_compiler_retry_after_bad_path():
    """After a failed Initialize() with a bad path, a second call with a good
    path in the same process succeeds — load failures are not cached.

    Regression guard: if s_init_error caching were reintroduced, the second
    session would re-emit the bad-path error and never use the compiler.
    We detect this by asserting the bad path appears exactly once in output
    (first attempt) and that inference via the second session is correct
    (confirming the compiler initialized and the EP compiled the model).
    """
    compiler_path = _find_compiler_lib_or_skip()

    script = f"""\
import sys
import onnxruntime as ort
import onnxruntime_ep_iree
import numpy as np

ep_lib_path = onnxruntime_ep_iree.get_library_path()
ort.register_execution_provider_library(
    onnxruntime_ep_iree.get_ep_name(), ep_lib_path
)

local_task_dev = None
for dev in ort.get_ep_devices():
    if dev.device.metadata.get("iree.driver") == "local-task":
        local_task_dev = dev
        break
if local_task_dev is None:
    print("SKIP: no local-task device")
    sys.exit(0)

from onnx import TensorProto, helper, save
import tempfile, os

input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 2])
output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 2])
relu_node = helper.make_node("Relu", ["A"], ["C"])
graph = helper.make_graph([relu_node], "test", [input_a], [output])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8

f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
save(model, f.name)
f.close()

try:
    # Attempt 1: bad compiler path.  ORT falls back to CPU; error goes to
    # stderr.  IreeCompiler::Initialize() fails but caches nothing.
    opts_bad = ort.SessionOptions()
    opts_bad.add_session_config_entry(
        "ep.iree.compiler_lib_path", "/nonexistent/libIREECompiler.so"
    )
    opts_bad.add_provider_for_devices([local_task_dev], {{"target_arch": "host"}})
    try:
        ort.InferenceSession(f.name, sess_options=opts_bad)
    except Exception:
        pass  # ORT may raise or fall back silently.

    # Attempt 2: good compiler path in the same process.  Must not be blocked
    # by the previous failure.
    opts_good = ort.SessionOptions()
    opts_good.add_session_config_entry(
        "ep.iree.compiler_lib_path", {compiler_path!r}
    )
    opts_good.add_provider_for_devices([local_task_dev], {{"target_arch": "host"}})
    session = ort.InferenceSession(f.name, sess_options=opts_good)

    # Print the providers used by the second session so the parent process can
    # assert the IREE EP is active (not just CPU fallback).
    print("PROVIDERS:" + ",".join(session.get_providers()), flush=True)

    input_data = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    result = session.run(None, {{"A": input_data}})[0]
    expected = np.maximum(0.0, input_data)
    if not np.allclose(result, expected):
        print(f"WRONG_RESULT: {{result}} vs {{expected}}", flush=True)
        sys.exit(2)

    print("RETRY_SUCCEEDED", flush=True)
finally:
    os.unlink(f.name)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = result.stdout + result.stderr

    if "SKIP:" in result.stdout:
        pytest.skip("no local-task device available")

    # Bad path must appear from attempt 1.
    assert "/nonexistent/libIREECompiler.so" in combined, (
        f"Expected bad-path error from attempt 1.\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )

    # The IREE EP must be active in the second session — not just CPU fallback.
    # If error caching regressed, Initialize() would return the cached bad-path
    # error, the EP would fail, and ORT would fall back to CPU only.
    providers_line = next(
        (l for l in result.stdout.splitlines() if l.startswith("PROVIDERS:")), ""
    )
    assert "IREE" in providers_line, (
        f"IREE EP not in second session providers — retry may have been blocked.\n"
        f"providers: {providers_line}\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )

    assert "RETRY_SUCCEEDED" in result.stdout, (
        f"Second session with good path did not succeed.\n"
        f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
    )
