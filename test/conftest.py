"""Pytest configuration and shared fixtures for IREE ONNX EP tests."""

import glob
import os
import pathlib
import re
import tempfile

import onnx
import onnxruntime as ort
import pytest

import onnxruntime_ep_iree


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store",
        default=None,
        help="GPU driver to use for GPU tests (e.g. 'hip', 'vulkan')",
    )
    parser.addoption(
        "--gpu-target",
        action="store",
        default=None,
        help="GPU compilation target (e.g. 'gfx1100', 'gfx1201', 'vulkan-spirv')",
    )
    parser.addoption(
        "--gpu-device-id",
        action="store",
        default=None,
        type=int,
        help="GPU device ID to use (e.g. 0, 1). Defaults to first device found.",
    )
    # ExternDispatch test options.
    parser.addoption("--ep-lib", default=None, help="Path to EP .so")
    parser.addoption("--kernel-dir", default="", help="Kernel .co dir")
    parser.addoption("--target-arch", default="gfx1100", help="GPU arch")
    parser.addoption("--device-index", type=int, default=0, help="Device index")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as requiring a GPU device")


def pytest_collection_modifyitems(config, items):
    gpu_driver = config.getoption("--gpu")
    if gpu_driver:
        return
    skip_gpu = pytest.mark.skip(
        reason="GPU tests require --gpu=<driver> (e.g. --gpu=hip)"
    )
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


_ep_registered = False


@pytest.fixture(scope="session", autouse=True)
def register_ep(request):
    """Register the IREE EP plugin once per test session.

    Uses --ep-lib if provided, otherwise uses the installed Python package.
    """
    global _ep_registered
    if not _ep_registered:
        ep_lib_path = request.config.getoption("--ep-lib")
        if ep_lib_path:
            ort.register_execution_provider_library("IREE", ep_lib_path)
        else:
            ep_lib_path = onnxruntime_ep_iree.get_library_path()
            ort.register_execution_provider_library(
                onnxruntime_ep_iree.get_ep_name(), ep_lib_path
            )
        _ep_registered = True


@pytest.fixture(scope="session")
def iree_device(register_ep):
    """Return an IREE EP device using the local-task driver."""
    device = _get_iree_device("local-task")
    if not device:
        pytest.fail("IREE EP device with local-task driver not found")
    return device


@pytest.fixture(scope="session")
def iree_gpu_device(request, register_ep):
    """Return an IREE EP device for the GPU driver specified via --gpu."""
    driver = request.config.getoption("--gpu")
    if not driver:
        pytest.skip("GPU tests require --gpu=<driver>")
    device_id = request.config.getoption("--gpu-device-id")
    device = _get_iree_device(driver, device_id=device_id)
    if not device:
        msg = f"IREE EP device with {driver} driver"
        if device_id is not None:
            msg += f" and device_id={device_id}"
        msg += " not found"
        pytest.fail(msg)
    return device


@pytest.fixture(scope="session")
def gpu_target(request):
    """Return the GPU compilation target specified via --gpu-target."""
    target = request.config.getoption("--gpu-target")
    if not target:
        driver = request.config.getoption("--gpu")
        if driver == "vulkan":
            return "vulkan-spirv"
        pytest.fail(
            f"--gpu-target is required for driver '{driver}' "
            f"(e.g. --gpu-target=gfx1100)"
        )
    return target


def _get_iree_device(driver, device_id=None):
    """Find an IREE EP device by driver name and optional device ID."""
    for dev in ort.get_ep_devices():
        if dev.device.metadata.get("iree.driver") != driver:
            continue
        if device_id is not None and dev.device.device_id != device_id:
            continue
        return dev
    return None


# ---------------------------------------------------------------------------
# ExternDispatch fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ep_lib(request):
    return request.config.getoption("--ep-lib")


@pytest.fixture(scope="session")
def kernel_dir(request):
    return request.config.getoption("--kernel-dir")


@pytest.fixture(scope="session")
def target_arch(request):
    return request.config.getoption("--target-arch")


@pytest.fixture(scope="session")
def device(request, register_ep):
    """Get any available IREE device. Prefers HIP, falls back to any."""
    ep_devices = ort.get_ep_devices()
    if not ep_devices:
        pytest.skip("No IREE devices available")
    idx = request.config.getoption("--device-index")
    # Prefer HIP if available.
    hip_devices = [
        dev for dev in ep_devices if dev.device.metadata.get("iree.driver", "") == "hip"
    ]
    if hip_devices and idx < len(hip_devices):
        return hip_devices[idx]
    # Fall back to any device.
    if idx < len(ep_devices):
        return ep_devices[idx]
    return ep_devices[0]


@pytest.fixture(scope="session")
def gpu_device(device):
    """Require a HIP device. Skips otherwise.

    ExternDispatch tests use pre-compiled HIP kernel objects (.co files)
    built by build_kernels.sh, so only HIP devices are supported.
    """
    driver = device.device.metadata.get("iree.driver", "")
    if driver != "hip":
        pytest.skip(f"ExternDispatch tests require a HIP device (got '{driver}')")
    return device


@pytest.fixture(scope="session")
def cpu_device(register_ep):
    """Return an IREE EP device using the local-task driver."""
    device = _get_iree_device("local-task")
    if not device:
        pytest.skip("No local-task device available")
    return device


@pytest.fixture(scope="session")
def vulkan_device(register_ep):
    """Return an IREE EP device using the vulkan driver."""
    device = _get_iree_device("vulkan")
    if not device:
        pytest.skip("No Vulkan device available")
    return device


# ---------------------------------------------------------------------------
# ExternDispatch helpers
# ---------------------------------------------------------------------------


def try_compile(model, device, kernel_dir, target_arch):
    """Try to compile a model. Returns (success, error_msg)."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = f.name
        onnx.save(model, model_path)

    try:
        sess_options = ort.SessionOptions()
        provider_options = {
            "target_arch": target_arch,
            "extern_kernel_path": kernel_dir,
        }
        sess_options.add_provider_for_devices([device], provider_options)
        ort.InferenceSession(model_path, sess_options=sess_options)
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        pathlib.Path(model_path).unlink(missing_ok=True)


def try_generate_mlir(model, device, kernel_dir, target_arch):
    """Generate MLIR for a model via the EP. Returns (mlir_str, error_msg).

    Uses save_intermediates to keep the MLIR file that the EP writes before
    invoking iree-compile.  Runs with a private temp directory (via TMPDIR)
    so that concurrent test runs cannot interfere with each other.

    Distinguishing MLIR-gen errors from iree-compile errors:
    - Session succeeds -> MLIR gen succeeded; find and return the MLIR file.
    - Error contains "iree-compile" -> MLIR gen succeeded but iree-compile
      failed; find and return the MLIR file.
    - Error without "iree-compile" -> MLIR gen itself failed; return error.

    Returns:
        (mlir_content, None) on successful MLIR generation.
        (None, error_msg) if MLIR generation itself fails (before iree-compile).
    """
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = f.name
        onnx.save(model, model_path)

    # Use a private temp directory so the EP's TempFile (which reads
    # std::filesystem::temp_directory_path() -> TMPDIR) writes only here.
    private_tmp = tempfile.mkdtemp(prefix="iree_ep_test_")
    old_tmpdir = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = private_tmp

    err = None
    try:
        sess_options = ort.SessionOptions()
        provider_options = {
            "target_arch": target_arch,
            "extern_kernel_path": kernel_dir,
            "save_intermediates": "1",
        }
        sess_options.add_provider_for_devices([device], provider_options)
        ort.InferenceSession(model_path, sess_options=sess_options)
        # Session succeeded -- MLIR gen worked and iree-compile worked.
    except Exception as e:
        err = str(e)
    finally:
        # Restore TMPDIR immediately.
        if old_tmpdir is not None:
            os.environ["TMPDIR"] = old_tmpdir
        else:
            os.environ.pop("TMPDIR", None)
        pathlib.Path(model_path).unlink(missing_ok=True)

    # Locate the MLIR file in the private temp directory.
    mlir_files = glob.glob(os.path.join(private_tmp, "iree_ep_*.mlir"))
    mlir_path = pathlib.Path(mlir_files[0]) if mlir_files else None

    def _cleanup():
        """Remove the entire private temp directory."""
        import shutil

        shutil.rmtree(private_tmp, ignore_errors=True)

    if err is None:
        # Session succeeded -- return MLIR content.
        if mlir_path:
            try:
                return mlir_path.read_text(), None
            finally:
                _cleanup()
        _cleanup()
        return None, "Session succeeded but no MLIR file found"

    # Session failed. Check whether MLIR gen succeeded (error from
    # iree-compile) or MLIR gen itself failed.
    if "iree-compile" in err:
        # iree-compile failed, but MLIR gen succeeded -- return the MLIR.
        if mlir_path:
            try:
                return mlir_path.read_text(), None
            finally:
                _cleanup()
        # Fallback: parse path from error message.
        m = re.search(r'iree-compile "([^"]+\.mlir)"', err)
        if m:
            fallback_path = pathlib.Path(m.group(1))
            if fallback_path.exists():
                try:
                    return fallback_path.read_text(), None
                finally:
                    fallback_path.unlink(missing_ok=True)
        _cleanup()
        return None, f"iree-compile failed and MLIR file not found: {err}"

    # Error did not mention iree-compile -- MLIR gen itself failed.
    _cleanup()
    return None, err
