"""Pytest configuration and shared fixtures for IREE ONNX EP tests."""

import pytest

import onnxruntime as ort
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
def register_ep():
    """Register the IREE EP plugin once per test session."""
    global _ep_registered
    if not _ep_registered:
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
