"""IREE Execution Provider for ONNX Runtime — Python helper package.

Provides functions to locate the native EP shared library and its registration name.
"""

import importlib
import os
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = [
    "get_library_path",
    "get_ep_name",
    "get_ep_names",
]

_LIB_NAMES = [
    "libonnxruntime_ep_iree.dylib",
    "libonnxruntime_ep_iree.so",
    "onnxruntime_ep_iree.dll",
]
_RUNTIME_ENV_VAR = "ONNXRUNTIME_EP_IREE_PY_RUNTIME"


def _find_lib_in(directory: Path) -> str | None:
    """Return the library path if exactly one matching library exists in *directory*."""
    if not directory.is_dir():
        return None
    found = [directory / name for name in _LIB_NAMES if (directory / name).exists()]
    if len(found) == 1:
        return str(found[0])
    return None


def _get_runtime_variant() -> str:
    runtime_variant = os.environ.get(_RUNTIME_ENV_VAR, "default").strip().lower()
    if runtime_variant not in {"default", "tracy"}:
        warnings.warn(
            f"Unknown value for {_RUNTIME_ENV_VAR} ({runtime_variant}): Using default"
        )
        return "default"
    return runtime_variant


def _get_runtime_package() -> ModuleType:
    runtime_variant = _get_runtime_variant()
    if runtime_variant == "tracy":
        runtime_package = importlib.import_module("onnxruntime_ep_iree_tracy")
        print(
            f"-- Using Tracy runtime ({_RUNTIME_ENV_VAR}=tracy)",
            file=sys.stderr,
        )
        return runtime_package
    return importlib.import_module("onnxruntime_ep_iree_default")


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _find_variant_build_library(runtime_variant: str) -> str | None:
    build_dir = _get_project_root() / "build" / "cmake" / runtime_variant
    return _find_lib_in(build_dir)


if TYPE_CHECKING:
    _runtime_package: ModuleType


def get_library_path() -> str:
    """Return the absolute path to the native IREE EP shared library.

    The runtime variant is selected in ``__init__`` via
    ``ONNXRUNTIME_EP_IREE_PY_RUNTIME`` and the shared library is resolved from
    the corresponding built runtime package.
    """
    runtime_variant = _get_runtime_variant()
    try:
        runtime_package = _get_runtime_package()
    except ModuleNotFoundError:
        runtime_package = None
        if runtime_variant == "tracy":
            build_result = _find_variant_build_library(runtime_variant)
            if not build_result:
                raise ModuleNotFoundError(
                    "Tracy runtime requested via "
                    f"{_RUNTIME_ENV_VAR}=tracy but it is not enabled in this build"
                ) from None
            return build_result

    if runtime_package is not None:
        result = _find_lib_in(Path(runtime_package.__file__).resolve().parent)
        if result:
            return result

    result = _find_variant_build_library(runtime_variant)
    if result:
        return result

    raise FileNotFoundError(
        "IREE EP library not found. "
        "Build the package with `pip install ./python` or "
        "`pip wheel -w dist ./python`."
    )


def get_ep_name() -> str:
    """Return the IREE execution provider registration name."""
    return "IREE"


def get_ep_names() -> list[str]:
    """Return a list of execution provider names provided by this package."""
    return [get_ep_name()]
