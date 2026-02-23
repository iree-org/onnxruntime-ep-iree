"""IREE Execution Provider for ONNX Runtime — Python helper package.

Provides functions to locate the native EP shared library and its registration name.
"""

from pathlib import Path

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


def _find_lib_in(directory: Path) -> str | None:
    """Return the library path if exactly one matching library exists in *directory*."""
    if not directory.is_dir():
        return None
    found = [directory / name for name in _LIB_NAMES if (directory / name).exists()]
    if len(found) == 1:
        return str(found[0])
    return None


def get_library_path() -> str:
    """Return the absolute path to the native IREE EP shared library.

    Lookup order:
      1. The project's ``build/`` directory, found by walking up from this package's
         location (``python/onnxruntime_ep_iree/ -> python/ -> project_root/ -> build/``).
         This is the primary path for editable installs — it always points to the
         latest build output, even after the C++ library is rebuilt.
      2. The package directory itself (for wheel-based installs where the library
         was bundled into the package).

    Raises ``FileNotFoundError`` if the library cannot be found in either location.
    """
    pkg_dir = Path(__file__).parent

    # 1. Check the project build directory (editable installs).
    #    Layout: <project_root>/python/onnxruntime_ep_iree/__init__.py
    project_root = pkg_dir.parent.parent
    result = _find_lib_in(project_root / "build")
    if result:
        return result

    # 2. Check the package directory (wheel installs).
    result = _find_lib_in(pkg_dir)
    if result:
        return result

    raise FileNotFoundError(
        "IREE EP library not found. "
        "Make sure the C++ library has been built (cmake -B build -GNinja && ninja -C build)."
    )



def get_ep_name() -> str:
    """Return the IREE execution provider registration name."""
    return "IREE"


def get_ep_names() -> list[str]:
    """Return a list of execution provider names provided by this package."""
    return [get_ep_name()]
