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
      1. The packaging build directory (``<project_root>/build/cmake/default``),
         which is where ``pip install`` / ``pip wheel`` builds the shared library.
      2. The package directory itself (for wheel-based installs where the library
         was bundled into the package).

    Raises ``FileNotFoundError`` if the library cannot be found in either location.
    """
    pkg_dir = Path(__file__).parent
    project_root = pkg_dir.parent.parent

    # Editable installs import Python from the source tree, so the native EP is
    # resolved from the packaging-owned build tree instead of site-packages.
    result = _find_lib_in(project_root / "build" / "cmake" / "default")
    if result:
        return result

    # Wheel installs bundle the shared library directly into the package.
    result = _find_lib_in(pkg_dir)
    if result:
        return result

    raise FileNotFoundError(
        "IREE EP library not found. "
        "Build the package with `pip install ./python` or "
        "`pip wheel -w dist ./python`. For editable installs, build the native library in "
        "`build/cmake/default`."
    )


def get_ep_name() -> str:
    """Return the IREE execution provider registration name."""
    return "IREE"


def get_ep_names() -> list[str]:
    """Return a list of execution provider names provided by this package."""
    return [get_ep_name()]
