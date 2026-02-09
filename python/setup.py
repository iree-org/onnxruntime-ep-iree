"""Setup script for the onnxruntime-ep-iree Python package.

Currently, we use ONNXRUNTIME_EP_IREE_BUILD_DIR to specify the build directory used by
the developer to built the library. The library is built externally via cmake
and not by this setup script. It is only required at install time, not at
runtime. For editable installs, setup.py only validates the setup.py exists in
the build directory, it doesn't copy or generate any files. At runtime,
``get_library_path()`` finds the library by walking up from the package's
source location to ``<project_root>/build/``. This means rebuilds are picked up
immediately without reinstalling. For wheel builds (``pip wheel`` / ``pip
install``), setup.py copies the library into the package directory so it is
bundled in the wheel.

Note that this is a very temporary setup. We are never going to ship like this.
The only reason we started out like this is because before this we were hardcoding
library paths, which didn't work when trying to develop the library on macos
vs linux (.dylib vs .so). We will eventually probably do something similar to
shortfin, where we have a devme.py file to get things setup for dev workflow,
and let the setup.py build a normal build and a tracy enabled build.
"""

import os
import pathlib
import shutil

from setuptools import Distribution, setup

script_dir = pathlib.Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Locate the pre-built native library via environment variable.
# ---------------------------------------------------------------------------
build_dir = os.environ.get("ONNXRUNTIME_EP_IREE_BUILD_DIR")
if not build_dir:
    raise EnvironmentError(
        "ONNXRUNTIME_EP_IREE_BUILD_DIR environment variable must be set to the "
        "directory containing the built libonnxruntime_ep_iree shared library."
    )

build_dir = pathlib.Path(build_dir).resolve()

# Find the library in the build directory.
lib_names = [
    "libonnxruntime_ep_iree.dylib",
    "libonnxruntime_ep_iree.so",
    "onnxruntime_ep_iree.dll",
]
found = [build_dir / name for name in lib_names if (build_dir / name).exists()]
if len(found) != 1:
    raise FileNotFoundError(
        f"Expected exactly one EP library in {build_dir}, "
        f"found: {[str(p) for p in found]}"
    )
ep_lib_path = found[0]

# ---------------------------------------------------------------------------
# For non-editable installs, copy the library into the package so it gets
# bundled into the wheel.  For editable installs this copy is unnecessary
# (get_library_path() finds it via relative path), but it's harmless and
# keeps setup.py simple — no need to detect which install mode we're in.
# ---------------------------------------------------------------------------
pkg_dir = script_dir / "onnxruntime_ep_iree"
dst = pkg_dir / ep_lib_path.name
if dst.resolve() != ep_lib_path.resolve():
    shutil.copyfile(ep_lib_path, dst)


# ---------------------------------------------------------------------------
# Setuptools configuration.
# ---------------------------------------------------------------------------
class BinaryDistribution(Distribution):
    """Mark the distribution as containing native code."""

    # Force setuptools to generate platform-specific wheels, because the package
    # contains a native shared library.
    def has_ext_modules(self):
        return True


setup(
    name="onnxruntime-ep-iree",
    version="0.1.0",
    description="Python helpers for the IREE ONNX Runtime Execution Provider",
    packages=["onnxruntime_ep_iree"],
    # Includes ``*.dylib``, ``*.so``, ``*.dll`` so that when the library IS copied
    # for wheel builds, it ends up in the installed package.
    package_data={
        "onnxruntime_ep_iree": ["*.dylib", "*.so", "*.dll"],
    },
    include_package_data=True,
    distclass=BinaryDistribution,
    python_requires=">=3.10",
)
