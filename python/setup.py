import os
import pathlib
import shutil
import subprocess
import sys
import traceback

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
SOURCE_DIR = SCRIPT_DIR.parent
# Keep packaging-owned builds in a single stable location so editable installs,
# wheel builds, and runtime library lookup all agree on where the native
# artifact lives.
DEFAULT_BUILD_DIR = SOURCE_DIR / "build" / "cmake" / "default"
CMAKE_EXE = os.environ.get("ONNXRUNTIME_EP_IREE_CMAKE", "cmake")
LIB_NAMES = [
    "libonnxruntime_ep_iree.dylib",
    "libonnxruntime_ep_iree.so",
    "onnxruntime_ep_iree.dll",
]


def add_env_cmake_setting(
    args: list[str], env_name: str, cmake_name: str | None = None
):
    value = os.getenv(env_name)
    if value:
        args.append(f"-D{cmake_name or env_name}={value}")


def maybe_nuke_cmake_cache(build_dir: pathlib.Path) -> str:
    ninja_path = shutil.which("ninja") or ""
    # PEP 517/660 builds can run under different virtualenvs or isolated build
    # environments across invocations. CMake caches both the Python executable
    # and the Ninja path, so preserve a tiny stamp and invalidate the cache when
    # either one changes underneath an existing build tree.
    expected_stamp = f"{sys.executable}\n{ninja_path}"
    stamp_file = build_dir / "python_stamp.txt"
    if stamp_file.exists() and stamp_file.read_text() == expected_stamp:
        return ninja_path

    cmake_cache = build_dir / "CMakeCache.txt"
    if cmake_cache.exists():
        cmake_cache.unlink()

    stamp_file.write_text(expected_stamp)
    return ninja_path


def find_library(root_dir: pathlib.Path) -> pathlib.Path:
    candidates = []
    # CMake install layouts differ slightly across platforms and generators, so
    # search the common staging locations and require a single unambiguous EP
    # library before packaging it.
    for directory in [
        root_dir,
        root_dir / "lib",
        root_dir / "bin",
        root_dir / "Release",
        root_dir / "Debug",
    ]:
        for lib_name in LIB_NAMES:
            lib_path = directory / lib_name
            if lib_path.exists():
                candidates.append(lib_path)
    unique_candidates = sorted(set(candidates), key=str)
    if len(unique_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one EP library under {root_dir}, "
            f"found: {[str(path) for path in unique_candidates]}"
        )
    return unique_candidates[0]


def build_native_extension() -> pathlib.Path:
    build_dir = DEFAULT_BUILD_DIR.resolve()
    stage_dir = build_dir / "python-install"
    build_type = os.environ.get("ONNXRUNTIME_EP_IREE_CMAKE_BUILD_TYPE", "Release")
    generator = os.environ.get("CMAKE_GENERATOR", "Ninja")
    build_dir.mkdir(parents=True, exist_ok=True)
    ninja_path = maybe_nuke_cmake_cache(build_dir)

    cmake_args = [
        "-S",
        str(SOURCE_DIR),
        "-B",
        str(build_dir),
        "-G",
        generator,
        f"-DCMAKE_BUILD_TYPE={build_type}",
        # Pass both spellings because the top-level project and fetched
        # dependencies may use different FindPython modules.
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    if generator.startswith("Ninja") and ninja_path:
        cmake_args.append(f"-DCMAKE_MAKE_PROGRAM={ninja_path}")
    add_env_cmake_setting(cmake_args, "ONNXRUNTIME_SOURCE_DIR")
    add_env_cmake_setting(cmake_args, "ONNXRUNTIME_VERSION")
    add_env_cmake_setting(
        cmake_args, "ONNXRUNTIME_EP_IREE_IREE_SOURCE_DIR", "IREE_SOURCE_DIR"
    )
    add_env_cmake_setting(
        cmake_args, "ONNXRUNTIME_EP_IREE_C_COMPILER", "CMAKE_C_COMPILER"
    )
    add_env_cmake_setting(
        cmake_args, "ONNXRUNTIME_EP_IREE_CXX_COMPILER", "CMAKE_CXX_COMPILER"
    )

    subprocess.check_call([CMAKE_EXE] + cmake_args)
    subprocess.check_call(
        [CMAKE_EXE, "--build", str(build_dir), "--config", build_type]
    )
    subprocess.check_call(
        [
            CMAKE_EXE,
            "--install",
            str(build_dir),
            "--config",
            build_type,
            "--prefix",
            str(stage_dir),
        ]
    )
    return find_library(stage_dir)


class CMakeBuildPy(_build_py):
    def run(self):
        super().run()
        try:
            built_library = build_native_extension()
        except subprocess.CalledProcessError:
            # Editable installs route through setuptools, which otherwise tends
            # to collapse native build failures into a generic packaging error.
            # Keep the underlying traceback visible and fail the install
            # immediately.
            print("Native build failed:")
            traceback.print_exc()
            sys.exit(1)
        target_dir = pathlib.Path(self.build_lib) / "onnxruntime_ep_iree"
        target_dir.mkdir(parents=True, exist_ok=True)
        # Wheels should contain exactly the freshly staged EP library and not a
        # leftover artifact from a previous platform or build configuration.
        for pattern in ("*.so", "*.dylib", "*.dll"):
            for candidate in target_dir.glob(pattern):
                candidate.unlink()
        shutil.copyfile(built_library, target_dir / built_library.name)


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
    cmdclass={
        "build_py": CMakeBuildPy,
    },
    # Include only the packaged EP shared library so stale local artifacts do
    # not leak into wheels.
    package_data={
        "onnxruntime_ep_iree": [
            "libonnxruntime_ep_iree.dylib",
            "libonnxruntime_ep_iree.so",
            "onnxruntime_ep_iree.dll",
        ],
    },
    include_package_data=False,
    distclass=BinaryDistribution,
    python_requires=">=3.10",
    zip_safe=False,
)
