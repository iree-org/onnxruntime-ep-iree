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
DEFAULT_BUILD_DIR = SOURCE_DIR / "build" / "cmake" / "default"
TRACY_BUILD_DIR = SOURCE_DIR / "build" / "cmake" / "tracy"
CMAKE_EXE = os.environ.get("ONNXRUNTIME_EP_IREE_CMAKE", "cmake")
ENABLE_TRACY = os.environ.get("ONNXRUNTIME_EP_IREE_ENABLE_TRACING", "").upper() in (
    "1",
    "ON",
    "TRUE",
)
SELECTOR_PACKAGE = "onnxruntime_ep_iree"
DEFAULT_PACKAGE = "onnxruntime_ep_iree_default"
TRACY_PACKAGE = "onnxruntime_ep_iree_tracy"
LIB_NAMES = [
    "libonnxruntime_ep_iree.dylib",
    "libonnxruntime_ep_iree.so",
    "onnxruntime_ep_iree.dll",
]
REL_SOURCE_DIR = pathlib.Path(os.path.relpath(SOURCE_DIR, SCRIPT_DIR))
REL_DEFAULT_PYTHON_DIR = pathlib.Path(
    os.path.relpath(DEFAULT_BUILD_DIR / "python", SCRIPT_DIR)
)
REL_TRACY_PYTHON_DIR = pathlib.Path(
    os.path.relpath(TRACY_BUILD_DIR / "python", SCRIPT_DIR)
)


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


def ensure_built_package(package_root: pathlib.Path) -> None:
    package_root.mkdir(parents=True, exist_ok=True)
    init_file = package_root / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")


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


def build_native_extension(
    build_dir: pathlib.Path, extra_cmake_args: list[str] | None = None
) -> pathlib.Path:
    build_dir = build_dir.resolve()
    stage_dir = build_dir / "python-install"
    extra_cmake_args = extra_cmake_args or []
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
    cmake_args.extend(extra_cmake_args)

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


def copy_built_library(
    built_library: pathlib.Path, package_root: pathlib.Path, build_py: _build_py
) -> None:
    ensure_built_package(package_root)
    wheel_target_dir = pathlib.Path(build_py.build_lib) / package_root.name
    wheel_target_dir.mkdir(parents=True, exist_ok=True)
    for directory in [package_root, wheel_target_dir]:
        for pattern in ("*.so", "*.dylib", "*.dll"):
            for candidate in directory.glob(pattern):
                candidate.unlink()
        shutil.copyfile(built_library, directory / built_library.name)


class CMakeBuildPy(_build_py):
    def run(self):
        super().run()
        try:
            default_package_root = DEFAULT_BUILD_DIR / "python" / DEFAULT_PACKAGE
            default_library = build_native_extension(DEFAULT_BUILD_DIR)
            copy_built_library(default_library, default_package_root, self)
            if ENABLE_TRACY:
                tracy_package_root = TRACY_BUILD_DIR / "python" / TRACY_PACKAGE
                tracy_library = build_native_extension(
                    TRACY_BUILD_DIR,
                    ["-DONNXRUNTIME_EP_IREE_ENABLE_TRACING=ON"],
                )
                copy_built_library(tracy_library, tracy_package_root, self)
        except subprocess.CalledProcessError:
            # Editable installs route through setuptools, which otherwise tends
            # to collapse native build failures into a generic packaging error.
            # Keep the underlying traceback visible and fail the install
            # immediately.
            print("Native build failed:")
            traceback.print_exc()
            sys.exit(1)


# ---------------------------------------------------------------------------
# Setuptools configuration.
# ---------------------------------------------------------------------------
class BinaryDistribution(Distribution):
    """Mark the distribution as containing native code."""

    # Force setuptools to generate platform-specific wheels, because the package
    # contains a native shared library.
    def has_ext_modules(self):
        return True


ensure_built_package(DEFAULT_BUILD_DIR / "python" / DEFAULT_PACKAGE)
if ENABLE_TRACY:
    ensure_built_package(TRACY_BUILD_DIR / "python" / TRACY_PACKAGE)

setup(
    name="onnxruntime-ep-iree",
    version="0.1.0",
    description="Python helpers for the IREE ONNX Runtime Execution Provider",
    packages=[SELECTOR_PACKAGE, DEFAULT_PACKAGE]
    + ([TRACY_PACKAGE] if ENABLE_TRACY else []),
    package_dir={
        SELECTOR_PACKAGE: SELECTOR_PACKAGE,
        DEFAULT_PACKAGE: str(REL_DEFAULT_PYTHON_DIR / DEFAULT_PACKAGE),
        **(
            {TRACY_PACKAGE: str(REL_TRACY_PYTHON_DIR / TRACY_PACKAGE)}
            if ENABLE_TRACY
            else {}
        ),
    },
    cmdclass={
        "build_py": CMakeBuildPy,
    },
    include_package_data=False,
    distclass=BinaryDistribution,
    python_requires=">=3.10",
    zip_safe=False,
)
