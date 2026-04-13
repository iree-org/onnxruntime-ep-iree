#!/usr/bin/env python3
"""Opinionated developer setup helper for onnxruntime-ep-iree.

First-time usage:
  rm -rf build
  python dev_me.py [--cmake=/path/to/cmake] [--clang=/path/to/clang] \
    [--onnxruntime=/path/to/onnxruntime] [--iree=/path/to/iree] \
    [--build-type=Debug] [--no-tracing]

Subsequent usage:
  python dev_me.py

This configures an editable install of the Python package into the active
Python environment using the package-owned build directory under
``build/cmake/default`` and, by default, also under ``build/cmake/tracy``.
If one or more configured build directories already exist under
``build/cmake/``, rerunning this script performs incremental rebuilds instead
of reconfiguring.
"""

from __future__ import annotations

import argparse
import importlib.util
from importlib import metadata
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


CMAKE_REQUIRED_VERSION = (3, 24)
PYTHON_REQUIRED_VERSION = (3, 10)
SETUPTOOLS_REQUIRED_VERSION = (77, 0)

DEFAULT_BUILD_TYPE = "RelWithDebInfo"


def parse_version_components(version_text: str) -> tuple[int, ...]:
    matches = re.findall(r"\d+", version_text)
    return tuple(int(match) for match in matches[:3])


def format_version(version: tuple[int, ...] | None) -> str:
    if not version:
        return "not found"
    return ".".join(str(part) for part in version)


def quote_command(parts: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in parts)


class EnvInfo:
    def __init__(self, args: argparse.Namespace):
        self.this_dir = Path(__file__).resolve().parent
        self.python_exe = sys.executable
        self.python_version = sys.version_info[:3]
        self.has_pip_module = importlib.util.find_spec("pip") is not None
        self.cmake_exe, self.cmake_version = self.find_cmake(args.cmake)
        self.ninja_exe = shutil.which("ninja")
        self.clang_exe, self.clangxx_exe = self.find_clang_pair(args.clang)
        self.onnxruntime_dir = self.find_onnxruntime(args.onnxruntime)
        self.iree_dir = self.find_iree(args.iree)
        self.setuptools_version = self.find_package_version("setuptools")
        self.wheel_version = self.find_package_version("wheel")
        self.configured_dirs = self.find_configured_dirs()

    def find_cmake(
        self, cmake_path: Path | None
    ) -> tuple[str | None, tuple[int, ...] | None]:
        candidates = [str(cmake_path)] if cmake_path else []
        if not candidates:
            default_cmake = shutil.which("cmake")
            if default_cmake:
                candidates.append(default_cmake)
        for candidate in candidates:
            try:
                output = subprocess.check_output(
                    [candidate, "--version"], stderr=subprocess.STDOUT, text=True
                )
            except (OSError, subprocess.CalledProcessError):
                continue
            match = re.search(r"cmake version ([^\s]+)", output)
            if match:
                return candidate, parse_version_components(match.group(1))
        return None, None

    def find_clang_pair(self, clang_path: Path | None) -> tuple[str | None, str | None]:
        if not clang_path:
            return None, None
        clang = Path(clang_path).resolve()
        if not clang.exists():
            print(f"ERROR: clang executable not found: {clang}")
            sys.exit(1)
        clangxx = clang.with_name("clang++")
        if not clangxx.exists():
            print(f"ERROR: matching clang++ executable not found next to {clang}")
            sys.exit(1)
        return str(clang), str(clangxx)

    def find_onnxruntime(self, onnxruntime_dir: Path | None) -> str | None:
        if onnxruntime_dir is None:
            return None

        candidate = onnxruntime_dir.resolve()
        header = (
            candidate
            / "include"
            / "onnxruntime"
            / "core"
            / "session"
            / "onnxruntime_c_api.h"
        )
        if not header.exists():
            print(
                "ERROR: --onnxruntime must point to an ONNX Runtime source tree "
                f"containing {header.relative_to(candidate)}"
            )
            sys.exit(1)
        return str(candidate)

    def find_iree(self, iree_dir: Path | None) -> str | None:
        if iree_dir is None:
            return None

        candidate = iree_dir.resolve()
        if not (candidate / "CMakeLists.txt").exists():
            print("ERROR: --iree must point to an IREE source tree")
            sys.exit(1)
        return str(candidate)

    def find_package_version(self, package_name: str) -> tuple[int, ...] | None:
        try:
            version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            return None
        return parse_version_components(version)

    def find_configured_dirs(self) -> list[Path]:
        build_root = self.this_dir / "build" / "cmake"
        if not build_root.exists():
            return []
        return sorted(
            path
            for path in build_root.iterdir()
            if path.is_dir() and (path / "CMakeCache.txt").exists()
        )

    def has_live_build_tool(self, build_dir: Path) -> bool:
        cache_file = build_dir / "CMakeCache.txt"
        if not cache_file.exists():
            return False
        for line in cache_file.read_text().splitlines():
            if line.startswith("CMAKE_MAKE_PROGRAM:"):
                build_tool = line.split("=", 1)[1]
                return bool(build_tool) and Path(build_tool).exists()
        return True

    def check_build_prereqs(self) -> None:
        if self.python_version < PYTHON_REQUIRED_VERSION:
            print(f"ERROR: python version too old: {self.python_exe}")
            print(
                f"  Required: {format_version(PYTHON_REQUIRED_VERSION)}, "
                f"Found: {format_version(self.python_version)}"
            )
            sys.exit(1)

        if self.cmake_version is None or self.cmake_version < CMAKE_REQUIRED_VERSION:
            print(f"ERROR: cmake not found or version too old: {self.cmake_exe}")
            print(
                f"  Required: {format_version(CMAKE_REQUIRED_VERSION)}, "
                f"Found: {format_version(self.cmake_version)}"
            )
            sys.exit(1)

    def check_configure_prereqs(self) -> None:
        self.check_build_prereqs()

        if not self.ninja_exe:
            print("ERROR: ninja not found on PATH")
            sys.exit(1)

        if (
            self.setuptools_version is None
            or self.setuptools_version < SETUPTOOLS_REQUIRED_VERSION
        ):
            print(
                "ERROR: setuptools is not installed or too old for editable builds. "
                f"Found {format_version(self.setuptools_version)}, "
                f"need at least {format_version(SETUPTOOLS_REQUIRED_VERSION)}"
            )
            sys.exit(1)

        if self.wheel_version is None:
            print("ERROR: wheel is not installed")
            sys.exit(1)

        if not self.has_pip_module:
            print(
                "ERROR: pip is not available in the active Python environment. "
                "If you are using uv, create the environment with `uv venv --seed`."
            )
            sys.exit(1)

    def __repr__(self) -> str:
        lines = [
            f"python: {self.python_exe} ({format_version(self.python_version)})",
            f"cmake: {self.cmake_exe} ({format_version(self.cmake_version)})",
            f"ninja: {self.ninja_exe or 'not found'}",
            f"clang: {self.clang_exe or 'system default'}",
            f"onnxruntime: {self.onnxruntime_dir or 'pinned fetch'}",
            f"iree: {self.iree_dir or 'pinned fetch'}",
            f"setuptools: {format_version(self.setuptools_version)}",
            f"wheel: {format_version(self.wheel_version)}",
        ]
        if self.configured_dirs:
            lines.append(
                "configured build dirs: "
                + ", ".join(
                    str(path.relative_to(self.this_dir))
                    for path in self.configured_dirs
                )
            )
        else:
            lines.append("configured build dirs: none")
        return "\n".join(lines)


def has_configure_overrides(args: argparse.Namespace) -> bool:
    return any(
        (
            args.cmake is not None,
            args.clang is not None,
            args.onnxruntime is not None,
            args.iree is not None,
            args.no_tracing,
            args.build_type != DEFAULT_BUILD_TYPE,
        )
    )


def configure_mode(env_info: EnvInfo, args: argparse.Namespace) -> None:
    print("Environment info:")
    print(env_info)
    env_info.check_configure_prereqs()

    env_vars = {
        "ONNXRUNTIME_EP_IREE_CMAKE": env_info.cmake_exe,
        "ONNXRUNTIME_EP_IREE_CMAKE_BUILD_TYPE": args.build_type,
        "ONNXRUNTIME_EP_IREE_ENABLE_TRACING": "OFF" if args.no_tracing else "ON",
    }
    if env_info.onnxruntime_dir:
        env_vars["ONNXRUNTIME_SOURCE_DIR"] = env_info.onnxruntime_dir
    if env_info.iree_dir:
        env_vars["ONNXRUNTIME_EP_IREE_IREE_SOURCE_DIR"] = env_info.iree_dir
    if env_info.clang_exe and env_info.clangxx_exe:
        env_vars["ONNXRUNTIME_EP_IREE_C_COMPILER"] = env_info.clang_exe
        env_vars["ONNXRUNTIME_EP_IREE_CXX_COMPILER"] = env_info.clangxx_exe

    setup_cmd = [
        env_info.python_exe,
        "-m",
        "pip",
        "install",
        # Reuse the active environment's build tooling instead of spinning up a
        # temporary isolated env. That keeps editable installs fast and ensures
        # rerunning dev_me.py rebuilds against the same cmake/ninja toolchain.
        "--no-build-isolation",
        "-v",
        "-e",
        str(env_info.this_dir / "python"),
    ]

    print("Executing setup:")
    for key, value in env_vars.items():
        print(f"  {key}={value}")
    print(f"  {quote_command(setup_cmd)}")

    actual_env = dict(os.environ)
    actual_env.update(env_vars)
    subprocess.check_call(setup_cmd, cwd=env_info.this_dir, env=actual_env)


def build_mode(env_info: EnvInfo, args: argparse.Namespace) -> None:
    env_info.check_build_prereqs()

    if has_configure_overrides(args):
        print(
            "NOTE: build/cmake is already configured; configure-time flags are being "
            "ignored. Delete build/ to reconfigure from scratch."
        )

    stale_dirs = [
        build_dir
        for build_dir in env_info.configured_dirs
        if not env_info.has_live_build_tool(build_dir)
    ]
    if stale_dirs:
        # The most common failure here is an old Ninja path cached from a prior
        # pip build environment. Re-running the editable install refreshes the
        # CMake cache using the current tool locations.
        print(
            "Detected stale build-tool paths in the existing CMake cache. "
            "Refreshing the editable install to reconfigure the build tree."
        )
        configure_mode(env_info, args)
        return

    # Rebuild every configured variant under build/cmake/.
    for build_dir in env_info.configured_dirs:
        print(f"Building {build_dir.relative_to(env_info.this_dir)}")
        subprocess.check_call(
            [env_info.cmake_exe, "--build", str(build_dir)],
            cwd=env_info.this_dir,
        )


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="onnxruntime-ep-iree dev", description="Development setup helper"
    )
    parser.add_argument("--cmake", type=Path, help="Path to cmake")
    parser.add_argument("--clang", type=Path, help="Path to clang")
    parser.add_argument(
        "--onnxruntime", type=Path, help="Path to an ONNX Runtime source checkout"
    )
    parser.add_argument("--iree", type=Path, help="Path to an IREE source checkout")
    parser.add_argument(
        "--build-type",
        default=DEFAULT_BUILD_TYPE,
        help=f"CMake build type (default: {DEFAULT_BUILD_TYPE})",
    )
    parser.add_argument(
        "--no-tracing",
        action="store_true",
        help="Disable the Tracy package variant during initial configure",
    )
    args = parser.parse_args(argv)

    env_info = EnvInfo(args)
    if env_info.configured_dirs:
        build_mode(env_info, args)
    else:
        configure_mode(env_info, args)


if __name__ == "__main__":
    main(sys.argv[1:])
