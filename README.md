# IREE ONNX Runtime Execution Provider

IREE based ONNX Runtime Execution Provider.

## Basic Python Installation

This path is recommended for Python users: install or wheel the Python package.

1. Create and activate a virtual environment

```bash
python3 -m venv .env
source .env/bin/activate
```

2. Build and install the Python package directly

```bash
pip install ./python
```

This runs CMake as part of the package build and places the native build tree in
`build/cmake/default` by default.

Today this path only packages the default variant. The entrypoint is intended to
grow a tracing variant later without changing the install command.

To get the shared library path after installation, run:

```bash
python3 -c 'import onnxruntime_ep_iree; print(onnxruntime_ep_iree.get_library_path())'
```

Optional environment variables:

```bash
# For build configuration, e.g. Debug or Release. Default is Release.
ONNXRUNTIME_EP_IREE_CMAKE_BUILD_TYPE=Debug

# For local source checkouts instead of the pinned fetched sources.
ONNXRUNTIME_SOURCE_DIR=/path/to/onnxruntime
ONNXRUNTIME_VERSION=1.24.1

# For local source checkouts instead of the pinned fetched sources.
ONNXRUNTIME_EP_IREE_IREE_SOURCE_DIR=/path/to/iree
```

## Recommended Dev Setup

For developers, we recommend using `dev_me.py` for editable installs and
rebuilds.

1. Create a virtual environment and install development dependencies

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements-dev.txt setuptools wheel
```

Tip: If using `uv`, seed `pip` into the environment so `dev_me.py` can pick it up:

```bash
uv venv --seed .env
source .env/bin/activate
uv pip install -r requirements-dev.txt setuptools wheel
```

2. Make sure `cmake` and `ninja` are available on your PATH

3. Start from a fresh build directory when doing the initial setup or when
   changing configure-time inputs

```bash
rm -rf build/
```

4. Run the dev helper

```bash
./dev_me.py
```

Pass `--onnxruntime=/path/to/onnxruntime` and/or `--iree=/path/to/iree`
explicitly if you want local source checkouts. Otherwise the package build
falls back to the pinned fetched sources.

Useful overrides:

```bash
./dev_me.py --build-type=Debug
./dev_me.py --onnxruntime=/path/to/onnxruntime --iree=/path/to/iree
./dev_me.py --clang=/path/to/clang
```

Behavior:

- On the first run, `dev_me.py` performs `pip install --no-build-isolation -e ./python`
  and configures the package-owned build tree under `build/cmake/default`.
- On later runs, `dev_me.py` performs incremental rebuilds of every configured
  build directory under `build/cmake/`.
- Delete `build/` to start over with a fresh configuration.

Today this path only creates the default editable build. The rebuild path is
already structured so future variants such as tracy will rebuild automatically
once we add them.

5. Run sample test

```bash
python test/test_ep_load.py
```

## Advanced Non-Python Usage

If you only want the shared library and do not care about the Python package,
build it directly with CMake:

```bash
cmake -S . -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_SOURCE_DIR=/path/to/onnxruntime \
  -DIREE_SOURCE_DIR=/path/to/iree
cmake --build build/native
```

If `ONNXRUNTIME_SOURCE_DIR` or `IREE_SOURCE_DIR` are omitted, CMake will fetch
the pinned source dependencies instead.
