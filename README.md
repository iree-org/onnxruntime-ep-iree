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

Set `ONNXRUNTIME_EP_IREE_ENABLE_TRACING=ON` to also build and package the Tracy
variant alongside the default one. Select the runtime in Python with
`ONNXRUNTIME_EP_IREE_PY_RUNTIME=default` or `tracy`.

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

## Runtime Provider Options

Python users normally pass IREE EP runtime options through
`SessionOptions.add_provider_for_devices`:

```python
sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices(
    [iree_device],
    {
        "target_arch": "gfx1100",
        "opt_level": "O3",
        "dim_specs": "batch(1,1), seq(1,131072,16)",
    },
)
session = ort.InferenceSession(model_path, sess_options=sess_options)
```

The Python provider option names are stored internally as ONNX Runtime session
configuration entries with an `ep.iree.` prefix. If setting session config
entries directly, use the full `ep.iree.*` names.

| Python provider option | Session config entry | Default | Description |
| --- | --- | --- | --- |
| `target_arch` | `ep.iree.target_arch` | Empty | Target architecture for IREE compilation. Required for `hip`, `cuda`, and `vulkan`; CPU backends use the host target. Examples: `gfx1100`, `sm_90`. |
| `opt_level` | `ep.iree.opt_level` | `O0` | IREE compiler optimization level, such as `O0`, `O1`, `O2`, or `O3`. |
| `save_intermediates` | `ep.iree.save_intermediates` | `0` | Set to `1` to keep generated MLIR, VMFB, and IRPA files for debugging. The EP logs the saved paths. |
| `enable_ep_context_cache` | `ep.iree.enable_ep_context_cache` | `0` | Set to `1` to keep compiled VMFB and IRPA artifacts for ONNX Runtime EP context caching when `save_intermediates` is not set. |
| `extern_kernel_path` | `ep.iree.extern_kernel_path` | Empty | Directory containing precompiled extern dispatch kernel objects, such as `.co` files. |
| `dim_specs` | `ep.iree.dim_specs` | Empty | Dimension specialization constraints. See the syntax below. |
| `compiler_lib_path` | `ep.iree.compiler_lib_path` | Empty | Absolute path to the IREE compiler shared library. This is process-global: the first successful compiler initialization wins. If unset, the EP auto-discovers the compiler library. |
| `enable_inplace_outputs` | `ep.iree.enable_inplace_outputs` | `1` | Set to `0` to disable in-place output emission for static-shape outputs. When enabled (the default), the generated MLIR uses the canonical Torch in-place pattern (`!torch.tensor` storage args + `torch.overwrite.tensor.contents` + `torch.copy.to_vtensor`) and the runtime pre-binds ORT-allocated output buffers at invoke time, eliminating one `stream.resource.alloca` and one post-execution copy per tied output. |

### Dimension Specialization (`dim_specs`)

`dim_specs` constrains symbolic dimensions and can describe one or more
specialized variants. Each spec has the form `name(min,max)` for an inclusive
range or `name(min,max,div)` for a range plus a divisibility constraint. Static
dimensions use the same value for `min` and `max`, such as `batch(1,1)`.
Commas separate specs within one variant, and semicolons separate variants.
Variants are checked in user-provided order at runtime, so the first matching
variant wins.

Examples:

```text
batch(1,1), seq(64,64)
seq(1,131072,16)
batch(1,1), seq(64,64); seq(1,131072,16)
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
  and configures the package-owned build tree under `build/cmake/default`, and
  also `build/cmake/tracy` unless `--no-tracing` is passed.
- On later runs, `dev_me.py` performs incremental rebuilds of every configured
  build directory under `build/cmake/`.
- Delete `build/` to start over with a fresh configuration.
- Select the Tracy-enabled editable runtime with
  `ONNXRUNTIME_EP_IREE_PY_RUNTIME=tracy`.

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
