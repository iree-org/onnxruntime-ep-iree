# IREE ONNX Runtime Execution Provider

ONNX Runtime Execution Provider using IREE.

## Installation

1. Build the plugin

```bash
mkdir build && cd build
cmake .. -GNinja
ninja
```

2. Create a virtual environment and install dependencies

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

3. Install Python Package

```bash
ONNXRUNTIME_EP_IREE_BUILD_DIR=$(pwd)/build uv pip install -e python/
```

4. Run sample test

```bash
python test/test_ep_load.py
```
