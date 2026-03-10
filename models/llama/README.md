# Llama 3.1 with IREE ONNX Runtime EP

Text generation using Llama 3.1 8B via the IREE Execution Provider for ONNX Runtime.

## Setup

### 1. Export the model to ONNX

Download the Llama 3.1 8B Instruct safetensors from Hugging Face and export to ONNX using the `onnxruntime-genai` model builder:

```bash
pip install onnxruntime-genai

python -m onnxruntime_genai.models.builder \
    -m meta-llama/Llama-3.1-8B-Instruct \
    -o llama3.1-onnx \
    -p fp32 \
    -e cpu
```

This produces a `llama3.1-onnx/` directory containing `model.onnx` and tokenizer files.

### 2. Run

```bash
python models/llama/run.py \
    --model llama3.1-onnx/model.onnx \
    --tokenizer llama3.1-onnx \
    --target gfx1100 \
    --driver hip
```

Use `--verbose` for detailed logging and `--max-tokens N` to control generation length (default: 20).
