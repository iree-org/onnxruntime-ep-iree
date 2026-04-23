# ResNet-50 with IREE ONNX Runtime EP

Image classification using ResNet-50 via the IREE Execution Provider for ONNX Runtime.

## Setup

### 1. Download the model and labels

Download the ONNX model from the ONNX Model Zoo mirror on Hugging Face and the ImageNet labels file:

```bash
mkdir -p resnet50-assets

curl -L \
  https://huggingface.co/onnxmodelzoo/resnet50_Opset18_torch_hub/resolve/main/resnet50_Opset18_torch_hub.onnx \
  -o resnet50-assets/model.onnx

curl -L \
  https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json \
  -o resnet50-assets/imagenet-simple-labels.json
```

If you already have the checked-in `examples/model.onnx` and `examples/imagenet-simple-labels.json`, the script uses those paths by default.

### 2. Run

From the `models/resnet` directory:

```bash
cd models/resnet
```

For CPU execution:

```bash
python run.py \
  --image images/dog.jpg \
  --image images/plane.jpg \
  --driver local-task \
  --target none
```

To use separately downloaded assets instead of the checked-in `examples/` copies:

```bash
python run.py \
  --model resnet50-assets/model.onnx \
  --labels resnet50-assets/imagenet-simple-labels.json \
  --image images/dog.jpg \
  --driver local-task \
  --target none
```

For GPU execution, pass the appropriate driver and target architecture, for example:

```bash
python run.py \
  --image images/dog.jpg \
  --driver hip \
  --target gfx1201
```

Use `--top-k N` to control how many predictions are printed and `--verbose` for detailed ONNX Runtime logging.
