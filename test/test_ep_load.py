"""Test that the IREE ONNX Runtime EP loads and runs correctly."""

import pathlib
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def test_ep_load(iree_device):
    """Test that the IREE EP plugin loads and runs inference correctly."""
    # Create a simple Add model: C = A + D (constant).
    a = np.float32(np.random.rand(64, 64))
    b = np.float32(np.random.rand(64, 64))

    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, list(a.shape))
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, list(b.shape))
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["D"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=list(b.shape),
            vals=b.flatten().tolist(),
        ),
    )
    add_node = helper.make_node("Add", inputs=["A", "D"], outputs=["C"])

    graph = helper.make_graph(
        [add_node, constant_node], "test_graph", [input_a], [output]
    )
    model = helper.make_model(
        graph, producer_name="iree_test", opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        model_path = f.name

    try:
        opts = ort.SessionOptions()
        opts.add_provider_for_devices([iree_device], {"target_arch": "host"})
        session = ort.InferenceSession(model_path, sess_options=opts)

        expected = a + b
        outputs = session.run(None, {"A": a})
        result = outputs[0]

        assert (
            result.shape == expected.shape
        ), f"Shape mismatch: {result.shape} vs {expected.shape}"
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    finally:
        pathlib.Path(model_path).unlink()
