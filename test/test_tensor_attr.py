"""Test tensor attribute support in MLIR generation.

ONNX ops like ConstantOfShape have tensor-valued attributes (e.g., the fill
value). These must be properly serialized as dense<> : tensor<> in MLIR.
Without proper handling, the EP emits "NYI" which causes compilation failure.

All tests use dynamic shapes to prevent ORT from constant-folding the
ConstantOfShape away before it reaches the EP's MLIR generation.
"""

import os
import struct
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from conftest import try_generate_mlir
from onnx import TensorProto, helper


def _make_dynamic_constant_of_shape_model(fill_value=0.0):
    """Build a model with dynamic shapes so ConstantOfShape survives ORT folding.

    Graph: Y = ConstantOfShape(Shape(X), value=fill_value) + X
    With X having dynamic shape, ORT cannot constant-fold ConstantOfShape.
    """
    shape_node = helper.make_node("Shape", inputs=["X"], outputs=["shape"])
    value_tensor = helper.make_tensor(
        "value", TensorProto.FLOAT, [1], [float(fill_value)]
    )
    cos_node = helper.make_node(
        "ConstantOfShape",
        inputs=["shape"],
        outputs=["filled"],
        value=value_tensor,
    )
    add_node = helper.make_node("Add", inputs=["filled", "X"], outputs=["Y"])

    # Dynamic batch dimension prevents ORT from folding ConstantOfShape.
    input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 4])
    output_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 4])

    graph = helper.make_graph(
        [shape_node, cos_node, add_node],
        "test_dynamic_cos",
        [input_info],
        [output_info],
    )
    model = helper.make_model(
        graph,
        producer_name="iree_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    return model


def _make_neg_inf_mask_model():
    """Build a model using ConstantOfShape with -inf fill value.

    This is the pattern used in attention masking: create a tensor filled with
    -inf, then use Where to select between -inf and 0 based on a condition.

    Graph: mask = Where(condition, 0.0, ConstantOfShape(-inf, shape_of(X)))
    """
    # Shape of input
    shape_node = helper.make_node("Shape", inputs=["X"], outputs=["shape"])

    # ConstantOfShape filled with -inf
    neg_inf = float("-inf")
    neg_inf_bytes = struct.pack("<f", neg_inf)
    value_tensor = helper.make_tensor(
        "value",
        TensorProto.FLOAT,
        [1],
        vals=neg_inf_bytes,
        raw=True,
    )
    cos_node = helper.make_node(
        "ConstantOfShape",
        inputs=["shape"],
        outputs=["neg_inf_tensor"],
        value=value_tensor,
    )

    # Zeros tensor (same shape)
    zero_tensor = helper.make_tensor("zero_val", TensorProto.FLOAT, [1], [0.0])
    cos_zero = helper.make_node(
        "ConstantOfShape",
        inputs=["shape"],
        outputs=["zero_tensor"],
        value=zero_tensor,
    )

    # condition: X > 0
    zero_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["zero_scalar"],
        value=helper.make_tensor("z", TensorProto.FLOAT, [], [0.0]),
    )
    greater_node = helper.make_node(
        "Greater", inputs=["X", "zero_scalar"], outputs=["condition"]
    )

    # Where(condition, zero_tensor, neg_inf_tensor)
    where_node = helper.make_node(
        "Where",
        inputs=["condition", "zero_tensor", "neg_inf_tensor"],
        outputs=["Y"],
    )

    # Dynamic first dimension prevents ORT from constant-folding ConstantOfShape.
    input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 3])
    output_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 3])

    graph = helper.make_graph(
        [shape_node, cos_node, cos_zero, zero_const, greater_node, where_node],
        "test_neg_inf_mask",
        [input_info],
        [output_info],
    )
    model = helper.make_model(
        graph,
        producer_name="iree_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    return model


def _run_model(model, inputs, iree_device):
    """Run model on both CPU and IREE, return (cpu_result, iree_result)."""
    model_dir = tempfile.mkdtemp()
    model_path = os.path.join(model_dir, "model.onnx")
    onnx.save(model, model_path)

    cpu_sess = None
    iree_sess = None
    try:
        # CPU reference
        cpu_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        cpu_result = cpu_sess.run(None, inputs)[0]

        # IREE EP
        opts = ort.SessionOptions()
        opts.add_provider_for_devices([iree_device], {"target_arch": "host"})
        iree_sess = ort.InferenceSession(model_path, sess_options=opts)
        iree_result = iree_sess.run(None, inputs)[0]

        return cpu_result, iree_result
    finally:
        del cpu_sess, iree_sess
        for f in os.listdir(model_dir):
            os.remove(os.path.join(model_dir, f))
        os.rmdir(model_dir)


def test_constant_of_shape_zero_fill(iree_device):
    """ConstantOfShape with value=0.0 and dynamic shapes."""
    model = _make_dynamic_constant_of_shape_model(0.0)
    x = np.random.rand(3, 4).astype(np.float32)
    cpu, iree = _run_model(model, {"X": x}, iree_device)
    # ConstantOfShape(0.0) + X = X
    np.testing.assert_allclose(iree, cpu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(iree, x, rtol=1e-5, atol=1e-5)


def test_constant_of_shape_nonzero_fill(iree_device):
    """ConstantOfShape with value=3.14 and dynamic shapes."""
    model = _make_dynamic_constant_of_shape_model(3.14)
    x = np.random.rand(3, 4).astype(np.float32)
    cpu, iree = _run_model(model, {"X": x}, iree_device)
    expected = np.full_like(x, 3.14) + x
    np.testing.assert_allclose(iree, cpu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(iree, expected, rtol=1e-4, atol=1e-4)


def test_constant_of_shape_neg_inf(iree_device):
    """ConstantOfShape with value=-inf (used in attention masks).

    This specifically tests that special float values are hex-encoded in MLIR,
    since MLIR's dense<> parser doesn't accept -inf as a literal.
    """
    model = _make_neg_inf_mask_model()
    # Mix of positive and negative values
    x = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    cpu, iree = _run_model(model, {"X": x}, iree_device)

    # Where X > 0: 0.0, else: -inf
    expected = np.where(x > 0, 0.0, float("-inf")).astype(np.float32)
    np.testing.assert_array_equal(iree, cpu)
    np.testing.assert_array_equal(iree, expected)


def test_constant_of_shape_mlir_content(iree_device):
    """Verify the generated MLIR contains proper tensor attributes.

    Uses dynamic shapes so ORT can't constant-fold the ConstantOfShape away.
    Checks that ConstantOfShape has a dense<> tensor attribute in the MLIR.
    """
    model = _make_dynamic_constant_of_shape_model(0.0)
    mlir_content, err = try_generate_mlir(model, iree_device, "", "host")
    assert err is None, f"MLIR generation failed: {err}"

    assert (
        "onnx.ConstantOfShape" in mlir_content
    ), "MLIR should contain onnx.ConstantOfShape op"
    assert (
        "torch.onnx.value = dense<" in mlir_content
    ), "ConstantOfShape should have dense<> tensor attribute"


def _make_constant_of_shape_cast_model(onnx_dtype, fill_value):
    """Build a model: Shape(X) -> ConstantOfShape(fill) -> Cast(float32) -> Y.

    Casting to float32 allows uniform comparison across all dtypes.
    Dynamic batch dimension prevents ORT from constant-folding ConstantOfShape.
    """
    value_tensor = helper.make_tensor("value", onnx_dtype, [1], [fill_value])
    shape_node = helper.make_node("Shape", inputs=["X"], outputs=["shape"])
    cos_node = helper.make_node(
        "ConstantOfShape",
        inputs=["shape"],
        outputs=["filled"],
        value=value_tensor,
    )
    cast_node = helper.make_node(
        "Cast", inputs=["filled"], outputs=["Y"], to=TensorProto.FLOAT
    )

    input_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 4])
    output_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 4])

    graph = helper.make_graph(
        [shape_node, cos_node, cast_node],
        "test_cos_cast",
        [input_info],
        [output_info],
    )
    model = helper.make_model(
        graph,
        producer_name="iree_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    return model


@pytest.mark.parametrize(
    "onnx_dtype,fill_value,expected_fill",
    [
        (TensorProto.FLOAT, 1.5, 1.5),
        (TensorProto.DOUBLE, 1.5, 1.5),
        (TensorProto.INT8, -42, -42.0),
        (TensorProto.INT16, 1000, 1000.0),
        (TensorProto.INT32, 100000, 100000.0),
        (TensorProto.INT64, 1000000, 1000000.0),
        (TensorProto.UINT8, 200, 200.0),
        # UINT16/UINT32/UINT64 are not supported by the IREE llvm-cpu backend
        # for ConstantOfShape lowering, so they are excluded here.
        (TensorProto.BOOL, 1, 1.0),
    ],
)
def test_constant_of_shape_dtype(iree_device, onnx_dtype, fill_value, expected_fill):
    """ConstantOfShape with various dtypes, cast to float32 for comparison.

    Verifies that dense<> literals for all supported types are parseable by
    iree-compile and produce correct values at runtime.
    """
    model = _make_constant_of_shape_cast_model(onnx_dtype, fill_value)
    x = np.ones((3, 4), dtype=np.float32)
    cpu, iree = _run_model(model, {"X": x}, iree_device)
    expected = np.full((3, 4), expected_fill, dtype=np.float32)
    np.testing.assert_array_equal(iree, cpu)
    np.testing.assert_array_equal(iree, expected)
