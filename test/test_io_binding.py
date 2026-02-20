"""Test io_binding with device memory for the IREE ONNX Runtime EP."""

import pathlib
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper


@pytest.mark.gpu
def test_io_binding(iree_gpu_device, gpu_target):
    """Test io_binding with device-allocated tensors on a GPU."""
    # Test data.
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    b = np.array(
        [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32
    )

    # Create a simple Add model.
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 4])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3, 4])
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["D"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=[3, 4],
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
        model_path = f.name
        onnx.save(model, model_path)

    try:
        opts = ort.SessionOptions()
        opts.add_provider_for_devices([iree_gpu_device], {"target_arch": gpu_target})
        session = ort.InferenceSession(model_path, sess_options=opts)

        io_binding = session.io_binding()

        input_shape = list(a.shape)
        output_shape = list(a.shape)

        device_id = iree_gpu_device.device.device_id
        vendor_id = 0x1EEE  # IREE vendor ID

        # Allocate input tensor on IREE device.
        input_tensor = ort.OrtValue.ortvalue_from_shape_and_type(
            input_shape,
            np.float32,
            device_type="gpu",
            device_id=device_id,
            vendor_id=vendor_id,
        )
        input_tensor.update_inplace(a)
        io_binding.bind_ortvalue_input("A", input_tensor)

        # Allocate output tensor on IREE device.
        output_tensor = ort.OrtValue.ortvalue_from_shape_and_type(
            output_shape,
            np.float32,
            device_type="gpu",
            device_id=device_id,
            vendor_id=vendor_id,
        )
        io_binding.bind_ortvalue_output("C", output_tensor)

        session.run_with_iobinding(io_binding)

        result = output_tensor.numpy()
        expected = a + b

        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    finally:
        pathlib.Path(model_path).unlink()
