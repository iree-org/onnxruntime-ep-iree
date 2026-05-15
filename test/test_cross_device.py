"""Tests for the cross IREE-device guard"""

import pathlib
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper


def _find_iree_device(driver):
    for d in ort.get_ep_devices():
        if d.ep_name == "IREE" and d.device.metadata.get("iree.driver") == driver:
            return d
    return None


@pytest.fixture(scope="module")
def cpu_device_pair(register_ep):
    """The two CPU-only IREE devices we need; skip if either is absent."""
    a = _find_iree_device("local-task")
    b = _find_iree_device("local-sync")
    if a is None or b is None:
        pytest.skip(
            "Cross IREE-device tests require both local-task and "
            "local-sync IREE EP devices."
        )
    return a, b


def _make_static_add_model():
    """`C = A + B` on shape (8, 8); B is a baked-in constant."""
    shape = [8, 8]
    rng = np.random.default_rng(seed=7)
    a = rng.standard_normal(shape, dtype=np.float32)
    b = rng.standard_normal(shape, dtype=np.float32)

    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, shape)
    output_c = helper.make_tensor_value_info("C", TensorProto.FLOAT, shape)
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["B"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=shape,
            vals=b.flatten().tolist(),
        ),
    )
    add_node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph(
        [constant_node, add_node],
        "cross_device_test_graph",
        [input_a],
        [output_c],
    )
    model = helper.make_model(
        graph,
        producer_name="iree_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    return model, a, b


def _alloc_on_iree_device(shape, device):
    return ort.OrtValue.ortvalue_from_shape_and_type(
        list(shape),
        np.float32,
        device_type="gpu",
        device_id=device.device.device_id,
        vendor_id=device.device.vendor_id,
    )


@pytest.mark.parametrize("bind_side", ["input", "output"])
def test_cross_device_binding(cpu_device_pair, bind_side):
    """An IO-bound tensor on a different IREE device than the session's
    must either round-trip correctly through OrtDataTransfer or raise an
    error that explicitly names the cross IREE-device case."""
    session_device, foreign_device = cpu_device_pair
    model, a, b = _make_static_add_model()

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        model_path = f.name

    try:
        sess_opts = ort.SessionOptions()
        sess_opts.add_provider_for_devices([session_device], {"target_arch": "host"})
        session = ort.InferenceSession(model_path, sess_options=sess_opts)

        io_binding = session.io_binding()
        if bind_side == "input":
            foreign_input = _alloc_on_iree_device(a.shape, foreign_device)
            foreign_input.update_inplace(a)
            io_binding.bind_ortvalue_input("A", foreign_input)
            io_binding.bind_output("C")
        else:
            foreign_output = _alloc_on_iree_device(a.shape, foreign_device)
            io_binding.bind_cpu_input("A", a)
            io_binding.bind_ortvalue_output("C", foreign_output)

        try:
            session.run_with_iobinding(io_binding)
        except Exception as e:
            assert "different IREE device" in str(e) or "device_id" in str(
                e
            ), f"Expected a cross-device error message, got: {e!r}"
            return

        if bind_side == "input":
            [out] = io_binding.copy_outputs_to_cpu()
        else:
            out = foreign_output.numpy()
        np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-5)
    finally:
        pathlib.Path(model_path).unlink(missing_ok=True)
