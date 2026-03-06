"""Test that zero-element tensors are rejected without corrupting device state.

Dispatching a zero-element buffer (shape with a 0 dimension) through IREE can
leave the HIP device queue in a broken state, hanging all subsequent dispatches.
The EP rejects zero-element inputs with a clear error instead.
"""

import pathlib
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper


def _make_mul_model(shape, factor=2.0):
    """Create a model that computes out = A * factor."""
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, list(shape))
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, list(shape))
    const = helper.make_tensor("factor", TensorProto.FLOAT, [1], [float(factor)])
    mul = helper.make_node("Mul", ["A", "factor"], ["out"])
    graph = helper.make_graph([mul], "mul", [input_a], [output], [const])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _save_model(model):
    """Save model to a temp file, return path."""
    f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, f.name)
    f.close()
    return f.name


@pytest.mark.gpu
def test_zero_element_gpu(iree_gpu_device, gpu_target):
    """Zero-element rejection on GPU — requires --gpu flag.

    Verifies that zero-element inference raises an error and that
    subsequent normal inference still works (device not corrupted).
    """
    provider_options = {"target_arch": gpu_target}
    model_paths = []

    try:
        # Normal inference works.
        path1 = _save_model(_make_mul_model([1024], 2.0))
        model_paths.append(path1)
        opts = ort.SessionOptions()
        opts.log_severity_level = 2
        opts.add_provider_for_devices([iree_gpu_device], provider_options)
        session = ort.InferenceSession(path1, sess_options=opts)

        a = np.ones(1024, dtype=np.float32)
        out = session.run(None, {"A": a})[0]
        np.testing.assert_allclose(out, 2.0 * a)

        # Zero-element inference raises an error.
        path_zero = _save_model(_make_mul_model([0], 2.0))
        model_paths.append(path_zero)
        opts_zero = ort.SessionOptions()
        opts_zero.log_severity_level = 2
        opts_zero.add_provider_for_devices([iree_gpu_device], provider_options)
        session_zero = ort.InferenceSession(path_zero, sess_options=opts_zero)

        with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument):
            session_zero.run(None, {"A": np.array([], dtype=np.float32)})

        # Normal inference still works after the error (device not corrupted).
        path2 = _save_model(_make_mul_model([1024], 3.0))
        model_paths.append(path2)
        opts2 = ort.SessionOptions()
        opts2.log_severity_level = 2
        opts2.add_provider_for_devices([iree_gpu_device], provider_options)
        session2 = ort.InferenceSession(path2, sess_options=opts2)

        out2 = session2.run(None, {"A": a})[0]
        np.testing.assert_allclose(out2, 3.0 * a)
    finally:
        for p in model_paths:
            pathlib.Path(p).unlink(missing_ok=True)
