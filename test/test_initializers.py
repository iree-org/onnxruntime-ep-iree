"""Test initializer handling: small inline, large IRPA parameter, and external."""

import glob
import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnx.external_data_helper import set_external_data
from onnx.numpy_helper import from_array

# Fixed seed for reproducibility.
np.random.seed(42)

# Test data. Four initializers, each handled differently:
#   D_small:     [1, 64] float32 = 256 bytes   -> inline dense<>
#   D_large:     [64, 64] float32 = 16384 bytes -> IRPA parameter
#   D_ext:       [64, 64] float32 = 16384 bytes -> external file (parameter, not in IRPA)
#   D_ext_small: [1, 64] float32 = 256 bytes    -> external file (inlined as dense<>)
# Graph: C = (((A + D_small) + D_large) + D_ext) + D_ext_small
SHAPE = [64, 64]
A_DATA = np.random.rand(*SHAPE).astype(np.float32)
B_SMALL = np.random.rand(1, 64).astype(np.float32)
B_LARGE = np.random.rand(*SHAPE).astype(np.float32)
B_EXT = np.random.rand(*SHAPE).astype(np.float32)
B_EXT_SMALL = np.random.rand(1, 64).astype(np.float32)
EXPECTED = (((A_DATA + B_SMALL) + B_LARGE) + B_EXT) + B_EXT_SMALL


def _create_model():
    """Create the test model and return (model_path, model_dir).

    Caller must clean up model_dir when done.
    """
    const_small = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["D_small"],
        value=helper.make_tensor(
            name="small_const",
            data_type=TensorProto.FLOAT,
            dims=[1, 64],
            vals=B_SMALL.flatten().tolist(),
        ),
    )
    const_large = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["D_large"],
        value=helper.make_tensor(
            name="large_const",
            data_type=TensorProto.FLOAT,
            dims=SHAPE,
            vals=B_LARGE.flatten().tolist(),
        ),
    )

    # D_ext and D_ext_small are graph initializers backed by external .bin files.
    model_dir = tempfile.mkdtemp()

    # Large external initializer.
    ext_data_filename = "ext_weights.bin"
    ext_data_path = os.path.join(model_dir, ext_data_filename)
    ext_tensor = from_array(B_EXT, name="D_ext")
    raw_data = ext_tensor.raw_data
    with open(ext_data_path, "wb") as f:
        f.write(raw_data)
    set_external_data(ext_tensor, location=ext_data_filename, length=len(raw_data))
    ext_tensor.ClearField("raw_data")
    ext_tensor.data_location = TensorProto.EXTERNAL

    # Small external initializer (should be inlined as dense<>).
    ext_small_filename = "ext_small_weights.bin"
    ext_small_path = os.path.join(model_dir, ext_small_filename)
    ext_small_tensor = from_array(B_EXT_SMALL, name="D_ext_small")
    raw_data_small = ext_small_tensor.raw_data
    with open(ext_small_path, "wb") as f:
        f.write(raw_data_small)
    set_external_data(
        ext_small_tensor, location=ext_small_filename, length=len(raw_data_small)
    )
    ext_small_tensor.ClearField("raw_data")
    ext_small_tensor.data_location = TensorProto.EXTERNAL

    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, SHAPE)
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, SHAPE)

    add1 = helper.make_node("Add", inputs=["A", "D_small"], outputs=["T1"])
    add2 = helper.make_node("Add", inputs=["T1", "D_large"], outputs=["T2"])
    add3 = helper.make_node("Add", inputs=["T2", "D_ext"], outputs=["T3"])
    add4 = helper.make_node("Add", inputs=["T3", "D_ext_small"], outputs=["C"])

    graph = helper.make_graph(
        [add1, add2, add3, add4, const_small, const_large],
        "test_graph",
        [input_a],
        [output],
        initializer=[ext_tensor, ext_small_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="iree_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    model_path = os.path.join(model_dir, "model.onnx")
    onnx.save(model, model_path)
    return model_path, model_dir


def _cleanup_model_dir(model_dir):
    for f in os.listdir(model_dir):
        os.remove(os.path.join(model_dir, f))
    os.rmdir(model_dir)


def _get_iree_files():
    """Return current sets of IREE temp MLIR and IRPA files."""
    temp_dir = tempfile.gettempdir()
    mlir = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir")))
    irpa = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.irpa")))
    return mlir, irpa


def _cleanup_iree_files(new_mlir, new_irpa):
    """Remove IREE temp files (MLIR, IRPA, VMFB)."""
    for f in new_mlir | new_irpa:
        try:
            os.remove(f)
        except OSError:
            pass
    temp_dir = tempfile.gettempdir()
    for f in glob.glob(os.path.join(temp_dir, "iree_ep_*.vmfb")):
        try:
            os.remove(f)
        except OSError:
            pass


def test_with_save_intermediates(iree_device):
    """Run with save_intermediates=1 and validate MLIR, IRPA, and inference."""
    model_path, model_dir = _create_model()
    mlir_before, irpa_before = _get_iree_files()

    try:
        opts = ort.SessionOptions()
        opts.add_provider_for_devices(
            [iree_device], {"target_arch": "host", "save_intermediates": "1"}
        )
        session = ort.InferenceSession(model_path, sess_options=opts)
        result = session.run(None, {"A": A_DATA})[0]

        np.testing.assert_allclose(result, EXPECTED, rtol=1e-5, atol=1e-5)

        # Validate generated MLIR.
        mlir_after, irpa_after = _get_iree_files()
        new_mlir = mlir_after - mlir_before
        new_irpa = irpa_after - irpa_before

        assert new_mlir, "No MLIR file saved"

        mlir_content = open(list(new_mlir)[0]).read()

        # D_small and D_ext_small should be inlined via dense<>.
        assert 'dense<"0x' in mlir_content, (
            "MLIR should contain inline dense<> attributes"
        )
        assert "dense_resource" not in mlir_content, (
            "MLIR should not contain dense_resource (replaced by dense<>)"
        )
        assert "dialect_resources" not in mlir_content, (
            "MLIR should not contain dialect_resources section"
        )

        # D_large and D_ext should use flow.parameter.named.
        assert 'flow.parameter.named<"model"::"D_large">' in mlir_content, (
            "MLIR should contain flow.parameter.named for D_large"
        )
        assert 'flow.parameter.named<"model"::"D_ext">' in mlir_content, (
            "MLIR should contain flow.parameter.named for D_ext"
        )
        # D_ext_small should NOT be a parameter.
        assert 'flow.parameter.named<"model"::"D_ext_small">' not in mlir_content, (
            "D_ext_small should be inlined, not a parameter"
        )

        # IRPA should contain only D_large's data (16384 bytes + header),
        # not D_ext's. If D_ext were copied it would be >32000 bytes.
        assert new_irpa, "No IRPA file was created"
        irpa_size = os.path.getsize(list(new_irpa)[0])
        assert irpa_size > 0, "IRPA should contain D_large data"
        assert irpa_size <= 20000, (
            f"IRPA too large ({irpa_size} bytes), external data may have been copied"
        )

        _cleanup_iree_files(new_mlir, new_irpa)
    finally:
        # Release the ORT session before cleanup. On Windows, the session
        # holds memory-mapped handles to external weight files which
        # prevents their deletion.
        del session
        _cleanup_model_dir(model_dir)


def test_without_save_intermediates(iree_device):
    """Run without save_intermediates and validate inference."""
    model_path, model_dir = _create_model()

    try:
        opts = ort.SessionOptions()
        opts.add_provider_for_devices([iree_device], {"target_arch": "host"})
        session = ort.InferenceSession(model_path, sess_options=opts)
        result = session.run(None, {"A": A_DATA})[0]

        np.testing.assert_allclose(result, EXPECTED, rtol=1e-5, atol=1e-5)
    finally:
        # Release the ORT session before cleanup. On Windows, the session
        # holds memory-mapped handles to external weight files which
        # prevents their deletion.
        del session
        _cleanup_model_dir(model_dir)
