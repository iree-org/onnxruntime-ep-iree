"""Test that intermediate file saving and cleanup works correctly."""

import glob
import os
import pathlib
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def _create_simple_model():
    """Create a simple Add model for testing."""
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[10, 20], [30, 40]], dtype=np.float32)

    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 2])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 2])
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["B"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=[2, 2],
            vals=b.flatten().tolist(),
        ),
    )
    add_node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])

    graph = helper.make_graph(
        [add_node, constant_node], "test_graph", [input_a], [output]
    )
    model = helper.make_model(
        graph, producer_name="iree_test", opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return model, a, b


def _get_iree_temp_files():
    """Return lists of IREE temp files (mlir, vmfb, irpa) in the temp directory."""
    temp_dir = tempfile.gettempdir()
    mlir_files = glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir"))
    vmfb_files = glob.glob(os.path.join(temp_dir, "iree_ep_*.vmfb"))
    irpa_files = glob.glob(os.path.join(temp_dir, "iree_ep_*.irpa"))
    return mlir_files, vmfb_files, irpa_files


def _cleanup_iree_temp_files():
    """Remove any leftover IREE temp files from previous runs."""
    mlir_files, vmfb_files, irpa_files = _get_iree_temp_files()
    for f in mlir_files + vmfb_files + irpa_files:
        try:
            os.remove(f)
        except OSError:
            pass


def _run_inference(model_path, device, save_intermediates):
    """Run inference with the given save_intermediates setting."""
    provider_options = {"target_arch": "host"}
    if save_intermediates:
        provider_options["save_intermediates"] = "1"

    opts = ort.SessionOptions()
    opts.add_provider_for_devices([device], provider_options)
    session = ort.InferenceSession(model_path, sess_options=opts)

    # Run inference to ensure compilation happens.
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    session.run(None, {"A": a})

    # Explicitly delete the session to trigger cleanup.
    del session


def _save_model(model):
    """Save ONNX model to a temp file. Returns path (caller must delete)."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        return f.name


def test_save_intermediates_enabled(iree_device):
    """Test that intermediate files are saved when save_intermediates=1."""
    _cleanup_iree_temp_files()
    mlir_before, vmfb_before, irpa_before = _get_iree_temp_files()

    model, _, _ = _create_simple_model()
    model_path = _save_model(model)

    try:
        _run_inference(model_path, iree_device, save_intermediates=True)

        mlir_after, vmfb_after, irpa_after = _get_iree_temp_files()

        new_mlir = set(mlir_after) - set(mlir_before)
        new_vmfb = set(vmfb_after) - set(vmfb_before)
        new_irpa = set(irpa_after) - set(irpa_before)

        assert new_mlir, "No MLIR file was saved"
        assert new_vmfb, "No VMFB file was saved"
        assert new_irpa, "No IRPA file was saved"

        # Verify files have content.
        for f in new_mlir:
            assert os.path.getsize(f) > 0, f"MLIR file is empty: {f}"
        for f in new_vmfb:
            assert os.path.getsize(f) > 0, f"VMFB file is empty: {f}"

        # Clean up the saved files.
        for f in list(new_mlir) + list(new_vmfb) + list(new_irpa):
            os.remove(f)
    finally:
        pathlib.Path(model_path).unlink()


def test_save_intermediates_disabled(iree_device):
    """Test that intermediate files are cleaned up when save_intermediates is not set."""
    _cleanup_iree_temp_files()
    mlir_before, vmfb_before, irpa_before = _get_iree_temp_files()

    model, _, _ = _create_simple_model()
    model_path = _save_model(model)

    try:
        _run_inference(model_path, iree_device, save_intermediates=False)

        mlir_after, vmfb_after, irpa_after = _get_iree_temp_files()

        new_mlir = set(mlir_after) - set(mlir_before)
        new_vmfb = set(vmfb_after) - set(vmfb_before)
        new_irpa = set(irpa_after) - set(irpa_before)

        # Clean up any leaked files before asserting so they don't affect other tests.
        leaked = list(new_mlir) + list(new_vmfb) + list(new_irpa)
        for f in leaked:
            try:
                os.remove(f)
            except OSError:
                pass

        assert not new_mlir, f"MLIR file was not cleaned up: {list(new_mlir)}"
        assert not new_vmfb, f"VMFB file was not cleaned up: {list(new_vmfb)}"
        assert not new_irpa, f"IRPA file was not cleaned up: {list(new_irpa)}"
    finally:
        pathlib.Path(model_path).unlink()
