"""Tests for in-place output emission (`ep.iree.enable_inplace_outputs`).

These tests lock in the contract for the session option:

  - unset:                     in-place ON  (the default)
  - explicit "1":              in-place ON
  - explicit "0":              in-place OFF (kill switch)

…and guard numerical correctness via a parity check so we never regress
to silently returning an unwritten output buffer (the failure mode if
the runtime-side device check ever drifts from the compiler emission).
"""

import pathlib
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from conftest import try_generate_mlir
from onnx import TensorProto, helper


def _make_static_add_model(shape=(64, 64)):
    """Return (model, A, B) for `C = A + B` with fully static shapes."""
    rng = np.random.default_rng(seed=42)
    a = rng.standard_normal(shape, dtype=np.float32)
    b = rng.standard_normal(shape, dtype=np.float32)

    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, list(shape))
    output_c = helper.make_tensor_value_info("C", TensorProto.FLOAT, list(shape))
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["B"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=list(shape),
            vals=b.flatten().tolist(),
        ),
    )
    add_node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph(
        [constant_node, add_node], "inplace_test_graph", [input_a], [output_c]
    )
    model = helper.make_model(
        graph,
        producer_name="iree_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    return model, a, b


def _has_inplace_pattern(mlir_text: str) -> bool:
    """All three markers must appear together for the pattern to be valid."""
    return (
        "!torch.tensor<" in mlir_text
        and "torch.overwrite.tensor.contents" in mlir_text
        and "torch.copy.to_vtensor" in mlir_text
    )


def test_inplace_outputs_default_on(iree_device):
    """Without setting the session option, the in-place pattern MUST be emitted."""
    model, _, _ = _make_static_add_model()
    mlir, err = try_generate_mlir(model, iree_device, "", "host")
    assert err is None, err
    assert _has_inplace_pattern(mlir), (
        "in-place pattern missing from MLIR while option was at its "
        "default (expected ON):\n" + mlir
    )


def test_inplace_outputs_explicit_disable(iree_device):
    """Setting the session option to "0" MUST suppress the in-place pattern."""
    model, _, _ = _make_static_add_model()
    mlir, err = try_generate_mlir(
        model,
        iree_device,
        "",
        "host",
        extra_provider_options={"enable_inplace_outputs": "0"},
    )
    assert err is None, err
    assert not _has_inplace_pattern(
        mlir
    ), "in-place pattern leaked into MLIR while option was explicitly disabled"


def test_inplace_outputs_numerical_parity(iree_device):
    """Both modes must produce bit-identical results for the same inputs."""
    model, a, b = _make_static_add_model()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        model_path = f.name

    def _run(enable_inplace_opt: str) -> np.ndarray:
        sess_opts = ort.SessionOptions()
        sess_opts.add_provider_for_devices(
            [iree_device],
            {
                "target_arch": "host",
                "enable_inplace_outputs": enable_inplace_opt,
            },
        )
        session = ort.InferenceSession(model_path, sess_options=sess_opts)
        [out] = session.run(None, {"A": a})
        return out

    try:
        out_off = _run("0")
        out_on = _run("1")
        np.testing.assert_array_equal(out_off, out_on)
        np.testing.assert_array_equal(out_on, a + b)
    finally:
        pathlib.Path(model_path).unlink(missing_ok=True)
