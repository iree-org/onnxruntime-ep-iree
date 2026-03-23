"""Test dim specialization: static, divisibility, multi-variant, and error handling."""

import re

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from conftest import try_generate_mlir
from onnx import TensorProto, helper

_rng = np.random.default_rng(42)

# ORT-specific exceptions (avoids bare `except Exception`).
# ORT has many flat exception types with no common base class.
_ort_state = ort.capi.onnxruntime_pybind11_state
OrtError = tuple(
    getattr(_ort_state, name)
    for name in dir(_ort_state)
    if isinstance(getattr(_ort_state, name), type)
    and issubclass(getattr(_ort_state, name), Exception)
)


def _save_model(model, tmp_path):
    """Save ONNX model into tmp_path. Pytest cleans up the directory."""
    path = str(tmp_path / "model.onnx")
    onnx.save(model, path)
    return path


def _make_matmul_model(batch_dim="batch", seq_dim="seq", k=16, n=16):
    """Create a MatMul ONNX model object (unsaved).

    input shape:  [batch, seq, K]  (batch and seq are symbolic/dynamic)
    weight shape: [K, N]           (static initializer)
    output shape: [batch, seq, N]

    Returns (model, weight_data).
    """
    weight_data = _rng.random((k, n)).astype(np.float32)

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [batch_dim, seq_dim, k]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_dim, seq_dim, n]
    )

    weight_init = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=[k, n],
        vals=weight_data.flatten().tolist(),
    )
    const_node = helper.make_node(
        "Constant", inputs=[], outputs=["W"], value=weight_init
    )
    matmul_node = helper.make_node("MatMul", inputs=["input", "W"], outputs=["output"])

    graph = helper.make_graph(
        [matmul_node, const_node], "test_graph", [input_info], [output_info]
    )
    model = helper.make_model(
        graph,
        producer_name="iree_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    return model, weight_data


def _create_matmul_model(tmp_path, **kwargs):
    """Create and save a MatMul model. Returns (model_path, weight_data)."""
    model, weight_data = _make_matmul_model(**kwargs)
    model_path = _save_model(model, tmp_path)
    return model_path, weight_data


def _make_add_model(batch_dim="batch", feat_a=10, feat_b=20):
    """Create an Add model with two inputs sharing a batch dim (unsaved).

    A shape: [batch, feat_a]
    B shape: [batch, feat_b]
    Uses MatMul(A, W) + B where W: [feat_a, feat_b].
    output shape: [batch, feat_b]

    Returns (model, weight_data).
    """
    weight_data = _rng.random((feat_a, feat_b)).astype(np.float32)

    a_info = helper.make_tensor_value_info("A", TensorProto.FLOAT, [batch_dim, feat_a])
    b_info = helper.make_tensor_value_info("B", TensorProto.FLOAT, [batch_dim, feat_b])
    out_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_dim, feat_b]
    )

    weight_init = helper.make_tensor(
        name="W",
        data_type=TensorProto.FLOAT,
        dims=[feat_a, feat_b],
        vals=weight_data.flatten().tolist(),
    )
    const_node = helper.make_node(
        "Constant", inputs=[], outputs=["weight"], value=weight_init
    )
    matmul_node = helper.make_node("MatMul", inputs=["A", "weight"], outputs=["AW"])
    add_node = helper.make_node("Add", inputs=["AW", "B"], outputs=["output"])

    graph = helper.make_graph(
        [const_node, matmul_node, add_node],
        "test_add_graph",
        [a_info, b_info],
        [out_info],
    )
    model = helper.make_model(
        graph,
        producer_name="iree_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    return model, weight_data


def _create_add_model(tmp_path, **kwargs):
    """Create and save an Add model. Returns (model_path, weight_data)."""
    model, weight_data = _make_add_model(**kwargs)
    model_path = _save_model(model, tmp_path)
    return model_path, weight_data


def _create_session(model_path, device, provider_options=None):
    """Create an ORT InferenceSession with the given IREE device."""
    opts = ort.SessionOptions()
    opts.add_provider_for_devices([device], provider_options or {})
    return ort.InferenceSession(model_path, sess_options=opts)


def _run_matmul(session, batch, m, weight_data):
    """Run inference and compare against numpy reference.

    Input shape: [batch, M, K] where K = weight_data.shape[0].
    """
    k = weight_data.shape[0]
    x = _rng.random((batch, m, k)).astype(np.float32)
    expected = x @ weight_data
    result = session.run(None, {"input": x})[0]
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


# ============================================================================
# E2E Tests
# ============================================================================


def test_static_specialization(iree_device, tmp_path):
    """Static dim_specs: matching and non-matching shapes produce correct results."""
    model_path, weight_data = _create_matmul_model(tmp_path)
    dim_specs = "batch(1,1), seq(64,64)"
    session = _create_session(
        model_path, iree_device, {"target_arch": "host", "dim_specs": dim_specs}
    )
    # Matching shape.
    _run_matmul(session, 1, 64, weight_data)
    # Non-matching shape must still work via generic fallback.
    _run_matmul(session, 2, 8, weight_data)


def test_divisibility_specialization(iree_device, tmp_path):
    """Divisibility dim_specs: seq(1,131072,16) works with seq=32."""
    model_path, weight_data = _create_matmul_model(tmp_path)
    dim_specs = "seq(1,131072,16)"
    session = _create_session(
        model_path, iree_device, {"target_arch": "host", "dim_specs": dim_specs}
    )
    # m=32 is divisible by 16.
    _run_matmul(session, 2, 32, weight_data)


def test_multi_variant_dispatch(iree_device, tmp_path):
    """Multi-variant compilation produces correct results for diverse input shapes.

    Compiles 3 variants (2 specialized + generic fallback) into one VMFB and
    verifies inference for shapes matching each constraint pattern.
    Dispatch checks variants in dim_specs order (first match wins).
    """
    model_path, weight_data = _create_matmul_model(tmp_path)
    dim_specs = "batch(1,1), seq(64,64); seq(1,131072,16)"
    session = _create_session(
        model_path, iree_device, {"target_arch": "host", "dim_specs": dim_specs}
    )
    # Shape matching the static variant.
    _run_matmul(session, 1, 64, weight_data)
    # Shape matching the divisibility variant.
    _run_matmul(session, 2, 32, weight_data)
    # Shape matching no specialized variant (generic fallback).
    _run_matmul(session, 3, 17, weight_data)


def test_partially_constrained_dims(iree_device, tmp_path):
    """dim_specs constrains only seq; batch remains fully dynamic."""
    model_path, weight_data = _create_matmul_model(tmp_path)
    dim_specs = "seq(1,131072,16)"
    session = _create_session(
        model_path, iree_device, {"target_arch": "host", "dim_specs": dim_specs}
    )
    # Varying batch with m divisible by 16.
    _run_matmul(session, 1, 32, weight_data)
    _run_matmul(session, 4, 64, weight_data)
    _run_matmul(session, 7, 16, weight_data)


def test_zero_size_dim_with_divisibility(iree_device, tmp_path):
    """seq(1,131072,16) with actual seq=0 should fall to generic."""
    model_path, weight_data = _create_matmul_model(tmp_path)
    dim_specs = "seq(1,131072,16)"
    session = _create_session(
        model_path, iree_device, {"target_arch": "host", "dim_specs": dim_specs}
    )
    try:
        k = weight_data.shape[0]
        x = np.zeros((1, 0, k), dtype=np.float32)
        expected = x @ weight_data
        result = session.run(None, {"input": x})[0]
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
    except OrtError as e:
        pytest.skip(f"Backend does not support zero-size tensors: {e}")


@pytest.mark.parametrize(
    "dim_specs",
    [
        pytest.param(""),
        pytest.param("   "),
    ],
)
def test_empty_dim_specs(dim_specs, iree_device, tmp_path):
    """Empty or whitespace dim_specs must behave like no specialization."""
    model_path, weight_data = _create_matmul_model(tmp_path)
    opts = {"target_arch": "host"}
    if dim_specs:
        opts["dim_specs"] = dim_specs
    session = _create_session(model_path, iree_device, opts)
    _run_matmul(session, 2, 8, weight_data)


def test_whitespace_and_divisibility_by_one(iree_device, tmp_path):
    """dim_specs with extra whitespace; divisibility by 1 (always matches)."""
    model_path, weight_data = _create_matmul_model(tmp_path)
    # Whitespace around delimiters.
    dim_specs = "  seq( 1 , 131072 , 1 )  "
    session = _create_session(
        model_path, iree_device, {"target_arch": "host", "dim_specs": dim_specs}
    )
    # div=1 matches any positive seq value.
    _run_matmul(session, 2, 7, weight_data)


# ============================================================================
# Error Tests
# ============================================================================


@pytest.mark.parametrize(
    "dim_specs",
    [
        pytest.param("batch"),
        pytest.param("batch=1"),
        pytest.param("(1,1)"),
        pytest.param("batch(1)"),
        pytest.param("batch(1,2,3,4)"),
        pytest.param("batch(abc,1)"),
        pytest.param("batch(1,abc)"),
        pytest.param("batch(1,2,abc)"),
        pytest.param("batch(0,1)"),
        pytest.param("batch(-1,1)"),
        pytest.param("batch(5,3)"),
        pytest.param("batch(1,2,0)"),
        pytest.param("batch(1,2,-1)"),
        pytest.param("batch(1,1"),
        pytest.param("batch(1,1)typo"),
        pytest.param("batch(1,1), batch(2,2)"),
        pytest.param("batch(,1)"),
        pytest.param("batch(1,)"),
        pytest.param("batch(1,1); seq(1,2,0)"),
    ],
)
def test_parse_error(dim_specs, iree_device, tmp_path):
    """Invalid dim_specs must be rejected by the parser.

    Each must cause the EP to reject the session (either via exception or
    CPU fallback). ORT silently falls back to CPU when EP creation fails,
    so we check both outcomes.
    """
    model_path, _ = _create_matmul_model(tmp_path)
    try:
        session = _create_session(
            model_path,
            iree_device,
            {"target_arch": "host", "dim_specs": dim_specs},
        )
    except OrtError:
        # Exception raised = parser detected the error.
        return

    # Session created without crash. ORT fell back to CPU -- verify
    # our EP is not active.
    providers = session.get_providers()
    assert not any(
        "IREE" in p for p in providers
    ), f"IREE EP should have rejected dim_specs={dim_specs!r}"


# ============================================================================
# MLIR Content Tests
# ============================================================================


def test_mlir_static_specialization(iree_device):
    """Static dim_specs produce util.assume.int in MLIR."""
    model, _ = _make_matmul_model()
    mlir, err = try_generate_mlir(
        model,
        iree_device,
        kernel_dir="",
        target_arch="host",
        extra_provider_options={"dim_specs": "batch(1,1), seq(64,64)"},
    )
    assert mlir is not None, err

    # Check for variant suffixed functions.
    assert "_variant0" in mlir, "MLIR should contain _variant0 function"

    # The fallback function has no suffix (just the graph name).
    func_count = mlir.count("func.func @")
    assert (
        func_count >= 2
    ), f"MLIR should contain at least 2 functions, got {func_count}"

    # Signatures should be generic (no type specialization).
    assert "vtensor<[?,?,16]" in mlir, "signatures should have dynamic dims [?,?,16]"

    # Static constraints should produce util.assume.int with umin == umax.
    assert (
        "util.assume.int" in mlir
    ), "variant should have util.assume.int for static dims"

    # Check static range for batch=1: umin = 1, umax = 1.
    assert (
        "<umin = 1, umax = 1>" in mlir
    ), "should have static assume <umin = 1, umax = 1>"

    # Check static range for seq=64: umin = 64, umax = 64.
    assert (
        "<umin = 64, umax = 64>" in mlir
    ), "should have static assume <umin = 64, umax = 64>"

    # Should have flow.tensor.tie_shape to apply the assumptions.
    assert "flow.tensor.tie_shape" in mlir, "should have flow.tensor.tie_shape"

    # Operand order in tie_shape should follow tensor dim order
    # (batch first, then seq).
    tie_line = next((l for l in mlir.split("\n") if "flow.tensor.tie_shape" in l), "")
    match = re.search(r"\{([^}]+)\}", tie_line)
    assert match, "could not parse tie_shape operands"
    operands = [x.strip() for x in match.group(1).split(",")]
    assert operands == [
        "%dim_assumed_0",
        "%dim_assumed_1",
    ], f"tie_shape operand order mismatch: {operands}"


def test_mlir_divisibility(iree_device):
    """Divisibility produces util.assume.int and flow.tensor.tie_shape."""
    model, _ = _make_matmul_model()
    mlir, err = try_generate_mlir(
        model,
        iree_device,
        kernel_dir="",
        target_arch="host",
        extra_provider_options={"dim_specs": "seq(1,131072,16)"},
    )
    assert mlir is not None, err

    # Should use util.assume.int (not torch.symbolic_int).
    assert "util.assume.int" in mlir, "MLIR should contain util.assume.int ops"

    # Should have range+div: umin = 1, umax = 131072, udiv = 16.
    assert "umin = 1" in mlir, "should have umin = 1"
    assert "umax = 131072" in mlir, "should have umax = 131072"
    assert "udiv = 16" in mlir, "should have udiv = 16"

    # Should have flow.tensor.tie_shape (not torch.bind_symbolic_shape).
    assert (
        "flow.tensor.tie_shape" in mlir
    ), "MLIR should contain flow.tensor.tie_shape ops"

    # Should NOT use the old torch.symbolic_int / torch.bind_symbolic_shape.
    assert "torch.symbolic_int" not in mlir, "should not use torch.symbolic_int anymore"
    assert (
        "torch.bind_symbolic_shape" not in mlir
    ), "should not use torch.bind_symbolic_shape anymore"


def test_mlir_mixed_static_and_divisibility(iree_device):
    """A single variant with both static and divisibility specs."""
    model, _ = _make_matmul_model()
    mlir, err = try_generate_mlir(
        model,
        iree_device,
        kernel_dir="",
        target_arch="host",
        extra_provider_options={"dim_specs": "batch(1,1), seq(1,131072,16)"},
    )
    assert mlir is not None, err

    # Static range assume for batch=1.
    assert (
        "<umin = 1, umax = 1>" in mlir
    ), "variant should have static assume for batch=1"

    # Divisibility assume for seq%16.
    assert "udiv = 16" in mlir, "should have udiv = 16 for seq divisibility"

    # Both should use flow.tensor.tie_shape.
    assert "flow.tensor.tie_shape" in mlir, "should have flow.tensor.tie_shape"


def test_mlir_shared_symbolic_dims(iree_device):
    """Two inputs sharing a symbolic dim must reuse the same canonical assume."""
    model, _ = _make_add_model(feat_a=10, feat_b=20)
    mlir, err = try_generate_mlir(
        model,
        iree_device,
        kernel_dir="",
        target_arch="host",
        extra_provider_options={"dim_specs": "batch(1,131072,4)"},
    )
    assert mlir is not None, err

    # The variant function should have exactly 2 flow.tensor.tie_shape
    # (one per input) and exactly 1 util.assume.int (canonical for "batch").
    # The generic fallback has no dim_specs so contributes 0 of each.
    tie_count = mlir.count("flow.tensor.tie_shape")
    assume_count = mlir.count("util.assume.int")

    assert tie_count == 2, f"expected 2 flow.tensor.tie_shape, got {tie_count}"
    assert (
        assume_count == 1
    ), f"expected 1 util.assume.int (canonical), got {assume_count}"

    # Both tie_shape ops should reference the same assumed SSA value.
    tie_lines = [l for l in mlir.split("\n") if "flow.tensor.tie_shape" in l]
    # Extract the operand list inside {}.
    tie_operands = [re.search(r"\{([^}]+)\}", l).group(1) for l in tie_lines]
    assert (
        len(set(tie_operands)) == 1
    ), f"tie_shape ops use different operands: {tie_operands}"
    canonical_operands = [x.strip() for x in tie_operands[0].split(",")]
    assert canonical_operands == [
        "%dim_assumed_0"
    ], f"expected canonical operand ['%dim_assumed_0'], got {canonical_operands}"


def test_mlir_partially_constrained_dims(iree_device):
    """Only constrained dims get util.assume.int; unconstrained dims do not."""
    model, _ = _make_matmul_model()
    mlir, err = try_generate_mlir(
        model,
        iree_device,
        kernel_dir="",
        target_arch="host",
        extra_provider_options={"dim_specs": "seq(1,131072,16)"},
    )
    assert mlir is not None, err

    # seq should be constrained.
    assert "util.assume.int" in mlir, "should have util.assume.int for seq"
    assert "udiv = 16" in mlir, "should have udiv = 16 for seq"

    # batch should NOT appear in any assume.int (it's unconstrained).
    assume_lines = [l for l in mlir.split("\n") if "util.assume.int" in l]
    for line in assume_lines:
        assert "batch" not in line.lower(), f"batch should not be in assume.int: {line}"


def test_mlir_multi_variant_ordering(iree_device):
    """Multi-variant MLIR has _variant0, _variant1, and 3+ functions in order."""
    model, _ = _make_matmul_model()
    mlir, err = try_generate_mlir(
        model,
        iree_device,
        kernel_dir="",
        target_arch="host",
        extra_provider_options={
            "dim_specs": "batch(1,1), seq(64,64); seq(1,131072,16)"
        },
    )
    assert mlir is not None, err

    # Should have _variant0 and _variant1.
    assert "_variant0" in mlir, "MLIR should contain _variant0"
    assert "_variant1" in mlir, "MLIR should contain _variant1"

    # Should have at least 3 functions (2 variants + generic fallback).
    func_count = mlir.count("func.func @")
    assert func_count >= 3, f"expected at least 3 functions, got {func_count}"

    # _variant0 should appear before _variant1 in the MLIR.
    assert mlir.index("_variant0") < mlir.index(
        "_variant1"
    ), "_variant0 should appear before _variant1"


def test_mlir_assume_ssa_uniqueness(iree_device):
    """Symbolic dims that sanitize identically get unique SSA names.

    'dim-1' and 'dim_1' both sanitize to 'dim_1', but the generated
    util.assume.int SSA names must not collide.
    """
    model, _ = _make_matmul_model(batch_dim="dim-1", seq_dim="dim_1")
    mlir, err = try_generate_mlir(
        model,
        iree_device,
        kernel_dir="",
        target_arch="host",
        extra_provider_options={"dim_specs": "dim-1(1,128), dim_1(1,256)"},
    )
    assert mlir is not None, err

    # Both constraints should produce util.assume.int ops.
    assume_lines = [l for l in mlir.split("\n") if "util.assume.int" in l]
    assert (
        len(assume_lines) == 2
    ), f"expected 2 util.assume.int ops, got {len(assume_lines)}"

    # Extract SSA names (the LHS of each assume line, before ' = ').
    ssa_names = [l.strip().split(" = ")[0] for l in assume_lines]

    assert (
        ssa_names[0] != ssa_names[1]
    ), f"SSA names should be unique but both are {ssa_names[0]}"
