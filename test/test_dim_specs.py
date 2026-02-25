"""Test dim specialization: static, divisibility, multi-variant, and error handling."""

import glob
import os
import re
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

np.random.seed(42)


def _save_model(model, tmp_path):
    """Save ONNX model into tmp_path. Pytest cleans up the directory."""
    path = str(tmp_path / "model.onnx")
    onnx.save(model, path)
    return path


def _create_matmul_model(tmp_path, batch_dim="batch", seq_dim="seq", hidden=16):
    """Create a MatMul model: output = input @ weight.

    input shape:  [batch, seq, hidden]  (batch and seq are symbolic/dynamic)
    weight shape: [hidden, hidden]      (static initializer)
    output shape: [batch, seq, hidden]
    """
    weight_data = np.random.rand(hidden, hidden).astype(np.float32)

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [batch_dim, seq_dim, hidden]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_dim, seq_dim, hidden]
    )

    weight_init = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=[hidden, hidden],
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

    model_path = _save_model(model, tmp_path)
    return model_path, weight_data


def _create_add_model(tmp_path, batch_dim="batch", feat_a=10, feat_b=20):
    """Create an Add model with two inputs sharing a batch dim.

    A shape: [batch, feat_a]
    B shape: [batch, feat_b]  (broadcast-added after Reshape)
    For simplicity, we use MatMul(A, W) + B where W: [feat_a, feat_b].
    output shape: [batch, feat_b]
    """
    weight_data = np.random.rand(feat_a, feat_b).astype(np.float32)

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

    model_path = _save_model(model, tmp_path)
    return model_path, weight_data


def _create_session(model_path, device, provider_options=None):
    """Create an ORT InferenceSession with the given IREE device."""
    opts = ort.SessionOptions()
    opts.add_provider_for_devices([device], provider_options or {})
    return ort.InferenceSession(model_path, sess_options=opts)


def _run_matmul(session, batch, seq, hidden, weight_data):
    """Run inference and compare against numpy reference."""
    x = np.random.rand(batch, seq, hidden).astype(np.float32)
    expected = x @ weight_data
    result = session.run(None, {"input": x})[0]
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def _get_new_mlir_files(before):
    """Return set of new MLIR files created since `before` snapshot.

    WARNING: Not safe for parallel pytest execution (e.g., pytest-xdist).
    Other processes creating iree_ep_*.mlir files in the system temp dir
    between the before/after snapshots would produce false positives.
    """
    temp_dir = tempfile.gettempdir()
    after = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir")))
    return after - before


def _snapshot_mlir_files():
    """Snapshot current MLIR files in temp dir."""
    temp_dir = tempfile.gettempdir()
    return set(glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir")))


def _cleanup_intermediates(mlir_before):
    """Clean up MLIR/VMFB/IRPA files created during a test."""
    temp_dir = tempfile.gettempdir()
    new_mlir = _get_new_mlir_files(mlir_before)
    for f in new_mlir:
        try:
            os.remove(f)
        except OSError:
            pass
    for pattern in ["iree_ep_*.vmfb", "iree_ep_*.irpa"]:
        for f in glob.glob(os.path.join(temp_dir, pattern)):
            try:
                os.remove(f)
            except OSError:
                pass


# ============================================================================
# Tests
# ============================================================================


def test_static_specialization(iree_device, tmp_path):
    """Static dim_specs: matching and non-matching shapes produce correct results."""
    hidden = 16
    model_path, weight_data = _create_matmul_model(tmp_path, hidden=hidden)
    dim_specs = "batch(1,1), seq(64,64)"
    session = _create_session(
        model_path, iree_device, {"target_arch": "host", "dim_specs": dim_specs}
    )
    # Matching shape.
    _run_matmul(session, 1, 64, hidden, weight_data)
    # Non-matching shape must still work via generic fallback.
    _run_matmul(session, 2, 8, hidden, weight_data)


def test_divisibility_specialization(iree_device, tmp_path):
    """Divisibility dim_specs: seq(0,131072,16) works with seq=32."""
    hidden = 16
    model_path, weight_data = _create_matmul_model(tmp_path, hidden=hidden)
    dim_specs = "seq(0,131072,16)"
    session = _create_session(
        model_path, iree_device, {"target_arch": "host", "dim_specs": dim_specs}
    )
    # seq=32 is divisible by 16.
    _run_matmul(session, 2, 32, hidden, weight_data)


def test_multi_variant_dispatch(iree_device, tmp_path):
    """Multi-variant compilation produces correct results for diverse input shapes.

    Compiles 3 variants (2 specialized + generic fallback) into one VMFB and
    verifies inference for shapes matching each constraint pattern.
    Dispatch checks variants in dim_specs order (first match wins).
    """
    hidden = 16
    model_path, weight_data = _create_matmul_model(tmp_path, hidden=hidden)
    dim_specs = "batch(1,1), seq(64,64); seq(0,131072,16)"
    session = _create_session(
        model_path, iree_device, {"target_arch": "host", "dim_specs": dim_specs}
    )
    # Shape matching the static variant.
    _run_matmul(session, 1, 64, hidden, weight_data)
    # Shape matching the divisibility variant.
    _run_matmul(session, 2, 32, hidden, weight_data)
    # Shape matching no specialized variant (generic fallback).
    _run_matmul(session, 3, 17, hidden, weight_data)


def test_parse_errors(iree_device, tmp_path):
    """Invalid dim_specs must be rejected by the parser.

    Tests a variety of malformed inputs. Each must cause the EP to reject the
    session (either via exception or CPU fallback). The old code threw
    std::runtime_error inside a noexcept function, causing std::terminate --
    no crash is the baseline requirement.
    """
    model_path, _ = _create_matmul_model(tmp_path)
    bad_inputs = [
        ("batch", "missing parentheses"),
        ("batch=1", "old format equals sign"),
        ("(1,1)", "empty name"),
        ("batch(1)", "one arg"),
        ("batch(1,2,3,4)", "four args"),
        ("batch(abc,1)", "non-numeric min"),
        ("batch(1,abc)", "non-numeric max"),
        ("batch(1,2,abc)", "non-numeric div"),
        ("batch(-1,1)", "negative min"),
        ("batch(5,3)", "max less than min"),
        ("batch(1,2,0)", "zero divisor"),
        ("batch(1,2,-1)", "negative divisor"),
        ("batch(1,1", "missing closing paren"),
        ("batch(1,1)typo", "trailing garbage after ')'"),
        ("batch(1,1), batch(2,2)", "duplicate key in variant"),
        ("batch(,1)", "empty min"),
        ("batch(1,)", "empty max"),
        ("batch(1,1); seq(1,2,0)", "zero divisor in variant 2"),
    ]

    for dim_specs, label in bad_inputs:
        try:
            session = _create_session(
                model_path,
                iree_device,
                {"target_arch": "host", "dim_specs": dim_specs},
            )
        except Exception:
            # Exception raised = parser detected the error.
            continue

        # Session created without crash. ORT fell back to CPU -- verify
        # our EP is not active.
        providers = session.get_providers()
        assert not any(
            "IREE" in p for p in providers
        ), f"IREE EP should have rejected {label!r}: {dim_specs!r}"


def test_save_intermediates_static(iree_device, tmp_path):
    """Validate that static specialization produces util.assume.int in MLIR."""
    hidden = 16
    model_path, weight_data = _create_matmul_model(tmp_path, hidden=hidden)
    mlir_before = _snapshot_mlir_files()

    dim_specs = "batch(1,1), seq(64,64)"
    session = _create_session(
        model_path,
        iree_device,
        {
            "target_arch": "host",
            "dim_specs": dim_specs,
            "save_intermediates": "1",
        },
    )

    _run_matmul(session, 1, 64, hidden, weight_data)

    new_mlir = _get_new_mlir_files(mlir_before)
    assert new_mlir, "No MLIR file saved"

    mlir_content = open(list(new_mlir)[0]).read()

    # Check for variant suffixed functions.
    assert "_variant0" in mlir_content, "MLIR should contain _variant0 function"

    # The fallback function has no suffix (just the graph name).
    func_count = mlir_content.count("func.func @")
    assert (
        func_count >= 2
    ), f"MLIR should contain at least 2 functions, got {func_count}"

    # Signatures should be generic (no type specialization).
    assert (
        "vtensor<[?,?,16]" in mlir_content
    ), "signatures should have dynamic dims [?,?,16]"

    # Static constraints should produce util.assume.int with umin == umax.
    assert (
        "util.assume.int" in mlir_content
    ), "variant should have util.assume.int for static dims"

    # Check static range for batch=1: umin = 1, umax = 1.
    assert (
        "<umin = 1, umax = 1>" in mlir_content
    ), "should have static assume <umin = 1, umax = 1>"

    # Check static range for seq=64: umin = 64, umax = 64.
    assert (
        "<umin = 64, umax = 64>" in mlir_content
    ), "should have static assume <umin = 64, umax = 64>"

    # Should have flow.tensor.tie_shape to apply the assumptions.
    assert "flow.tensor.tie_shape" in mlir_content, "should have flow.tensor.tie_shape"

    # Operand order in tie_shape should follow tensor dim order
    # (batch first, then seq).
    tie_line = next(
        (l for l in mlir_content.split("\n") if "flow.tensor.tie_shape" in l), ""
    )
    match = re.search(r"\{([^}]+)\}", tie_line)
    assert match, "could not parse tie_shape operands"
    operands = [x.strip() for x in match.group(1).split(",")]
    assert operands == [
        "%batch_assumed",
        "%seq_assumed",
    ], f"tie_shape operand order mismatch: {operands}"

    _cleanup_intermediates(mlir_before)


def test_save_intermediates_divisibility(iree_device, tmp_path):
    """Validate that divisibility produces util.assume.int and flow.tensor.tie_shape."""
    hidden = 16
    model_path, weight_data = _create_matmul_model(tmp_path, hidden=hidden)
    mlir_before = _snapshot_mlir_files()

    dim_specs = "seq(0,131072,16)"
    session = _create_session(
        model_path,
        iree_device,
        {
            "target_arch": "host",
            "dim_specs": dim_specs,
            "save_intermediates": "1",
        },
    )

    _run_matmul(session, 2, 32, hidden, weight_data)

    new_mlir = _get_new_mlir_files(mlir_before)
    assert new_mlir, "No MLIR file saved"

    mlir_content = open(list(new_mlir)[0]).read()

    # Should use util.assume.int (not torch.symbolic_int).
    assert "util.assume.int" in mlir_content, "MLIR should contain util.assume.int ops"

    # Should have range+div: umin = 0, umax = 131072, udiv = 16.
    assert "umin = 0" in mlir_content, "should have umin = 0"
    assert "umax = 131072" in mlir_content, "should have umax = 131072"
    assert "udiv = 16" in mlir_content, "should have udiv = 16"

    # Should have flow.tensor.tie_shape (not torch.bind_symbolic_shape).
    assert (
        "flow.tensor.tie_shape" in mlir_content
    ), "MLIR should contain flow.tensor.tie_shape ops"

    # Should NOT use the old torch.symbolic_int / torch.bind_symbolic_shape.
    assert (
        "torch.symbolic_int" not in mlir_content
    ), "should not use torch.symbolic_int anymore"
    assert (
        "torch.bind_symbolic_shape" not in mlir_content
    ), "should not use torch.bind_symbolic_shape anymore"

    _cleanup_intermediates(mlir_before)


def test_empty_dim_specs(iree_device, tmp_path):
    """Empty or whitespace dim_specs must behave like no specialization."""
    hidden = 16
    model_path, weight_data = _create_matmul_model(tmp_path, hidden=hidden)
    for dim_specs in ["", "   "]:
        opts = {"target_arch": "host"}
        if dim_specs:
            opts["dim_specs"] = dim_specs
        session = _create_session(model_path, iree_device, opts)
        _run_matmul(session, 2, 8, hidden, weight_data)


def test_mixed_static_divisibility(iree_device, tmp_path):
    """A single variant with both static and divisibility specs."""
    hidden = 16
    model_path, weight_data = _create_matmul_model(tmp_path, hidden=hidden)
    mlir_before = _snapshot_mlir_files()

    # batch(1,1) (static) + seq(0,131072,16) (range+div) in one variant.
    dim_specs = "batch(1,1), seq(0,131072,16)"
    session = _create_session(
        model_path,
        iree_device,
        {
            "target_arch": "host",
            "dim_specs": dim_specs,
            "save_intermediates": "1",
        },
    )

    # Matching input: batch=1, seq=32 (divisible by 16).
    _run_matmul(session, 1, 32, hidden, weight_data)

    # Check MLIR has both static and divisibility assumes.
    new_mlir = _get_new_mlir_files(mlir_before)
    assert new_mlir, "No MLIR file saved"

    mlir_content = open(list(new_mlir)[0]).read()

    # Static range assume for batch=1.
    assert (
        "<umin = 1, umax = 1>" in mlir_content
    ), "variant should have static assume for batch=1"

    # Divisibility assume for seq=%16.
    assert "udiv = 16" in mlir_content, "should have udiv = 16 for seq divisibility"

    # Both should use flow.tensor.tie_shape.
    assert "flow.tensor.tie_shape" in mlir_content, "should have flow.tensor.tie_shape"

    _cleanup_intermediates(mlir_before)


def test_shared_symbolic_dims(iree_device, tmp_path):
    """Two inputs sharing a symbolic dim must reuse the same canonical assume."""
    feat_a, feat_b = 10, 20
    model_path, weight_data = _create_add_model(tmp_path, feat_a=feat_a, feat_b=feat_b)
    mlir_before = _snapshot_mlir_files()

    dim_specs = "batch(0,131072,4)"
    session = _create_session(
        model_path,
        iree_device,
        {
            "target_arch": "host",
            "dim_specs": dim_specs,
            "save_intermediates": "1",
        },
    )

    # Run inference: batch=8 (divisible by 4).
    a = np.random.rand(8, feat_a).astype(np.float32)
    b = np.random.rand(8, feat_b).astype(np.float32)
    expected = a @ weight_data + b
    result = session.run(None, {"A": a, "B": b})[0]
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    # Check MLIR for the new ops.
    new_mlir = _get_new_mlir_files(mlir_before)
    assert new_mlir, "No MLIR file saved"

    mlir_content = open(list(new_mlir)[0]).read()

    # The variant function should have exactly 2 flow.tensor.tie_shape
    # (one per input) and exactly 1 util.assume.int (canonical for "batch").
    # The generic fallback has no dim_specs so contributes 0 of each.
    tie_count = mlir_content.count("flow.tensor.tie_shape")
    assume_count = mlir_content.count("util.assume.int")

    assert tie_count == 2, f"expected 2 flow.tensor.tie_shape, got {tie_count}"
    assert (
        assume_count == 1
    ), f"expected 1 util.assume.int (canonical), got {assume_count}"

    # Both tie_shape ops should reference the same assumed SSA value.
    tie_lines = [l for l in mlir_content.split("\n") if "flow.tensor.tie_shape" in l]
    # Extract the operand list inside {}.
    tie_operands = [re.search(r"\{([^}]+)\}", l).group(1) for l in tie_lines]
    assert (
        len(set(tie_operands)) == 1
    ), f"tie_shape ops use different operands: {tie_operands}"
    canonical_operands = [x.strip() for x in tie_operands[0].split(",")]
    assert canonical_operands == [
        "%batch_assumed"
    ], f"expected canonical operand ['%batch_assumed'], got {canonical_operands}"

    _cleanup_intermediates(mlir_before)


def test_shared_dim_conflict_fallback(iree_device, tmp_path):
    """Inputs sharing a symbolic dim with different runtime values must not crash.

    If input A has batch=4 and input B has batch=8, dispatch must not select a
    specialized variant (whose tie_shape assumes all batch dims are equal).
    The generic fallback has no tie_shape assumptions and is safe.

    Note: Mismatched shared symbolic dims is a model-level error. This test
    verifies that the EP handles it gracefully (no crash/segfault) rather than
    applying incorrect tie_shape assumptions from a specialized variant.
    """
    feat_a, feat_b = 10, 20
    model_path, weight_data = _create_add_model(tmp_path, feat_a=feat_a, feat_b=feat_b)
    dim_specs = "batch(0,131072,4)"
    session = _create_session(
        model_path, iree_device, {"target_arch": "host", "dim_specs": dim_specs}
    )

    # Run with matching batch dims first (sanity check).
    a = np.random.rand(8, feat_a).astype(np.float32)
    b = np.random.rand(8, feat_b).astype(np.float32)
    expected = a @ weight_data + b
    result = session.run(None, {"A": a, "B": b})[0]
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    # Run with conflicting batch dims (A:batch=4, B:batch=8).
    # Both satisfy the spec individually, but they disagree on "batch".
    # The EP should fall through to the generic fallback without crashing.
    a2 = np.random.rand(4, feat_a).astype(np.float32)
    b2 = np.random.rand(8, feat_b).astype(np.float32)
    try:
        result2 = session.run(None, {"A": a2, "B": b2})[0]
        # If it runs, verify the result shape is reasonable.
        assert result2.shape[1] == feat_b, f"unexpected output shape: {result2.shape}"
    except Exception:
        # Some backends may reject shape mismatches at the IREE level.
        # That's acceptable -- the key is no segfault/abort.
        pass


def test_whitespace_and_divisibility_by_one(iree_device, tmp_path):
    """dim_specs with extra whitespace; divisibility by 1 (always matches)."""
    hidden = 16
    model_path, weight_data = _create_matmul_model(tmp_path, hidden=hidden)
    # Whitespace around delimiters.
    dim_specs = "  seq( 0 , 131072 , 1 )  "
    session = _create_session(
        model_path, iree_device, {"target_arch": "host", "dim_specs": dim_specs}
    )
    # div=1 matches any positive seq value.
    _run_matmul(session, 2, 7, hidden, weight_data)


def test_unknown_dim_spec_rejected(iree_device, tmp_path):
    """dim_specs referencing a non-existent symbolic dim must be rejected.

    If a user makes a typo (e.g., "batc" instead of "batch"), the EP should
    reject it at compile time rather than silently ignoring the constraint.
    """
    model_path, _ = _create_matmul_model(tmp_path)
    bad_inputs = [
        ("batc(1,1), seq(64,64)", "typo in dim name"),
        ("foo(5,5)", "non-existent dim"),
        ("batch(1,1); xyz(0,131072,16)", "unknown dim in variant 2"),
    ]

    for dim_specs, label in bad_inputs:
        try:
            session = _create_session(
                model_path,
                iree_device,
                {"target_arch": "host", "dim_specs": dim_specs},
            )
        except Exception:
            continue

        # Session created without crash. ORT fell back to CPU -- verify
        # our EP is not active.
        providers = session.get_providers()
        assert not any(
            "IREE" in p for p in providers
        ), f"IREE EP should have rejected {label!r}: {dim_specs!r}"
