#!/usr/bin/env python3
"""Test dim specialization: static, divisibility, multi-variant, and error handling."""

import glob
import os
import pathlib
import re
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper

import test_utils

np.random.seed(42)


def create_matmul_model(batch_dim="batch", seq_dim="seq", hidden=16):
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

    model_path = test_utils.save_model(model)
    return model_path, weight_data


def create_add_model(batch_dim="batch", feat_a=10, feat_b=20):
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

    model_path = test_utils.save_model(model)
    return model_path, weight_data


def run_matmul(session, batch, seq, hidden, weight_data):
    """Run inference and compare against numpy reference."""
    x = np.random.rand(batch, seq, hidden).astype(np.float32)
    expected = x @ weight_data
    result = session.run(None, {"input": x})[0]
    return np.allclose(result, expected, rtol=1e-4, atol=1e-4)


# ============================================================================
# Tests
# ============================================================================


def test_static_specialization():
    """Static dim_specs: matching and non-matching shapes produce correct results."""
    print("\n=== test_static_specialization ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    hidden = 16
    model_path, weight_data = create_matmul_model(hidden=hidden)
    try:
        dim_specs = "batch=1,seq=64"
        session = test_utils.create_session(
            model_path, device, {"target_arch": "host", "dim_specs": dim_specs}
        )
        if not run_matmul(session, 1, 64, hidden, weight_data):
            print("FAIL: matching shape inference mismatch")
            return False
        print("  Matching shape (batch=1, seq=64): OK")

        # Non-matching shape must still work via generic fallback.
        if not run_matmul(session, 2, 8, hidden, weight_data):
            print("FAIL: non-matching shape inference mismatch")
            return False
        print("  Non-matching shape (batch=2, seq=8): OK")

        print("  Static specialization: PASS")
        return True
    finally:
        pathlib.Path(model_path).unlink()


def test_divisibility_specialization():
    """Divisibility dim_specs: seq=%16 works with seq=32."""
    print("\n=== test_divisibility_specialization ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    hidden = 16
    model_path, weight_data = create_matmul_model(hidden=hidden)
    try:
        dim_specs = "seq=%16"
        session = test_utils.create_session(
            model_path, device, {"target_arch": "host", "dim_specs": dim_specs}
        )
        # seq=32 is divisible by 16.
        if not run_matmul(session, 2, 32, hidden, weight_data):
            print("FAIL: inference result mismatch")
            return False

        print("  Divisibility specialization (seq=%16, seq=32): PASS")
        return True
    finally:
        pathlib.Path(model_path).unlink()


def test_multi_variant_dispatch():
    """Multi-variant compilation produces correct results for diverse input shapes.

    Compiles 3 variants (2 specialized + generic fallback) into one VMFB and
    verifies inference for shapes matching each constraint pattern.
    """
    print("\n=== test_multi_variant_dispatch ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    hidden = 16
    model_path, weight_data = create_matmul_model(hidden=hidden)
    try:
        dim_specs = "batch=1,seq=64;seq=%16"
        session = test_utils.create_session(
            model_path, device, {"target_arch": "host", "dim_specs": dim_specs}
        )

        # Shape matching the static variant.
        if not run_matmul(session, 1, 64, hidden, weight_data):
            print("FAIL: (batch=1, seq=64) result mismatch")
            return False
        print("  Shape (batch=1, seq=64): OK")

        # Shape matching the divisibility variant.
        if not run_matmul(session, 2, 32, hidden, weight_data):
            print("FAIL: (batch=2, seq=32) result mismatch")
            return False
        print("  Shape (batch=2, seq=32): OK")

        # Shape matching no specialized variant.
        if not run_matmul(session, 3, 17, hidden, weight_data):
            print("FAIL: (batch=3, seq=17) result mismatch")
            return False
        print("  Shape (batch=3, seq=17): OK")

        print("  Multi-variant dispatch: PASS")
        return True
    finally:
        pathlib.Path(model_path).unlink()


def _assert_ep_rejected(model_path, device, dim_specs, label):
    """Assert that invalid dim_specs causes EP rejection (not a crash).

    When ParseDimSpecs returns an error, ORT either raises a Python
    exception or silently falls back to CPUExecutionProvider. Either outcome
    proves the parser rejected the input. The critical requirement is that the
    process does not crash (the old code called std::terminate).
    """
    try:
        session = test_utils.create_session(
            model_path, device, {"target_arch": "host", "dim_specs": dim_specs}
        )
    except Exception as e:
        # Exception raised = parser detected the error.
        print(f"    {label}: OK (exception)")
        return True

    # Session created without crash. ORT fell back to CPU -- verify our EP
    # is not active.
    providers = session.get_providers()
    if any("IREE" in p for p in providers):
        print(f"    {label}: FAIL (IREE EP should have rejected this)")
        return False
    print(f"    {label}: OK (EP rejected, CPU fallback)")
    return True


def test_parse_errors():
    """Invalid dim_specs must be rejected by the parser.

    Tests a variety of malformed inputs. Each must cause the EP to reject the
    session (either via exception or CPU fallback). The old code threw
    std::runtime_error inside a noexcept function, causing std::terminate --
    no crash is the baseline requirement.
    """
    print("\n=== test_parse_errors ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    model_path, _ = create_matmul_model()
    try:
        bad_inputs = [
            # Missing '=' sign.
            ("batch", "missing equals"),
            ("batch,seq", "multiple specs missing equals"),
            # Empty key or value.
            ("=1", "empty key"),
            ("batch=", "empty value"),
            # Invalid divisor values.
            ("batch=%0", "zero divisor"),
            ("batch=%-1", "negative divisor"),
            ("batch=%abc", "non-numeric divisor"),
            ("batch=%", "bare percent"),
            ("batch=%1.5", "float divisor"),
            # Non-numeric static values.
            ("batch=hello", "non-numeric value"),
            ("batch=1.5", "float value"),
            # Zero divisor in later variant.
            ("a=1;b=%0", "zero divisor in variant 2"),
            # Invalid static dim values.
            ("batch=0", "zero static dim"),
            ("batch=-1", "negative static dim"),
        ]

        all_ok = True
        for dim_specs, label in bad_inputs:
            if not _assert_ep_rejected(model_path, device, dim_specs, label):
                all_ok = False

        if all_ok:
            print("  All parse error cases: PASS")
        return all_ok
    finally:
        pathlib.Path(model_path).unlink()


def test_save_intermediates_static():
    """Validate that static specialization produces concrete dims in MLIR."""
    print("\n=== test_save_intermediates_static ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    hidden = 16
    model_path, weight_data = create_matmul_model(hidden=hidden)

    # Track existing MLIR files to find the new one.
    temp_dir = tempfile.gettempdir()
    mlir_before = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir")))

    try:
        dim_specs = "batch=1,seq=64"
        session = test_utils.create_session(
            model_path,
            device,
            {
                "target_arch": "host",
                "dim_specs": dim_specs,
                "save_intermediates": "1",
            },
        )

        # Verify inference still works.
        if not run_matmul(session, 1, 64, hidden, weight_data):
            print("FAIL: inference result mismatch")
            return False

        # Find the newly created MLIR file.
        mlir_after = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir")))
        new_mlir = mlir_after - mlir_before
        if not new_mlir:
            print("FAIL: No MLIR file saved")
            return False

        mlir_content = open(list(new_mlir)[0]).read()

        # Static specialization should produce concrete dims in the function
        # signature, not "?" for the specialized dims.
        # Check for variant suffixed functions.
        if "_variant0" not in mlir_content:
            print("FAIL: MLIR should contain _variant0 function")
            return False
        print("  _variant0 function present")

        # The fallback function has no suffix (just the graph name).
        func_count = mlir_content.count("func.func @")
        if func_count < 2:
            print(f"FAIL: MLIR should contain at least 2 functions, got {func_count}")
            return False
        print(f"  {func_count} functions present (variant + fallback)")

        # The variant0 function should have concrete dims [1,64,16].
        if "vtensor<[1,64,16]" not in mlir_content:
            print("FAIL: variant0 should have concrete dims [1,64,16]")
            return False
        print("  Static dims [1,64,16] in variant0 signature")

        # The generic fallback should have "?" for the dynamic dims.
        if "vtensor<[?,?,16]" not in mlir_content:
            print("FAIL: generic fallback should have dynamic dims [?,?,16]")
            return False
        print("  Dynamic dims [?,?,16] in generic fallback")

        # Clean up MLIR files.
        for f in new_mlir:
            try:
                os.remove(f)
            except OSError:
                pass
        # Also clean up VMFB/IRPA files.
        for pattern in ["iree_ep_*.vmfb", "iree_ep_*.irpa"]:
            for f in glob.glob(os.path.join(temp_dir, pattern)):
                try:
                    os.remove(f)
                except OSError:
                    pass

        print("  MLIR validation: PASS")
        return True
    finally:
        pathlib.Path(model_path).unlink()


def test_save_intermediates_divisibility():
    """Validate that divisibility produces symbolic_int and bind_symbolic_shape."""
    print("\n=== test_save_intermediates_divisibility ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    hidden = 16
    model_path, weight_data = create_matmul_model(hidden=hidden)

    temp_dir = tempfile.gettempdir()
    mlir_before = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir")))

    try:
        dim_specs = "seq=%16"
        session = test_utils.create_session(
            model_path,
            device,
            {
                "target_arch": "host",
                "dim_specs": dim_specs,
                "save_intermediates": "1",
            },
        )

        if not run_matmul(session, 2, 32, hidden, weight_data):
            print("FAIL: inference result mismatch")
            return False

        mlir_after = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir")))
        new_mlir = mlir_after - mlir_before
        if not new_mlir:
            print("FAIL: No MLIR file saved")
            return False

        mlir_content = open(list(new_mlir)[0]).read()

        if "torch.symbolic_int" not in mlir_content:
            print("FAIL: MLIR should contain torch.symbolic_int ops")
            return False
        print("  torch.symbolic_int present")

        if "torch.bind_symbolic_shape" not in mlir_content:
            print("FAIL: MLIR should contain torch.bind_symbolic_shape ops")
            return False
        print("  torch.bind_symbolic_shape present")

        # The affine map for seq should contain the divisor multiplication.
        # The symbol index depends on registration order, so just check "* 16".
        if "* 16" not in mlir_content:
            print("FAIL: affine map should contain '* 16' for seq=%16")
            return False
        print("  Affine map contains divisor '* 16'")

        # Clean up.
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

        print("  MLIR divisibility ops: PASS")
        return True
    finally:
        pathlib.Path(model_path).unlink()


def _cleanup_intermediates(temp_dir, mlir_before):
    """Clean up MLIR/VMFB/IRPA files created during a test."""
    mlir_after = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir")))
    new_mlir = mlir_after - mlir_before
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


def test_empty_dim_specs():
    """Empty or whitespace dim_specs must behave like no specialization."""
    print("\n=== test_empty_dim_specs ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    hidden = 16
    model_path, weight_data = create_matmul_model(hidden=hidden)
    try:
        for label, dim_specs in [("empty string", ""), ("whitespace only", "   ")]:
            opts = {"target_arch": "host"}
            if dim_specs:
                opts["dim_specs"] = dim_specs
            session = test_utils.create_session(model_path, device, opts)
            if not run_matmul(session, 2, 8, hidden, weight_data):
                print(f"FAIL: {label} inference mismatch")
                return False
            print(f"  {label}: OK")

        print("  Empty dim_specs backward compatibility: PASS")
        return True
    finally:
        pathlib.Path(model_path).unlink()


def test_mixed_static_divisibility():
    """A single variant with both static and divisibility specs."""
    print("\n=== test_mixed_static_divisibility ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    hidden = 16
    model_path, weight_data = create_matmul_model(hidden=hidden)
    temp_dir = tempfile.gettempdir()
    mlir_before = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir")))

    try:
        # batch=1 (static) + seq=%16 (divisibility) in one variant.
        dim_specs = "batch=1,seq=%16"
        session = test_utils.create_session(
            model_path,
            device,
            {
                "target_arch": "host",
                "dim_specs": dim_specs,
                "save_intermediates": "1",
            },
        )

        # Matching input: batch=1, seq=32 (divisible by 16).
        if not run_matmul(session, 1, 32, hidden, weight_data):
            print("FAIL: inference result mismatch")
            return False
        print("  Inference (batch=1, seq=32): OK")

        # Check MLIR has concrete batch=1 and symbolic_int for seq.
        mlir_after = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir")))
        new_mlir = mlir_after - mlir_before
        if not new_mlir:
            print("FAIL: No MLIR file saved")
            return False

        mlir_content = open(list(new_mlir)[0]).read()

        if "vtensor<[1," not in mlir_content:
            print("FAIL: variant should have concrete batch=1")
            return False
        print("  Concrete batch=1 in variant: OK")

        if "torch.symbolic_int" not in mlir_content:
            print("FAIL: should have symbolic_int for seq divisibility")
            return False
        print("  symbolic_int for seq: OK")

        _cleanup_intermediates(temp_dir, mlir_before)
        print("  Mixed static+divisibility: PASS")
        return True
    finally:
        pathlib.Path(model_path).unlink()


def test_shared_symbolic_dims():
    """Two inputs sharing a symbolic dim must bind to the same symbol."""
    print("\n=== test_shared_symbolic_dims ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    feat_a, feat_b = 10, 20
    model_path, weight_data = create_add_model(feat_a=feat_a, feat_b=feat_b)
    temp_dir = tempfile.gettempdir()
    mlir_before = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir")))

    try:
        dim_specs = "batch=%4"
        session = test_utils.create_session(
            model_path,
            device,
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
        if not np.allclose(result, expected, rtol=1e-4, atol=1e-4):
            print("FAIL: inference result mismatch")
            return False
        print("  Inference (batch=8, shared dim): OK")

        # Check MLIR: should have exactly one torch.symbolic_int for "batch".
        mlir_after = set(glob.glob(os.path.join(temp_dir, "iree_ep_*.mlir")))
        new_mlir = mlir_after - mlir_before
        if not new_mlir:
            print("FAIL: No MLIR file saved")
            return False

        mlir_content = open(list(new_mlir)[0]).read()

        # The variant function should have exactly 1 torch.symbolic_int (for
        # "batch") and exactly 2 torch.bind_symbolic_shape (one per input).
        # The generic fallback has no dim_specs so contributes 0 of each.
        sym_int_count = mlir_content.count("torch.symbolic_int")
        bind_count = mlir_content.count("torch.bind_symbolic_shape")

        if sym_int_count != 1:
            print(f"FAIL: expected 1 torch.symbolic_int, got {sym_int_count}")
            return False
        print(f"  torch.symbolic_int count: {sym_int_count}")

        if bind_count != 2:
            print(f"FAIL: expected 2 torch.bind_symbolic_shape, got {bind_count}")
            return False
        print(f"  torch.bind_symbolic_shape count: {bind_count}")

        # Both bind ops should reference the same symbol (shared batch dim).
        bind_lines = [
            l for l in mlir_content.split("\n") if "torch.bind_symbolic_shape" in l
        ]
        sym_refs = [re.search(r"\[([^\]]+)\]", l).group(1) for l in bind_lines]
        if len(set(sym_refs)) != 1:
            print(f"FAIL: bind ops reference different symbols: {sym_refs}")
            return False
        print(f"  Both inputs share symbol: {sym_refs[0]}")

        _cleanup_intermediates(temp_dir, mlir_before)
        print("  Shared symbolic dims: PASS")
        return True
    finally:
        pathlib.Path(model_path).unlink()


def test_whitespace_and_divisibility_by_one():
    """dim_specs with extra whitespace; divisibility by 1 (always matches)."""
    print("\n=== test_whitespace_and_divisibility_by_one ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    hidden = 16
    model_path, weight_data = create_matmul_model(hidden=hidden)
    try:
        # Whitespace around delimiters.
        dim_specs = "  seq = %1  "
        session = test_utils.create_session(
            model_path, device, {"target_arch": "host", "dim_specs": dim_specs}
        )
        # %1 matches any positive seq value.
        if not run_matmul(session, 2, 7, hidden, weight_data):
            print("FAIL: inference result mismatch")
            return False
        print("  Whitespace + %1 divisibility: OK")

        print("  Whitespace and divisibility by 1: PASS")
        return True
    finally:
        pathlib.Path(model_path).unlink()


def test_unknown_dim_spec_rejected():
    """dim_specs referencing a non-existent symbolic dim must be rejected.

    If a user makes a typo (e.g., "batc" instead of "batch"), the EP should
    reject it at compile time rather than silently ignoring the constraint.
    """
    print("\n=== test_unknown_dim_spec_rejected ===")

    device = test_utils.get_iree_device()
    if not device:
        print("ERROR: IREE device not found")
        return False

    model_path, _ = create_matmul_model()
    try:
        bad_inputs = [
            # Typo in dim name.
            ("batc=1,seq=64", "typo in dim name"),
            # Completely unknown dim.
            ("foo=5", "non-existent dim"),
            # Unknown dim in later variant.
            ("batch=1;xyz=%16", "unknown dim in variant 2"),
        ]

        all_ok = True
        for dim_specs, label in bad_inputs:
            if not _assert_ep_rejected(model_path, device, dim_specs, label):
                all_ok = False

        if all_ok:
            print("  All unknown dim_spec cases: PASS")
        return all_ok
    finally:
        pathlib.Path(model_path).unlink()


def main():
    """Run all dim specialization tests."""
    print("Testing dim specialization (static, divisibility, multi-variant)")
    print("=" * 60)

    test_utils.register_ep()

    results = []
    results.append(("static_specialization", test_static_specialization()))
    results.append(("divisibility_specialization", test_divisibility_specialization()))
    results.append(("multi_variant_dispatch", test_multi_variant_dispatch()))
    results.append(("parse_errors", test_parse_errors()))
    results.append(("save_intermediates_static", test_save_intermediates_static()))
    results.append(
        ("save_intermediates_divisibility", test_save_intermediates_divisibility())
    )
    results.append(("empty_dim_specs", test_empty_dim_specs()))
    results.append(("mixed_static_divisibility", test_mixed_static_divisibility()))
    results.append(("shared_symbolic_dims", test_shared_symbolic_dims()))
    results.append(
        ("whitespace_and_div_by_1", test_whitespace_and_divisibility_by_one())
    )
    results.append(("unknown_dim_spec_rejected", test_unknown_dim_spec_rejected()))

    print("\n" + "=" * 60)
    print("Summary:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n=== All tests PASSED ===")
        return 0
    else:
        print("\n=== Some tests FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
