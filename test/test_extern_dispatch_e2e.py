#!/usr/bin/env python3
"""End-to-end ExternDispatch tests through the IREE ORT EP.

Tests exercise full model creation -> MLIR gen -> iree-compile -> inference
on a real HIP device. Uses 256-element tensors with workgroup_count=[4,1,1]
to exercise multi-workgroup dispatch. Three test models:

  1. Sigmoid (single-in single-out, static):
     Input(256xf32) -> Relu -> sigmoid -> Relu -> sigmoid -> Output

  2. AddMul (multi-in multi-out, static):
     (A(256xf32), B(256xf32)) -> add_mul -> (Sum, Prod)
     -> Relu(Sum) -> sigmoid -> SigmoidOut
     Final outputs: SigmoidOut, Prod

  3. Dynamic Sigmoid (ONNX-idiomatic $N input references):
     Input(256xf32) -> Relu -> relu_out
     Shape(Input) -> Gather(0) -> n_elements
     ceildiv(n_elements, 64) -> wg_x
     ExternDispatch(relu_out, n_elements, wg_x,
                    push_constants=["$1"], workgroup_count=["$2","1","1"])

Usage:
  # Step 1: Build kernels (if not already built)
  ./test/build_kernels.sh --arch gfx1100

  # Step 2: Run with pytest (fixtures provided by conftest.py)
  pytest test/test_extern_dispatch_e2e.py \
      --ep-lib /path/to/libonnxruntime_ep_iree.so \
      --kernel-dir test/build \
      --target-arch gfx1100 \
      --device-index 1 -v

  # Or run standalone
  python test/test_extern_dispatch_e2e.py \
      --ep-lib /path/to/libonnxruntime_ep_iree.so \
      --kernel-dir test/build \
      --target-arch gfx1100
"""

import argparse
import pathlib
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

F32 = TensorProto.FLOAT
I64 = TensorProto.INT64


def vi(name, shape):
    return helper.make_tensor_value_info(name, F32, shape)


N = 256  # Element count — exercises multi-workgroup dispatch.
WG_SIZE = 64  # Threads per workgroup (one RDNA wavefront).
WG_COUNT = str(N // WG_SIZE)  # 256/64 = 4 workgroups.


def make_extern(name, inputs, outputs, kernel_name, push_constants):
    return helper.make_node(
        "ExternDispatch",
        inputs=inputs,
        outputs=outputs,
        name=name,
        domain="com.iree",
        kernel_name=kernel_name,
        kernel_object="kernel_{arch}.co",
        workgroup_size=[WG_SIZE, 1, 1],
        push_constants=push_constants,
        workgroup_count=[WG_COUNT, "1", "1"],
    )


def build_model(nodes, inputs, outputs, value_info=None):
    graph = helper.make_graph(
        nodes,
        "test_extern_dispatch",
        inputs,
        outputs,
        value_info=value_info or [],
    )
    model = helper.make_model(
        graph,
        producer_name="extern_dispatch_e2e_test",
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.iree", 1),
        ],
    )
    model.ir_version = 8
    return model


def _run_with_device(model, feeds, device, kernel_dir, target_arch):
    """Run a model with a pre-discovered device. Returns list of output arrays."""
    import onnxruntime as ort

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = f.name
        onnx.save(model, model_path)

    try:
        sess_options = ort.SessionOptions()
        provider_options = {
            "target_arch": target_arch,
            "extern_kernel_path": kernel_dir,
            "save_intermediates": "1",
        }
        sess_options.add_provider_for_devices([device], provider_options)
        session = ort.InferenceSession(model_path, sess_options=sess_options)
        return session.run(None, feeds)
    finally:
        pathlib.Path(model_path).unlink(missing_ok=True)


_ep_registered = False


def run_model(model, feeds, ep_lib, kernel_dir, target_arch, device_index=0):
    """Run a model end-to-end on HIP (standalone CLI). Returns list of output arrays."""
    import onnxruntime as ort

    global _ep_registered
    if not _ep_registered:
        ort.register_execution_provider_library("IREE", ep_lib)
        _ep_registered = True
    ep_devices = ort.get_ep_devices()
    hip_devices = [
        d for d in ep_devices
        if d.device.metadata.get("iree.driver", "") == "hip"
    ]
    if not hip_devices or device_index >= len(hip_devices):
        raise RuntimeError(
            f"No HIP device at index {device_index} "
            f"({len(hip_devices)} available)"
        )

    return _run_with_device(
        model, feeds, hip_devices[device_index], kernel_dir, target_arch
    )


# ---------------------------------------------------------------------------
# Test 1: Sigmoid (single-in single-out, chained with ONNX Relu)
# ---------------------------------------------------------------------------

def create_sigmoid_model():
    """Input -> Relu -> sigmoid -> Relu -> sigmoid -> Output."""
    n = str(N)
    nodes = [
        helper.make_node("Relu", ["input"], ["relu1_out"]),
        make_extern("ext1", ["relu1_out"], ["sig1_out"],
                     "sigmoid_extern", [n]),
        helper.make_node("Relu", ["sig1_out"], ["relu2_out"]),
        make_extern("ext2", ["relu2_out"], ["output"],
                     "sigmoid_extern", [n]),
    ]
    return build_model(
        nodes,
        inputs=[vi("input", [N])],
        outputs=[vi("output", [N])],
        value_info=[vi("relu1_out", [N]), vi("sig1_out", [N]),
                    vi("relu2_out", [N])],
    )


def sigmoid_expected(input_data):
    after_relu1 = np.maximum(input_data, 0)
    after_sig1 = 1.0 / (1.0 + np.exp(-after_relu1))
    after_relu2 = np.maximum(after_sig1, 0)  # no-op (sigmoid > 0)
    return 1.0 / (1.0 + np.exp(-after_relu2))


# ---------------------------------------------------------------------------
# Test 2: AddMul (multi-in multi-out)
#   (A, B) -> add_mul_extern -> (Sum, Prod)
#   Sum -> Relu -> sigmoid -> SigmoidOut
#   Outputs: SigmoidOut, Prod
# ---------------------------------------------------------------------------

def create_addmul_model():
    """(A, B) -> add_mul -> (sum, prod); sum -> Relu -> sigmoid -> out1; out2=prod."""
    n = str(N)
    nodes = [
        make_extern("addmul", ["a", "b"], ["sum_out", "prod_out"],
                     "add_mul_extern", [n]),
        helper.make_node("Relu", ["sum_out"], ["relu_out"]),
        make_extern("sig", ["relu_out"], ["sigmoid_out"],
                     "sigmoid_extern", [n]),
    ]
    return build_model(
        nodes,
        inputs=[vi("a", [N]), vi("b", [N])],
        outputs=[vi("sigmoid_out", [N]), vi("prod_out", [N])],
        value_info=[vi("sum_out", [N]), vi("relu_out", [N])],
    )


def addmul_expected(a, b):
    sum_ab = a + b
    prod_ab = a * b
    relu_sum = np.maximum(sum_ab, 0)
    sigmoid_out = 1.0 / (1.0 + np.exp(-relu_sum))
    return sigmoid_out, prod_ab


# ---------------------------------------------------------------------------
# Test 3: Dynamic Sigmoid (ONNX-idiomatic $N input references)
#   input(256xf32) -> Relu -> relu_out
#   Shape(input) -> Gather(0) -> n_elements (scalar i64)
#   ceildiv(n_elements, WG_SIZE) -> wg_x (scalar i64)
#   ExternDispatch(relu_out, n_elements, wg_x,
#                  push_constants=["$1"], workgroup_count=["$2","1","1"])
# ---------------------------------------------------------------------------

def _make_i64_scalar(name, val):
    """Create a scalar (rank-0) i64 constant tensor."""
    return numpy_helper.from_array(np.int64(val), name=name)


def create_dynamic_sigmoid_model():
    """Sigmoid with Shape->Gather->Div computing push constant and workgroup count."""
    nodes = [
        helper.make_node("Relu", ["input"], ["relu_out"]),
        # Compute n_elements = Shape(input)[0]
        helper.make_node("Shape", ["input"], ["shape_out"]),
        helper.make_node("Constant", [], ["gather_idx"],
                         value=_make_i64_scalar("", 0)),
        helper.make_node("Gather", ["shape_out", "gather_idx"],
                         ["n_elements"], axis=0),
        # Compute wg_x = ceildiv(n_elements, WG_SIZE)
        helper.make_node("Constant", [], ["const_wg_m1"],
                         value=_make_i64_scalar("", WG_SIZE - 1)),
        helper.make_node("Constant", [], ["const_wg"],
                         value=_make_i64_scalar("", WG_SIZE)),
        helper.make_node("Add", ["n_elements", "const_wg_m1"],
                         ["n_plus_wg_m1"]),
        helper.make_node("Div", ["n_plus_wg_m1", "const_wg"], ["wg_x"]),
        # ExternDispatch with $N references
        helper.make_node(
            "ExternDispatch",
            inputs=["relu_out", "n_elements", "wg_x"],
            outputs=["output"],
            name="ext_dynamic",
            domain="com.iree",
            kernel_name="sigmoid_extern",
            kernel_object="kernel_{arch}.co",
            workgroup_size=[WG_SIZE, 1, 1],
            push_constants=["$1"],
            workgroup_count=["$2", "1", "1"],
        ),
    ]
    return build_model(
        nodes,
        inputs=[vi("input", [N])],
        outputs=[vi("output", [N])],
        value_info=[
            vi("relu_out", [N]),
            helper.make_tensor_value_info("shape_out", I64, [1]),
            helper.make_tensor_value_info("n_elements", I64, []),
            helper.make_tensor_value_info("n_plus_wg_m1", I64, []),
            helper.make_tensor_value_info("wg_x", I64, []),
        ],
    )


def dynamic_sigmoid_expected(input_data):
    after_relu = np.maximum(input_data, 0)
    return 1.0 / (1.0 + np.exp(-after_relu))


# ---------------------------------------------------------------------------
# Pytest interface (fixtures provided by conftest.py)
# ---------------------------------------------------------------------------

class TestSigmoidE2E:
    """End-to-end: Relu -> sigmoid -> Relu -> sigmoid."""

    def test_sigmoid(self, gpu_device, kernel_dir, target_arch):
        rng = np.random.default_rng(42)
        input_data = rng.standard_normal(N).astype(np.float32)
        model = create_sigmoid_model()
        expected = sigmoid_expected(input_data)

        result = _run_with_device(
            model, {"input": input_data}, gpu_device, kernel_dir, target_arch
        )[0]
        np.testing.assert_allclose(result, expected, atol=1e-3)


class TestAddMulE2E:
    """End-to-end: add_mul (MIMO) -> Relu -> sigmoid."""

    def test_addmul(self, gpu_device, kernel_dir, target_arch):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(N).astype(np.float32)
        b = rng.standard_normal(N).astype(np.float32)
        model = create_addmul_model()
        exp_sig, exp_prod = addmul_expected(a, b)

        outputs = _run_with_device(
            model, {"a": a, "b": b}, gpu_device, kernel_dir, target_arch
        )
        np.testing.assert_allclose(outputs[0], exp_sig, atol=1e-3)
        np.testing.assert_allclose(outputs[1], exp_prod, atol=1e-3)


class TestDynamicDispatchE2E:
    """End-to-end: dynamic $N refs with Shape->Gather->Div for workgroup count."""

    def test_dynamic_sigmoid(self, gpu_device, kernel_dir, target_arch):
        rng = np.random.default_rng(42)
        input_data = rng.standard_normal(N).astype(np.float32)
        model = create_dynamic_sigmoid_model()
        expected = dynamic_sigmoid_expected(input_data)

        result = _run_with_device(
            model, {"input": input_data}, gpu_device, kernel_dir, target_arch
        )[0]
        np.testing.assert_allclose(result, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="E2E ExternDispatch tests"
    )
    parser.add_argument("--ep-lib", required=True,
                        help="Path to libonnxruntime_ep_iree.so")
    parser.add_argument("--kernel-dir", required=True,
                        help="Directory containing kernel .co files")
    parser.add_argument("--target-arch", default="gfx1100",
                        help="GPU target arch")
    parser.add_argument("--device-index", type=int, default=0,
                        help="HIP device index")
    parser.add_argument("--model-only", action="store_true",
                        help="Only save models, don't run inference")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for --model-only")
    args = parser.parse_args()

    if args.model_only:
        out = pathlib.Path(args.output_dir)
        onnx.save(create_sigmoid_model(), str(out / "sigmoid_e2e.onnx"))
        onnx.save(create_addmul_model(), str(out / "addmul_e2e.onnx"))
        onnx.save(create_dynamic_sigmoid_model(),
                  str(out / "dynamic_sigmoid_e2e.onnx"))
        print(f"Models saved to {out}")
        return

    passed = 0
    failed = 0

    # Test 1: Sigmoid
    print(f"=== Test 1: Sigmoid ({N} elements, {WG_COUNT} workgroups) ===")
    rng = np.random.default_rng(42)
    input_data = rng.standard_normal(N).astype(np.float32)
    expected = sigmoid_expected(input_data)
    try:
        result = run_model(
            create_sigmoid_model(),
            {"input": input_data},
            args.ep_lib, args.kernel_dir, args.target_arch, args.device_index,
        )[0]
        print(f"  Shape: {input_data.shape}")
        print(f"  First 4 expected: {expected[:4]}")
        print(f"  First 4 got:      {result[:4]}")
        if np.allclose(result, expected, atol=1e-3):
            print("  PASSED")
            passed += 1
        else:
            print(f"  FAILED: max diff = {np.max(np.abs(result - expected))}")
            failed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 2: AddMul (MIMO)
    print(f"\n=== Test 2: AddMul MIMO ({N} elements, {WG_COUNT} workgroups) ===")
    a = rng.standard_normal(N).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    exp_sig, exp_prod = addmul_expected(a, b)
    try:
        outputs = run_model(
            create_addmul_model(),
            {"a": a, "b": b},
            args.ep_lib, args.kernel_dir, args.target_arch, args.device_index,
        )
        sigmoid_out, prod_out = outputs[0], outputs[1]
        print(f"  Shape: {a.shape}")
        print(f"  First 4 expected sig:  {exp_sig[:4]}")
        print(f"  First 4 got sig:       {sigmoid_out[:4]}")
        print(f"  First 4 expected prod: {exp_prod[:4]}")
        print(f"  First 4 got prod:      {prod_out[:4]}")
        sig_ok = np.allclose(sigmoid_out, exp_sig, atol=1e-3)
        prod_ok = np.allclose(prod_out, exp_prod, atol=1e-3)
        if sig_ok and prod_ok:
            print("  PASSED")
            passed += 1
        else:
            if not sig_ok:
                print(f"  FAILED sigmoid: max diff = "
                      f"{np.max(np.abs(sigmoid_out - exp_sig))}")
            if not prod_ok:
                print(f"  FAILED prod: max diff = "
                      f"{np.max(np.abs(prod_out - exp_prod))}")
            failed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 3: Dynamic Sigmoid ($N input refs)
    print(f"\n=== Test 3: Dynamic Sigmoid ($N refs, {N} elements) ===")
    rng3 = np.random.default_rng(42)
    input_data3 = rng3.standard_normal(N).astype(np.float32)
    expected3 = dynamic_sigmoid_expected(input_data3)
    try:
        result3 = run_model(
            create_dynamic_sigmoid_model(),
            {"input": input_data3},
            args.ep_lib, args.kernel_dir, args.target_arch, args.device_index,
        )[0]
        print(f"  Shape: {input_data3.shape}")
        print(f"  First 4 expected: {expected3[:4]}")
        print(f"  First 4 got:      {result3[:4]}")
        if np.allclose(result3, expected3, atol=1e-3):
            print("  PASSED")
            passed += 1
        else:
            print(f"  FAILED: max diff = {np.max(np.abs(result3 - expected3))}")
            failed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
