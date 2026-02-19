#!/usr/bin/env python3
"""ExternDispatch MLIR generation tests for the IREE ORT EP.

Tests verify the generated MLIR (input to iree-compile) rather than running
the full compilation pipeline. This avoids requiring iree-compile on PATH.

Run with:
  pytest test/test_extern_dispatch.py \
      --ep-lib /path/to/libonnxruntime_ep_iree.so -v
"""

import pytest
from conftest import try_generate_mlir
from onnx import TensorProto, helper

# ---------------------------------------------------------------------------
# ONNX dtype constants for parametrize
# ---------------------------------------------------------------------------
F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
F64 = TensorProto.DOUBLE
BF16 = TensorProto.BFLOAT16
I32 = TensorProto.INT32
I64 = TensorProto.INT64
I8 = TensorProto.INT8

# Map ONNX dtype to expected MLIR type string.
MLIR_TYPES = {
    F32: "f32",
    F16: "f16",
    F64: "f64",
    BF16: "bf16",
    I32: "i32",
    I8: "i8",
}


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def make_extern_node(
    inputs,
    outputs,
    kernel_name="sigmoid_extern",
    kernel_object="kernel_{arch}.co",
    workgroup_size=None,
    push_constants=None,
    workgroup_count=None,
):
    """Create an ExternDispatch node with given params."""
    kwargs = {}
    if kernel_name is not None:
        kwargs["kernel_name"] = kernel_name
    if kernel_object is not None:
        kwargs["kernel_object"] = kernel_object
    if workgroup_size is not None:
        kwargs["workgroup_size"] = workgroup_size
    if push_constants is not None:
        kwargs["push_constants"] = push_constants
    if workgroup_count is not None:
        kwargs["workgroup_count"] = workgroup_count
    return helper.make_node(
        "ExternDispatch",
        inputs=inputs,
        outputs=outputs,
        domain="com.iree",
        **kwargs,
    )


def make_model(nodes, inputs, outputs, value_info=None):
    """Build a complete ONNX model with com.iree opset."""
    graph = helper.make_graph(
        nodes,
        "test_extern_dispatch",
        inputs,
        outputs,
        value_info=value_info or [],
    )
    model = helper.make_model(
        graph,
        producer_name="extern_dispatch_test",
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.iree", 1),
        ],
    )
    model.ir_version = 8
    return model


def vi(name, dtype, shape):
    """Shorthand for make_tensor_value_info."""
    return helper.make_tensor_value_info(name, dtype, shape)


# ===================================================================
# Valid configuration tests — check generated MLIR
# ===================================================================


class TestValidConfigs:
    """Tests that should generate valid MLIR."""

    def test_basic_single_extern(self, gpu_device, kernel_dir, target_arch):
        """Basic: Relu -> ExternDispatch(sigmoid)."""
        relu = helper.make_node("Relu", ["input"], ["relu_out"])
        ext = make_extern_node(
            ["relu_out"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [relu, ext],
            [vi("input", F32, [4])],
            [vi("output", F32, [4])],
            value_info=[vi("relu_out", F32, [4])],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir
        assert "#hal.executable.target" in mlir
        assert "onnx.Relu" in mlir
        assert "torch_c.to_builtin_tensor" in mlir
        assert "torch_c.from_builtin_tensor" in mlir
        assert '"sigmoid_extern"' in mlir
        assert "tensor<4xf32>" in mlir

    def test_extern_only_no_onnx_ops(
        self, gpu_device, kernel_dir, target_arch
    ):
        """ExternDispatch as the only node (no standard ONNX ops)."""
        ext = make_extern_node(
            ["input"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext],
            [vi("input", F32, [4])],
            [vi("output", F32, [4])],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir
        assert "onnx.Relu" not in mlir

    def test_no_push_constants(self, gpu_device, kernel_dir, target_arch):
        """ExternDispatch with zero push constants."""
        ext = make_extern_node(
            ["input"],
            ["output"],
            workgroup_size=[64, 1, 1],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext],
            [vi("input", F32, [4])],
            [vi("output", F32, [4])],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir

    def test_multiple_push_constants(
        self, gpu_device, kernel_dir, target_arch
    ):
        """Three literal push constants."""
        ext = make_extern_node(
            ["input"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["4", "8", "16"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext],
            [vi("input", F32, [4])],
            [vi("output", F32, [4])],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir
        # Three push constants as i32.
        assert "4 : i32" in mlir
        assert "8 : i32" in mlir
        assert "16 : i32" in mlir

    def test_push_constant_from_input_ref(
        self, gpu_device, kernel_dir, target_arch
    ):
        """Push constant from $1 (scalar i64 input)."""
        ext = make_extern_node(
            ["input", "n_elements"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["$1"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext],
            [vi("input", F32, [4]), vi("n_elements", I64, [])],
            [vi("output", F32, [4])],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir
        assert "tensor.extract" in mlir
        assert "arith.trunci" in mlir

    def test_dynamic_workgroup_x(self, gpu_device, kernel_dir, target_arch):
        """Workgroup count X from $2 (scalar i64 input)."""
        ext = make_extern_node(
            ["input", "n_elements", "wg_x"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["$1"],
            workgroup_count=["$2", "1", "1"],
        )
        model = make_model(
            [ext],
            [vi("input", F32, [4]),
             vi("n_elements", I64, []),
             vi("wg_x", I64, [])],
            [vi("output", F32, [4])],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir
        # Dynamic values use tensor.extract from scalar inputs.
        assert "tensor.extract" in mlir
        assert "arith.index_cast" in mlir

    def test_two_chained_externs(self, gpu_device, kernel_dir, target_arch):
        """Two ExternDispatches in sequence (SSA numbering)."""
        ext1 = make_extern_node(
            ["input"],
            ["mid"],
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
            workgroup_count=["1", "1", "1"],
        )
        ext2 = make_extern_node(
            ["mid"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext1, ext2],
            [vi("input", F32, [4])],
            [vi("output", F32, [4])],
            value_info=[vi("mid", F32, [4])],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        # Two dispatches.
        assert mlir.count("hal.dispatch.extern") == 2

    def test_mixed_onnx_and_extern(self, gpu_device, kernel_dir, target_arch):
        """Relu -> ExternDispatch -> Relu (interleaved)."""
        relu1 = helper.make_node("Relu", ["input"], ["relu1_out"])
        ext = make_extern_node(
            ["relu1_out"],
            ["sigmoid_out"],
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
            workgroup_count=["1", "1", "1"],
        )
        relu2 = helper.make_node("Relu", ["sigmoid_out"], ["output"])
        model = make_model(
            [relu1, ext, relu2],
            [vi("input", F32, [4])],
            [vi("output", F32, [4])],
            value_info=[
                vi("relu1_out", F32, [4]),
                vi("sigmoid_out", F32, [4]),
            ],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir
        assert mlir.count("onnx.Relu") == 2

    def test_two_inputs(self, gpu_device, kernel_dir, target_arch):
        """ExternDispatch with two input bindings."""
        ext = make_extern_node(
            ["a", "b"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext],
            [vi("a", F32, [4]), vi("b", F32, [4])],
            [vi("output", F32, [4])],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir
        # Two to_builtin_tensor bridges for two inputs.
        assert mlir.count("torch_c.to_builtin_tensor") == 2

    def test_two_outputs(self, gpu_device, kernel_dir, target_arch):
        """ExternDispatch producing two output bindings."""
        ext = make_extern_node(
            ["input"],
            ["out_a", "out_b"],
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext],
            [vi("input", F32, [4])],
            [vi("out_a", F32, [4]), vi("out_b", F32, [4])],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir
        # Two from_builtin_tensor bridges for two outputs.
        assert mlir.count("torch_c.from_builtin_tensor") == 2

    def test_multi_input_multi_output(self, gpu_device, kernel_dir, target_arch):
        """ExternDispatch with two inputs and two outputs."""
        ext = make_extern_node(
            ["a", "b"],
            ["out_a", "out_b"],
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext],
            [vi("a", F32, [4]), vi("b", F32, [4])],
            [vi("out_a", F32, [4]), vi("out_b", F32, [4])],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir
        assert mlir.count("torch_c.to_builtin_tensor") == 2
        assert mlir.count("torch_c.from_builtin_tensor") == 2

    def test_input_ref_later_input(self, gpu_device, kernel_dir, target_arch):
        """Input ref $2 referencing a later (non-first) input."""
        ext = make_extern_node(
            ["a", "b", "n_val"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["$2"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext],
            [vi("a", F32, [4]), vi("b", F32, [8]), vi("n_val", I64, [])],
            [vi("output", F32, [4])],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir
        assert "tensor.extract" in mlir

    def test_mixed_dtypes_single_dispatch(
        self, gpu_device, kernel_dir, target_arch
    ):
        """Mixed data dtypes (f32 + bf16) and mixed scalar dtypes (i32 + i64)
        in a single ExternDispatch node."""
        ext = make_extern_node(
            ["data_f32", "data_bf16", "scalar_i32", "scalar_i64"],
            ["out_f32"],
            workgroup_size=[64, 1, 1],
            push_constants=["$2", "$3"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext],
            [
                vi("data_f32", F32, [4]),
                vi("data_bf16", BF16, [4]),
                vi("scalar_i32", I32, []),
                vi("scalar_i64", I64, []),
            ],
            [vi("out_f32", F32, [4])],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir
        # Both data bindings present with correct types.
        assert "tensor<4xf32>" in mlir
        assert "tensor<4xbf16>" in mlir
        # Both scalar extracts present.
        assert "tensor.extract" in mlir
        # i32 scalar: extracted directly (no trunci needed).
        # i64 scalar: extracted then truncated to i32.
        assert "arith.trunci" in mlir

    # --- Data types ---

    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (F16, [4]),
            (F64, [4]),
            (BF16, [4]),
            (I32, [4]),
            (I8, [4]),
        ],
        ids=["f16", "f64", "bf16", "i32", "i8"],
    )
    def test_data_types(
        self, dtype, shape, gpu_device, kernel_dir, target_arch
    ):
        """ExternDispatch generates correct MLIR types."""
        ext = make_extern_node(
            ["input"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext],
            [vi("input", dtype, shape)],
            [vi("output", dtype, shape)],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir
        expected_type = f"tensor<4x{MLIR_TYPES[dtype]}>"
        assert expected_type in mlir

    # --- Tensor shapes ---

    @pytest.mark.parametrize(
        "shape,expected_shape_str",
        [
            ([1], "1x"),
            ([128], "128x"),
            ([4, 8], "4x8x"),
            ([2, 3, 4], "2x3x4x"),
        ],
        ids=["scalar_1", "vec_128", "mat_4x8", "tensor_2x3x4"],
    )
    def test_shapes(
        self, shape, expected_shape_str, gpu_device, kernel_dir, target_arch
    ):
        """ExternDispatch generates correct tensor shapes."""
        n = 1
        for d in shape:
            n *= d
        ext = make_extern_node(
            ["input"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=[str(n)],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext],
            [vi("input", F32, shape)],
            [vi("output", F32, shape)],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir
        expected_type = f"tensor<{expected_shape_str}f32>"
        assert expected_type in mlir

    # --- Workgroup configurations ---

    @pytest.mark.parametrize(
        "wg_size,wg_count",
        [
            ([32, 1, 1], ["1", "1", "1"]),
            ([64, 2, 1], ["1", "1", "1"]),
            ([64, 1, 1], ["2", "1", "1"]),
            ([64, 1, 1], ["1", "2", "1"]),
            ([64, 1, 1], ["1", "1", "2"]),
            ([256, 1, 1], ["1", "1", "1"]),
        ],
        ids=[
            "wg32",
            "wg64x2",
            "count_2x1x1",
            "count_1x2x1",
            "count_1x1x2",
            "wg256",
        ],
    )
    def test_workgroup_configs(
        self, wg_size, wg_count, gpu_device, kernel_dir, target_arch
    ):
        """Various valid workgroup size/count combinations."""
        ext = make_extern_node(
            ["input"],
            ["output"],
            workgroup_size=wg_size,
            push_constants=["4"],
            workgroup_count=wg_count,
        )
        model = make_model(
            [ext],
            [vi("input", F32, [4])],
            [vi("output", F32, [4])],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is not None, err
        assert "hal.dispatch.extern" in mlir
        wg_str = f"workgroup_size = [{wg_size[0]} : index, {wg_size[1]} : index, {wg_size[2]} : index]"
        assert wg_str in mlir


# ===================================================================
# Error tests — MLIR generation should fail with clear messages
# ===================================================================


class TestMissingAttributes:
    """Missing required attributes should produce clear errors."""

    def test_missing_kernel_name(self, gpu_device, kernel_dir, target_arch):
        node = helper.make_node(
            "ExternDispatch",
            inputs=["input"],
            outputs=["output"],
            domain="com.iree",
            kernel_object="kernel_{arch}.co",
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [node], [vi("input", F32, [4])], [vi("output", F32, [4])]
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is None
        assert "kernel_name" in err

    def test_missing_kernel_object(self, gpu_device, kernel_dir, target_arch):
        node = helper.make_node(
            "ExternDispatch",
            inputs=["input"],
            outputs=["output"],
            domain="com.iree",
            kernel_name="sigmoid_extern",
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [node], [vi("input", F32, [4])], [vi("output", F32, [4])]
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is None
        assert "kernel_object" in err

    def test_missing_workgroup_size(self, gpu_device, kernel_dir, target_arch):
        node = helper.make_node(
            "ExternDispatch",
            inputs=["input"],
            outputs=["output"],
            domain="com.iree",
            kernel_name="sigmoid_extern",
            kernel_object="kernel_{arch}.co",
            push_constants=["4"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [node], [vi("input", F32, [4])], [vi("output", F32, [4])]
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is None
        assert "workgroup_size" in err

    def test_missing_workgroup_count(
        self, gpu_device, kernel_dir, target_arch
    ):
        node = helper.make_node(
            "ExternDispatch",
            inputs=["input"],
            outputs=["output"],
            domain="com.iree",
            kernel_name="sigmoid_extern",
            kernel_object="kernel_{arch}.co",
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
        )
        model = make_model(
            [node], [vi("input", F32, [4])], [vi("output", F32, [4])]
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is None
        assert "workgroup_count" in err


class TestInvalidWorkgroupCount:
    """Invalid workgroup_count specs."""

    @pytest.mark.parametrize(
        "wg_count,expected_err",
        [
            (["1", "1"], "exactly 3"),
            (["1", "1", "1", "1"], "exactly 3"),
        ],
        ids=["too_few", "too_many"],
    )
    def test_wrong_element_count(
        self, wg_count, expected_err, gpu_device, kernel_dir, target_arch
    ):
        ext = make_extern_node(
            ["input"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
            workgroup_count=wg_count,
        )
        model = make_model(
            [ext], [vi("input", F32, [4])], [vi("output", F32, [4])]
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is None
        assert expected_err in err

    @pytest.mark.parametrize(
        "wg_count",
        [
            ["abc", "1", "1"],
            ["1", "abc", "1"],
            ["1", "1", "xyz"],
        ],
        ids=["non_numeric_x", "non_numeric_y", "non_numeric_z"],
    )
    def test_non_numeric(
        self, wg_count, gpu_device, kernel_dir, target_arch
    ):
        ext = make_extern_node(
            ["input"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
            workgroup_count=wg_count,
        )
        model = make_model(
            [ext], [vi("input", F32, [4])], [vi("output", F32, [4])]
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is None
        assert "not a valid" in err

    @pytest.mark.parametrize(
        "wg_count",
        [
            ["1", "$1", "1"],
            ["1", "1", "$1"],
        ],
        ids=["dynamic_y", "dynamic_z"],
    )
    def test_dynamic_y_or_z(
        self, wg_count, gpu_device, kernel_dir, target_arch
    ):
        ext = make_extern_node(
            ["input", "n_val"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["4"],
            workgroup_count=wg_count,
        )
        model = make_model(
            [ext],
            [vi("input", F32, [4]), vi("n_val", I64, [])],
            [vi("output", F32, [4])],
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is None
        assert "Y and Z must be integer literals" in err


class TestInvalidWorkgroupSize:
    """Invalid workgroup_size specs."""

    @pytest.mark.parametrize(
        "wg_size,expected_err",
        [
            ([64], "exactly 3"),
            ([64, 1], "exactly 3"),
            ([64, 1, 1, 1], "exactly 3"),
        ],
        ids=["one", "two", "four"],
    )
    def test_wrong_element_count(
        self, wg_size, expected_err, gpu_device, kernel_dir, target_arch
    ):
        ext = make_extern_node(
            ["input"],
            ["output"],
            workgroup_size=wg_size,
            push_constants=["4"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext], [vi("input", F32, [4])], [vi("output", F32, [4])]
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is None
        assert expected_err in err

    @pytest.mark.parametrize(
        "wg_size",
        [
            [0, 1, 1],
            [64, 0, 1],
            [64, 1, -1],
        ],
        ids=["zero_x", "zero_y", "negative_z"],
    )
    def test_non_positive(
        self, wg_size, gpu_device, kernel_dir, target_arch
    ):
        ext = make_extern_node(
            ["input"],
            ["output"],
            workgroup_size=wg_size,
            push_constants=["4"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext], [vi("input", F32, [4])], [vi("output", F32, [4])]
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is None
        assert "must be positive" in err


class TestInvalidPushConstants:
    """Invalid push_constants specs."""

    def test_non_numeric(self, gpu_device, kernel_dir, target_arch):
        ext = make_extern_node(
            ["input"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["abc"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext], [vi("input", F32, [4])], [vi("output", F32, [4])]
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is None
        assert "not a valid" in err

    def test_oob_input_reference(self, gpu_device, kernel_dir, target_arch):
        """$5 with only 1 input should error."""
        ext = make_extern_node(
            ["input"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["$5"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext], [vi("input", F32, [4])], [vi("output", F32, [4])]
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is None
        assert "only 1 inputs available" in err

    def test_malformed_input_ref(self, gpu_device, kernel_dir, target_arch):
        """Malformed $ syntax (no number after $)."""
        ext = make_extern_node(
            ["input"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["$abc"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext], [vi("input", F32, [4])], [vi("output", F32, [4])]
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is None
        assert "not a valid" in err

    def test_non_scalar_input_ref(self, gpu_device, kernel_dir, target_arch):
        """$0 on a non-scalar tensor (rank > 0) should error."""
        ext = make_extern_node(
            ["input"],
            ["output"],
            workgroup_size=[64, 1, 1],
            push_constants=["$0"],
            workgroup_count=["1", "1", "1"],
        )
        model = make_model(
            [ext], [vi("input", F32, [4])], [vi("output", F32, [4])]
        )
        mlir, err = try_generate_mlir(
            model, gpu_device, kernel_dir, target_arch
        )
        assert mlir is None
        assert "rank" in err.lower() or "scalar" in err.lower()
