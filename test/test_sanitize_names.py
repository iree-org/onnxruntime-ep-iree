"""Test that SanitizeName hex-escaping produces distinct, valid MLIR names."""

import pathlib
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from conftest import try_generate_mlir
from onnx import TensorProto, helper


def _make_add_model(input_names):
    """Create a model that adds [4, 4] float32 inputs together."""
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [4, 4])
        for name in input_names
    ]
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [4, 4])

    nodes = []
    prev = input_names[0]
    for i in range(1, len(input_names)):
        out_name = "out" if i == len(input_names) - 1 else f"tmp{i}"
        nodes.append(
            helper.make_node("Add", inputs=[prev, input_names[i]], outputs=[out_name])
        )
        prev = out_name

    graph = helper.make_graph(nodes, "test", inputs, [output])
    model = helper.make_model(
        graph,
        producer_name="sanitize_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    return model


def _generate_mlir(input_names, cpu_device, target_arch):
    """Helper: build model and return MLIR text, failing on error or compile failure."""
    model = _make_add_model(input_names)
    mlir, err = try_generate_mlir(
        model, cpu_device, "", target_arch, assert_compiles=True
    )
    assert err is None, f"MLIR generation failed: {err}"
    return mlir


def test_dash_vs_underscore_mlir(cpu_device, target_arch):
    """'input-1' and 'input_1' must produce distinct SSA names."""
    mlir = _generate_mlir(["input-1", "input_1"], cpu_device, target_arch)

    # 'input-1' -> 'input$2D$1' (dash escaped)
    # 'input_1' -> 'input_1'    (unchanged)
    assert "%input$2D$1" in mlir
    assert "%input_1" in mlir


def test_dot_in_name_mlir(cpu_device, target_arch):
    """Dots are hex-escaped: '.' -> '$2E$'."""
    mlir = _generate_mlir(["x.0", "x_0"], cpu_device, target_arch)

    assert "%x$2E$0" in mlir
    assert "%x_0" in mlir


def test_leading_digit_mlir(cpu_device, target_arch):
    """Names starting with a digit get the digit escaped."""
    mlir = _generate_mlir(["0abc", "abc"], cpu_device, target_arch)

    # '0abc' -> '$30$abc' (leading '0' escaped)
    assert "%$30$abc" in mlir
    assert "%abc" in mlir


def test_leading_digit_no_collision_mlir(cpu_device, target_arch):
    """'0abc' and '_0abc' must not collide."""
    mlir = _generate_mlir(["0abc", "_0abc"], cpu_device, target_arch)

    # '0abc'  -> '$30$abc' (leading digit escaped)
    # '_0abc' -> '_0abc'   (unchanged)
    assert "%$30$abc" in mlir
    assert "%_0abc" in mlir


def test_escape_ambiguity_mlir(cpu_device, target_arch):
    """Literal '_2D_' must not collide with escaped '-'.

    '$' is the escape delimiter, so 'a_2D_b' passes through unchanged
    while 'a-b' becomes 'a$2D$b'. No collision.
    """
    mlir = _generate_mlir(["a_2D_b", "a-b"], cpu_device, target_arch)

    assert "%a_2D_b" in mlir
    assert "%a$2D$b" in mlir


def test_dollar_escaped_mlir(cpu_device, target_arch):
    """'$' in a name is escaped to '$24$', so it can't collide with escape sequences.

    'a$2D$b' (literal) -> 'a$24$2D$24$b' (both '$' escaped)
    'a-b'              -> 'a$2D$b'        (dash escaped)
    """
    mlir = _generate_mlir(["a$2D$b", "a-b"], cpu_device, target_arch)

    assert "%a$24$2D$24$b" in mlir
    assert "%a$2D$b" in mlir
