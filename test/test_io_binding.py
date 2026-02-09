#!/usr/bin/env python3
"""Test io_binding with device memory for the IREE ONNX Runtime EP."""

import pathlib
import sys
import tempfile
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import onnxruntime_ep_iree


def test_io_binding():
    """Test io_binding with device-allocated tensors."""

    # Enable debug logging.
    ort.set_default_logger_severity(0)

    # Get the path to the built EP library
    ep_lib_path = onnxruntime_ep_iree.get_library_path()
    print(f"EP library path: {ep_lib_path}")

    # Register the EP plugin
    ort.register_execution_provider_library(
        onnxruntime_ep_iree.get_ep_name(), ep_lib_path
    )
    print("EP plugin registered successfully")

    # Get IREE device
    iree_driver = "vulkan"
    ep_devices = ort.get_ep_devices()
    iree_device = None
    for dev in ep_devices:
        if dev.device.metadata.get("iree.driver") == iree_driver:
            iree_device = dev
            break

    if not iree_device:
        print("ERROR: IREE EP not found in EP devices")
        return False

    print(f"IREE EP available (vendor={iree_device.ep_vendor})")
    print(f"  Device metadata: {iree_device.device.metadata}")
    print(f"  Device type: {iree_device.device.type}")
    print(f"  Device ID: {iree_device.device.device_id}")

    # Test data
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    b = np.array(
        [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], dtype=np.float32
    )

    # Create a simple Add model
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 4])
    output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3, 4])
    constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["D"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=[3, 4],
            vals=b.flatten().tolist(),
        ),
    )
    add_node = helper.make_node("Add", inputs=["A", "D"], outputs=["C"])

    graph = helper.make_graph(
        [add_node, constant_node], "test_graph", [input_a], [output]
    )
    model = helper.make_model(
        graph, producer_name="iree_test", opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = f.name
        onnx.save(model, model_path)

    try:
        # Create session with IREE EP
        sess_options = ort.SessionOptions()

        # Provider options for IREE EP.
        provider_options = {
            "target_arch": "vulkan-spirv",
        }
        sess_options.add_provider_for_devices([iree_device], provider_options)

        session = ort.InferenceSession(model_path, sess_options=sess_options)

        # Test io_binding with IREE device memory.
        # Allocate input on IREE device, run inference, keep output on device.

        io_binding = session.io_binding()

        input_shape = list(a.shape)
        output_shape = list(a.shape)  # Same shape for Add output
        print(f"\nTesting io_binding with IREE device memory")
        print(f"  Input shape: {input_shape}")
        print(f"  Output shape: {output_shape}")

        # Get IREE device info from the ep_device
        # vendor_id for IREE EP is 0x1EEE
        device_id = iree_device.device.device_id
        vendor_id = 0x1EEE
        print(f"  IREE device_id: {device_id}, vendor_id: 0x{vendor_id:X}")

        # Allocate input tensor on IREE device using vendor_id
        # device_type must be "gpu" when using vendor_id
        input_tensor = ort.OrtValue.ortvalue_from_shape_and_type(
            input_shape,
            np.float32,
            device_type="gpu",
            device_id=device_id,
            vendor_id=vendor_id,
        )
        print(f"  Input tensor allocated on IREE device")

        # Copy input data from numpy to device tensor
        input_tensor.update_inplace(a)
        print(f"  Input data copied to device")

        # Bind input from device memory
        io_binding.bind_ortvalue_input("A", input_tensor)
        print("  Input bound via io_binding")

        # Allocate output tensor on IREE device
        output_tensor = ort.OrtValue.ortvalue_from_shape_and_type(
            output_shape,
            np.float32,
            device_type="gpu",
            device_id=device_id,
            vendor_id=vendor_id,
        )
        print(f"  Output tensor allocated on IREE device")

        # Bind output to pre-allocated device memory
        io_binding.bind_ortvalue_output("C", output_tensor)
        print("  Output bound to IREE device")

        # Run inference with io_binding
        print("\nRunning inference with io_binding...")
        session.run_with_iobinding(io_binding)
        print("Inference completed")

        # Output is in our pre-allocated output_tensor (still on device)
        print(f"  Output tensor device: {output_tensor.device_name()}")

        # Copy output to CPU to verify results
        # The numpy() method should handle D2H transfer
        result = output_tensor.numpy()

        # Verify output
        expected = a + b

        if result.shape != expected.shape:
            print(f"FAIL: Shape mismatch: {result.shape} vs {expected.shape}")
            return False

        if not np.allclose(result, expected, rtol=1e-5, atol=1e-5):
            print(f"FAIL: Values mismatch")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")
            return False

        print("\nInference output verified!")
        print(f"  Input A: {a.flatten()[:4]}...")
        print(f"  Constant B: {b.flatten()[:4]}...")
        print(f"  Output:  {result.flatten()[:4]}...")

    finally:
        pathlib.Path(model_path).unlink()

    print("\n=== Test PASSED ===")
    return True


if __name__ == "__main__":
    success = test_io_binding()
    sys.exit(0 if success else 1)
