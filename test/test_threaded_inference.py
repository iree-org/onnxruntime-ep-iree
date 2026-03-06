"""Test concurrent inference correctness on a shared device.

Verifies that multiple threads running different models on the same device
produce correct results. This catches cross-thread buffer interleaving bugs
in the zero-copy output buffer reuse path: if thread A's output buffer were
stolen by thread B, results would be silently wrong.
"""

import pathlib
import tempfile
import threading

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper


def _make_mul_model(shape, factor, name):
    """Create a model that computes out = A * factor."""
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, shape)
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)
    const = helper.make_tensor("factor", TensorProto.FLOAT, [1], [float(factor)])
    mul = helper.make_node("Mul", ["A", "factor"], ["out"])
    graph = helper.make_graph([mul], name, [input_a], [output], [const])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _save_model(model):
    """Save model to a temp file, return path."""
    f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx.save(model, f.name)
    f.close()
    return f.name


def _run_threaded_inference(device, provider_options, iterations=100, num_threads=4):
    """Run two models concurrently and verify both produce correct results."""
    size = 1024
    factor_a, factor_b = 2.0, 3.0

    model_a_path = _save_model(_make_mul_model([size], factor_a, "mul2"))
    model_b_path = _save_model(_make_mul_model([size], factor_b, "mul3"))

    try:
        opts_a = ort.SessionOptions()
        opts_a.log_severity_level = 2
        opts_a.add_provider_for_devices([device], provider_options)
        session_a = ort.InferenceSession(model_a_path, sess_options=opts_a)

        opts_b = ort.SessionOptions()
        opts_b.log_severity_level = 2
        opts_b.add_provider_for_devices([device], provider_options)
        session_b = ort.InferenceSession(model_b_path, sess_options=opts_b)

        # Verify single-threaded correctness first.
        test_input = np.ones(size, dtype=np.float32)
        out_a = session_a.run(None, {"A": test_input})[0]
        out_b = session_b.run(None, {"A": test_input})[0]
        np.testing.assert_allclose(out_a, factor_a * test_input)
        np.testing.assert_allclose(out_b, factor_b * test_input)

        # Concurrent inference with barrier synchronization.
        barrier = threading.Barrier(num_threads)
        errors = []

        def worker(session, expected_factor, worker_id):
            inp = np.ones(size, dtype=np.float32) * (worker_id + 1)
            expected = inp * expected_factor
            for i in range(iterations):
                try:
                    barrier.wait(timeout=5)
                    result = session.run(None, {"A": inp})[0]
                    if not np.allclose(result, expected):
                        errors.append(
                            f"Worker {worker_id} iter {i}: "
                            f"expected sum={expected.sum():.1f}, "
                            f"got sum={result.sum():.1f}"
                        )
                        return
                except threading.BrokenBarrierError:
                    errors.append(f"Worker {worker_id} iter {i}: barrier broken")
                    return
                except Exception as e:
                    errors.append(f"Worker {worker_id} iter {i}: {e}")
                    return

        threads = []
        for i in range(num_threads):
            factor = factor_a if i % 2 == 0 else factor_b
            session = session_a if i % 2 == 0 else session_b
            t = threading.Thread(target=worker, args=(session, factor, i))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        hung = [i for i, t in enumerate(threads) if t.is_alive()]
        if hung:
            errors.append(f"Threads {hung} still alive after 60s timeout")

        assert not errors, f"{len(errors)} thread error(s):\n" + "\n".join(errors[:10])
    finally:
        pathlib.Path(model_a_path).unlink()
        pathlib.Path(model_b_path).unlink()


def test_threaded_inference_cpu(iree_device):
    """Threaded correctness test on CPU (local-task) — runs in CI."""
    _run_threaded_inference(
        iree_device, {"target_arch": "host"}, iterations=200, num_threads=8
    )


@pytest.mark.gpu
def test_threaded_inference_gpu(iree_gpu_device, gpu_target):
    """Threaded correctness test on GPU — requires --gpu flag."""
    _run_threaded_inference(
        iree_gpu_device, {"target_arch": gpu_target}, iterations=200, num_threads=8
    )
