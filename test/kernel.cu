// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Minimal HIP kernel for validating hal.dispatch.extern mixed with torch ONNX
// dialect in a single IREE module.
//
// This kernel computes element-wise sigmoid: output[i] = 1 / (1 + exp(-input[i]))
// It serves as the simplest possible extern dispatch to validate the approach
// before building a fused MoE top-k routing kernel.
//
// ABI contract (must match the MLIR layout declaration):
//   Arguments: ([ordered bindings...], [push constants...])
//   Bindings:
//     binding0 (ReadOnly)  - input  float[N]
//     binding1             - output float[N]
//   Push constants:
//     N (i32) - number of elements
//
// NOTE: kernels must be exported with C naming (no C++ mangling).
// NOTE: all push constants must be i32.
// NOTE: IREE guarantees no aliasing so __restrict__ is safe.

#include <hip/hip_runtime.h>

extern "C" __global__ void sigmoid_extern(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    output[tid] = 1.0f / (1.0f + expf(-input[tid]));
  }
}

// Multi-input multi-output kernel: element-wise add and multiply.
//
// ABI contract (must match the MLIR layout declaration):
//   Bindings:
//     binding0 (ReadOnly)  - a     float[N]
//     binding1 (ReadOnly)  - b     float[N]
//     binding2             - sum   float[N]  (a + b)
//     binding3             - prod  float[N]  (a * b)
//   Push constants:
//     N (i32) - number of elements
extern "C" __global__ void add_mul_extern(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ sum,
    float* __restrict__ prod,
    int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    sum[tid] = a[tid] + b[tid];
    prod[tid] = a[tid] * b[tid];
  }
}
