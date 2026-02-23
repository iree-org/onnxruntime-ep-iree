//===- iree_compile.h -----------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Provides in-process MLIR-to-VMFB compilation via the IREE compiler C API.
// The compiler shared library (libIREECompiler.so) is loaded dynamically at
// runtime — no compile-time linking to the compiler is required.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXRUNTIME_EP_IREE_SRC_IREE_COMPILE_H_
#define ONNXRUNTIME_EP_IREE_SRC_IREE_COMPILE_H_

#include <string>
#include <vector>

#include "ort_import.h"

namespace onnxruntime::iree {

// Manages the IREE compiler lifecycle and provides compilation.
//
// The compiler is a process-global singleton — Initialize() loads the shared
// library once for the lifetime of the process. All state is protected by an
// internal mutex.
//
// Thread safety:
//   Initialize()    — Thread-safe. Multiple threads may call concurrently;
//     exactly one performs initialization. Load failures are not cached —
//     the caller may retry with a different path.
//   IsInitialized() — Thread-safe.
//   CompileToVmfb() — Thread-safe (each call creates its own compiler
//     session/invocation). Requires successful Initialize() first.
//
// Lifetime: ireeCompilerGlobalShutdown() is called automatically via RAII when
// the process-global CompilerState is destroyed at EP DSO unload. This is safe
// because the IREE loader intentionally never calls dlclose, so libIREECompiler
// .so persists until process exit regardless of EP DSO lifetime. The IREE C
// API's final shutdown permanently disables the compiler for the process, so
// unloading and reloading the EP plugin in the same process is not supported.
class IreeCompiler {
 public:
  // Load the compiler shared library and initialize global state.
  //
  // library_path: absolute path to libIREECompiler.so/.dylib/IREECompiler.dll.
  //   If empty, auto-discovers the library using this search order:
  //     1. Session option ep.iree.compiler_lib_path (passed as library_path)
  //     2. Environment variable IREE_EP_COMPILER_LIB (full path to library)
  //     3. Environment variable IREE_EP_COMPILER_LIB_DIR (directory to search)
  //     4. pip-installed iree-base-compiler (relative to EP shared library)
  //     5. Bare dlopen("libIREECompiler.so") via standard search paths
  //
  // Thread-safe. First successful call wins; subsequent calls are no-ops.
  // On failure, returns an error and allows the caller to retry with a
  // different path (no IREE global state is set up on load failure).
  static OrtStatus* Initialize(const std::string& library_path);

  // Returns true if Initialize() has been called successfully.
  static bool IsInitialized();

  // Compile an MLIR file to VMFB bytecode using the IREE compiler C API.
  //
  // Args:
  //   mlir_path: Path to the input MLIR file.
  //   vmfb_path: Path where the output VMFB should be written.
  //   flags: Compiler flags (e.g., ["--iree-hal-target-device=hip",
  //          "--iree-hip-target=gfx1100"]).
  //
  // Returns nullptr on success, OrtStatus* with error message on failure.
  // Requires successful Initialize() first.
  static OrtStatus* CompileToVmfb(const std::string& mlir_path,
                                  const std::string& vmfb_path,
                                  const std::vector<std::string>& flags);
};

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_IREE_COMPILE_H_
