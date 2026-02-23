//===- iree_compile.cc ----------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// In-process MLIR-to-VMFB compilation via the IREE compiler C API.
// The compiler shared library is loaded dynamically at runtime.
//
//===----------------------------------------------------------------------===//

#include "iree_compile.h"

#include <filesystem>
#include <format>
#include <mutex>
#include <string>
#include <vector>

#include "iree/compiler/embedding_api.h"
#include "iree/compiler/loader.h"
#include "platform_utils.h"

namespace onnxruntime::iree {

// RAII scope guard for arbitrary cleanup.
template <typename F>
struct ScopeExit {
  F cleanup;
  explicit ScopeExit(F f) : cleanup(std::move(f)) {}
  ~ScopeExit() { cleanup(); }
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;
};
template <typename F>
ScopeExit<F> MakeScopeExit(F f) {
  return ScopeExit<F>(std::move(f));
}

// Compiler state token — existence means the compiler is loaded and
// initialized. Destructor calls ireeCompilerGlobalShutdown() at EP DSO unload
// (when the process-global unique_ptr is destroyed).
//
// Ordering guarantee: libIREECompiler.so is guaranteed to still be loaded
// when this destructor runs because the IREE loader (loader.cpp) intentionally
// never calls dlclose — the handle is leaked and libIREECompiler.so persists
// until process exit.
//
// Unloading and reloading the EP in the same process is not supported — the
// IREE final shutdown permanently disables the compiler for the process.
struct CompilerState {
  ~CompilerState() { ireeCompilerGlobalShutdown(); }
};

static std::mutex s_mutex;                      // Protects s_state.
static std::unique_ptr<CompilerState> s_state;  // Non-null iff initialized.

// Platform-specific compiler library names.
static constexpr const char* kCompilerLibNames[] = {
#if defined(__APPLE__)
    "libIREECompiler.dylib",
#elif defined(_WIN32)
    "IREECompiler.dll",
#else
    "libIREECompiler.so",
#endif
};

// Try to find the compiler library relative to our own shared library.
// For pip installs, both the EP and the compiler are under site-packages:
//   site-packages/onnxruntime_ep_iree/libonnxruntime_ep_iree.so
//   site-packages/iree/compiler/_mlir_libs/libIREECompiler.so
//
// For editable installs, the EP is in a build/ directory and the compiler
// is in the venv's site-packages. We walk up looking for iree/compiler/.
static std::string FindCompilerLibRelativeToSelf() {
  std::string self_lib = GetSelfLibraryPath();
  if (self_lib.empty()) {
    return {};
  }

  std::filesystem::path self_path(self_lib);
  // Walk up to find a directory that contains iree/compiler/_mlir_libs/.
  // Start from our parent dir and go up a few levels.
  for (auto dir = self_path.parent_path(); dir.has_parent_path();
       dir = dir.parent_path()) {
    for (const char* name : kCompilerLibNames) {
      auto candidate = dir / "iree" / "compiler" / "_mlir_libs" / name;
      if (std::filesystem::exists(candidate)) {
        return candidate.string();
      }
    }
    // Don't walk above the filesystem root or too many levels.
    if (dir == dir.parent_path()) break;
  }
  return {};
}

// Search a directory for a compiler library file. Returns full path or empty.
static std::string FindLibInDir(const std::filesystem::path& dir) {
  for (const char* name : kCompilerLibNames) {
    auto candidate = dir / name;
    if (std::filesystem::exists(candidate)) {
      return candidate.string();
    }
  }
  return {};
}

// Resolve the compiler library path using the discovery chain.
// Returns the resolved path and appends all attempted locations to `attempts`
// for diagnostic purposes.
static std::string ResolveCompilerLibPath(const std::string& explicit_path,
                                          std::vector<std::string>& attempts) {
  // 1. Explicit path from caller (session option ep.iree.compiler_lib_path).
  if (!explicit_path.empty()) {
    attempts.push_back(std::format("session option: {}", explicit_path));
    return explicit_path;
  }

  // 2. Runtime environment variable override.
  //    IREE_EP_COMPILER_LIB — full path to the library file.
  //    IREE_EP_COMPILER_LIB_DIR — directory to search.
  std::string env_lib = GetEnv("IREE_EP_COMPILER_LIB");
  if (!env_lib.empty()) {
    attempts.push_back(std::format("env IREE_EP_COMPILER_LIB: {}", env_lib));
    return env_lib;
  }
  std::string env_dir = GetEnv("IREE_EP_COMPILER_LIB_DIR");
  if (!env_dir.empty()) {
    std::string found = FindLibInDir(env_dir);
    attempts.push_back(
        std::format("env IREE_EP_COMPILER_LIB_DIR: {}", env_dir));
    if (!found.empty()) return found;
  }

  // 3. Search relative to our own .so (pip-installed iree-base-compiler).
  std::string relative = FindCompilerLibRelativeToSelf();
  if (!relative.empty()) {
    attempts.push_back(std::format("relative to EP library: {}", relative));
    return relative;
  }
  attempts.push_back("relative to EP library: (not found)");

  // 4. Bare library name — rely on standard dlopen search paths.
  attempts.push_back(std::format("dlopen fallback: {}", kCompilerLibNames[0]));
  return kCompilerLibNames[0];
}

/*static*/
OrtStatus* IreeCompiler::Initialize(const std::string& library_path) {
  std::lock_guard<std::mutex> lock(s_mutex);
  if (s_state) return nullptr;

  std::vector<std::string> attempts;
  std::string resolved = ResolveCompilerLibPath(library_path, attempts);

  if (!ireeCompilerLoadLibrary(resolved.c_str())) {
    // Do not cache this error — no IREE global state was set up, so the
    // caller may retry with a different path (e.g. a corrected session option).
    std::string msg = std::format(
        "IREE EP: Failed to load compiler library '{}'. Search order:\n",
        resolved);
    for (size_t i = 0; i < attempts.size(); ++i) {
      msg += std::format("  {}. {}\n", i + 1, attempts[i]);
    }
    msg +=
        "Ensure iree-base-compiler is installed or set "
        "IREE_EP_COMPILER_LIB_DIR / ep.iree.compiler_lib_path.";
    return Ort::Status(msg.c_str(), ORT_FAIL).release();
  }

  ireeCompilerGlobalInitialize();
  s_state = std::make_unique<CompilerState>();
  return nullptr;
}

/*static*/
bool IreeCompiler::IsInitialized() {
  std::lock_guard<std::mutex> lock(s_mutex);
  return s_state != nullptr;
}

/*static*/
OrtStatus* IreeCompiler::CompileToVmfb(const std::string& mlir_path,
                                       const std::string& vmfb_path,
                                       const std::vector<std::string>& flags) {
  {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (!s_state) {
      return Ort::Status(
                 "IREE EP: Compiler not initialized. Call "
                 "IreeCompiler::Initialize() first.",
                 ORT_FAIL)
          .release();
    }
  }

  iree_compiler_session_t* session = nullptr;
  iree_compiler_invocation_t* inv = nullptr;
  iree_compiler_output_t* output = nullptr;
  auto on_exit = MakeScopeExit([&]() {
    if (output) ireeCompilerOutputDestroy(output);
    if (inv) ireeCompilerInvocationDestroy(inv);
    if (session) ireeCompilerSessionDestroy(session);
  });

  // 1. Create a compiler session.
  session = ireeCompilerSessionCreate();
  if (!session) {
    return Ort::Status("IREE EP: Failed to create compiler session.", ORT_FAIL)
        .release();
  }

  // 2. Set compilation flags on the session.
  if (!flags.empty()) {
    std::vector<const char*> argv;
    argv.reserve(flags.size());
    for (const auto& f : flags) {
      argv.push_back(f.c_str());
    }
    iree_compiler_error_t* err = ireeCompilerSessionSetFlags(
        session, static_cast<int>(argv.size()), argv.data());
    if (err) {
      std::string msg = std::format("IREE EP: Failed to set compiler flags: {}",
                                    ireeCompilerErrorGetMessage(err));
      ireeCompilerErrorDestroy(err);
      return Ort::Status(msg.c_str(), ORT_FAIL).release();
    }
  }

  // 3. Create an invocation and enable diagnostics.
  inv = ireeCompilerInvocationCreate(session);
  if (!inv) {
    return Ort::Status("IREE EP: Failed to create compiler invocation (OOM?).",
                       ORT_FAIL)
        .release();
  }
  ireeCompilerInvocationEnableConsoleDiagnostics(inv);

  // 4. Open the MLIR source file.
  iree_compiler_source_t* source = nullptr;
  iree_compiler_error_t* err =
      ireeCompilerSourceOpenFile(session, mlir_path.c_str(), &source);
  if (err) {
    std::string msg =
        std::format("IREE EP: Failed to open MLIR source '{}': {}", mlir_path,
                    ireeCompilerErrorGetMessage(err));
    ireeCompilerErrorDestroy(err);
    return Ort::Status(msg.c_str(), ORT_FAIL).release();
  }

  // 5. Parse the MLIR source. On success, inv takes ownership of source and
  // destroys it when inv is destroyed (via on_exit). On failure, source is not
  // consumed and must be destroyed explicitly.
  if (!ireeCompilerInvocationParseSource(inv, source)) {
    ireeCompilerSourceDestroy(source);
    return Ort::Status(std::format("IREE EP: Failed to parse MLIR source '{}'",
                                   mlir_path)
                           .c_str(),
                       ORT_FAIL)
        .release();
  }

  // 6. Run the standard compilation pipeline.
  if (!ireeCompilerInvocationPipeline(inv, IREE_COMPILER_PIPELINE_STD)) {
    return Ort::Status(
               "IREE EP: Compilation pipeline failed. "
               "Check console diagnostics for details.",
               ORT_FAIL)
        .release();
  }

  // 7. Write the compiled VMFB output to file.
  err = ireeCompilerOutputOpenFile(vmfb_path.c_str(), &output);
  if (err) {
    std::string msg =
        std::format("IREE EP: Failed to open output file '{}': {}", vmfb_path,
                    ireeCompilerErrorGetMessage(err));
    ireeCompilerErrorDestroy(err);
    return Ort::Status(msg.c_str(), ORT_FAIL).release();
  }

  err = ireeCompilerInvocationOutputVMBytecode(inv, output);
  if (err) {
    std::string msg = std::format("IREE EP: Failed to emit VM bytecode: {}",
                                  ireeCompilerErrorGetMessage(err));
    ireeCompilerErrorDestroy(err);
    return Ort::Status(msg.c_str(), ORT_FAIL).release();
  }

  // 8. Keep the output file (without this, it's deleted on destroy).
  ireeCompilerOutputKeep(output);

  // on_exit destroys output (file kept), inv, and session.
  return nullptr;
}

}  // namespace onnxruntime::iree
