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

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdlib>
#include <filesystem>
#include <format>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "iree/compiler/embedding_api.h"
#include "iree/compiler/loader.h"

namespace onnxruntime::iree {

static std::once_flag s_init_flag;
static bool s_initialized = false;

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

// Get the filesystem path of the shared library containing this function.
// Returns empty string on failure.
static std::string GetSelfLibraryPath() {
#if defined(_WIN32)
  HMODULE hm = NULL;
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(&GetSelfLibraryPath),
                          &hm)) {
    return {};
  }
  char path[MAX_PATH];
  DWORD len = GetModuleFileNameA(hm, path, sizeof(path));
  if (len == 0 || len >= sizeof(path)) {
    return {};
  }
  return std::string(path, len);
#else
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void*>(&GetSelfLibraryPath), &dl_info) == 0 ||
      dl_info.dli_fname == nullptr) {
    return {};
  }
  return dl_info.dli_fname;
#endif
}

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

// Get an environment variable value. Returns empty string if not set.
static std::string GetEnv(const char* name) {
#if defined(_WIN32)
  // MSVC deprecates getenv; use _dupenv_s.
  char* value = nullptr;
  size_t len = 0;
  if (_dupenv_s(&value, &len, name) == 0 && value != nullptr) {
    std::string result(value);
    free(value);
    return result;
  }
  return {};
#else
  const char* value = std::getenv(name);
  return value ? std::string(value) : std::string{};
#endif
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
// Returns the resolved path and appends all attempted locations to |attempts|
// for diagnostic purposes.
static std::string ResolveCompilerLibPath(
    const std::string& explicit_path, std::vector<std::string>& attempts) {
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
    attempts.push_back(std::format("env IREE_EP_COMPILER_LIB_DIR: {}", env_dir));
    if (!found.empty()) return found;
  }

  // 3. Build-time configured directory.
#ifdef IREE_COMPILER_LIB_DIR
  {
    std::filesystem::path dir(IREE_COMPILER_LIB_DIR);
    attempts.push_back(std::format("build-time IREE_COMPILER_LIB_DIR: {}",
                                   dir.string()));
    std::string found = FindLibInDir(dir);
    if (!found.empty()) return found;
  }
#endif

  // 4. Search relative to our own .so (pip-installed iree-base-compiler).
  std::string relative = FindCompilerLibRelativeToSelf();
  if (!relative.empty()) {
    attempts.push_back(std::format("relative to EP library: {}", relative));
    return relative;
  }
  attempts.push_back("relative to EP library: (not found)");

  // 5. Bare library name — rely on standard dlopen search paths.
  attempts.push_back(
      std::format("dlopen fallback: {}", kCompilerLibNames[0]));
  return kCompilerLibNames[0];
}

// Actual initialization logic, called exactly once via std::call_once.
// IMPORTANT: Throws on failure so std::call_once does NOT consume the flag,
// allowing a subsequent call (e.g. with a corrected path) to retry.
static void DoInitialize(const std::string& library_path) {
  std::vector<std::string> attempts;
  std::string resolved = ResolveCompilerLibPath(library_path, attempts);

  if (!ireeCompilerLoadLibrary(resolved.c_str())) {
    std::string msg = std::format(
        "IREE EP: Failed to load compiler library '{}'. Search order:\n",
        resolved);
    for (size_t i = 0; i < attempts.size(); ++i) {
      msg += std::format("  {}. {}\n", i + 1, attempts[i]);
    }
    msg += "Ensure iree-base-compiler is installed or set "
           "IREE_EP_COMPILER_LIB_DIR / ep.iree.compiler_lib_path.";
    throw std::runtime_error(msg);
  }

  ireeCompilerGlobalInitialize();
  s_initialized = true;
}

/*static*/
OrtStatus* IreeCompiler::Initialize(const std::string& library_path) {
  if (s_initialized) return nullptr;

  try {
    std::call_once(s_init_flag, DoInitialize, library_path);
  } catch (const std::runtime_error& e) {
    return Ort::Status(e.what(), ORT_FAIL).release();
  }
  return nullptr;
}

/*static*/
void IreeCompiler::Shutdown() {
  if (!s_initialized) return;
  ireeCompilerGlobalShutdown();
  s_initialized = false;
  // Note: After shutdown, the compiler cannot be re-initialized in the same
  // process (the IREE compiler C API does not support it). Only call this
  // at process exit for explicit cleanup.
}

/*static*/
bool IreeCompiler::IsInitialized() { return s_initialized; }

/*static*/
OrtStatus* IreeCompiler::CompileToVmfb(
    const std::string& mlir_path, const std::string& vmfb_path,
    const std::vector<std::string>& flags) {
  if (!s_initialized) {
    return Ort::Status(
               "IREE EP: Compiler not initialized. Call "
               "IreeCompiler::Initialize() first.",
               ORT_FAIL)
        .release();
  }

  // 1. Create a compiler session.
  iree_compiler_session_t* session = ireeCompilerSessionCreate();
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
      std::string msg =
          std::format("IREE EP: Failed to set compiler flags: {}",
                      ireeCompilerErrorGetMessage(err));
      ireeCompilerErrorDestroy(err);
      ireeCompilerSessionDestroy(session);
      return Ort::Status(msg.c_str(), ORT_FAIL).release();
    }
  }

  // 3. Create an invocation and enable diagnostics.
  iree_compiler_invocation_t* inv = ireeCompilerInvocationCreate(session);
  if (!inv) {
    ireeCompilerSessionDestroy(session);
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
    std::string msg = std::format("IREE EP: Failed to open MLIR source '{}': {}",
                                  mlir_path, ireeCompilerErrorGetMessage(err));
    ireeCompilerErrorDestroy(err);
    ireeCompilerInvocationDestroy(inv);
    ireeCompilerSessionDestroy(session);
    return Ort::Status(msg.c_str(), ORT_FAIL).release();
  }

  // 5. Parse the MLIR source.
  if (!ireeCompilerInvocationParseSource(inv, source)) {
    ireeCompilerSourceDestroy(source);
    ireeCompilerInvocationDestroy(inv);
    ireeCompilerSessionDestroy(session);
    return Ort::Status(
               std::format("IREE EP: Failed to parse MLIR source '{}'",
                           mlir_path)
                   .c_str(),
               ORT_FAIL)
        .release();
  }
  // Source is consumed by parse — do not destroy it separately.

  // 6. Run the standard compilation pipeline.
  if (!ireeCompilerInvocationPipeline(inv, IREE_COMPILER_PIPELINE_STD)) {
    ireeCompilerInvocationDestroy(inv);
    ireeCompilerSessionDestroy(session);
    return Ort::Status("IREE EP: Compilation pipeline failed. "
                       "Check console diagnostics for details.",
                       ORT_FAIL)
        .release();
  }

  // 7. Write the compiled VMFB output to file.
  iree_compiler_output_t* output = nullptr;
  err = ireeCompilerOutputOpenFile(vmfb_path.c_str(), &output);
  if (err) {
    std::string msg =
        std::format("IREE EP: Failed to open output file '{}': {}", vmfb_path,
                    ireeCompilerErrorGetMessage(err));
    ireeCompilerErrorDestroy(err);
    ireeCompilerInvocationDestroy(inv);
    ireeCompilerSessionDestroy(session);
    return Ort::Status(msg.c_str(), ORT_FAIL).release();
  }

  err = ireeCompilerInvocationOutputVMBytecode(inv, output);
  if (err) {
    std::string msg =
        std::format("IREE EP: Failed to emit VM bytecode: {}",
                    ireeCompilerErrorGetMessage(err));
    ireeCompilerErrorDestroy(err);
    ireeCompilerOutputDestroy(output);
    ireeCompilerInvocationDestroy(inv);
    ireeCompilerSessionDestroy(session);
    return Ort::Status(msg.c_str(), ORT_FAIL).release();
  }

  // 8. Keep the output file (without this, it's deleted on destroy).
  ireeCompilerOutputKeep(output);

  // 9. Cleanup.
  ireeCompilerOutputDestroy(output);
  ireeCompilerInvocationDestroy(inv);
  ireeCompilerSessionDestroy(session);

  return nullptr;
}

}  // namespace onnxruntime::iree
