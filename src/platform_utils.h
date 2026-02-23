//===- platform_utils.h ---------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Cross-platform utilities: environment variable access and self-library path
// resolution. Used for compiler discovery logic.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXRUNTIME_EP_IREE_SRC_PLATFORM_UTILS_H_
#define ONNXRUNTIME_EP_IREE_SRC_PLATFORM_UTILS_H_

#include <cstdlib>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace onnxruntime::iree {

// Get an environment variable value. Returns empty string if not set.
inline std::string GetEnv(const char* name) {
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

// Get the filesystem path of the shared library containing this function.
// Returns empty string on failure. Uses dladdr (POSIX) or GetModuleFileName
// (Windows) on the function's own address to locate the enclosing library.
inline std::string GetSelfLibraryPath() {
#if defined(_WIN32)
  HMODULE hm = NULL;
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(&GetSelfLibraryPath), &hm)) {
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

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_PLATFORM_UTILS_H_
