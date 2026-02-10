//===- temp_file.cc -------------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RAII wrapper for temporary files.
//
//===----------------------------------------------------------------------===//

#include "temp_file.h"

#include <filesystem>
#include <fstream>
#include <random>

namespace onnxruntime::iree {

TempFile::TempFile(std::string_view suffix) {
  // Generate random hex string for unique naming.
  std::random_device rd;
  uint64_t random_value = (static_cast<uint64_t>(rd()) << 32) | rd();

  // Build path: temp_dir/iree_ep_{random}{suffix}
  std::filesystem::path path = std::filesystem::temp_directory_path();
  path /= "iree_ep_" + std::to_string(random_value) + std::string(suffix);
  path_ = path.string();

  // Create empty file to reserve the path.
  std::ofstream file(path_);
}

TempFile::~TempFile() {
  if (!path_.empty() && !keep_) {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
}

TempFile::TempFile(TempFile&& other) noexcept
    : path_(std::move(other.path_)), keep_(other.keep_) {
  other.path_.clear();
  other.keep_ = false;
}

TempFile& TempFile::operator=(TempFile&& other) noexcept {
  if (this != &other) {
    // Clean up current file if not marked to keep.
    if (!path_.empty() && !keep_) {
      std::error_code ec;
      std::filesystem::remove(path_, ec);
    }
    path_ = std::move(other.path_);
    keep_ = other.keep_;
    other.path_.clear();
    other.keep_ = false;
  }
  return *this;
}

}  // namespace onnxruntime::iree
