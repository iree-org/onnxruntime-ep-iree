//===- temp_file.h --------------------------------------------------------===//
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

#ifndef ONNXRUNTIME_EP_IREE_SRC_TEMP_FILE_H_
#define ONNXRUNTIME_EP_IREE_SRC_TEMP_FILE_H_

#include <string>
#include <string_view>

namespace onnxruntime::iree {

// RAII wrapper for temporary files.
//
// Creates a uniquely-named temporary file with the specified suffix.
// The file is automatically deleted when the TempFile object is destroyed.
// TempFile is move-only.
class TempFile {
 public:
  explicit TempFile(std::string_view suffix);
  ~TempFile();

  // Move-only.
  TempFile(TempFile&& other) noexcept;
  TempFile& operator=(TempFile&& other) noexcept;

  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  [[nodiscard]] const std::string& Path() const { return path_; }

  // Marks the file to be kept (not deleted) when TempFile is destroyed.
  void Keep() { keep_ = true; }

 private:
  std::string path_;
  bool keep_ = false;
};

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_TEMP_FILE_H_
