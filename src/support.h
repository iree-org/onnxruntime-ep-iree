//===- support.h ----------------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ErrorOr<T>: a type representing either a value T or an error message.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXRUNTIME_EP_IREE_SRC_SUPPORT_H_
#define ONNXRUNTIME_EP_IREE_SRC_SUPPORT_H_

#include <concepts>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <type_traits>
#include <variant>

namespace onnxruntime::iree {

// Holds an error message. Always represents a failure.
struct [[nodiscard]] ErrorObject {
  std::string message;
};

// Represents either a value T or an ErrorObject.
//
// On success, provides pointer/optional<T>-like semantics to the underlying T.
// On failure, the ErrorObject is accessible via getError().
//
// Usage:
//   ErrorOr<std::string> format() {
//     if (bad) return error("something went wrong");
//     return result;
//   }
//
//   OrtStatus* caller() {
//     IREE_ORT_ASSIGN_OR_RETURN(std::string s, format());
//     // use s ...
//   }
template <typename T>
class [[nodiscard]] ErrorOr {
 public:
  // Construct successful case from anything T is constructible from.
  template <typename U>
    requires std::constructible_from<T, U&&>
  ErrorOr(U&& val) : storage_(std::in_place_type<T>, std::forward<U>(val)) {}

  // Move constructor.
  ErrorOr(ErrorOr&& other) noexcept : storage_(std::move(other.storage_)) {}

  // Move constructor for compatible types (e.g. ErrorOr<const char*> →
  // ErrorOr<std::string>).
  template <typename U>
    requires std::is_constructible_v<T, U>
  ErrorOr(ErrorOr<U>&& other)
      : storage_(
            other.hasValue()
                ? Storage(std::in_place_type<T>, std::move(*other))
                : Storage(std::in_place_type<ErrorObject>,
                          std::move(std::get<ErrorObject>(other.storage_)))) {}

  // Construct from ErrorObject to support returning error(...).
  ErrorOr(ErrorObject err)
      : storage_(std::in_place_type<ErrorObject>, std::move(err)) {}

  // No copies or assignment.
  ErrorOr(const ErrorOr&) = delete;
  ErrorOr& operator=(const ErrorOr&) = delete;
  ErrorOr& operator=(ErrorOr&&) = delete;
  template <typename U>
  ErrorOr& operator=(const ErrorOr<U>&) = delete;
  template <typename U>
  ErrorOr& operator=(ErrorOr<U>&&) = delete;

  // Dereference – must only call when isOk().
  T& operator*() {
    assertHasValue();
    return std::get<T>(storage_);
  }
  const T& operator*() const {
    assertHasValue();
    return std::get<T>(storage_);
  }

  // Member access – must only call when isOk().
  T* operator->() {
    assertHasValue();
    return &std::get<T>(storage_);
  }
  const T* operator->() const {
    assertHasValue();
    return &std::get<T>(storage_);
  }

  // Returns the contained ErrorObject – must only call when isError().
  const ErrorObject& getError() const {
    return std::get<ErrorObject>(storage_);
  }

 private:
  using Storage = std::variant<T, ErrorObject>;
  Storage storage_;

  bool hasValue() const { return std::holds_alternative<T>(storage_); }

  void assertHasValue() const {
    if (!hasValue()) {
      std::cerr << "ErrorOr<T> is in error state and cannot be dereferenced. "
                   "Check isOk() before dereferencing.\n";
      std::abort();
    }
  }

  template <typename U>
  friend class ErrorOr;

  template <typename U>
  friend bool isOk(const ErrorOr<U>&);
  template <typename U>
  friend bool isError(const ErrorOr<U>&);
};

template <typename T>
inline bool isOk(const ErrorOr<T>& e) {
  return e.hasValue();
}

template <typename T>
inline bool isError(const ErrorOr<T>& e) {
  return !e.hasValue();
}

// Factory: create an ErrorObject. Accepts std::format arguments directly.
// Usage: return error("bad attribute '{}': {}", name, reason);
template <typename... Args>
inline ErrorObject error(std::format_string<Args...> fmt, Args&&... args) {
  return ErrorObject{std::format(fmt, std::forward<Args>(args)...)};
}

}  // namespace onnxruntime::iree

// Helper for macro token concatenation.
#define IREE_EP_CONCAT_IMPL(a, b) a##b
#define IREE_EP_CONCAT(a, b) IREE_EP_CONCAT_IMPL(a, b)

// Evaluates `expr` (which must return ErrorOr<T>), propagates the ErrorObject
// if in error state, otherwise binds the value to `varDecl`.
//
// For use inside ErrorOr<T>-returning functions. For OrtStatus*-returning
// functions use IREE_ORT_ASSIGN_OR_RETURN instead (in iree_ort_utils.h).
//
// Usage (in a function returning ErrorOr<U>):
//   IREE_EP_ASSIGN_OR_RETURN(std::string s, FormatAttribute(attr));
#define IREE_EP_ASSIGN_OR_RETURN_IMPL(tmp, varDecl, expr) \
  auto tmp = (expr);                                      \
  if (isError(tmp)) return (tmp).getError();              \
  varDecl = std::move(*tmp)

#define IREE_EP_ASSIGN_OR_RETURN(varDecl, expr)                             \
  IREE_EP_ASSIGN_OR_RETURN_IMPL(IREE_EP_CONCAT(_iree_ep_err_or_, __LINE__), \
                                varDecl, expr)

#endif  // ONNXRUNTIME_EP_IREE_SRC_SUPPORT_H_
