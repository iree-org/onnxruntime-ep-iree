# C++ Coding Style Guide

## Overview

This document defines the C++ coding standards for the IREE ONNX Execution Provider project. These standards are based on the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with specific adaptations for this project.

The primary goals of this style guide are:
- **Readability**: Code is read far more often than it is written
- **Consistency**: Uniform style across the codebase reduces cognitive load
- **Maintainability**: Clear, well-organized code is easier to modify and debug
- **Avoiding Surprises**: Code should behave as readers expect

## C++ Version

- **Target**: C++20
- **Avoid**: non-standard compiler extensions

## Header Files

### Self-Contained Headers

All header files must be self-contained and compile independently. Use the `.h` extension for all headers.

**Include Guards**: Use the format `_<PROJECT>___<PATH>___<FILE>__H_`

```cpp
// For file: onnxruntime_ep_iree/runtime/session.h
#ifndef ONNXRUNTIME_EP_IREE_RUNTIME_SESSION_H_
#define ONNXRUNTIME_EP_IREE_RUNTIME_SESSION_H_

// ... header content ...

#endif  // ONNXRUNTIME_EP_IREE_RUNTIME_SESSION_H_
```

### Inline Functions

Only define short functions (≤10 lines) inline at their public declaration point. Longer function definitions belong in `.cc` files or implementation detail sections.

```cpp
// Good: Short, simple inline function
class Tensor {
 public:
  size_t Size() const { return size_; }

 private:
  size_t size_;
};

// Bad: Complex logic should be in .cc file
class Session {
 public:
  Status Initialize(const Config& config) {
    // 20 lines of initialization logic...
  }
};
```

## Scoping

### Namespaces

Place all code in namespaces with unique, project-based names.

**Project namespace**: `onnxruntime::iree`

```cpp
namespace onnxruntime::iree {

class Session {
  // No indentation inside namespace
};

}  // namespace onnxruntime::iree
```

**Nested namespaces**:

```cpp
namespace onnxruntime::iree {
namespace runtime {

class Executor {
  // Implementation
};

}  // namespace runtime
}  // namespace onnxruntime::iree
```

**Prohibited**:
- `using namespace` directives
- Inline namespaces
- Declaring anything in `std::` namespace

### Internal Linkage

For code used only within a `.cc` file, use unnamed namespaces or `static`:

```cpp
// session.cc
namespace {

constexpr int kDefaultTimeout = 1000;

Status ValidateConfig(const Config& config) {
  // Internal helper function
}

}  // namespace

namespace onnxruntime::iree {

Status Session::Initialize(const Config& config) {
  return ValidateConfig(config);
}

}  // namespace onnxruntime::iree
```

### Local Variables

- Declare in the narrowest scope possible
- Initialize at declaration
- Declare close to first use

```cpp
// Good
int ComputeHash(const std::string& input) {
  int hash = 0;
  for (char c : input) {
    hash = hash * 31 + c;
  }
  return hash;
}

// Bad: Declaration far from use
int ComputeHash(const std::string& input) {
  int hash;
  int multiplier;
  // ... many lines ...
  hash = 0;
  multiplier = 31;
  // ...
}
```

### Static and Global Variables

Objects with static storage duration are **forbidden unless trivially destructible**.

**Rule of thumb**: A global variable satisfies requirements if its declaration could be `constexpr`.

```cpp
// Good: Trivially destructible
constexpr int kMaxBatchSize = 128;
constexpr std::string_view kDefaultDevice = "cpu";

// Bad: Non-trivial destructor
static std::string kDefaultDevice = "cpu";  // FORBIDDEN
static std::unique_ptr<Session> kSession;   // FORBIDDEN

// Good alternative for function-local static
const Session& GetDefaultSession() {
  static Session* session = new Session();  // Never destroyed, but acceptable
  return *session;
}
```

Use `constexpr` or `constinit` for compile-time constants.

## Classes

### Constructors

- Avoid complex work in constructors
- Never call virtual methods from constructors
- For initialization that can fail, use factory functions or `Init()` methods

```cpp
// Good: Factory function for fallible initialization
class Session {
 public:
  static absl::StatusOr<std::unique_ptr<Session>> Create(
      const Config& config) {
    auto session = std::make_unique<Session>();
    auto status = session->Initialize(config);
    if (!status.ok()) {
      return status;
    }
    return session;
  }

 private:
  Session() = default;
  Status Initialize(const Config& config);
};

// Bad: Complex initialization in constructor
class Session {
 public:
  Session(const Config& config) {
    // What if this fails?
    CompileModel(config.model_path());
  }
};
```

### Implicit Conversions

Mark conversion operators and single-argument constructors `explicit` unless specifically designing for implicit conversion.

```cpp
class Tensor {
 public:
  // Explicit: Prevents accidental conversions
  explicit Tensor(size_t size);

  // Copy/move constructors are NOT explicit
  Tensor(const Tensor&) = default;
  Tensor(Tensor&&) = default;

  // Explicit conversion operator
  explicit operator bool() const { return data_ != nullptr; }

 private:
  void* data_;
};
```

### Copyable and Movable Types

Make ownership semantics clear in the public API. Explicitly declare or delete copy/move operations.

```cpp
// Copyable type
class TensorMetadata {
 public:
  TensorMetadata(const TensorMetadata&) = default;
  TensorMetadata& operator=(const TensorMetadata&) = default;
  TensorMetadata(TensorMetadata&&) = default;
  TensorMetadata& operator=(TensorMetadata&&) = default;
};

// Move-only type
class Session {
 public:
  Session(Session&&) = default;
  Session& operator=(Session&&) = default;

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
};

// Non-copyable, non-movable
class DeviceContext {
 public:
  DeviceContext(const DeviceContext&) = delete;
  DeviceContext& operator=(const DeviceContext&) = delete;
  DeviceContext(DeviceContext&&) = delete;
  DeviceContext& operator=(DeviceContext&&) = delete;
};
```

### Structs vs. Classes

- Use `struct` for passive data with public fields and no invariants
- Use `class` for types requiring functionality or invariants

```cpp
// Good: struct for passive data
struct TensorShape {
  std::vector<int64_t> dimensions;
  DataType dtype;
};

// Good: class with invariants
class Tensor {
 public:
  Tensor(const TensorShape& shape);
  void* Data() const;

 private:
  void* data_;
  TensorShape shape_;
};
```

### Inheritance

- **Prefer composition over inheritance**
- When using inheritance, make it public
- Use pure abstract base classes for interfaces
- Limit `protected` members
- Annotate overrides with `override` or `final` (not `virtual`)

```cpp
// Good: Interface (pure abstract base class)
class IExecutor {
 public:
  virtual ~IExecutor() = default;
  virtual Status Execute(const Request& request) = 0;
};

// Good: Implementation with override annotation
class SyncExecutor : public IExecutor {
 public:
  Status Execute(const Request& request) override;
};

// Good: Final class cannot be inherited from
class IreeDevice final {
 public:
  explicit IreeDevice(iree_device_t* device);
  // ...
};
```

### Operator Overloading

Define operators only when semantics are obvious and consistent with built-in operators.

**Prohibited**:
- `operator&&`, `operator||`, `operator,`
- Unary `operator&`
- User-defined literals

```cpp
// Good: Obvious semantics
class Tensor {
 public:
  bool operator==(const Tensor& other) const;
  bool operator!=(const Tensor& other) const;
};

// Bad: Unclear semantics
class Session {
 public:
  Session operator+(const Model& model);  // What does this mean?
};
```

### Access Control

- Make data members `private` unless they are constants
- Use `const` accessors where appropriate
- Test fixtures in `.cc` files may use `protected`

```cpp
class Session {
 public:
  const std::string& Name() const { return name_; }
  void SetName(std::string name) { name_ = std::move(name); }

 private:
  std::string name_;
  std::unique_ptr<IreeContext> context_;
};
```

### Declaration Order

Within a class, group declarations as follows (omit empty sections):

1. Types and type aliases
2. Static constants
3. Factory functions
4. Constructors and assignment operators
5. Destructor
6. All other functions
7. Data members

```cpp
class Session {
 public:
  // Types
  using RequestCallback = std::function<void(const Response&)>;

  // Constants
  static constexpr int kDefaultTimeout = 1000;

  // Factory
  static std::unique_ptr<Session> Create(const Config& config);

  // Constructors
  Session();
  explicit Session(const Config& config);

  // Destructor
  ~Session();

  // Other methods
  Status Execute(const Request& request);
  void RegisterCallback(RequestCallback callback);

 private:
  // Private methods
  Status Initialize();

  // Data members
  std::unique_ptr<IreeContext> context_;
  RequestCallback callback_;
};
```

## Functions

### Parameters and Return Values

**Inputs**:
- Pass small values by value
- Pass large objects by `const` reference
- Optional inputs: `std::optional<T>` or `const T*`

**Outputs**:
- Prefer return values over output parameters
- Return by value (rely on RVO/move semantics)
- Output parameters: non-const references
- Optional outputs: non-const pointers

**Parameter ordering**: All inputs before outputs

```cpp
// Good: Clear input/output separation
Status CompileModel(
    const std::string& model_path,     // Input
    const CompilerOptions& options,    // Input
    std::vector<uint8_t>& bytecode);   // Output

// Better: Return by value
absl::StatusOr<std::vector<uint8_t>> CompileModel(
    const std::string& model_path,
    const CompilerOptions& options);

// Good: Optional input
Status LoadModel(
    const std::string& model_path,
    const CompilerOptions* options = nullptr);
```

### Function Length

Keep functions small and focused. Functions exceeding ~40 lines should be reviewed for possible decomposition.

```cpp
// Good: Short, focused function
Status ValidateShape(const TensorShape& shape) {
  if (shape.dimensions.empty()) {
    return Status(StatusCode::kInvalidArgument, "Empty shape");
  }
  for (int64_t dim : shape.dimensions) {
    if (dim <= 0) {
      return Status(StatusCode::kInvalidArgument, "Invalid dimension");
    }
  }
  return Status::OK();
}
```

### Default Arguments

Allowed on non-virtual functions when defaults never change. Banned on virtual functions.

```cpp
// Good
Status LoadModel(
    const std::string& path,
    bool optimize = true,
    int timeout_ms = 1000);

// Bad: Virtual function with default argument
class IExecutor {
 public:
  virtual Status Execute(const Request& request,
                        int timeout_ms = 1000) = 0;  // BAD
};
```

## Ownership and Smart Pointers

### Ownership Model

Prefer single, fixed ownership with explicit transfer.

**Use `std::unique_ptr` for exclusive ownership**:

```cpp
// Factory returning owned object
std::unique_ptr<Session> CreateSession(const Config& config);

// Consumer takes ownership
void ProcessSession(std::unique_ptr<Session> session);

// Class owns member
class ExecutionContext {
 private:
  std::unique_ptr<IreeDevice> device_;
  std::unique_ptr<IreeInstance> instance_;
};
```

**Use `std::shared_ptr` only with very good reason** (typically immutable shared data):

```cpp
// Acceptable: Shared immutable data
class ModelCache {
 public:
  std::shared_ptr<const CompiledModel> Get(const std::string& key);

 private:
  std::map<std::string, std::shared_ptr<const CompiledModel>> cache_;
};
```

**Never use `std::auto_ptr`** (deprecated).

### Raw Pointers

Use raw pointers for non-owning references:

```cpp
class Executor {
 public:
  // Non-owning reference (device outlives executor)
  explicit Executor(IreeDevice* device);

 private:
  IreeDevice* device_;  // Not owned
};
```

## C++ Language Features

### Rvalue References

Use rvalue references only for:
- Move constructors and move assignment operators
- `&&`-qualified methods consuming `*this`
- Perfect forwarding with `std::forward`
- Paired overloads when significantly more efficient

```cpp
// Good: Move constructor
class Tensor {
 public:
  Tensor(Tensor&& other) noexcept
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
};

// Good: Perfect forwarding
template <typename... Args>
std::unique_ptr<Session> MakeSession(Args&&... args) {
  return std::make_unique<Session>(std::forward<Args>(args)...);
}
```

### Exceptions

**Exceptions are PROHIBITED** in this project. Use alternative error handling:

- Return status codes (`Status`)
- Factory functions returning `std::optional` or `std::unique_ptr`
- Assertions for programmer errors

```cpp
// Good: StatusOr for fallible operations
absl::StatusOr<Tensor> AllocateTensor(const TensorShape& shape) {
  if (!IsValidShape(shape)) {
    return absl::InvalidArgumentError("Invalid shape");
  }
  return Tensor(shape);
}

// Good: Assertions for programmer errors
void SetData(void* data) {
  assert(data != nullptr);
  data_ = data;
}
```

### noexcept

Use `noexcept` when semantically correct and beneficial:

```cpp
// Good: Move operations should be noexcept
class Tensor {
 public:
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;

  void Swap(Tensor& other) noexcept;
};
```

### RTTI

Avoid RTTI (`dynamic_cast`, `typeid`). Prefer virtual functions or visitor patterns.

```cpp
// Bad: Using RTTI
if (auto* sync_exec = dynamic_cast<SyncExecutor*>(executor)) {
  // ...
}

// Good: Virtual function
class IExecutor {
 public:
  virtual bool IsAsync() const = 0;
};
```

### Casting

Use C++-style casts:

```cpp
// Good: Brace initialization for arithmetic
int64_t large_value = int64_t{42} << 32;

// Good: static_cast for pointer hierarchy
IExecutor* executor = static_cast<IExecutor*>(impl);

// Good: const_cast to remove const (use sparingly)
void ProcessBuffer(char* buffer) {
  // ...
}
void Caller() {
  const char* const_buffer = GetBuffer();
  ProcessBuffer(const_cast<char*>(const_buffer));
}

// Good: reinterpret_cast for unsafe conversions
void* opaque = reinterpret_cast<void*>(handle);
```

**Never use C-style casts** except casting to `void`.

### Pre/Post-increment

Use prefix form (`++i`) unless postfix semantics are specifically needed.

```cpp
// Good
for (int i = 0; i < n; ++i) {
  // ...
}

// Acceptable when postfix needed
int old_value = counter++;
```

## Naming Conventions

### File Names

- Lowercase with underscores
- C++ files: `.cc`
- Headers: `.h`

Examples: `session.h`, `session.cc`, `jit_compiler.h`, `model_executor.cc`

### Type Names

Use **UpperCamelCase** for all type names:

```cpp
class Session { };
struct TensorShape { };
enum class DataType { };
using StatusCallback = std::function<void(Status)>;
template <typename T> class Buffer { };
```

### Variable Names

Use **snake_case** (lowercase with underscores):

```cpp
std::string model_path;
int batch_size;
const TensorShape& input_shape;
```

**Class member variables**: Add trailing underscore:

```cpp
class Session {
 private:
  std::string name_;
  std::unique_ptr<IreeContext> context_;
  int timeout_ms_;
};
```

### Constant Names

Use **leading `k` + UpperCamelCase**:

```cpp
constexpr int kMaxBatchSize = 128;
constexpr std::string_view kDefaultDevice = "cpu";
const int kDefaultTimeout = 1000;
```

### Function Names

Use **UpperCamelCase**:

```cpp
void Initialize();
Status CompileModel(const std::string& path);
bool IsReady() const;
```

**Accessors and mutators**:

```cpp
class Session {
 public:
  const std::string& Name() const { return name_; }  // Accessor
  void SetName(std::string name) { name_ = std::move(name); }  // Mutator

 private:
  std::string name_;
};
```

### Namespace Names

Use **lowercase**:

```cpp
namespace onnxruntime::iree {
namespace runtime {
namespace compiler {
```

### Enumerator Names

Use **leading `k` + UpperCamelCase**:

```cpp
enum class Status {
  kOk,
  kInvalidArgument,
  kNotFound,
  kInternal
};
```

### Macro Names

Use **ALL_CAPS_WITH_UNDERSCORES**:

```cpp
#define ONNXRUNTIME_EP_IREE_VERSION_MAJOR 1
#define ONNXRUNTIME_EP_IREE_ASSERT(condition) \
  do { \
    if (!(condition)) { \
      /* ... */ \
    } \
  } while (0)
```

## Formatting

### Line Length

- **Preferred**: 80 characters
- Exceptions: URLs, long string literals

### Indentation

- **Use 2 spaces**; never tabs
- No indentation inside namespaces

```cpp
namespace onnxruntime::iree {

class Session {
 public:
  void Method() {
    if (condition) {
      DoSomething();
    }
  }
};

}  // namespace onnxruntime::iree
```

### Pointer and Reference Alignment

Attach `*` and `&` to the type:

```cpp
// Good
char* buffer;
const std::string& name;
Tensor&& temp;

// Bad
char *buffer;
const std::string &name;
```

### Braces

Always use braces for control statements, even single-line bodies:

```cpp
// Good
if (condition) {
  DoSomething();
}

// Bad
if (condition)
  DoSomething();
```

Opening brace on same line as statement:

```cpp
if (condition) {
  DoSomething();
} else if (other_condition) {
  DoSomethingElse();
} else {
  DoDefault();
}

for (int i = 0; i < n; ++i) {
  Process(i);
}

while (Running()) {
  Update();
}
```

### Class Format

```cpp
class Session {
 public:
  Session();
  explicit Session(const Config& config);
  ~Session();

  // Public methods
  Status Initialize();
  Status Execute(const Request& request);

 protected:
  // Protected methods (rare)
  virtual void OnInitialize();

 private:
  // Private methods
  void InternalHelper();

  // Data members
  std::string name_;
  std::unique_ptr<IreeContext> context_;
};
```

## Comments

Comments are always sentences and should end with a period.

### File Comments

Include a brief description at the top of each file:

```cpp
// session.h - ONNX Runtime Session implementation using IREE
//
// This file defines the Session class which manages the lifecycle of
// an ONNX model execution using IREE as the backend runtime.
```

### Class Comments

Describe the purpose and important invariants:

```cpp
// Manages compilation and execution of ONNX models using IREE.
//
// Sessions are move-only and maintain exclusive ownership of compiled
// artifacts and runtime resources. All methods are thread-safe unless
// otherwise noted.
class Session {
  // ...
};
```

### Function Comments

Document non-obvious behavior, ownership transfer, lifetime requirements:

```cpp
// Compiles the ONNX model at the given path and returns the bytecode.
//
// The model file must remain accessible during compilation. Compilation
// happens synchronously and may take several seconds for large models.
//
// Returns:
//   - OK with bytecode on success
//   - InvalidArgument if model is malformed
//   - NotFound if path doesn't exist
//   - Internal if compilation fails
absl::StatusOr<std::vector<uint8_t>> CompileModel(
    const std::string& model_path,
    const CompilerOptions& options);
```

### Implementation Comments

Explain tricky or non-obvious code:

```cpp
Status Session::Execute(const Request& request) {
  // IREE requires 32-byte alignment for input buffers
  auto aligned_buffer = AllocateAligned(request.size(), 32);

  // The semaphore must be signaled before releasing the device lock,
  // otherwise we risk a race with the async executor.
  iree_hal_semaphore_signal(semaphore_, current_value_);
  device_lock_.unlock();

  return ProcessResponse();
}
```

### TODO Comments

Mark unfinished work clearly:

```cpp
// TODO(username): Implement async execution path
// TODO(username): Add support for dynamic shapes (bug #123)
```

### Deprecation Comments

Mark deprecated code clearly:

```cpp
// DEPRECATED: Use ExecuteAsync() instead.
// This synchronous method will be removed in v2.0.
Status Execute(const Request& request);
```

## The `auto` Keyword

Use `auto` to avoid redundant or overly verbose type names:

```cpp
// Good: Type is obvious from right-hand side
auto session = CreateSession(config);
auto it = tensors.begin();
auto result = CompileModel(path);

// Bad: Type is unclear
auto value = GetValue();  // What type is this?
auto result = Process();  // What does this return?

// Good: Explicit type for clarity
std::unique_ptr<Session> session = CreateSession(config);
Status status = Validate(model);
```

Always qualify `auto` with `const`, `*`, `&`, or `&&` as appropriate:

```cpp
const auto& name = session.Name();
auto* device = GetDevice();
auto&& forwarding_ref = std::forward<T>(value);
```

## Lambda Expressions

Use lambdas for callbacks and local functions. Keep them simple:

```cpp
// Good: Simple lambda
session.ExecuteAsync([](const Response& response) {
  ProcessResponse(response);
});

// Good: Explicit return type when needed
auto comparator = [](const Tensor& a, const Tensor& b) -> bool {
  return a.Size() < b.Size();
};

// Bad: Complex lambda (extract to named function)
auto processor = [&](const Request& request) {
  // 30 lines of complex logic...
};  // Move to named function instead
```

Capture carefully:
- Prefer explicit captures over `[&]` or `[=]`
- Understand lifetime implications of reference captures

```cpp
// Good: Explicit captures
int timeout = 1000;
session.ExecuteAsync([timeout](const Response& response) {
  if (response.latency_ms() > timeout) {
    LogSlowResponse(response);
  }
});

// Dangerous: Capturing reference to local variable
std::string path = GetPath();
session.ExecuteAsync([&path]() {  // BAD: path might be destroyed
  LoadModel(path);
});
```

## Templates

Minimize template usage. Use templates for:
- Generic containers and algorithms
- Type-safe wrappers
- Performance-critical code requiring specialization

**Avoid**:
- Complex template metaprogramming
- Overly recursive templates
- Excessive specializations

Keep template definitions in headers:

```cpp
// tensor_buffer.h
template <typename T>
class TensorBuffer {
 public:
  explicit TensorBuffer(size_t size) : data_(size) {}

  T& operator[](size_t index) { return data_[index]; }
  const T& operator[](size_t index) const { return data_[index]; }

 private:
  std::vector<T> data_;
};
```

For complex templates, explicitly instantiate in `.cc` files:

```cpp
// tensor_buffer.cc
#include "tensor_buffer.h"

// Explicit instantiations for common types
template class TensorBuffer<float>;
template class TensorBuffer<int32_t>;
template class TensorBuffer<int64_t>;
```

## Preprocessor

Minimize preprocessor usage. Prefer:

- `const`/`constexpr` over `#define` for constants
- Inline functions over function-like macros
- Enums over numeric defines

```cpp
// Good
constexpr int kMaxBatchSize = 128;

inline int Square(int x) { return x * x; }

enum class LogLevel {
  kDebug,
  kInfo,
  kWarning,
  kError
};

// Bad
#define MAX_BATCH_SIZE 128
#define SQUARE(x) ((x) * (x))
#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO 1
```

When macros are necessary:
- Use ALL_CAPS names
- Protect multi-statement macros with `do { ... } while (0)`
- Document non-obvious behavior

```cpp
#define ONNXRUNTIME_EP_IREE_CHECK_OK(expr)           \
  do {                                         \
    auto status = (expr);                      \
    if (!status.ok()) {                        \
      return status;                           \
    }                                          \
  } while (0)
```

## Tools

### Static Analysis

Enable compiler warnings and static analysis:

```cmake
target_compile_options(onnxruntime_ep_iree PRIVATE
  -Wall
  -Wextra
  -Werror
  -Wpedantic
  -Wno-unused-parameter
)
```

## Summary

This style guide prioritizes:

1. **Readability** - Code should be immediately understandable
2. **Consistency** - Uniform patterns reduce cognitive load
3. **Simplicity** - Avoid clever code; prefer straightforward solutions
4. **Modularity** - Clean interfaces and separation of concerns
5. **Maintainability** - Code that's easy to modify and debug

When in doubt, follow the principle: **optimize for the reader, not the writer**.

For questions or clarifications on style, refer to the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
