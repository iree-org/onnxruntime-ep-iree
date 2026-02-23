cmake_minimum_required(VERSION 3.24)
project(onnxruntime_ep_iree VERSION 0.1.0 LANGUAGES CXX)

# C++20 required (for std::format)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

################################################################################
# ONNX Runtime Dependency
#
# We only need the ONNX Runtime C++ API headers for the EP interface. The
# headers are fetched from GitHub - no linking to the ONNX Runtime library is
# required since the EP is loaded as a plugin and receives the ORT API via
# function pointers at runtime.
#
# To point to a local ONNX Runtime source directory, specify:
#
#     -DONNXRUNTIME_SOURCE_DIR=</path/to/onnxruntime/source>
#
################################################################################

set(ONNXRUNTIME_VERSION "1.24.1")
set(ONNXRUNTIME_SOURCE_DIR "" CACHE FILEPATH "Path to ONNX Runtime source")

if(ONNXRUNTIME_SOURCE_DIR)
  message(STATUS "Using existing ONNX Runtime sources: ${ONNXRUNTIME_SOURCE_DIR}")
  set(IREE_EP_ONNX_SRC_DIR "${ONNXRUNTIME_SOURCE_DIR}")
else()
  message(STATUS "Fetching ONNX Runtime headers v${ONNXRUNTIME_VERSION}")
  FetchContent_Declare(
    onnxruntime
    URL "https://github.com/microsoft/onnxruntime/archive/refs/tags/v${ONNXRUNTIME_VERSION}.zip"
    # Point to include/ which has no CMakeLists.txt, preventing any build.
    SOURCE_SUBDIR include
  )
  FetchContent_MakeAvailable(onnxruntime)
  set(IREE_EP_ONNX_SRC_DIR "${onnxruntime_SOURCE_DIR}")
endif()

# Verify ONNX Runtime headers exist
if(NOT EXISTS "${IREE_EP_ONNX_SRC_DIR}/include/onnxruntime/core/session/onnxruntime_c_api.h")
  message(FATAL_ERROR
    "ONNX Runtime headers not found at: ${IREE_EP_ONNX_SRC_DIR}\n"
    "Please set ONNXRUNTIME_SOURCE_DIR to point to your ONNX Runtime checkout.\n"
    "Example: cmake -DONNXRUNTIME_SOURCE_DIR=/path/to/onnxruntime ..")
endif()

################################################################################
# IREE Dependency
#
# The EP takes a source dependency on the IREE runtime for interfacing with its
# C-API. The runtime is structured as a modular set of library components
# designed to be (statically) linked into applications directly and compiled
# with LTO style optimizations.
#
# To point to a local IREE source directory, simply specify:
#
#     -DIREE_SOURCE_DIR=</path/to/iree/source>
#
# When not specified, FetchContent the IREE sources from GitHub at the tag
# specified by `IREE_GIT_TAG`.
#
# The EP dynamically loads libIREECompiler.so at runtime via dlopen (no
# compile-time linking to the compiler). The loader source and C API headers
# come from the IREE source tree (compiler/bindings/c/).
#
################################################################################

# Version pins for dependencies.
# Prefer to keep the IREE_GIT_TAG synced with the python-installed IREE version.
# At a minimum, the compiler from those packages must be compatible with the
# runtime at this source ref.
set(IREE_GIT_TAG "v3.10.0")
set(IREE_SOURCE_DIR "" CACHE FILEPATH "Path to IREE source")

# Set IREE build flags.
set(IREE_VISIBILITY_HIDDEN ON)
set(IREE_BUILD_COMPILER OFF)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)
# Disable missing submodules error because we are only building the runtime.
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF)
# Disable default drivers and explicitly enable used ones.
set(IREE_HAL_DRIVER_DEFAULTS OFF)
# Enable local sync and task drivers for CPU execution.
set(IREE_HAL_DRIVER_LOCAL_SYNC ON)
set(IREE_HAL_DRIVER_LOCAL_TASK ON)
# Enable HIP driver for ROCm GPU execution.
set(IREE_HAL_DRIVER_HIP ON)
set(IREE_HIP_TEST_TARGET_CHIP "" CACHE STRING "Enable ROCm testing on specific architecture (e.g. gfx942)")
# Enable Vulkan driver for GPU execution (disabled on Apple — IREE's Vulkan
# dynamic_symbols.cc has a pointer alignment error on macOS ARM64).
if(NOT APPLE)
  set(IREE_HAL_DRIVER_VULKAN ON)
endif()

if(IREE_SOURCE_DIR)
  message(STATUS "Using existing IREE sources: ${IREE_SOURCE_DIR}")
  # Normalize so downstream code can use iree_SOURCE_DIR unconditionally
  # (FetchContent sets this automatically; local source needs it explicitly).
  set(iree_SOURCE_DIR "${IREE_SOURCE_DIR}")
  add_subdirectory(${IREE_SOURCE_DIR} iree SYSTEM EXCLUDE_FROM_ALL)
else()
  message(STATUS "Fetching IREE sources from tag ${IREE_GIT_TAG}")
  # TODO: Do we need the benchmark submodule?
  set(IREE_SUBMODULES "third_party/benchmark third_party/flatcc")
  # Add hip-build-deps submodule only if HIP driver is enabled.
  if(IREE_HAL_DRIVER_HIP)
    list(APPEND IREE_SUBMODULES "third_party/hip-build-deps")
  endif()
  # Add vulkan_headers submodule only if Vulkan driver is enabled.
  if(IREE_HAL_DRIVER_VULKAN)
    list(APPEND IREE_SUBMODULES "third_party/vulkan_headers")
  endif()
  FetchContent_Declare(
    iree
    GIT_REPOSITORY https://github.com/iree-org/iree.git
    GIT_TAG "${IREE_GIT_TAG}"
    GIT_SUBMODULES ${IREE_SUBMODULES}
    GIT_SHALLOW TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
  )
  FetchContent_GetProperties(iree)
  if(NOT iree_POPULATED)
    FetchContent_MakeAvailable(iree)
  endif()
endif()

################################################################################
# Compiler Loader
#
# The EP uses the IREE compiler C API (embedding_api.h) via a dynamic loader
# (loader.h / loader.cpp). At runtime, the EP loads libIREECompiler.so via
# dlopen — there is no compile-time linking to the compiler. The loader source
# and C API headers come from the IREE source tree.
#
# At runtime, the compiler library is resolved in this order:
#   1. Session option: ep.iree.compiler_lib_path (explicit path)
#   2. Env var: IREE_EP_COMPILER_LIB (full path to library file)
#   3. Env var: IREE_EP_COMPILER_LIB_DIR (directory to search)
#   4. Relative to EP .so: iree/compiler/_mlir_libs/ (pip co-installs)
#   5. Fallback: dlopen("libIREECompiler.so") via standard search paths
#
################################################################################

# Helper: check whether a Python module is importable in the active environment.
# Sets ${result_var} to TRUE/FALSE in the caller's scope.
function(check_python_import module result_var)
  set(Python3_FIND_VIRTUALENV FIRST)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  unset(Python3_FIND_VIRTUALENV)
  if(NOT Python3_FOUND)
    message(STATUS "Python3 interpreter not found; skipping ${module} check")
    set(${result_var} FALSE PARENT_SCOPE)
    return()
  endif()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import importlib.util; exit(0 if importlib.util.find_spec('${module}') else 1)"
    RESULT_VARIABLE _import_result
    OUTPUT_QUIET ERROR_QUIET
  )
  if(_import_result EQUAL 0)
    set(${result_var} TRUE PARENT_SCOPE)
  else()
    set(${result_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

# Check for iree-base-compiler. Not required at build time — the EP discovers
# libIREECompiler.so at runtime. A warning is printed if not found.
# Use iree.compiler._mlir_libs (not iree.compiler) because the IREE build tree
# creates a namespace package that makes find_spec('iree.compiler') succeed
# even without iree-base-compiler installed.
check_python_import("iree.compiler._mlir_libs" _iree_compiler_installed)
if(_iree_compiler_installed)
  message(STATUS "iree-base-compiler is installed")
else()
  message(WARNING
    "iree-base-compiler is not installed in the active Python environment.\n"
    "The EP will still build but will fail to compile models at runtime unless "
    "iree-base-compiler is installed or IREE_EP_COMPILER_LIB / "
    "IREE_EP_COMPILER_LIB_DIR env var is set.")
endif()

set(IREE_COMPILER_BINDINGS_DIR "${iree_SOURCE_DIR}/compiler/bindings/c")

# Build shared library (EP plugin)
add_library(onnxruntime_ep_iree SHARED
  src/plugin_entry.cc
  src/iree_ep_factory.cc
  src/iree_ep.cc
  src/dim_spec.cc
  src/iree_compile.cc
  src/mlir_gen.cc
  src/iree_ort_utils.cc
  src/iree_allocator.cc
  src/iree_data_transfer.cc
  src/temp_file.cc
)

# Build IREE compiler loader as a static library. The loader's trampoline
# functions use IREE_EMBED_EXPORTED (visibility("default")) which we can't
# override at compile time. Instead, --exclude-libs localizes all symbols
# from this archive when linking into our shared library.
add_library(iree_compiler_loader STATIC
  ${IREE_COMPILER_BINDINGS_DIR}/iree/compiler/loader/loader.cpp
)
set_target_properties(iree_compiler_loader PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(iree_compiler_loader PRIVATE
  ${IREE_COMPILER_BINDINGS_DIR}
)
if(MSVC)
  target_compile_options(iree_compiler_loader PRIVATE /w)
else()
  target_compile_options(iree_compiler_loader PRIVATE -w)
endif()

# Statically link IREE runtime and io/parameter libraries.
# Link dl for dlopen/dlsym (compiler loader).
target_link_libraries(onnxruntime_ep_iree PRIVATE
  iree_runtime_unified
  iree_io_parameter_index
  iree_io_parameter_index_provider
  iree_io_formats_irpa_irpa
  iree_modules_io_parameters_parameters
  iree_compiler_loader
  ${CMAKE_DL_LIBS}
)

# Localize all symbols from the compiler loader static library so its
# IREE_EMBED_EXPORTED trampolines don't leak into our public symbol table.
if(UNIX AND NOT APPLE)
  # GNU ld: hide all symbols from the loader's static archive.
  target_link_options(onnxruntime_ep_iree PRIVATE
    "LINKER:--exclude-libs,libiree_compiler_loader.a"
  )
elseif(APPLE)
  # macOS ld: generate an unexported symbols list from the loader's trampolines.
  # The ireeCompiler* symbols are generated by loader.cpp with
  # visibility("default") — we re-hide them here.
  set(_unexport_file "${CMAKE_CURRENT_BINARY_DIR}/iree_loader_unexported.txt")
  file(WRITE "${_unexport_file}" "_ireeCompiler*\n")
  target_link_options(onnxruntime_ep_iree PRIVATE
    "LINKER:-unexported_symbols_list,${_unexport_file}"
  )
endif()
# On Windows, MSVC only exports symbols explicitly marked __declspec(dllexport),
# so the loader trampolines (which use visibility("default"), a no-op on MSVC)
# are not exported. No additional linker flags needed.

# Include directories
target_include_directories(onnxruntime_ep_iree PRIVATE
  ${IREE_EP_ONNX_SRC_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${IREE_COMPILER_BINDINGS_DIR}
)

# Compiler flags
if(MSVC)
  target_compile_options(onnxruntime_ep_iree PRIVATE /W3)
  target_compile_definitions(onnxruntime_ep_iree PRIVATE
    _CRT_SECURE_NO_WARNINGS
  )
else()
  target_compile_options(onnxruntime_ep_iree PRIVATE
    -Wall
    -Wextra
    -Werror
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3>
  )
endif()

# Hide all symbols by default. Only plugin entry points (CreateEpFactories,
# ReleaseEpFactory) are exported via explicit visibility("default") attributes.
# VISIBILITY_INLINES_HIDDEN suppresses weak inline/template symbols (e.g.
# std::format, libstdc++ internals) that would otherwise leak into the ABI.
set_target_properties(onnxruntime_ep_iree PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  C_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# Install
install(TARGETS onnxruntime_ep_iree
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Print configuration
message(STATUS "========================================")
message(STATUS "IREE ONNX Execution Provider Configuration")
message(STATUS "========================================")
message(STATUS "  C++ Standard:        ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "  ONNX Runtime Source: ${IREE_EP_ONNX_SRC_DIR}")
message(STATUS "  IREE Source:         ${IREE_SOURCE_DIR}")
message(STATUS "  Install Prefix:      ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")
