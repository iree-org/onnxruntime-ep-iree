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
# Enable Vulkan driver for GPU execution.
set(IREE_HAL_DRIVER_VULKAN ON)

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
#   4. Build-time default: -DIREE_COMPILER_LIB_DIR (baked into binary)
#   5. Relative to EP .so: iree/compiler/_mlir_libs/ (pip co-installs)
#   6. Fallback: dlopen("libIREECompiler.so") via standard search paths
#
################################################################################

# Cache var with env-var default: users can set IREE_COMPILER_LIB_DIR in the
# environment OR pass -DIREE_COMPILER_LIB_DIR=... on the cmake command line.
# The explicit -D flag always wins over the environment variable.
set(IREE_COMPILER_LIB_DIR "$ENV{IREE_COMPILER_LIB_DIR}" CACHE PATH
  "Directory containing libIREECompiler.so (compile-time default for runtime search)")

# When ON, cmake will FATAL_ERROR if the compiler library is not found at
# configure time. Default OFF for development flexibility — the EP can still
# discover the library at runtime via session options or dlopen search paths.
option(IREE_EP_REQUIRE_COMPILER_LIB
  "Require IREE compiler library to be found at configure time" OFF)

# Auto-detect the pip-installed iree-base-compiler library directory.
# Uses Python3_FIND_VIRTUALENV FIRST so venv interpreters are preferred.
if(NOT IREE_COMPILER_LIB_DIR)
  set(Python3_FIND_VIRTUALENV FIRST)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_FOUND)
    message(STATUS "Using Python interpreter for compiler detection: ${Python3_EXECUTABLE}")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
        "import iree.compiler._mlir_libs as m; from pathlib import Path; print(Path(m.__file__).parent)"
      OUTPUT_VARIABLE _pip_compiler_lib_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _pip_result
    )
    if(_pip_result EQUAL 0 AND _pip_compiler_lib_dir)
      # Persist in cache so subsequent reconfigures (e.g. without venv active)
      # don't lose the detected path.
      set(IREE_COMPILER_LIB_DIR "${_pip_compiler_lib_dir}" CACHE PATH
        "Directory containing libIREECompiler.so (compile-time default for runtime search)" FORCE)
      message(STATUS "Auto-detected pip-installed IREE compiler: ${IREE_COMPILER_LIB_DIR}")
    else()
      message(STATUS "iree-base-compiler not found via ${Python3_EXECUTABLE}")
    endif()
  else()
    message(STATUS "Python3 interpreter not found; skipping pip compiler detection")
  endif()
endif()

# Validate the detected/configured directory.
if(IREE_COMPILER_LIB_DIR)
  # Resolve to real path (canonicalize symlinks, relative paths).
  get_filename_component(IREE_COMPILER_LIB_DIR "${IREE_COMPILER_LIB_DIR}" REALPATH)

  # Check that the directory actually contains a compiler library.
  set(_iree_compiler_lib_found FALSE)
  foreach(_lib_name "libIREECompiler.so" "libIREECompiler.dylib" "IREECompiler.dll")
    if(EXISTS "${IREE_COMPILER_LIB_DIR}/${_lib_name}")
      set(_iree_compiler_lib_found TRUE)
      message(STATUS "IREE compiler library: ${IREE_COMPILER_LIB_DIR}/${_lib_name}")
      break()
    endif()
  endforeach()
  if(NOT _iree_compiler_lib_found)
    message(WARNING
      "IREE_COMPILER_LIB_DIR is set to '${IREE_COMPILER_LIB_DIR}' "
      "but no compiler library (libIREECompiler.so/.dylib/IREECompiler.dll) was found there.")
    # Clear invalid directory so it's not baked into the binary.
    set(IREE_COMPILER_LIB_DIR "")
  endif()
  unset(_iree_compiler_lib_found)
endif()

# Final status/error reporting.
if(IREE_COMPILER_LIB_DIR)
  message(STATUS "IREE compiler library directory: ${IREE_COMPILER_LIB_DIR}")
elseif(IREE_EP_REQUIRE_COMPILER_LIB)
  message(FATAL_ERROR
    "IREE compiler library not found and IREE_EP_REQUIRE_COMPILER_LIB is ON.\n"
    "Either:\n"
    "  - Set IREE_COMPILER_LIB_DIR env var or -DIREE_COMPILER_LIB_DIR=<path>, or\n"
    "  - Activate a virtualenv with iree-base-compiler before running cmake, or\n"
    "  - Set -DIREE_EP_REQUIRE_COMPILER_LIB=OFF to allow runtime discovery")
else()
  message(WARNING
    "IREE compiler library not found at configure time.\n"
    "The EP will rely on runtime discovery (session option, env var, or dlopen search paths).\n"
    "To bake the path into the binary, either:\n"
    "  - Set IREE_COMPILER_LIB_DIR env var or -DIREE_COMPILER_LIB_DIR=<path>, or\n"
    "  - Activate a virtualenv with iree-base-compiler before running cmake")
endif()

set(IREE_COMPILER_BINDINGS_DIR "${iree_SOURCE_DIR}/compiler/bindings/c")

# Build shared library (EP plugin)
add_library(onnxruntime_ep_iree SHARED
  src/plugin_entry.cc
  src/iree_ep_factory.cc
  src/iree_ep.cc
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

# Pass build-time compiler library directory as a compile definition.
if(IREE_COMPILER_LIB_DIR)
  target_compile_definitions(onnxruntime_ep_iree PRIVATE
    IREE_COMPILER_LIB_DIR="${IREE_COMPILER_LIB_DIR}"
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
