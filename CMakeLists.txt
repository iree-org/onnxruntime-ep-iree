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
# We choose not to take a source dependency on the IREE compiler, which is
# quite heavy and can be slow to build. It is left to the consumer of the EP to
# make sure iree compiler tooling scripts are available in their environment.
# This can be generally done by `pip install iree-base-compiler`.

# We may in future decide to build the IREE compiler from source.
#
################################################################################

# Version pins for dependencies.
# Prefer to keep the IREE_GIT_TAG synced with the python-installed IREE version.
# At a minimum, the compiler from those packages must be compatible with the
# runtime at this source ref.
set(IREE_GIT_TAG "v3.10.0")
set(IREE_SOURCE_DIR "" CACHE FILEPATH "Path to IREE source")

# Set IREE build flags.
set(IREE_VISIBILITY_HIDDEN OFF)
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

# Statically link IREE runtime and io/parameter libraries.
target_link_libraries(onnxruntime_ep_iree PRIVATE
  iree_runtime_unified
  iree_io_parameter_index
  iree_io_parameter_index_provider
  iree_io_formats_irpa_irpa
  iree_modules_io_parameters_parameters
)


# Include directories
target_include_directories(onnxruntime_ep_iree PRIVATE
  ${IREE_EP_ONNX_SRC_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
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

# Export symbols for plugin loading
if(MSVC)
  # Windows: Use .def file (ORT_EXPORT is defined as empty in ORT headers on Windows)
  set_target_properties(onnxruntime_ep_iree PROPERTIES
    LINK_FLAGS "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/src/symbols.def"
  )
elseif(UNIX AND NOT APPLE)
  # Linux: Use version script
  set_target_properties(onnxruntime_ep_iree PROPERTIES
    LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/symbols.lds"
  )
elseif(APPLE)
  # macOS: Export all symbols
  set_target_properties(onnxruntime_ep_iree PROPERTIES
    LINK_FLAGS "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/src/symbols.txt"
  )
endif()

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
