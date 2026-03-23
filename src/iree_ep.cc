//===- iree_ep.cc ---------------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements IREE-based compilation and execution for ONNX models.
// Uses IREE runtime API for loading VMFB modules and executing functions.
//
//===----------------------------------------------------------------------===//

#include "iree_ep.h"

#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "iree/modules/io/parameters/module.h"
#include "iree/runtime/api.h"
#include "iree_compile.h"
#include "iree_ep_factory.h"
#include "iree_ort_utils.h"
#include "mlir_gen.h"
#include "temp_file.h"

namespace onnxruntime::iree {

// Builds symbolic dimension mappings from graph inputs. Returns ALL occurrences
// of each symbolic dimension (not deduplicated). ComputeImpl reads actual
// values from these mappings at runtime for variant dispatch.
static std::vector<IreeNodeComputeInfo::SymbolicDimMapping>
BuildSymbolicDimMappings(const Ort::ConstGraph& graph) {
  std::vector<IreeNodeComputeInfo::SymbolicDimMapping> mappings;

  auto inputs = graph.GetInputs();
  auto initializers = graph.GetInitializers();
  std::unordered_set<std::string> init_names;
  for (const auto& init : initializers) {
    init_names.insert(init.GetName());
  }

  size_t input_index = 0;
  for (const auto& input : inputs) {
    if (init_names.count(input.GetName())) continue;
    auto type_info = input.TypeInfo();
    if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      auto shape = tensor_info.GetShape();
      auto sym_dims = tensor_info.GetSymbolicDimensions();
      for (size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] >= 0 || d >= sym_dims.size() || sym_dims[d] == nullptr ||
            sym_dims[d][0] == '\0')
          continue;
        mappings.push_back({input_index, d, std::string(sym_dims[d])});
      }
    }
    input_index++;
  }
  return mappings;
}

static std::vector<std::string> GenerateCompileFlags(
    const IreeEp::Config& config) {
  std::vector<std::string> flags;

  if (config.backend == "llvm-cpu") {
    flags.push_back("--iree-hal-target-device=local");
    flags.push_back("--iree-hal-local-target-device-backends=llvm-cpu");
    flags.push_back("--iree-llvmcpu-target-cpu=host");
    if (config.opt_level == "O2" || config.opt_level == "O3") {
      flags.push_back("--iree-opt-data-tiling");
    }
  } else if (config.backend == "vulkan") {
    flags.push_back("--iree-hal-target-device=vulkan");
  } else if (config.backend == "cuda") {
    flags.push_back("--iree-hal-target-device=cuda");
    if (!config.target_arch.empty()) {
      flags.push_back("--iree-cuda-target=" + config.target_arch);
    }
  } else if (config.backend == "hip") {
    flags.push_back("--iree-hal-target-device=hip");
    if (!config.target_arch.empty()) {
      flags.push_back("--iree-hip-target=" + config.target_arch);
    }
  } else {
    assert(false && "Unsupported backend, should have been caught earlier");
  }

  // Extern dispatch support: add search path for pre-compiled kernel objects.
  if (!config.extern_kernel_path.empty()) {
    flags.push_back("--iree-hal-executable-object-search-path=" +
                    config.extern_kernel_path);
  }

  flags.push_back("--iree-opt-level=" + config.opt_level);
  return flags;
}

// ============================================================================
// IreeEp Implementation
// ============================================================================

IreeEp::IreeEp(IreeEpFactory& factory, const std::string& name,
               const Config& config, const OrtLogger& logger,
               uint32_t device_id)
    : OrtEp{},
      ApiPtrs(static_cast<const ApiPtrs&>(factory)),
      factory_(factory),
      name_(name),
      config_(config),
      logger_(&logger),
      device_id_(device_id) {
  // Set ORT version we support.
  ort_version_supported = ORT_API_VERSION;

  // Set function pointers for EP interface.
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
}

IreeEp::~IreeEp() {
  // Note: Avoid using logger during cleanup - ORT logging infrastructure may
  // be torn down before EP destructors run during Python interpreter shutdown.
}

iree_hal_device_t* IreeEp::IreeDevice() const {
  return factory_.GetDeviceForId(device_id_);
}

iree_runtime_instance_t* IreeEp::IreeInstance() const {
  return factory_.IreeInstance();
}

/*static*/
const char* ORT_API_CALL IreeEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const IreeEp*>(this_ptr);
  return ep->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL
IreeEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                          OrtEpGraphSupportInfo* graph_support_info) noexcept {
  auto* ep = static_cast<IreeEp*>(this_ptr);

  // Use the C++ wrapper for easier API access,
  Ort::ConstGraph graph{ort_graph};

  // Get all nodes in the graph.
  std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  if (nodes.empty()) {
    return nullptr;  // Empty graph, nothing to claim.
  }

  // Collect all nodes - we claim the entire graph.
  std::vector<const OrtNode*> nodes_to_fuse;
  nodes_to_fuse.reserve(nodes.size());
  for (const auto& node : nodes) {
    nodes_to_fuse.push_back(node);
  }

  // Create fusion options for compiling EP.
  OrtNodeFusionOptions node_fusion_options = {};
  node_fusion_options.ort_version_supported = ORT_API_VERSION;

  // Drop constant initializers - EP will save them during Compile().
  // This reduces memory usage and allows weight preprocessing/
  node_fusion_options.drop_constant_initializers = true;

  // Register all nodes as a single fused subgraph.
  OrtStatus* status = ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(
      graph_support_info, nodes_to_fuse.data(), nodes_to_fuse.size(),
      &node_fusion_options);

  return status;
}

/*static*/
OrtStatus* ORT_API_CALL IreeEp::CompileImpl(
    OrtEp* this_ptr, const OrtGraph** graphs, const OrtNode** /*fused_nodes*/,
    size_t count, OrtNodeComputeInfo** node_compute_infos,
    OrtNode** /*ep_context_nodes*/) noexcept {
  auto* ep = static_cast<IreeEp*>(this_ptr);

  if (count == 0 || graphs == nullptr) {
    return Ort::Status("IREE EP: No graphs provided to compile.",
                       ORT_INVALID_ARGUMENT)
        .release();
  }

  // TODO: Do we need to handle multiple graphs?
  if (count != 1) {
    return Ort::Status(("IREE EP: Expected exactly 1 graph, got " +
                        std::to_string(count) + ".")
                           .c_str(),
                       ORT_INVALID_ARGUMENT)
        .release();
  }
  Ort::ConstGraph graph{graphs[0]};

  // Determine how many variants we need (specialized + generic fallback).
  const auto& dim_spec_variants = ep->config_.dim_spec_variants;
  size_t num_specialized = dim_spec_variants.size();

  // Variants are dispatched in user-specified order (first match wins).
  // Preserving caller order makes precedence explicit for overlaps.

  // Create temp files for intermediate artifacts.
  TempFile mlir_file(".mlir");
  TempFile vmfb_file(".vmfb");
  TempFile irpa_file(".irpa");

  // save_intermediates: keep all artifacts on disk for debugging/inspection.
  if (ep->config_.save_intermediates) {
    mlir_file.Keep();
    vmfb_file.Keep();
    irpa_file.Keep();
    ORT_CXX_LOGF_NOEXCEPT(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                          "IREE EP: Saving MLIR to: %s",
                          mlir_file.Path().c_str());
    ORT_CXX_LOGF_NOEXCEPT(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                          "IREE EP: Saving VMFB to: %s",
                          vmfb_file.Path().c_str());
    ORT_CXX_LOGF_NOEXCEPT(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                          "IREE EP: Saving IRPA to: %s",
                          irpa_file.Path().c_str());
  }

  // enable_ep_context_cache: keep only compiled artifacts (VMFB, IRPA) needed
  // for caching. MLIR is a transient intermediate — only the final compiled
  // artifacts are needed to skip recompilation on subsequent runs.
  if (!ep->config_.save_intermediates && ep->config_.enable_ep_context_cache) {
    vmfb_file.Keep();
    irpa_file.Keep();
    ORT_CXX_LOGF_NOEXCEPT(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                          "IREE EP: Saving VMFB to: %s",
                          vmfb_file.Path().c_str());
    ORT_CXX_LOGF_NOEXCEPT(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                          "IREE EP: Saving IRPA to: %s",
                          irpa_file.Path().c_str());
  }

  // Build symbolic dimension mappings for runtime dispatch.
  auto dim_mappings = BuildSymbolicDimMappings(graph);

  // Phase 1: Generate MLIR from the graph.
  // Also builds an IRPA parameter archive for large initializers.
  ParameterIndexPtr parameter_index;
  ParameterProviderPtr parameter_provider;

  std::vector<std::pair<std::string, DimSpecVariant>> mlir_variants;
  for (size_t i = 0; i < num_specialized; ++i) {
    mlir_variants.emplace_back("_variant" + std::to_string(i),
                               dim_spec_variants[i]);
  }

  // Add a generic fallback variant.
  mlir_variants.emplace_back("", DimSpecVariant{});

  // function_names[i] is the MLIR function name for mlir_variants[i].
  std::vector<std::string> function_names;
  TargetConfig target_config =
      TargetConfig::Create(ep->config_.target_arch, ep->config_.backend);
  ORT_CXX_LOGF_NOEXCEPT(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                        "IREE EP: Generating MLIR (%zu variants)",
                        mlir_variants.size());
  IREE_ORT_RETURN_IF_MAYBE_ERROR(
      GenerateMlir(graph, ep->ort_api, mlir_file.Path(), irpa_file.Path(),
                   mlir_variants, function_names, parameter_index,
                   parameter_provider, std::move(target_config)));

  // Phase 2: Compile MLIR to VMFB.
  std::vector<std::string> flags = GenerateCompileFlags(ep->config_);
  ORT_CXX_LOG_NOEXCEPT(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                       "IREE EP: Generating VMFB");
  ORT_RETURN_IF_ERROR(
      CompileToVmfb(mlir_file.Path(), vmfb_file.Path(), flags, ep->ort_api));
  ORT_CXX_LOG_NOEXCEPT(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                       "IREE EP: VMFB Generated Successfully");

  // Phase 3: Read VMFB into memory. Session creation is deferred to first
  // execution, which allows the VMFB to be cached and loaded on a different
  // device.
  std::ifstream vmfb_stream(vmfb_file.Path(), std::ios::binary | std::ios::ate);
  if (!vmfb_stream) {
    return Ort::Status("IREE EP: Failed to open VMFB file", ORT_FAIL).release();
  }
  std::streampos vmfb_pos = vmfb_stream.tellg();
  if (vmfb_pos == std::streampos(-1) || vmfb_pos <= 0) {
    return Ort::Status("IREE EP: Failed to determine VMFB file size", ORT_FAIL)
        .release();
  }
  auto vmfb_size = static_cast<size_t>(vmfb_pos);
  vmfb_stream.seekg(0, std::ios::beg);
  std::vector<uint8_t> vmfb_data(vmfb_size);
  if (!vmfb_stream.read(reinterpret_cast<char*>(vmfb_data.data()),
                        static_cast<std::streamsize>(vmfb_size))) {
    return Ort::Status("IREE EP: Failed to read VMFB file contents", ORT_FAIL)
        .release();
  }

  // Build variant info for lazy init.
  // Format: "module.{graph_name}{variant_suffix}" for VMFB function lookup.
  std::vector<IreeNodeComputeInfo::VariantInfo> variant_infos;
  variant_infos.reserve(function_names.size());
  for (size_t i = 0; i < function_names.size(); ++i) {
    variant_infos.push_back(
        {"module." + function_names[i], mlir_variants[i].second});
  }

  // Create NodeComputeInfo with compiled artifacts. Session creation and VMFB
  // loading are deferred to first ComputeImpl call (lazy initialization).
  size_t num_variants = variant_infos.size();
  auto* info = new IreeNodeComputeInfo(
      *ep, std::move(vmfb_data), std::move(parameter_index),
      std::move(parameter_provider), std::move(variant_infos),
      std::move(dim_mappings));
  node_compute_infos[0] = info;

  ORT_CXX_LOGF_NOEXCEPT(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                        "IREE EP: Compilation complete (%zu variants)",
                        num_variants);
  return nullptr;
}

/*static*/
void ORT_API_CALL IreeEp::ReleaseNodeComputeInfosImpl(
    OrtEp* /*this_ptr*/, OrtNodeComputeInfo** node_compute_infos,
    size_t num_node_compute_infos) noexcept {
  // Delete all node compute infos we created
  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    if (node_compute_infos[i] != nullptr) {
      delete static_cast<IreeNodeComputeInfo*>(node_compute_infos[i]);
    }
  }
}

// ============================================================================
// Dim Spec Dispatch Helpers
// ============================================================================

static bool VariantMatchesDimSpecs(
    const IreeNodeComputeInfo::Variant& variant,
    const std::unordered_map<std::string, int64_t>& dim_values) {
  for (const auto& spec : variant.dim_specs) {
    auto it = dim_values.find(spec.symbolic_name);
    if (it == dim_values.end()) {
      continue;
    }
    int64_t actual = it->second;
    if (actual < spec.min || actual > spec.max) {
      return false;
    }
    if (spec.div > 1 && actual % spec.div != 0) {
      return false;
    }
  }
  return true;
}

static std::unordered_map<std::string, int64_t> CollectDimValues(
    const Ort::KernelContext& ctx,
    const std::vector<IreeNodeComputeInfo::SymbolicDimMapping>& dim_mappings) {
  std::unordered_map<std::string, int64_t> dim_values;
  for (const auto& mapping : dim_mappings) {
    auto shape = ctx.GetInput(mapping.input_index)
                     .GetTensorTypeAndShapeInfo()
                     .GetShape();
    dim_values.try_emplace(mapping.symbolic_name, shape[mapping.dim_index]);
  }
  return dim_values;
}

static const IreeNodeComputeInfo::Variant* SelectMatchingVariant(
    const std::vector<IreeNodeComputeInfo::Variant>& variants,
    const std::unordered_map<std::string, int64_t>& dim_values) {
  // Variants are in user-specified order (first match wins).
  for (const auto& variant : variants) {
    if (VariantMatchesDimSpecs(variant, dim_values)) {
      return &variant;
    }
  }
  return nullptr;
}

// ============================================================================
// IreeNodeComputeInfo Implementation
// ============================================================================

IreeNodeComputeInfo::IreeNodeComputeInfo(
    IreeEp& ep_ref, std::vector<uint8_t> vmfb_data,
    ParameterIndexPtr parameter_index, ParameterProviderPtr parameter_provider,
    std::vector<VariantInfo> variant_infos,
    std::vector<SymbolicDimMapping> dim_mappings)
    : ep(ep_ref),
      vmfb_data_(std::move(vmfb_data)),
      parameter_index_(std::move(parameter_index)),
      parameter_provider_(std::move(parameter_provider)),
      variant_infos_(std::move(variant_infos)),
      dim_mappings_(std::move(dim_mappings)) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

IreeNodeComputeInfo::~IreeNodeComputeInfo() {
  // Note: Avoid using logger during cleanup - ORT logging infrastructure may
  // be torn down before our destructors run during Python interpreter shutdown.
  // Release session before vmfb_data_ is destroyed, since the session
  // references VMFB bytes directly (loaded with iree_allocator_null).
  // Redundant with current member declaration order but defensive against
  // future reordering.
  session_.Reset();
}

OrtStatus* IreeNodeComputeInfo::InitializeSession() {
  // Phase 1: Create IREE runtime session with device.
  iree_runtime_session_options_t session_opts;
  iree_runtime_session_options_initialize(&session_opts);
  IREE_ORT_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      ep.IreeInstance(), &session_opts, ep.IreeDevice(),
      iree_runtime_instance_host_allocator(ep.IreeInstance()),
      session_.ForOutput()));

  // Phase 2: Register io_parameters module if we have parameters.
  if (parameter_provider_) {
    VmModulePtr parameters_module;
    iree_io_parameter_provider_t* provider_raw = parameter_provider_.Get();
    IREE_ORT_RETURN_IF_ERROR(iree_io_parameters_module_create(
        iree_runtime_instance_vm_instance(ep.IreeInstance()), 1, &provider_raw,
        iree_runtime_instance_host_allocator(ep.IreeInstance()),
        parameters_module.ForOutput()));
    IREE_ORT_RETURN_IF_ERROR(iree_runtime_session_append_module(
        session_.Get(), parameters_module.Get()));
  }

  // Phase 3: Load VMFB from memory (avoids disk round-trip).
  iree_const_byte_span_t flatbuffer_data = {
      vmfb_data_.data(), static_cast<iree_host_size_t>(vmfb_data_.size())};
  IREE_ORT_RETURN_IF_ERROR(
      iree_runtime_session_append_bytecode_module_from_memory(
          session_.Get(), flatbuffer_data, iree_allocator_null()));

  // Phase 4: Lookup all variant functions and populate variants_.
  variants_.reserve(variant_infos_.size());
  for (const auto& vi : variant_infos_) {
    iree_vm_function_t function;
    IREE_ORT_RETURN_IF_ERROR(iree_runtime_session_lookup_function(
        session_.Get(), iree_make_cstring_view(vi.function_name.c_str()),
        &function));
    variants_.push_back({function, vi.dim_specs});
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL IreeNodeComputeInfo::CreateStateImpl(
    OrtNodeComputeInfo* /*this_ptr*/,
    OrtNodeComputeContext* /*compute_context*/, void** compute_state) noexcept {
  // No per-invocation state needed - session/function stored in
  // NodeComputeInfo.
  *compute_state = nullptr;
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL IreeNodeComputeInfo::ComputeImpl(
    OrtNodeComputeInfo* this_ptr, void* /*compute_state*/,
    OrtKernelContext* kernel_context) noexcept {
  auto* info = static_cast<IreeNodeComputeInfo*>(this_ptr);

  // Lazy-init: create session on first invocation (thread-safe).
  std::call_once(info->init_flag_, [info]() {
    OrtStatus* status = info->InitializeSession();
    if (status) {
      info->init_error_ = Ort::Status(status).GetErrorMessage();
    }
  });
  if (!info->init_error_.empty()) {
    return Ort::Status(info->init_error_.c_str(), ORT_FAIL).release();
  }

  Ort::KernelContext ctx(kernel_context);

  auto dim_values = CollectDimValues(ctx, info->dim_mappings_);
  const Variant* selected = SelectMatchingVariant(info->variants_, dim_values);
  assert(selected && "At least the generic fallback variant should match");

  iree_hal_device_t* device = info->ep.IreeDevice();
  iree_hal_allocator_t* allocator =
      iree_runtime_session_device_allocator(info->session_.Get());

  // Convert ORT inputs to IREE buffer views.
  std::vector<HalBufferViewPtr> input_views;
  size_t input_count = ctx.GetInputCount();
  input_views.reserve(input_count);

  for (size_t i = 0; i < input_count; ++i) {
    Ort::ConstValue input = ctx.GetInput(i);
    iree_hal_buffer_view_t* view = nullptr;
    ORT_RETURN_IF_ERROR(OrtTensorToIreeBufferView(
        input, device, allocator, iree_allocator_system(), &view,
        info->ep.ep_api, info->ep.Logger()));
    input_views.emplace_back(view);
  }

  // Initialize the call.
  RuntimeCall call;
  IREE_ORT_RETURN_IF_ERROR(iree_runtime_call_initialize(
      info->session_.Get(), selected->function, call.Get()));
  call.MarkInitialized();

  // Push input buffer views.
  for (auto& view : input_views) {
    IREE_ORT_RETURN_IF_ERROR(
        iree_runtime_call_inputs_push_back_buffer_view(call.Get(), view.Get()));
  }

  // Invoke the function.
  IREE_ORT_RETURN_IF_ERROR(
      iree_runtime_call_invoke(call.Get(), IREE_RUNTIME_CALL_FLAG_RESERVED));

  // Pop outputs and copy to ORT tensors.
  //
  // TODO(perf): Currently IREE allocates its own output buffers, then we copy
  // to ORT's pre-allocated device buffers (D2D copy). The way to properly
  // eliminate this is by passing mutable dps buffers as part of the iree input
  // signature and writing to them. The problem is that ORT doesn't give us a
  // good way to infer the output shape. I'm not sure what the right fix is.
  // Maybe we could have a custom iree allocator that does the job for us?
  // I'm just not sure how to do this properly.
  iree_vm_list_t* output_list = iree_runtime_call_outputs(call.Get());
  iree_host_size_t output_count = iree_vm_list_size(output_list);

  for (size_t i = 0; i < output_count; ++i) {
    // Pop output buffer view.
    iree_hal_buffer_view_t* output_view = nullptr;
    IREE_ORT_RETURN_IF_ERROR(iree_runtime_call_outputs_pop_front_buffer_view(
        call.Get(), &output_view));
    HalBufferViewPtr output_view_ptr(output_view);

    // Get shape and element type from IREE buffer view.
    std::vector<int64_t> shape = GetBufferViewShape(output_view);
    iree_hal_element_type_t iree_dtype =
        iree_hal_buffer_view_element_type(output_view);
    ONNXTensorElementDataType onnx_dtype = IreeToOnnxElementType(iree_dtype);

    if (onnx_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      return Ort::Status("IREE EP: Unsupported output element type",
                         ORT_NOT_IMPLEMENTED)
          .release();
    }

    // Allocate ORT output tensor and copy data from IREE buffer.
    Ort::UnownedValue output = ctx.GetOutput(i, shape.data(), shape.size());
    ORT_RETURN_IF_ERROR(IreeBufferViewToOrtTensor(
        output_view, output, device, info->ep.ep_api, info->ep.Logger()));
  }
  return nullptr;
}

/*static*/
void ORT_API_CALL IreeNodeComputeInfo::ReleaseStateImpl(
    OrtNodeComputeInfo* /*this_ptr*/, void* /*compute_state*/) noexcept {
  // No per-invocation state to release.
}

}  // namespace onnxruntime::iree
