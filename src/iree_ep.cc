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

#include <algorithm>
#include <charconv>
#include <format>
#include <numeric>
#include <string_view>
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

// ============================================================================
// Parsing for dim_specs
// ============================================================================

// Parses dim_specs in the format "batch=1,seq=%16;batch=2,seq=%16".
// Semicolons separate variants, commas separate specs within a variant,
// equals separates key from value. Values are integers (static) or %N
// (divisibility).
OrtStatus* ParseDimSpecs(const std::string& spec_str,
                         std::vector<DimSpecVariant>& out) {
  out.clear();

  auto err = [](std::string msg) -> OrtStatus* {
    return Ort::Status(msg.c_str(), ORT_INVALID_ARGUMENT).release();
  };

  auto trim = [](std::string_view sv) -> std::string_view {
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front())))
      sv.remove_prefix(1);
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back())))
      sv.remove_suffix(1);
    return sv;
  };

  // Pops the next trimmed token before `delim` from `sv`, advancing past it.
  auto pop_token = [&trim](std::string_view& sv,
                           char delim) -> std::string_view {
    auto pos = sv.find(delim);
    auto token = trim(sv.substr(0, pos));
    sv = (pos == std::string_view::npos) ? std::string_view{}
                                         : sv.substr(pos + 1);
    return token;
  };

  // Strict integer parse (no-throw, no allocation). All chars must be consumed.
  auto parse_int = [](std::string_view sv, int64_t& result) -> bool {
    auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), result);
    return ec == std::errc{} && ptr == sv.data() + sv.size();
  };

  // Parses a single "key=value" or "key=%N" into a DimSpec.
  auto parse_spec = [&](std::string_view spec, DimSpec& dim) -> OrtStatus* {
    auto eq = spec.find('=');
    if (eq == std::string_view::npos)
      return err(
          std::format("dim_specs: missing '=' in \"{}\"", std::string(spec)));
    auto key = trim(spec.substr(0, eq));
    auto val = trim(spec.substr(eq + 1));
    if (key.empty())
      return err(
          std::format("dim_specs: empty key in \"{}\"", std::string(spec)));
    if (val.empty())
      return err(std::format("dim_specs: empty value for key \"{}\"",
                             std::string(key)));

    if (val.starts_with('%')) {
      int64_t divisor = 0;
      if (val.size() < 2 || !parse_int(val.substr(1), divisor))
        return err(
            std::format("dim_specs: invalid divisor in \"{}\" for key \"{}\"",
                        std::string(val), std::string(key)));
      if (divisor <= 0)
        return err(
            std::format("dim_specs: divisor must be > 0, got {} for key \"{}\"",
                        divisor, std::string(key)));
      dim = {std::string(key), DimSpec::Kind::kDivisibleBy, divisor};
      return nullptr;
    }

    int64_t value = 0;
    if (!parse_int(val, value))
      return err(std::format("dim_specs: invalid number \"{}\" for key \"{}\"",
                             std::string(val), std::string(key)));
    if (value <= 0)
      return err(std::format(
          "dim_specs: static dim must be > 0, got {} for key \"{}\"", value,
          std::string(key)));
    dim = {std::string(key), DimSpec::Kind::kStatic, value};
    return nullptr;
  };

  // Parses a comma-separated list of specs into a variant.
  auto parse_variant = [&](std::string_view str,
                           DimSpecVariant& variant) -> OrtStatus* {
    while (!str.empty()) {
      auto spec = pop_token(str, ',');
      if (spec.empty()) continue;
      DimSpec dim;
      ORT_RETURN_IF_ERROR(parse_spec(spec, dim));
      variant.push_back(std::move(dim));
    }
    return nullptr;
  };

  std::string_view input = trim(spec_str);
  if (input.empty()) return nullptr;

  while (!input.empty()) {
    auto variant_str = pop_token(input, ';');
    if (variant_str.empty()) continue;
    DimSpecVariant variant;
    ORT_RETURN_IF_ERROR(parse_variant(variant_str, variant));
    if (!variant.empty()) out.push_back(std::move(variant));
  }

  return nullptr;
}

// Returns a specificity score for a variant. Higher = more specific.
// Static specs count as 2, divisibility specs count as 1.
static int VariantSpecificity(const DimSpecVariant& variant) {
  int score = 0;
  for (const auto& spec : variant) {
    score += (spec.kind == DimSpec::Kind::kStatic) ? 2 : 1;
  }
  return score;
}

// Builds symbolic dimension mappings from graph inputs. This tells the runtime
// which (input_index, dim_index) corresponds to each symbolic dimension name.
static std::vector<IreeNodeComputeInfo::SymbolicDimMapping>
BuildSymbolicDimMappings(const Ort::ConstGraph& graph) {
  std::vector<IreeNodeComputeInfo::SymbolicDimMapping> mappings;
  std::unordered_set<std::string> seen;

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
        std::string name(sym_dims[d]);
        if (seen.count(name)) continue;
        seen.insert(name);
        mappings.push_back({input_index, d, name});
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
  Ort::ConstGraph graph{graphs[0]};

  // Determine how many variants we need (specialized + generic fallback).
  const auto& dim_spec_variants = ep->config_.dim_spec_variants;
  size_t num_specialized = dim_spec_variants.size();

  // Sort specialized variants by specificity (most specific first).
  std::vector<size_t> sorted_indices(num_specialized);
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&](size_t a, size_t b) {
              return VariantSpecificity(dim_spec_variants[a]) >
                     VariantSpecificity(dim_spec_variants[b]);
            });

  // Create temp files: one combined MLIR, one VMFB, one IRPA.
  TempFile mlir_file(".mlir");
  TempFile vmfb_file(".vmfb");
  TempFile irpa_file(".irpa");

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

  // Phase 1: Generate MLIR.
  ParameterIndexPtr parameter_index;
  ParameterProviderPtr parameter_provider;

  std::vector<std::pair<std::string, DimSpecVariant>> mlir_variants;
  for (size_t i = 0; i < num_specialized; ++i) {
    size_t variant_idx = sorted_indices[i];
    mlir_variants.emplace_back("_variant" + std::to_string(i),
                               dim_spec_variants[variant_idx]);
  }

  // Add a generic fallback variant.
  mlir_variants.emplace_back("", DimSpecVariant{});

  // function_names[i] is the MLIR function name for mlir_variants[i].
  std::vector<std::string> function_names;
  ORT_CXX_LOGF_NOEXCEPT(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                        "IREE EP: Generating MLIR (%zu variants)",
                        mlir_variants.size());
  ORT_RETURN_IF_ERROR(GenerateMlir(
      graph, ep->ort_api, mlir_file.Path(), irpa_file.Path(), mlir_variants,
      function_names, parameter_index, parameter_provider));

  // Phase 2: Compile the single MLIR to one VMFB.
  std::vector<std::string> flags = GenerateCompileFlags(ep->config_);
  ORT_CXX_LOG_NOEXCEPT(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                       "IREE EP: Compiling VMFB...");
  ORT_RETURN_IF_ERROR(
      CompileToVmfb(mlir_file.Path(), vmfb_file.Path(), flags, ep->ort_api));
  ORT_CXX_LOG_NOEXCEPT(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                       "IREE EP: VMFB compiled successfully");

  // Phase 3: Create session, load VMFB, lookup functions.
  RuntimeSessionPtr shared_session;
  {
    iree_runtime_session_options_t session_opts;
    iree_runtime_session_options_initialize(&session_opts);
    IREE_ORT_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
        ep->factory_.IreeInstance(), &session_opts, ep->IreeDevice(),
        iree_runtime_instance_host_allocator(ep->factory_.IreeInstance()),
        shared_session.ForOutput()));

    if (parameter_provider) {
      VmModulePtr parameters_module;
      iree_io_parameter_provider_t* provider_raw = parameter_provider.Get();
      IREE_ORT_RETURN_IF_ERROR(iree_io_parameters_module_create(
          iree_runtime_instance_vm_instance(ep->factory_.IreeInstance()), 1,
          &provider_raw,
          iree_runtime_instance_host_allocator(ep->factory_.IreeInstance()),
          parameters_module.ForOutput()));
      IREE_ORT_RETURN_IF_ERROR(iree_runtime_session_append_module(
          shared_session.Get(), parameters_module.Get()));
    }

    IREE_ORT_RETURN_IF_ERROR(
        iree_runtime_session_append_bytecode_module_from_file(
            shared_session.Get(), vmfb_file.Path().c_str()));
  }

  // Lookup each variant function using the exact names from MLIR generation.
  // IREE VMFB functions are in the "module" namespace.
  std::vector<IreeNodeComputeInfo::Variant> variants;
  variants.reserve(function_names.size());
  for (size_t i = 0; i < function_names.size(); ++i) {
    std::string vmfb_name = "module." + function_names[i];
    iree_vm_function_t function;
    IREE_ORT_RETURN_IF_ERROR(iree_runtime_session_lookup_function(
        shared_session.Get(), iree_make_cstring_view(vmfb_name.c_str()),
        &function));
    variants.push_back({function, mlir_variants[i].second});
  }

  // Build symbolic dimension mappings for runtime dispatch.
  auto dim_mappings = BuildSymbolicDimMappings(graph);

  // Validate that all dim_spec keys reference known symbolic dims.
  {
    std::unordered_set<std::string> known_dims;
    for (const auto& m : dim_mappings) {
      known_dims.insert(m.symbolic_name);
    }
    for (const auto& variant : dim_spec_variants) {
      for (const auto& spec : variant) {
        if (known_dims.contains(spec.symbolic_name)) continue;
        return Ort::Status(
                   std::format("dim_specs: key \"{}\" does not match any "
                               "symbolic dimension in the graph inputs",
                               spec.symbolic_name)
                       .c_str(),
                   ORT_INVALID_ARGUMENT)
            .release();
      }
    }
  }

  size_t num_variants = variants.size();
  auto* info =
      new IreeNodeComputeInfo(*ep, std::move(shared_session),
                              std::move(variants), std::move(dim_mappings));
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
// IreeNodeComputeInfo Implementation
// ============================================================================

IreeNodeComputeInfo::IreeNodeComputeInfo(
    IreeEp& ep_ref, RuntimeSessionPtr session, std::vector<Variant> variants,
    std::vector<SymbolicDimMapping> dim_mappings)
    : ep(ep_ref),
      session_(std::move(session)),
      variants_(std::move(variants)),
      dim_mappings_(std::move(dim_mappings)) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

IreeNodeComputeInfo::~IreeNodeComputeInfo() {
  // Note: Avoid using logger during cleanup - ORT logging infrastructure may
  // be torn down before our destructors run during Python interpreter shutdown.
  // Explicitly release sessions to ensure proper cleanup ordering.
  variants_.clear();
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
  Ort::KernelContext ctx(kernel_context);

  // --- Runtime variant dispatch ---
  // Build actual dim values from input shapes using the symbolic dim mappings.
  std::unordered_map<std::string, int64_t> dim_values;
  for (const auto& m : info->dim_mappings_) {
    if (m.input_index >= ctx.GetInputCount()) continue;
    auto shape =
        ctx.GetInput(m.input_index).GetTensorTypeAndShapeInfo().GetShape();
    if (m.dim_index >= shape.size()) continue;
    dim_values[m.symbolic_name] = shape[m.dim_index];
  }

  // Select best matching variant (iterate most specific to least).
  // The generic fallback is always last and always matches.
  const Variant* selected = nullptr;
  for (const auto& v : info->variants_) {
    bool match = true;
    for (const auto& spec : v.dim_specs) {
      auto it = dim_values.find(spec.symbolic_name);
      if (it == dim_values.end()) continue;
      if (spec.kind == DimSpec::Kind::kStatic && it->second != spec.value) {
        match = false;
        break;
      }
      if (spec.kind == DimSpec::Kind::kDivisibleBy &&
          (it->second <= 0 || it->second % spec.value != 0)) {
        match = false;
        break;
      }
    }
    if (!match) continue;
    selected = &v;
    break;
  }
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
