//===- mlir_gen.cc --------------------------------------------------------===//
//
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// MLIR text generation from OrtGraph.
//
//===----------------------------------------------------------------------===//

#include "mlir_gen.h"

#include <cassert>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "iree/io/file_handle.h"
#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree_ort_utils.h"

namespace onnxruntime::iree {
namespace {

// Initializers smaller than this are inlined via dense<> DenseElementsAttr.
// Larger ones become IREE parameters backed by an IRPA archive.
constexpr size_t kMaxInlineInitializerSize = 256;

// Encodes raw bytes as a hex string: "0xAABBCC...".
std::string HexEncode(const uint8_t* data, size_t size) {
  constexpr char hex_chars[] = "0123456789abcdef";
  std::string result;
  result.reserve(2 + size * 2);
  result = "0x";
  for (size_t i = 0; i < size; ++i) {
    result += hex_chars[(data[i] >> 4) & 0xF];
    result += hex_chars[data[i] & 0xF];
  }
  return result;
}

// Tracks a large initializer that will become an IREE parameter.
struct ParameterInitializer {
  std::string sanitized_name;
  size_t initializer_index;  // Index into initializers_ vector.
};

// Callback for iree_io_build_parameter_archive to create the IRPA file.
iree_status_t IrpaFileOpenCallback(void* user_data, iree_io_physical_offset_t,
                                   iree_io_physical_size_t,
                                   iree_io_file_handle_t** out_file_handle) {
  auto* path = static_cast<const std::string*>(user_data);
  return iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_OVERWRITE,
      iree_make_string_view(path->data(), path->size()),
      iree_allocator_system(), out_file_handle);
}

// Joins a vector of strings with a separator.
std::string Join(const std::vector<std::string>& parts,
                 const std::string& sep) {
  std::ostringstream ss;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      ss << sep;
    }
    ss << parts[i];
  }
  return ss.str();
}

// Returns MLIR element type string for an ONNX tensor element type.
// If signless is true, returns signless types (i64) for all integers.
// Otherwise returns signed (si64) or unsigned (ui64) types for torch dialect.
ErrorOr<std::string> GetElementType(ONNXTensorElementDataType dtype,
                                    bool signless = false) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "f32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "f64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "f16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "bf16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return signless ? "i8" : "si8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return signless ? "i16" : "si16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return signless ? "i32" : "si32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return signless ? "i64" : "si64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return signless ? "i8" : "ui8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return signless ? "i16" : "ui16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return signless ? "i32" : "ui32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return signless ? "i64" : "ui64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "i1";
    default:
      return error("Unsupported element type: {}", static_cast<int>(dtype));
  }
}

// Formats a tensor type as !torch.vtensor<[dims],dtype>.
// Dynamic dims are always emitted as "?".
ErrorOr<std::string> FormatTensorType(const Ort::ConstTypeInfo& type_info) {
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return error("Non-tensor type {} not supported",
                 static_cast<int>(type_info.GetONNXType()));
  }

  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  auto shape = tensor_info.GetShape();
  auto dtype = tensor_info.GetElementType();

  std::ostringstream ss;
  ss << "!torch.vtensor<[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    if (shape[i] < 0) {
      // TODO: Ensure that dynamic dimensions are actually represented as -1. I
      // checked and they seem to but an example to check would be good.
      ss << "?";
    } else {
      ss << shape[i];
    }
  }
  IREE_EP_ASSIGN_OR_RETURN(std::string elem_type, GetElementType(dtype));
  ss << "]," << elem_type << ">";
  return ss.str();
}

// Formats a tensor type as tensor<dimsxdtype> (standard MLIR format).
// Uses signless integer types as required by MLIR tensor dialect.
// Dynamic dims are always emitted as "?".
ErrorOr<std::string> FormatMlirTensorType(const Ort::ConstTypeInfo& type_info) {
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return error("Non-tensor type {} not supported",
                 static_cast<int>(type_info.GetONNXType()));
  }

  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  auto shape = tensor_info.GetShape();
  auto dtype = tensor_info.GetElementType();

  std::ostringstream ss;
  ss << "tensor<";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      ss << "?";
    } else {
      ss << shape[i];
    }
    ss << "x";
  }
  IREE_EP_ASSIGN_OR_RETURN(std::string elem_type,
                           GetElementType(dtype, /*signless=*/true));
  ss << elem_type << ">";
  return ss.str();
}

// Tries to parse an entire string as a non-negative integer.
// Returns true and sets value on success. Returns false on any invalid input
// (empty, non-digit characters, overflow). Uses std::from_chars — no
// exceptions.
bool ParseUnsigned(std::string_view s, size_t& value) {
  if (s.empty()) return false;
  size_t parsed{};
  auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), parsed);
  if (ec != std::errc{} || ptr != s.data() + s.size()) return false;
  value = parsed;
  return true;
}

// Parses an input reference of the form "$N" where N is a non-negative integer.
// Returns true and sets input_idx = N. Returns false for literal values
// (e.g., "4") or malformed specs (e.g., "$abc", "$").
bool ParseInputRef(const std::string& spec, size_t& input_idx) {
  if (spec.size() < 2 || spec[0] != '$') return false;
  return ParseUnsigned(std::string_view(spec).substr(1), input_idx);
}

// MLIR generator class.
class MlirGenerator {
 public:
  MlirGenerator(const Ort::ConstGraph& graph, std::ostream& out,
                const std::string& irpa_path, TargetConfig target_config)
      : graph_(graph),
        out_(out),
        irpa_path_(irpa_path),
        target_config_(std::move(target_config)) {}

  // A single (suffix, dim_specs) variant for MLIR generation.
  struct VariantInfo {
    std::string suffix;           // Function name suffix (e.g., "_variant0").
    const DimSpecVariant* specs;  // Dim specs for this variant.
  };

  // Generates an MLIR module containing one function per variant.
  // All functions share the same module (and thus the same parameter
  // references), so when compiled to a single VMFB the weights are shared.
  // For the unspecialized case, pass a single variant with empty suffix/specs.
  // Populates function_names with the MLIR function name for each variant.
  OrtStatus* Generate(const std::vector<VariantInfo>& variants,
                      std::vector<std::string>& function_names) {
    CollectMetadata();
    function_names.clear();
    function_names.reserve(variants.size());

    EmitModulePrologue();
    for (const auto& v : variants) {
      ConfigureForVariant(*v.specs, v.suffix);
      function_names.push_back(graph_name_);
      OrtStatus* status = EmitFunctionHeader();
      if (status) return status;
      status = EmitFunctionBody();
      if (status) return status;
      out_ << "  }\n";  // Close function.
    }
    out_ << "}\n";  // Close module.
    return nullptr;
  }

  // Builds an IRPA parameter archive for large initializers and creates a
  // parameter provider. Call after Generate(). If no parameters are needed,
  // the output pointers remain null.
  OrtStatus* BuildParameterArchive(ParameterIndexPtr& out_index,
                                   ParameterProviderPtr& out_provider);

 private:
  void CollectMetadata() {
    // Get IR version.
    ir_version_ = graph_.GetOnnxIRVersion();

    // Get opset version (find default domain).
    auto opsets = graph_.GetOperatorSets();
    for (const auto& opset : opsets) {
      if (opset.domain.empty() || opset.domain == "ai.onnx") {
        opset_version_ = opset.version;
        break;
      }
    }

    // Collect inputs (excluding initializers).
    auto inputs = graph_.GetInputs();
    auto initializers = graph_.GetInitializers();

    // Build set of initializer names.
    std::unordered_set<std::string> init_names;
    for (const auto& init : initializers) {
      init_names.insert(init.GetName());
    }

    // Graph inputs are those not in initializers.
    for (const auto& input : inputs) {
      std::string name = input.GetName();
      if (init_names.find(name) == init_names.end()) {
        graph_inputs_.push_back(input);
      }
    }

    // Graph outputs.
    graph_outputs_ = graph_.GetOutputs();

    // Initializers.
    initializers_ = initializers;

    // Identify large initializers that need IRPA parameter backing.
    for (size_t i = 0; i < initializers_.size(); ++i) {
      auto tensor_info =
          initializers_[i].TypeInfo().GetTensorTypeAndShapeInfo();
      size_t byte_size = tensor_info.GetElementCount() *
                         OnnxElementTypeSize(tensor_info.GetElementType());
      if (byte_size > kMaxInlineInitializerSize) {
        parameter_initializers_.push_back(
            {SanitizeName(initializers_[i].GetName()), i});
      }
    }

    // Collect symbolic dimension names for graph inputs.
    // These are used for dim spec matching (static specialization and
    // divisibility constraints).
    for (const auto& input : graph_inputs_) {
      if (input.TypeInfo().GetONNXType() != ONNX_TYPE_TENSOR) {
        input_symbolic_dims_.emplace_back();
        continue;
      }
      input_symbolic_dims_.push_back(
          input.TypeInfo().GetTensorTypeAndShapeInfo().GetSymbolicDimensions());
    }
  }

  // Configures the generator for a variant (dim specs, function name suffix).
  // Must be called before EmitFunctionHeader/EmitFunctionBody for each variant.
  void ConfigureForVariant(const DimSpecVariant& specs,
                           const std::string& suffix) {
    // Build constraint lookup: symbolic_name -> DimSpec.
    constraint_specs_.clear();
    for (const auto& spec : specs) {
      constraint_specs_[spec.symbolic_name] = spec;
    }

    // Determine which inputs have at least one constrained dynamic dim.
    constrained_inputs_.clear();
    for (size_t i = 0; i < graph_inputs_.size(); ++i) {
      if (graph_inputs_[i].TypeInfo().GetONNXType() != ONNX_TYPE_TENSOR)
        continue;
      auto shape =
          graph_inputs_[i].TypeInfo().GetTensorTypeAndShapeInfo().GetShape();
      const auto& sym_dims = input_symbolic_dims_[i];
      for (size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] < 0 && d < sym_dims.size() && sym_dims[d] &&
            sym_dims[d][0] != '\0' && constraint_specs_.contains(sym_dims[d])) {
          constrained_inputs_.insert(i);
          break;
        }
      }
    }

    // ONNX allows empty graph names; MLIR functions need a valid name.
    std::string raw_name = graph_.GetName();
    graph_name_ = raw_name.empty() ? "main" : SanitizeName(raw_name);
    graph_name_ += suffix;
  }

  // Emits the module-level prologue.
  void EmitModulePrologue() {
    // Define the #executable_target alias for hal.dispatch.extern objects()
    // clauses. This is harmless when no extern dispatches are present.
    // We intentionally do NOT set hal.device.targets on the module — device
    // targeting is handled by iree-compile CLI flags (--iree-hal-target-device,
    // --iree-hip-target, etc.).
    if (!target_config_.hal_backend.empty()) {
      out_ << "#executable_target = #hal.executable.target<\""
           << target_config_.hal_backend << "\", \""
           << target_config_.hal_format << "\", {target_arch = \""
           << target_config_.target_arch << "\"}>\n";
    }
    out_ << "module {\n";
  }

  // Emits the function signature with current dim specs and function name.
  // Constrained input args are renamed to %name__orig so EmitDimConstraints()
  // can rebind the original name after applying shape assumptions.
  OrtStatus* EmitFunctionHeader() {
    std::ostringstream args;
    for (size_t i = 0; i < graph_inputs_.size(); ++i) {
      if (i > 0) {
        args << ", ";
      }
      std::string name = SanitizeName(graph_inputs_[i].GetName());
      IREE_ORT_ASSIGN_OR_RETURN(std::string type,
                                FormatTensorType(graph_inputs_[i].TypeInfo()));
      if (constrained_inputs_.contains(i)) {
        args << "%" << name << "__orig: " << type;
      } else {
        args << "%" << name << ": " << type;
      }
    }

    // Build return types.
    std::ostringstream ret_types;
    for (size_t i = 0; i < graph_outputs_.size(); ++i) {
      if (i > 0) {
        ret_types << ", ";
      }
      IREE_ORT_ASSIGN_OR_RETURN(std::string ret_type,
                                FormatTensorType(graph_outputs_[i].TypeInfo()));
      ret_types << ret_type;
    }

    constexpr std::string_view func_schema =
        R"(  func.func @{0}({1}) -> ({2})
      attributes {{
        torch.onnx_meta.ir_version = {3} : si64,
        torch.onnx_meta.opset_version = {4} : si64,
        torch.onnx_meta.producer_name = "onnxruntime-ep-iree",
        torch.onnx_meta.producer_version = ""
      }} {{
)";

    out_ << std::format(func_schema,
                        graph_name_,      // {0}
                        args.str(),       // {1}
                        ret_types.str(),  // {2}
                        ir_version_,      // {3}
                        opset_version_);  // {4}
    return nullptr;
  }

  OrtStatus* EmitFunctionBody() {
    // Emit dim constraints (util.assume.int + flow.tensor.tie_shape).
    OrtStatus* dim_status = EmitDimConstraints();
    if (dim_status) return dim_status;

    // Emit initializers as flow.tensor.constant ops.
    for (const auto& init : initializers_) {
      OrtStatus* status = EmitInitializer(init);
      if (status) return status;
    }

    // Emit nodes.
    auto nodes = graph_.GetNodes();
    for (const auto& node : nodes) {
      OrtStatus* status = EmitNode(node);
      if (status) return status;
    }

    // Emit return.
    return EmitReturn();
  }

  // Emits an initializer as a flow.tensor.constant with a
  // torch_c.from_builtin_tensor cast. Small initializers use dense<> with
  // inline hex-encoded data. Large initializers use #flow.parameter.named
  // (data stored in IRPA archive).
  //
  // Output format (small):
  //   %__raw_name = flow.tensor.constant dense<"0x..."> : tensor<...>
  //   %name = torch_c.from_builtin_tensor %__raw_name : tensor<...>
  //       -> !torch.vtensor<[...],dtype>
  //
  // Output format (large):
  //   %__raw_name = flow.tensor.constant
  //       #flow.parameter.named<"model"::"name"> : tensor<...>
  //   %name = torch_c.from_builtin_tensor %__raw_name : tensor<...>
  //       -> !torch.vtensor<[...],dtype>
  OrtStatus* EmitInitializer(const Ort::ConstValueInfo& init) {
    std::string name = SanitizeName(init.GetName());
    IREE_ORT_ASSIGN_OR_RETURN(std::string vtensor_type,
                              FormatTensorType(init.TypeInfo()));
    IREE_ORT_ASSIGN_OR_RETURN(std::string tensor_type,
                              FormatMlirTensorType(init.TypeInfo()));

    auto tensor_info = init.TypeInfo().GetTensorTypeAndShapeInfo();
    size_t byte_size = tensor_info.GetElementCount() *
                       OnnxElementTypeSize(tensor_info.GetElementType());

    if (byte_size <= kMaxInlineInitializerSize) {
      // Small: inline with dense<> DenseElementsAttr.
      Ort::ConstValue tensor_value{nullptr};
      auto status = init.GetInitializer(tensor_value);
      if (!status.IsOK()) {
        return status.release();
      }
      const auto* data =
          static_cast<const uint8_t*>(tensor_value.GetTensorRawData());
      std::string hex = HexEncode(data, tensor_value.GetTensorSizeInBytes());

      constexpr std::string_view schema =
          R"(    %__raw_{0} = flow.tensor.constant dense<"{3}"> : {1}
    %{0} = torch_c.from_builtin_tensor %__raw_{0} : {1} -> {2}
)";
      out_ << std::format(schema, name, tensor_type, vtensor_type, hex);
    } else {
      // Large: parameter reference. Data stored in IRPA archive.
      constexpr std::string_view schema =
          R"(    %__raw_{0} = flow.tensor.constant #flow.parameter.named<"model"::"{0}"> : {1}
    %{0} = torch_c.from_builtin_tensor %__raw_{0} : {1} -> {2}
)";
      out_ << std::format(schema, name, tensor_type, vtensor_type);
    }
    return nullptr;
  }

  OrtStatus* EmitNode(const Ort::ConstNode& node) {
    if (node.GetDomain() == "com.iree" &&
        node.GetOperatorType() == "ExternDispatch") {
      return EmitExternDispatch(node, extern_id_++);
    }

    std::string op_type = node.GetOperatorType();
    auto inputs = node.GetInputs();
    auto outputs = node.GetOutputs();
    auto attrs = node.GetAttributes();

    // Build output SSA names and types.
    std::ostringstream out_names;
    std::ostringstream out_types;
    bool first_output = true;
    size_t valid_output_count = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (!outputs[i]) {
        // Skip invalid outputs (optional outputs can be empty/null).
        continue;
      }
      std::string output_name = outputs[i].GetName();
      if (output_name.empty()) {
        // Skip empty outputs.
        continue;
      }
      if (!first_output) {
        out_names << ", ";
        out_types << ", ";
      }
      first_output = false;
      valid_output_count++;
      out_names << "%" << SanitizeName(output_name);
      IREE_ORT_ASSIGN_OR_RETURN(std::string out_type,
                                FormatTensorType(outputs[i].TypeInfo()));
      out_types << out_type;
    }

    // Build input SSA references.
    std::ostringstream in_names;
    std::ostringstream in_types;
    bool first_input = true;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i]) {
        // Skip invalid inputs (optional inputs can be empty/null).
        continue;
      }
      std::string input_name = inputs[i].GetName();
      if (input_name.empty()) {
        continue;
      }
      if (!first_input) {
        in_names << ", ";
        in_types << ", ";
      }
      first_input = false;
      in_names << "%" << SanitizeName(input_name);
      IREE_ORT_ASSIGN_OR_RETURN(std::string in_type,
                                FormatTensorType(inputs[i].TypeInfo()));
      in_types << in_type;
    }

    // Build attributes.
    IREE_ORT_ASSIGN_OR_RETURN(std::string attr_str, FormatAttributes(attrs));

    // Format output types: wrap in parentheses if multiple outputs.
    std::string out_types_str = out_types.str();
    if (valid_output_count > 1 && !out_types_str.empty()) {
      out_types_str = "(" + out_types_str + ")";
    }

    // Emit the operator.
    constexpr std::string_view schema =
        R"(    {0} = torch.operator "onnx.{1}"({2}) {{{3}}} : ({4}) -> {5}
)";
    out_ << std::format(schema,
                        out_names.str(),  // {0}
                        op_type,          // {1}
                        in_names.str(),   // {2}
                        attr_str,         // {3}
                        in_types.str(),   // {4}
                        out_types_str);   // {5}
    return nullptr;
  }

  ErrorOr<std::string> FormatAttributes(
      const std::vector<Ort::ConstOpAttr>& attrs) {
    if (attrs.empty()) {
      return "";
    }

    std::ostringstream ss;
    for (size_t i = 0; i < attrs.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      IREE_EP_ASSIGN_OR_RETURN(std::string formatted,
                               FormatAttribute(attrs[i]));
      ss << formatted;
    }
    return ss.str();
  }

  ErrorOr<std::string> FormatAttribute(const Ort::ConstOpAttr& attr) {
    std::string name = attr.GetName();
    OrtOpAttrType type = attr.GetType();

    switch (type) {
      case ORT_OP_ATTR_INT: {
        int64_t value = 0;
        attr.GetValue(value);
        return std::format("torch.onnx.{0} = {1} : si64", name, value);
      }
      case ORT_OP_ATTR_FLOAT: {
        float value = 0.0f;
        attr.GetValue(value);
        return std::format("torch.onnx.{0} = {1:e} : f32", name, value);
      }
      case ORT_OP_ATTR_STRING: {
        std::string value;
        attr.GetValue(value);
        return std::format("torch.onnx.{0} = \"{1}\"", name, value);
      }
      case ORT_OP_ATTR_INTS: {
        std::vector<int64_t> values;
        attr.GetValueArray<int64_t>(values);
        std::vector<std::string> str_values(values.size());
        std::transform(values.begin(), values.end(), str_values.begin(),
                       [](int64_t v) { return std::format("{0} : si64", v); });
        return std::format("torch.onnx.{0} = [{1}]", name,
                           Join(str_values, ", "));
      }
      case ORT_OP_ATTR_TENSOR: {
        Ort::Value tensor_value{nullptr};
        auto status = attr.GetTensorAttributeAsOrtValue(tensor_value);
        if (!status.IsOK()) {
          return error("Failed to read tensor attribute '{}': {}", name,
                       status.GetErrorMessage());
        }
        auto type_info = tensor_value.GetTensorTypeAndShapeInfo();
        auto shape = type_info.GetShape();
        auto dtype = type_info.GetElementType();
        size_t count = type_info.GetElementCount();
        const auto* raw =
            static_cast<const uint8_t*>(tensor_value.GetTensorRawData());

        // Only splat (single-element) tensor attributes are supported.
        // Per the ONNX spec, ConstantOfShape.value is the only TENSOR
        // attribute that reaches an EP and it is always 1-element.
        if (count != 1) {
          return error(
              "Unsupported tensor attribute '{}': expected 1-element splat, "
              "got {} elements",
              name, count);
        }

        // Format as dense<hex> : tensor<shape x dtype>.
        // Hex encoding is lossless for all types and matches EmitInitializer.
        std::string dense = std::format(
            "dense<\"{}\">", HexEncode(raw, OnnxElementTypeSize(dtype)));

        // Build tensor type using signed integer types (matching
        // torch.onnx.value format).
        std::ostringstream type_ss;
        type_ss << "tensor<";
        for (size_t j = 0; j < shape.size(); ++j) {
          type_ss << shape[j] << "x";
        }
        IREE_EP_ASSIGN_OR_RETURN(std::string elem_type,
                                 GetElementType(dtype, /*signless=*/false));
        type_ss << elem_type << ">";

        return std::format("torch.onnx.{0} = {1} : {2}", name, dense,
                           type_ss.str());
      }
      default:
        return error("Unsupported attribute type {} for '{}'",
                     static_cast<int>(type), name);
    }
  }

  // Emits ops to resolve a spec string to an SSA value. The spec is either:
  // - A literal integer (emits arith.constant)
  // - "$N" input reference (extracts scalar from input tensor N)
  // as_i32: if true, result is i32; otherwise index.
  // context: used in error messages, e.g. "push_constants[0]".
  OrtStatus* EmitResolvedValue(
      const std::string& spec, const std::string& result_name, bool as_i32,
      const std::vector<std::string>& raw_input_names,
      const std::vector<std::string>& raw_input_types,
      const std::vector<std::string>& raw_input_elem_types,
      const std::vector<size_t>& raw_input_ranks, const std::string& context) {
    size_t input_idx;
    if (ParseInputRef(spec, input_idx)) {
      if (input_idx >= raw_input_names.size()) {
        return MakeError(
            "ExternDispatch: {} = '{}' references input[{}] but only {} "
            "inputs available",
            context, spec, input_idx, raw_input_names.size());
      }
      if (raw_input_names[input_idx].empty()) {
        return MakeError(
            "ExternDispatch: {} = '{}' references input[{}] which is "
            "null or empty",
            context, spec, input_idx);
      }
      if (raw_input_ranks[input_idx] != 0) {
        return MakeError(
            "ExternDispatch: {} = '{}' references input[{}] which has "
            "rank {} (must be a scalar tensor, rank 0)",
            context, spec, input_idx, raw_input_ranks[input_idx]);
      }
      const std::string& elem_type = raw_input_elem_types[input_idx];
      if (elem_type != "i64" && elem_type != "i32") {
        return MakeError(
            "ExternDispatch: {} = '{}' references input[{}] with element "
            "type {} (must be i32 or i64)",
            context, spec, input_idx, elem_type);
      }
      // Extract scalar from the rank-0 tensor.
      if (as_i32 && elem_type == "i32") {
        // Already i32 — extract directly into result, no cast needed.
        out_ << std::format("    %{} = tensor.extract %{}[] : {}\n",
                            result_name, raw_input_names[input_idx],
                            raw_input_types[input_idx]);
      } else {
        std::string scalar = result_name + "_scalar";
        out_ << std::format("    %{} = tensor.extract %{}[] : {}\n", scalar,
                            raw_input_names[input_idx],
                            raw_input_types[input_idx]);
        // Cast to the target type.
        if (as_i32) {
          // elem_type is i64 here (i32 case handled above).
          out_ << std::format("    %{} = arith.trunci %{} : i64 to i32\n",
                              result_name, scalar);
        } else {
          out_ << std::format("    %{} = arith.index_cast %{} : {} to index\n",
                              result_name, scalar, elem_type);
        }
      }
    } else {
      size_t value;
      if (!ParseUnsigned(spec, value)) {
        return MakeError(
            "ExternDispatch: {} = '{}' is not a valid integer literal "
            "or input reference ($N)",
            context, spec);
      }
      if (as_i32 && value > static_cast<size_t>(INT32_MAX)) {
        return MakeError("ExternDispatch: {} = '{}' exceeds i32 range (max {})",
                         context, spec, INT32_MAX);
      }
      out_ << std::format("    %{} = arith.constant {} : {}\n", result_name,
                          spec, as_i32 ? "i32" : "index");
    }
    return nullptr;
  }

  // Emits a hal.dispatch.extern for a com.iree:ExternDispatch ONNX node.
  OrtStatus* EmitExternDispatch(const Ort::ConstNode& node, int extern_id) {
    std::string prefix = std::format("__extern_{}", extern_id);
    auto inputs = node.GetInputs();
    auto outputs = node.GetOutputs();

    // Parse ExternDispatch attributes.
    std::string kernel_name;
    std::string kernel_object;
    std::vector<int64_t> workgroup_size;
    std::vector<std::string> push_constants;
    std::vector<std::string> workgroup_count;

    for (const auto& attr : node.GetAttributes()) {
      std::string name = attr.GetName();
      if (name == "kernel_name") {
        attr.GetValue(kernel_name);
      } else if (name == "kernel_object") {
        attr.GetValue(kernel_object);
      } else if (name == "workgroup_size") {
        attr.GetValueArray<int64_t>(workgroup_size);
      } else if (name == "push_constants") {
        attr.GetValueArray<std::string>(push_constants);
      } else if (name == "workgroup_count") {
        attr.GetValueArray<std::string>(workgroup_count);
      }
    }

    // Validate backend supports extern dispatch (must have a HAL target).
    if (target_config_.hal_backend.empty()) {
      return MakeError(
          "ExternDispatch: backend '{}' does not support extern dispatch",
          target_config_.backend);
    }

    // Validate required attributes.
    if (kernel_name.empty()) {
      return MakeError(
          "ExternDispatch: missing required 'kernel_name' attribute");
    }
    if (kernel_object.empty()) {
      return MakeError(
          "ExternDispatch: missing required 'kernel_object' attribute");
    }
    if (workgroup_size.size() != 3) {
      return MakeError(
          "ExternDispatch: 'workgroup_size' must have exactly 3 elements "
          "(X, Y, Z), got {}",
          workgroup_size.size());
    }
    for (size_t i = 0; i < 3; ++i) {
      if (workgroup_size[i] <= 0) {
        return MakeError(
            "ExternDispatch: workgroup_size[{}] = {} must be positive", i,
            workgroup_size[i]);
      }
    }
    if (workgroup_count.size() != 3) {
      return MakeError(
          "ExternDispatch: 'workgroup_count' must have "
          "exactly 3 elements (X, Y, Z), got {}",
          workgroup_count.size());
    }

    // Pre-scan push_constants and workgroup_count for $N input references.
    // Inputs referenced by $N are scalar tensors consumed as values, NOT
    // data buffers — they must be excluded from dispatch args and bindings.
    std::unordered_set<size_t> scalar_input_indices;
    for (const auto& spec : push_constants) {
      size_t idx;
      if (ParseInputRef(spec, idx)) scalar_input_indices.insert(idx);
    }
    for (const auto& spec : workgroup_count) {
      size_t idx;
      if (ParseInputRef(spec, idx)) scalar_input_indices.insert(idx);
    }

    // Step 1: Bridge ALL inputs from torch to builtin tensor.
    // Vectors are indexed by original ONNX input position so that $N input
    // refs resolve correctly. Scalar inputs (referenced by $N) are bridged
    // too — they need builtin tensors for tensor.extract.
    std::vector<std::string> raw_input_names(inputs.size());
    std::vector<std::string> raw_input_types(inputs.size());
    std::vector<std::string> raw_input_elem_types(inputs.size());
    std::vector<size_t> raw_input_ranks(inputs.size(), 0);
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i]) continue;
      std::string input_name = inputs[i].GetName();
      if (input_name.empty()) continue;

      std::string ssa = SanitizeName(input_name);
      IREE_ORT_ASSIGN_OR_RETURN(std::string vtensor_type,
                                FormatTensorType(inputs[i].TypeInfo()));
      IREE_ORT_ASSIGN_OR_RETURN(std::string tensor_type,
                                FormatMlirTensorType(inputs[i].TypeInfo()));
      auto tensor_info = inputs[i].TypeInfo().GetTensorTypeAndShapeInfo();
      std::string raw = std::format("{}_raw_{}", prefix, i);
      raw_input_names[i] = raw;
      raw_input_types[i] = tensor_type;
      IREE_ORT_ASSIGN_OR_RETURN(
          raw_input_elem_types[i],
          GetElementType(tensor_info.GetElementType(), /*signless=*/true));
      raw_input_ranks[i] = tensor_info.GetShape().size();

      out_ << std::format(
          "    %{} = torch_c.to_builtin_tensor %{} : {} -> {}\n", raw, ssa,
          vtensor_type, tensor_type);
    }

    // Step 2: Emit push constants.
    std::vector<std::string> pc_names;
    for (size_t i = 0; i < push_constants.size(); ++i) {
      std::string pc = std::format("{}_pc{}", prefix, i);
      pc_names.push_back(pc);
      OrtStatus* status = EmitResolvedValue(
          push_constants[i], pc, /*as_i32=*/true, raw_input_names,
          raw_input_types, raw_input_elem_types, raw_input_ranks,
          std::format("push_constants[{}]", i));
      if (status) return status;
    }

    // Step 3: Compute workload values from workgroup_count[0..2].
    // All three dimensions are resolved (literal or $N input reference) and
    // passed as workload arguments to hal.dispatch.extern. The count region
    // receives them as block arguments and returns them directly.
    std::vector<std::string> workload_names;
    for (size_t i = 0; i < 3; ++i) {
      std::string wl = std::format("{}_workload_{}", prefix, i);
      workload_names.push_back(wl);
      OrtStatus* status = EmitResolvedValue(
          workgroup_count[i], wl, /*as_i32=*/false, raw_input_names,
          raw_input_types, raw_input_elem_types, raw_input_ranks,
          std::format("workgroup_count[{}]", i));
      if (status) return status;
    }

    // Step 4: Build output names and types.
    // ExternDispatch is opaque — ORT cannot infer its output shapes. The ONNX
    // model must declare explicit type info for all ExternDispatch outputs
    // (via graph value_info entries).
    std::vector<std::string> out_raw_names;
    std::vector<std::string> out_raw_types;
    std::vector<std::string> out_vtensor_types;
    std::vector<std::string> out_ssa_names;
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (!outputs[i]) continue;
      std::string output_name = outputs[i].GetName();
      if (output_name.empty()) continue;
      out_raw_names.push_back(std::format("{}_out{}", prefix, i));
      IREE_ORT_ASSIGN_OR_RETURN(std::string raw_type,
                                FormatMlirTensorType(outputs[i].TypeInfo()));
      out_raw_types.push_back(std::move(raw_type));
      IREE_ORT_ASSIGN_OR_RETURN(std::string vtensor_type,
                                FormatTensorType(outputs[i].TypeInfo()));
      out_vtensor_types.push_back(std::move(vtensor_type));
      out_ssa_names.push_back(SanitizeName(output_name));
    }

    if (out_raw_names.empty()) {
      return MakeError(
          "ExternDispatch: node has no valid outputs (all outputs are "
          "null or have empty names)");
    }

    // Step 5: Emit hal.dispatch.extern.
    // Pre-build variable parts for the dispatch template.
    std::vector<std::string> dispatch_outs;
    for (const auto& name : out_raw_names) {
      dispatch_outs.push_back("%" + name);
    }

    std::vector<std::string> dispatch_args;
    for (const auto& pc : pc_names) {
      dispatch_args.push_back("%" + pc);
    }
    for (size_t i = 0; i < raw_input_names.size(); ++i) {
      if (!raw_input_names[i].empty() && !scalar_input_indices.count(i)) {
        dispatch_args.push_back("%" + raw_input_names[i]);
      }
    }

    std::vector<std::string> dispatch_arg_types;
    for (size_t i = 0; i < pc_names.size(); ++i) {
      dispatch_arg_types.push_back("i32");
    }
    for (size_t i = 0; i < raw_input_types.size(); ++i) {
      if (!raw_input_types[i].empty() && !scalar_input_indices.count(i)) {
        dispatch_arg_types.push_back(raw_input_types[i]);
      }
    }

    std::string ret_types_str;
    if (out_raw_types.size() == 1) {
      ret_types_str = out_raw_types[0];
    } else {
      ret_types_str = "(" + Join(out_raw_types, ", ") + ")";
    }

    std::vector<std::string> bindings;
    for (size_t i = 0; i < raw_input_names.size(); ++i) {
      if (!raw_input_names[i].empty() && !scalar_input_indices.count(i)) {
        bindings.push_back(
            "        #hal.pipeline.binding<storage_buffer, ReadOnly>");
      }
    }
    for (size_t i = 0; i < out_raw_names.size(); ++i) {
      bindings.push_back("        #hal.pipeline.binding<storage_buffer>");
    }

    std::vector<std::string> wg_size_parts;
    for (auto s : workgroup_size) {
      wg_size_parts.push_back(std::format("{} : index", s));
    }

    constexpr std::string_view dispatch_schema =
        R"(    {0} = hal.dispatch.extern "{1}"[%{2}, %{3}, %{4}]({5})
        : ({6}) -> {7}
      count(%device: !hal.device, %wl_x: index, %wl_y: index, %wl_z: index) -> (index, index, index) {{
        hal.return %wl_x, %wl_y, %wl_z : index, index, index
      }}
      layout(#hal.pipeline.layout<constants = {8}, bindings = [
{9}      ]>)
      objects({{
        #executable_target ordinal(0) = [
          #hal.executable.object<{{path = "{10}"}}>
        ]
      }})
      attributes {{workgroup_size = [{11}]}}
)";
    out_ << std::format(dispatch_schema, Join(dispatch_outs, ", "),  // {0}
                        kernel_name,                                 // {1}
                        workload_names[0],                           // {2}
                        workload_names[1],                           // {3}
                        workload_names[2],                           // {4}
                        Join(dispatch_args, ", "),                   // {5}
                        Join(dispatch_arg_types, ", "),              // {6}
                        ret_types_str,                               // {7}
                        pc_names.size(),                             // {8}
                        Join(bindings, ",\n") + "\n",                // {9}
                        kernel_object,                               // {10}
                        Join(wg_size_parts, ", "));                  // {11}

    // Step 6: Bridge outputs back to torch types.
    for (size_t i = 0; i < out_raw_names.size(); ++i) {
      out_ << std::format(
          "    %{} = torch_c.from_builtin_tensor %{} : {} -> {}\n",
          out_ssa_names[i], out_raw_names[i], out_raw_types[i],
          out_vtensor_types[i]);
    }
    return nullptr;
  }

  OrtStatus* EmitReturn() {
    std::ostringstream ret_values;
    std::ostringstream ret_types;
    for (size_t i = 0; i < graph_outputs_.size(); ++i) {
      if (i > 0) {
        ret_values << ", ";
        ret_types << ", ";
      }
      ret_values << "%" << SanitizeName(graph_outputs_[i].GetName());
      IREE_ORT_ASSIGN_OR_RETURN(std::string ret_type,
                                FormatTensorType(graph_outputs_[i].TypeInfo()));
      ret_types << ret_type;
    }

    out_ << std::format("    return {0} : {1}\n", ret_values.str(),
                        ret_types.str());
    return nullptr;
  }

  // Info about a constrained input's dynamic dimensions.
  struct DynDimInfo {
    size_t dim_idx;
    std::string dim_ssa;        // e.g., %input__d0
    std::string symbolic_name;  // empty if unconstrained
  };

  struct InputConstraintInfo {
    std::string name;
    std::string vtensor_type;
    std::string tensor_type;
    std::vector<DynDimInfo> dynamic_dims;
  };

  // Canonical assume: one per constrained symbolic dim name.
  struct CanonicalAssumeInfo {
    std::string symbolic_name;
    std::string ssa_name;        // e.g., %batch_assumed
    std::string source_dim_ssa;  // dim SSA from first occurrence
    const DimSpec* spec;
  };

  ErrorOr<std::vector<InputConstraintInfo>> CollectConstrainedInputInfos(
      std::vector<CanonicalAssumeInfo>& out_canonical_assumes,
      std::unordered_map<std::string, size_t>& out_assume_index) const {
    std::vector<InputConstraintInfo> infos;
    infos.reserve(constrained_inputs_.size());

    for (size_t i = 0; i < graph_inputs_.size(); ++i) {
      if (!constrained_inputs_.contains(i)) continue;

      InputConstraintInfo info;
      info.name = SanitizeName(graph_inputs_[i].GetName());
      IREE_EP_ASSIGN_OR_RETURN(info.vtensor_type,
                               FormatTensorType(graph_inputs_[i].TypeInfo()));
      IREE_EP_ASSIGN_OR_RETURN(
          info.tensor_type, FormatMlirTensorType(graph_inputs_[i].TypeInfo()));

      auto shape =
          graph_inputs_[i].TypeInfo().GetTensorTypeAndShapeInfo().GetShape();
      const auto& sym_dims = input_symbolic_dims_[i];
      for (size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] >= 0) continue;  // static dim in ONNX model, skip

        std::string sym_name;
        if (d < sym_dims.size() && sym_dims[d] && sym_dims[d][0] != '\0') {
          sym_name = sym_dims[d];
        }

        std::string dim_ssa = std::format("%{}__d{}", info.name, d);
        info.dynamic_dims.push_back({d, dim_ssa, sym_name});

        auto spec_it = constraint_specs_.find(sym_name);
        if (sym_name.empty() || spec_it == constraint_specs_.end()) {
          continue;
        }
        if (!out_assume_index.contains(sym_name)) {
          out_assume_index[sym_name] = out_canonical_assumes.size();
          out_canonical_assumes.push_back(CanonicalAssumeInfo{
              sym_name,
              std::format("%dim_assumed_{}", out_canonical_assumes.size()),
              dim_ssa, &spec_it->second});
        }
      }

      infos.push_back(std::move(info));
    }

    return infos;
  }

  void EmitInputDimExtraction(const std::vector<InputConstraintInfo>& infos) {
    std::unordered_set<size_t> emitted_constants;
    for (const auto& info : infos) {
      out_ << std::format(
          "    %{}__builtin = torch_c.to_builtin_tensor %{}__orig : {} -> {}\n",
          info.name, info.name, info.vtensor_type, info.tensor_type);

      for (const auto& dim : info.dynamic_dims) {
        if (!emitted_constants.contains(dim.dim_idx)) {
          out_ << std::format("    %c{} = arith.constant {} : index\n",
                              dim.dim_idx, dim.dim_idx);
          emitted_constants.insert(dim.dim_idx);
        }
        out_ << std::format("    {} = tensor.dim %{}__builtin, %c{} : {}\n",
                            dim.dim_ssa, info.name, dim.dim_idx,
                            info.tensor_type);
      }
    }
  }

  void EmitCanonicalAssumes(
      const std::vector<CanonicalAssumeInfo>& canonical_assumes) {
    for (const auto& assume : canonical_assumes) {
      if (assume.spec->div > 1) {
        out_ << std::format(
            "    {} = util.assume.int {}<umin = {}, umax = {}, udiv = {}> "
            ": index\n",
            assume.ssa_name, assume.source_dim_ssa, assume.spec->min,
            assume.spec->max, assume.spec->div);
      } else {
        out_ << std::format(
            "    {} = util.assume.int {}<umin = {}, umax = {}> : index\n",
            assume.ssa_name, assume.source_dim_ssa, assume.spec->min,
            assume.spec->max);
      }
    }
  }

  void EmitTieShapeRebinding(
      const std::vector<InputConstraintInfo>& infos,
      const std::vector<CanonicalAssumeInfo>& canonical_assumes,
      const std::unordered_map<std::string, size_t>& assume_index) {
    for (const auto& info : infos) {
      std::ostringstream operands;
      for (size_t j = 0; j < info.dynamic_dims.size(); ++j) {
        if (j > 0) operands << ", ";
        const auto& dim = info.dynamic_dims[j];
        auto it = assume_index.find(dim.symbolic_name);
        if (it != assume_index.end()) {
          operands << canonical_assumes[it->second].ssa_name;
        } else {
          operands << dim.dim_ssa;
        }
      }

      out_ << std::format(
          "    %{}__tied = flow.tensor.tie_shape %{}__builtin : {}{{{}}}\n",
          info.name, info.name, info.tensor_type, operands.str());
      out_ << std::format(
          "    %{} = torch_c.from_builtin_tensor %{}__tied : {} -> {}\n",
          info.name, info.name, info.tensor_type, info.vtensor_type);
    }
  }

  // Emits util.assume.int + flow.tensor.tie_shape ops for constrained dims.
  // Range-only specs (div == 0): util.assume.int with umin, umax.
  // Range+div specs (div > 0): util.assume.int with umin, umax, udiv.
  // For a symbolic dim name shared across multiple inputs, a single canonical
  // assumed SSA value is emitted and reused in all corresponding tie_shape ops.
  OrtStatus* EmitDimConstraints() {
    if (constraint_specs_.empty()) return nullptr;

    std::vector<CanonicalAssumeInfo> canonical_assumes;
    std::unordered_map<std::string, size_t> assume_index;

    IREE_ORT_ASSIGN_OR_RETURN(auto infos, CollectConstrainedInputInfos(
                                              canonical_assumes, assume_index));
    if (infos.empty()) return nullptr;

    EmitInputDimExtraction(infos);
    EmitCanonicalAssumes(canonical_assumes);
    EmitTieShapeRebinding(infos, canonical_assumes, assume_index);
    return nullptr;
  }

  // Member variables.
  const Ort::ConstGraph& graph_;
  std::ostream& out_;
  std::string irpa_path_;
  TargetConfig target_config_;
  // Lookup map: symbolic_name -> DimSpec for all specs in current variant.
  std::unordered_map<std::string, DimSpec> constraint_specs_;
  // Input indices that have at least one constrained dynamic dim.
  std::unordered_set<size_t> constrained_inputs_;
  std::string graph_name_;
  int64_t ir_version_ = 8;
  int64_t opset_version_ = 17;

  std::vector<Ort::ConstValueInfo> graph_inputs_;
  std::vector<Ort::ConstValueInfo> graph_outputs_;
  std::vector<Ort::ConstValueInfo> initializers_;
  std::vector<ParameterInitializer> parameter_initializers_;

  // Extern dispatch state.
  int extern_id_ = 0;
  // Symbolic dimension names per graph input (parallel to graph_inputs_).
  std::vector<std::vector<const char*>> input_symbolic_dims_;
};

// Builds an IRPA parameter archive for large initializers.
//
// Large inline initializers are copied into an IRPA (IREE Parameter Archive)
// file on disk. We write to an IRPA file rather than keeping data in memory
// because ORT does not guarantee that initializer tensor data remains valid
// beyond the Compile() call. By persisting to disk via IRPA, the data is
// accessed at runtime through IREE's parameter index with file-backed entries.
//
// External initializers (already backed by external files) are added to the
// parameter index directly, pointing to their original files without copying.
//
// The resulting parameter provider is registered with the IREE session so that
// the compiled module can resolve #flow.parameter.named references at runtime.
OrtStatus* MlirGenerator::BuildParameterArchive(
    ParameterIndexPtr& out_index, ParameterProviderPtr& out_provider) {
  if (parameter_initializers_.empty()) {
    return nullptr;
  }

  iree_allocator_t allocator = iree_allocator_system();

  // Build source index from ORT tensor data wrapped as file handles.
  // The tensor data is valid for the duration of this call (we are inside
  // CompileImpl). iree_io_build_parameter_archive copies it to the IRPA file.
  ParameterIndexPtr source_index;
  IREE_ORT_RETURN_IF_ERROR(
      iree_io_parameter_index_create(allocator, source_index.ForOutput()));

  for (const auto& param : parameter_initializers_) {
    const auto& init = initializers_[param.initializer_index];

    // Skip external initializers — added to target index later.
    // Note: GetExternalInitializerInfo returns OK with null output for
    // non-external initializers, so we must check both status and pointer.
    Ort::ExternalInitializerInfo ext_info(nullptr);
    ORT_RETURN_IF_ERROR(init.GetExternalInitializerInfo(ext_info).release());
    if (ext_info) {
      continue;
    }

    Ort::ConstValue tensor(nullptr);
    auto status = init.GetInitializer(tensor);
    if (!status.IsOK()) {
      return MakeError("Failed to get initializer: {}", init.GetName());
    }

    auto* data = const_cast<uint8_t*>(
        static_cast<const uint8_t*>(tensor.GetTensorRawData()));
    size_t size = tensor.GetTensorSizeInBytes();

    FileHandlePtr handle;
    iree_byte_span_t span = {data, static_cast<iree_host_size_t>(size)};
    IREE_ORT_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ, span,
        iree_io_file_handle_release_callback_null(), allocator,
        handle.ForOutput()));

    iree_io_parameter_index_entry_t entry = {};
    entry.key = iree_make_string_view(param.sanitized_name.data(),
                                      param.sanitized_name.size());
    entry.length = size;
    entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
    entry.storage.file.handle = handle.Get();
    entry.storage.file.offset = 0;
    IREE_ORT_RETURN_IF_ERROR(
        iree_io_parameter_index_add(source_index.Get(), &entry));
  }

  // Build IRPA archive from source index.
  ParameterIndexPtr target_index;
  IREE_ORT_RETURN_IF_ERROR(
      iree_io_parameter_index_create(allocator, target_index.ForOutput()));

  if (iree_io_parameter_index_count(source_index.Get()) > 0) {
    iree_io_parameter_archive_file_open_callback_t file_open = {
        IrpaFileOpenCallback,
        const_cast<std::string*>(&irpa_path_),
    };
    IREE_ORT_RETURN_IF_ERROR(iree_io_build_parameter_archive(
        source_index.Get(), target_index.Get(), file_open, 0, allocator));
  }

  // Add external initializer entries directly to target index.
  for (const auto& param : parameter_initializers_) {
    const auto& init = initializers_[param.initializer_index];

    Ort::ExternalInitializerInfo ext_info(nullptr);
    ORT_RETURN_IF_ERROR(init.GetExternalInitializerInfo(ext_info).release());
    if (!ext_info) {
      continue;
    }

    FileHandlePtr ext_handle;
    // External data paths are relative to the model directory.
    std::filesystem::path model_dir =
        std::filesystem::path(graph_.GetModelPath()).parent_path();
    std::string filepath = (model_dir / ext_info.GetFilePath()).string();
    IREE_ORT_RETURN_IF_ERROR(iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ,
        iree_make_string_view(filepath.data(), filepath.size()), allocator,
        ext_handle.ForOutput()));

    iree_io_parameter_index_entry_t entry = {};
    entry.key = iree_make_string_view(param.sanitized_name.data(),
                                      param.sanitized_name.size());
    entry.length = ext_info.GetByteSize();
    entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
    entry.storage.file.handle = ext_handle.Get();
    entry.storage.file.offset = static_cast<uint64_t>(ext_info.GetFileOffset());
    IREE_ORT_RETURN_IF_ERROR(
        iree_io_parameter_index_add(target_index.Get(), &entry));
  }

  ParameterProviderPtr provider;
  IREE_ORT_RETURN_IF_ERROR(iree_io_parameter_index_provider_create(
      iree_make_cstring_view("model"), target_index.Get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      allocator, provider.ForOutput()));

  out_index = std::move(target_index);
  out_provider = std::move(provider);
  return nullptr;
}

}  // namespace

/*static*/
TargetConfig TargetConfig::Create(const std::string& target_arch,
                                  const std::string& backend) {
  TargetConfig config;
  config.target_arch = target_arch;
  config.backend = backend;
  if (backend == "hip") {
    config.hal_backend = "rocm";
    config.hal_format = "rocm-hsaco-fb";
  } else if (backend == "cuda") {
    config.hal_backend = "cuda";
    config.hal_format = "cuda-nvptx-fb";
  } else if (backend == "vulkan") {
    config.hal_backend = "vulkan-spirv";
    config.hal_format = "vulkan-spirv-fb";
  }
  return config;
}

OrtStatus* GenerateMlir(
    const Ort::ConstGraph& graph, const OrtApi& /*ort_api*/,
    const std::string& mlir_path, const std::string& irpa_path,
    const std::vector<std::pair<std::string, DimSpecVariant>>& variants,
    std::vector<std::string>& out_function_names, ParameterIndexPtr& out_index,
    ParameterProviderPtr& out_provider, TargetConfig target_config) {
  std::ofstream file(mlir_path);
  if (!file.is_open()) {
    return MakeError("Failed to open output file: {}", mlir_path);
  }

  MlirGenerator gen(graph, file, irpa_path, std::move(target_config));

  std::vector<MlirGenerator::VariantInfo> infos;
  infos.reserve(variants.size());
  for (const auto& [suffix, specs] : variants) {
    infos.push_back({suffix, &specs});
  }
  OrtStatus* gen_status = gen.Generate(infos, out_function_names);
  if (gen_status) return gen_status;

  file.close();
  if (file.fail()) {
    return MakeError("Failed to write to file: {}", mlir_path);
  }

  return gen.BuildParameterArchive(out_index, out_provider);
}

}  // namespace onnxruntime::iree
