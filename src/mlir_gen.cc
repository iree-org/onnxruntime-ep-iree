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
#include "iree_ep.h"
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

// Sanitizes an ONNX name to be a valid MLIR SSA identifier.
// MLIR identifiers must match [a-zA-Z_][a-zA-Z0-9_$]*.
std::string SanitizeName(const std::string& name) {
  assert(!name.empty() && "Unexpected empty name");
  std::string result;
  result.reserve(name.size());
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$') {
      result += c;
    } else {
      result += '_';
    }
  }
  // Ensure starts with letter or underscore.
  if (!result.empty() && std::isdigit(static_cast<unsigned char>(result[0]))) {
    result = "_" + result;
  }
  return result.empty() ? "_unnamed" : result;
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
std::string GetElementType(ONNXTensorElementDataType dtype,
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
      return "NYI";
  }
}

// Formats a tensor type as !torch.vtensor<[dims],dtype>.
// When static_specs is provided, dynamic dims whose symbolic name matches a
// kStatic spec are replaced with the concrete value.
std::string FormatTensorType(
    const Ort::ConstTypeInfo& type_info,
    const std::unordered_map<std::string, const DimSpec*>& static_specs = {},
    const std::vector<const char*>& symbolic_dims = {}) {
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return "NYI";  // NYI: non-tensor types.
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
      // Check if this dynamic dim has a static specialization.
      if (i < symbolic_dims.size() && symbolic_dims[i] != nullptr &&
          symbolic_dims[i][0] != '\0') {
        auto it = static_specs.find(symbolic_dims[i]);
        if (it != static_specs.end()) {
          ss << it->second->value;
          continue;
        }
      }
      ss << "?";
    } else {
      ss << shape[i];
    }
  }
  ss << "]," << GetElementType(dtype) << ">";
  return ss.str();
}

// Formats a tensor type as tensor<dimsxdtype> (standard MLIR format).
// Uses signless integer types as required by MLIR tensor dialect.
// When static_specs is provided, dynamic dims whose symbolic name matches a
// kStatic spec are replaced with the concrete value.
std::string FormatMlirTensorType(
    const Ort::ConstTypeInfo& type_info,
    const std::unordered_map<std::string, const DimSpec*>& static_specs = {},
    const std::vector<const char*>& symbolic_dims = {}) {
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return "NYI";
  }

  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  auto shape = tensor_info.GetShape();
  auto dtype = tensor_info.GetElementType();

  std::ostringstream ss;
  ss << "tensor<";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      // Check if this dynamic dim has a static specialization.
      if (i < symbolic_dims.size() && symbolic_dims[i] != nullptr &&
          symbolic_dims[i][0] != '\0') {
        auto it = static_specs.find(symbolic_dims[i]);
        if (it != static_specs.end()) {
          ss << it->second->value;
          ss << "x";
          continue;
        }
      }
      ss << "?";
    } else {
      ss << shape[i];
    }
    ss << "x";
  }
  ss << GetElementType(dtype, /*signless=*/true) << ">";
  return ss.str();
}

// MLIR generator class.
class MlirGenerator {
 public:
  MlirGenerator(const Ort::ConstGraph& graph, std::ostream& out,
                const std::string& irpa_path)
      : graph_(graph), out_(out), irpa_path_(irpa_path) {}

  // A single (suffix, dim_specs) variant for MLIR generation.
  struct VariantInfo {
    std::string suffix;           // Function name suffix (e.g., "_variant0").
    const DimSpecVariant* specs;  // Dim specs for this variant.
  };

  // Generates an MLIR module containing one function per variant.
  // All functions share the same module (and thus the same parameter
  // references), so when compiled to a single VMFB the weights are shared.
  // For the unspecialized case, pass a single variant with empty suffix/specs.
  // Returns the MLIR function name for each variant (parallel to input).
  std::vector<std::string> Generate(const std::vector<VariantInfo>& variants) {
    CollectMetadata();
    std::vector<std::string> function_names;
    function_names.reserve(variants.size());
    out_ << "module {\n";
    for (const auto& v : variants) {
      ConfigureForVariant(*v.specs, v.suffix);
      function_names.push_back(graph_name_);
      EmitFunctionHeader();
      EmitFunctionBody();
      out_ << "  }\n";  // Close function.
    }
    out_ << "}\n";  // Close module.
    return function_names;
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

    std::unordered_set<std::string> init_names;
    for (const auto& init : initializers) {
      init_names.insert(init.GetName());
    }
    for (const auto& input : inputs) {
      if (!init_names.contains(input.GetName())) {
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

    // Collect symbolic dimension names for graph inputs and outputs.
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
    for (const auto& output : graph_outputs_) {
      if (output.TypeInfo().GetONNXType() != ONNX_TYPE_TENSOR) {
        output_symbolic_dims_.emplace_back();
        continue;
      }
      output_symbolic_dims_.push_back(output.TypeInfo()
                                          .GetTensorTypeAndShapeInfo()
                                          .GetSymbolicDimensions());
    }
  }

  // Configures the generator for a variant (dim specs, function name suffix).
  // Must be called before EmitFunctionHeader/EmitFunctionBody for each variant.
  void ConfigureForVariant(const DimSpecVariant& specs,
                           const std::string& suffix) {
    dim_specs_ = specs;
    static_specs_.clear();
    specialized_types_.clear();
    for (const auto& spec : dim_specs_) {
      if (spec.kind == DimSpec::Kind::kStatic) {
        static_specs_[spec.symbolic_name] = &spec;
      }
    }
    graph_name_ = SanitizeName(graph_.GetName());
    if (graph_name_.empty()) graph_name_ = "main";
    graph_name_ += suffix;
  }

  // Emits the function signature with current dim specs and function name.
  void EmitFunctionHeader() {
    // Build function arguments (apply static specialization to signature).
    // Also populate specialized_types_ so node emissions use consistent types.
    std::ostringstream args;
    for (size_t i = 0; i < graph_inputs_.size(); ++i) {
      if (i > 0) {
        args << ", ";
      }
      std::string name = SanitizeName(graph_inputs_[i].GetName());
      std::string type = FormatTensorType(
          graph_inputs_[i].TypeInfo(), static_specs_, input_symbolic_dims_[i]);
      args << "%" << name << ": " << type;
      if (!static_specs_.empty()) {
        specialized_types_[name] = type;
      }
    }

    // Build return types (apply static specialization to signature).
    std::ostringstream ret_types;
    for (size_t i = 0; i < graph_outputs_.size(); ++i) {
      if (i > 0) {
        ret_types << ", ";
      }
      std::string out_name = SanitizeName(graph_outputs_[i].GetName());
      std::string type =
          FormatTensorType(graph_outputs_[i].TypeInfo(), static_specs_,
                           output_symbolic_dims_[i]);
      ret_types << type;
      if (!static_specs_.empty()) {
        specialized_types_[out_name] = type;
      }
    }

    constexpr std::string_view schema = R"(  func.func @{0}({1}) -> ({2})
      attributes {{
        torch.onnx_meta.ir_version = {3} : si64,
        torch.onnx_meta.opset_version = {4} : si64,
        torch.onnx_meta.producer_name = "onnxruntime-ep-iree",
        torch.onnx_meta.producer_version = ""
      }} {{
)";

    out_ << std::format(schema,
                        graph_name_,      // {0}
                        args.str(),       // {1}
                        ret_types.str(),  // {2}
                        ir_version_,      // {3}
                        opset_version_);  // {4}
  }

  void EmitFunctionBody() {
    // Emit divisibility constraints (torch.symbolic_int + bind_symbolic_shape).
    EmitDivisibilityConstraints();

    // Emit initializers as flow.tensor.constant ops.
    for (const auto& init : initializers_) {
      EmitInitializer(init);
    }

    // Emit nodes.
    auto nodes = graph_.GetNodes();
    for (const auto& node : nodes) {
      EmitNode(node);
    }

    // Emit return.
    EmitReturn();
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
  void EmitInitializer(const Ort::ConstValueInfo& init) {
    std::string name = SanitizeName(init.GetName());
    std::string vtensor_type = FormatTensorType(init.TypeInfo());
    std::string tensor_type = FormatMlirTensorType(init.TypeInfo());

    auto tensor_info = init.TypeInfo().GetTensorTypeAndShapeInfo();
    size_t byte_size = tensor_info.GetElementCount() *
                       OnnxElementTypeSize(tensor_info.GetElementType());

    if (byte_size > kMaxInlineInitializerSize) {
      // Large: reference IRPA parameter archive.
      constexpr std::string_view schema =
          R"(    %__raw_{0} = flow.tensor.constant #flow.parameter.named<"model"::"{0}"> : {1}
    %{0} = torch_c.from_builtin_tensor %__raw_{0} : {1} -> {2}
)";
      out_ << std::format(schema, name, tensor_type, vtensor_type);
      return;
    }

    // Small: inline with dense<> DenseElementsAttr.
    Ort::ConstValue tensor_value{nullptr};
    auto status = init.GetInitializer(tensor_value);
    if (!status.IsOK()) return;

    const auto* data =
        static_cast<const uint8_t*>(tensor_value.GetTensorRawData());
    std::string hex = HexEncode(data, tensor_value.GetTensorSizeInBytes());

    constexpr std::string_view schema =
        R"(    %__raw_{0} = flow.tensor.constant dense<"{3}"> : {1}
    %{0} = torch_c.from_builtin_tensor %__raw_{0} : {1} -> {2}
)";
    out_ << std::format(schema, name, tensor_type, vtensor_type, hex);
  }

  void EmitNode(const Ort::ConstNode& node) {
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
        continue;
      }
      std::string output_name = outputs[i].GetName();
      if (output_name.empty()) {
        continue;
      }
      if (!first_output) {
        out_names << ", ";
        out_types << ", ";
      }
      first_output = false;
      valid_output_count++;
      std::string sanitized = SanitizeName(output_name);
      out_names << "%" << sanitized;

      // If a node output corresponds to a graph output with a specialized type,
      // use the specialized type to maintain SSA type consistency.
      auto it = specialized_types_.find(sanitized);
      if (it != specialized_types_.end()) {
        out_types << it->second;
      } else {
        out_types << FormatTensorType(outputs[i].TypeInfo());
      }
    }

    // Build input SSA references.
    std::ostringstream in_names;
    std::ostringstream in_types;
    bool first_input = true;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i]) {
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
      std::string sanitized = SanitizeName(input_name);
      in_names << "%" << sanitized;

      // If a node input references a graph input with a specialized type,
      // use the specialized type to maintain SSA type consistency.
      auto it = specialized_types_.find(sanitized);
      if (it != specialized_types_.end()) {
        in_types << it->second;
      } else {
        in_types << FormatTensorType(inputs[i].TypeInfo());
      }
    }

    // Build attributes.
    std::string attr_str = FormatAttributes(attrs);

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
  }

  std::string FormatAttributes(const std::vector<Ort::ConstOpAttr>& attrs) {
    if (attrs.empty()) {
      return "";
    }

    std::ostringstream ss;
    for (size_t i = 0; i < attrs.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << FormatAttribute(attrs[i]);
    }
    return ss.str();
  }

  std::string FormatAttribute(const Ort::ConstOpAttr& attr) {
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
      default:
        return std::format("torch.onnx.{0} = \"NYI\"", name);
    }
  }

  void EmitReturn() {
    std::ostringstream ret_values;
    std::ostringstream ret_types;
    for (size_t i = 0; i < graph_outputs_.size(); ++i) {
      if (i > 0) {
        ret_values << ", ";
        ret_types << ", ";
      }
      ret_values << "%" << SanitizeName(graph_outputs_[i].GetName());
      ret_types << FormatTensorType(graph_outputs_[i].TypeInfo(), static_specs_,
                                    output_symbolic_dims_[i]);
    }

    out_ << std::format("    return {0} : {1}\n", ret_values.str(),
                        ret_types.str());
  }

  // Emits torch.symbolic_int and torch.bind_symbolic_shape ops for
  // kDivisibleBy dim specs. This tells the compiler that certain dynamic
  // dimensions are multiples of a given divisor.
  void EmitDivisibilityConstraints() {
    // Build divisor lookup: symbolic_name -> divisor value.
    std::unordered_map<std::string, int64_t> divisors;
    for (const auto& spec : dim_specs_) {
      if (spec.kind == DimSpec::Kind::kDivisibleBy)
        divisors[spec.symbolic_name] = spec.value;
    }
    if (divisors.empty()) return;

    // True if dim d of an input is dynamic with a symbolic name.
    auto is_named_dynamic = [](int64_t shape_d, const auto& sym_dims,
                               size_t d) {
      return shape_d < 0 && d < sym_dims.size() && sym_dims[d] &&
             sym_dims[d][0] != '\0';
    };

    // Each unique symbolic dim gets a torch.symbolic_int declaration.
    // Constrained dims are registered first for deterministic ordering.
    struct Symbol {
      std::string label;
      int64_t divisor;  // 0 = unconstrained, >0 = divisibility
    };
    std::vector<Symbol> symbols;
    std::unordered_map<std::string, size_t> sym_index;

    auto register_sym = [&](const std::string& name, int64_t divisor) {
      if (auto [it, ok] = sym_index.try_emplace(name, symbols.size()); ok)
        symbols.push_back({"s_" + SanitizeName(name), divisor});
    };

    for (const auto& [name, divisor] : divisors) register_sym(name, divisor);

    for (size_t i = 0; i < graph_inputs_.size(); ++i) {
      if (graph_inputs_[i].TypeInfo().GetONNXType() != ONNX_TYPE_TENSOR)
        continue;
      auto shape =
          graph_inputs_[i].TypeInfo().GetTensorTypeAndShapeInfo().GetShape();
      const auto& sym_dims = input_symbolic_dims_[i];
      for (size_t d = 0; d < shape.size(); ++d) {
        if (!is_named_dynamic(shape[d], sym_dims, d)) continue;
        if (!static_specs_.contains(sym_dims[d])) register_sym(sym_dims[d], 0);
      }
    }

    // Emit torch.symbolic_int declarations.
    for (size_t i = 0; i < symbols.size(); ++i) {
      out_ << std::format(
          "    %_sym_{0} = torch.symbolic_int \"{1}\" "
          "{{min_val = 1, max_val = 9223372036854775807}} : !torch.int\n",
          i, symbols[i].label);
    }

    // For each graph input, emit torch.bind_symbolic_shape with an affine map
    // that relates each dynamic dim to its symbolic parameter.
    for (size_t i = 0; i < graph_inputs_.size(); ++i) {
      if (graph_inputs_[i].TypeInfo().GetONNXType() != ONNX_TYPE_TENSOR)
        continue;
      auto shape =
          graph_inputs_[i].TypeInfo().GetTensorTypeAndShapeInfo().GetShape();
      const auto& sym_dims = input_symbolic_dims_[i];

      // Single pass: collect referenced symbols and build per-dim expressions.
      std::vector<size_t> referenced;
      std::unordered_map<std::string, size_t> local_idx;
      std::vector<std::string> dim_exprs;

      auto get_local = [&](const std::string& name) -> size_t {
        auto [it, ok] = local_idx.try_emplace(name, referenced.size());
        if (ok) referenced.push_back(sym_index[name]);
        return it->second;
      };

      for (size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] >= 0) {
          dim_exprs.push_back(std::to_string(shape[d]));
          continue;
        }
        if (!is_named_dynamic(shape[d], sym_dims, d)) {
          std::string anon = std::format("__anon_{}_d{}", i, d);
          register_sym(anon, 0);
          dim_exprs.push_back(std::format("s{}", get_local(anon)));
          continue;
        }
        std::string name = sym_dims[d];
        if (auto it = static_specs_.find(name); it != static_specs_.end()) {
          dim_exprs.push_back(std::to_string(it->second->value));
          continue;
        }
        size_t idx = get_local(name);
        int64_t div = symbols[referenced[idx]].divisor;
        dim_exprs.push_back(div > 0 ? std::format("s{} * {}", idx, div)
                                    : std::format("s{}", idx));
      }
      if (referenced.empty()) continue;

      // Build affine_map and emit.
      std::string params, exprs, sym_refs;
      for (size_t j = 0; j < referenced.size(); ++j) {
        if (j) params += ", ";
        params += std::format("s{}", j);
      }
      for (size_t j = 0; j < dim_exprs.size(); ++j) {
        if (j) exprs += ", ";
        exprs += dim_exprs[j];
      }
      for (size_t j = 0; j < referenced.size(); ++j) {
        if (j) sym_refs += ", ";
        sym_refs += std::format("%_sym_{}", referenced[j]);
      }

      std::string input_name = SanitizeName(graph_inputs_[i].GetName());
      auto spec_it = specialized_types_.find(input_name);
      std::string vtensor_type =
          spec_it != specialized_types_.end()
              ? spec_it->second
              : FormatTensorType(graph_inputs_[i].TypeInfo());

      out_ << std::format(
          "    torch.bind_symbolic_shape %{0}, [{1}], "
          "affine_map<()[{2}] -> ({3})> : {4}\n",
          input_name, sym_refs, params, exprs, vtensor_type);
    }
  }

  // Member variables.
  const Ort::ConstGraph& graph_;
  std::ostream& out_;
  std::string irpa_path_;
  DimSpecVariant dim_specs_;

  // Lookup map: symbolic_name -> DimSpec* for kStatic specs.
  std::unordered_map<std::string, const DimSpec*> static_specs_;

  // Map from sanitized SSA name -> specialized vtensor type string.
  // Populated in EmitFunctionHeader for graph inputs/outputs that have static
  // specializations. Used by EmitNode to maintain SSA type consistency.
  std::unordered_map<std::string, std::string> specialized_types_;

  std::string graph_name_;
  int64_t ir_version_ = 8;
  int64_t opset_version_ = 17;

  std::vector<Ort::ConstValueInfo> graph_inputs_;
  std::vector<Ort::ConstValueInfo> graph_outputs_;
  std::vector<Ort::ConstValueInfo> initializers_;
  std::vector<ParameterInitializer> parameter_initializers_;

  // Symbolic dimension names per graph input/output (parallel to
  // graph_inputs_/graph_outputs_).
  std::vector<std::vector<const char*>> input_symbolic_dims_;
  std::vector<std::vector<const char*>> output_symbolic_dims_;
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
      return Ort::Status(
                 std::format("Failed to get initializer: {}", init.GetName())
                     .c_str(),
                 ORT_FAIL)
          .release();
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

OrtStatus* GenerateMlir(
    const Ort::ConstGraph& graph, const OrtApi& /*ort_api*/,
    const std::string& mlir_path, const std::string& irpa_path,
    const std::vector<std::pair<std::string, DimSpecVariant>>& variants,
    std::vector<std::string>& out_function_names, ParameterIndexPtr& out_index,
    ParameterProviderPtr& out_provider) {
  std::ofstream file(mlir_path);
  if (!file.is_open()) {
    return Ort::Status(
               std::format("Failed to open output file: {}", mlir_path).c_str(),
               ORT_FAIL)
        .release();
  }

  MlirGenerator gen(graph, file, irpa_path);

  std::vector<MlirGenerator::VariantInfo> infos;
  infos.reserve(variants.size());
  for (const auto& [suffix, specs] : variants) {
    infos.push_back({suffix, &specs});
  }
  out_function_names = gen.Generate(infos);

  file.close();
  if (file.fail()) {
    return Ort::Status(
               std::format("Failed to write to file: {}", mlir_path).c_str(),
               ORT_FAIL)
        .release();
  }

  return gen.BuildParameterArchive(out_index, out_provider);
}

}  // namespace onnxruntime::iree
