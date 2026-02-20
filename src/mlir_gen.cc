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
// Dynamic dims are always emitted as "?".
std::string FormatTensorType(const Ort::ConstTypeInfo& type_info) {
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
    ss << (shape[i] < 0 ? "?" : std::to_string(shape[i]));
  }
  ss << "]," << GetElementType(dtype) << ">";
  return ss.str();
}

// Formats a tensor type as tensor<dimsxdtype> (standard MLIR format).
// Uses signless integer types as required by MLIR tensor dialect.
// Dynamic dims are always emitted as "?".
std::string FormatMlirTensorType(const Ort::ConstTypeInfo& type_info) {
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return "NYI";
  }

  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  auto shape = tensor_info.GetShape();
  auto dtype = tensor_info.GetElementType();

  std::ostringstream ss;
  ss << "tensor<";
  for (size_t i = 0; i < shape.size(); ++i) {
    ss << (shape[i] < 0 ? "?" : std::to_string(shape[i]));
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

    // Build constraint lookup: symbolic_name -> DimSpec*.
    constraint_specs_.clear();
    for (const auto& spec : dim_specs_) {
      constraint_specs_[spec.symbolic_name] = &spec;
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

    graph_name_ = SanitizeName(graph_.GetName());
    if (graph_name_.empty()) graph_name_ = "main";
    graph_name_ += suffix;
  }

  // Emits the function signature with current dim specs and function name.
  // Constrained input args are renamed to %name__orig so EmitDimConstraints()
  // can rebind the original name after applying shape assumptions.
  void EmitFunctionHeader() {
    std::ostringstream args;
    for (size_t i = 0; i < graph_inputs_.size(); ++i) {
      if (i > 0) {
        args << ", ";
      }
      std::string name = SanitizeName(graph_inputs_[i].GetName());
      std::string type = FormatTensorType(graph_inputs_[i].TypeInfo());
      if (constrained_inputs_.contains(i)) {
        args << "%" << name << "__orig: " << type;
      } else {
        args << "%" << name << ": " << type;
      }
    }

    std::ostringstream ret_types;
    for (size_t i = 0; i < graph_outputs_.size(); ++i) {
      if (i > 0) {
        ret_types << ", ";
      }
      ret_types << FormatTensorType(graph_outputs_[i].TypeInfo());
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
    // Emit dim constraints (util.assume.int + flow.tensor.tie_shape).
    EmitDimConstraints();

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
      out_types << FormatTensorType(outputs[i].TypeInfo());
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
      in_types << FormatTensorType(inputs[i].TypeInfo());
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
      ret_types << FormatTensorType(graph_outputs_[i].TypeInfo());
    }

    out_ << std::format("    return {0} : {1}\n", ret_values.str(),
                        ret_types.str());
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

  std::vector<InputConstraintInfo> CollectConstrainedInputInfos(
      std::vector<CanonicalAssumeInfo>& out_canonical_assumes,
      std::unordered_map<std::string, size_t>& out_assume_index) const {
    std::vector<InputConstraintInfo> infos;
    infos.reserve(constrained_inputs_.size());

    for (size_t i = 0; i < graph_inputs_.size(); ++i) {
      if (!constrained_inputs_.contains(i)) continue;

      InputConstraintInfo info;
      info.name = SanitizeName(graph_inputs_[i].GetName());
      info.vtensor_type = FormatTensorType(graph_inputs_[i].TypeInfo());
      info.tensor_type = FormatMlirTensorType(graph_inputs_[i].TypeInfo());

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
          out_canonical_assumes.push_back(
              {sym_name, std::format("%{}_assumed", SanitizeName(sym_name)),
               dim_ssa, spec_it->second});
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
      if (assume.spec->div > 0) {
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
  void EmitDimConstraints() {
    if (constraint_specs_.empty()) return;

    std::vector<CanonicalAssumeInfo> canonical_assumes;
    std::unordered_map<std::string, size_t> assume_index;

    auto infos = CollectConstrainedInputInfos(canonical_assumes, assume_index);
    if (infos.empty()) return;

    EmitInputDimExtraction(infos);
    EmitCanonicalAssumes(canonical_assumes);
    EmitTieShapeRebinding(infos, canonical_assumes, assume_index);
  }

  // Member variables.
  const Ort::ConstGraph& graph_;
  std::ostream& out_;
  std::string irpa_path_;
  DimSpecVariant dim_specs_;

  // Lookup map: symbolic_name -> DimSpec* for all specs in current variant.
  std::unordered_map<std::string, const DimSpec*> constraint_specs_;

  // Input indices that have at least one constrained dynamic dim.
  std::unordered_set<size_t> constrained_inputs_;

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
