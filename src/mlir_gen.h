// mlir_gen.h - MLIR text generation from OrtGraph.
//
// Generates torch-mlir ONNX dialect MLIR text directly from an OrtGraph,
// bypassing the need for ONNX protobuf dependencies.
//
// Note that we make a conscious decision to avoid using ONNX protobuf here.
// There are some utilities to convert OrtGraph to ONNX protobuf implemented in
// https://github.com/microsoft/onnxruntime/pull/25292, but even they recommend
// that EP's should directly convert OrtGraph to their own internal
// representation. We also want to be light on dependencies and don't want to
// take a dependency on protobuf and onnx. Since we don't use ONNX protobuf, we
// have to directly emit torch onnx dialect instead of using torch-mlir's onnx
// importer.

#ifndef ONNXRUNTIME_EP_IREE_SRC_MLIR_GEN_H_
#define ONNXRUNTIME_EP_IREE_SRC_MLIR_GEN_H_

#include <string>

#include "iree_wrappers.h"
#include "ort_import.h"

namespace onnxruntime::iree {

// Generates MLIR text from an OrtGraph and writes it to the specified file.
// Small initializers are inlined in the MLIR. Large initializers are emitted as
// parameter references and their data is written to an IRPA archive at
// irpa_path. out_index and out_provider are populated with the parameter index
// and provider for the archive. They remain null if no parameters are needed.
OrtStatus* GenerateMlir(const Ort::ConstGraph& graph, const OrtApi& ort_api,
                        const std::string& mlir_path,
                        const std::string& irpa_path,
                        ParameterIndexPtr& out_index,
                        ParameterProviderPtr& out_provider);

}  // namespace onnxruntime::iree

#endif  // ONNXRUNTIME_EP_IREE_SRC_MLIR_GEN_H_
