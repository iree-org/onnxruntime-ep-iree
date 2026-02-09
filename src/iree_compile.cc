// iree_compile.cc - MLIR to VMFB compilation via iree-compile CLI.

#include "iree_compile.h"

#include <cstdlib>
#include <format>
#include <numeric>
#include <string>

namespace onnxruntime::iree {

OrtStatus* CompileToVmfb(const std::string& mlir_path,
                         const std::string& vmfb_path,
                         const std::vector<std::string>& flags,
                         const OrtApi& /*ort_api*/) {
  // Build command: iree-compile {mlir_path} {flags} -o {vmfb_path}
  std::string flags_str = std::accumulate(
      flags.begin(), flags.end(), std::string(),
      [](const std::string& a, const std::string& b) { return a + " " + b; });
  std::string command = std::format(R"(iree-compile "{}" {} -o "{}")",
                                    mlir_path, flags_str, vmfb_path);

  // Execute the command.
  int result = std::system(command.c_str());

  if (result != 0) {
    std::string error_msg = std::format(
        R"(IREE EP: iree-compile failed with exit code {}. Command: {})",
        result, command);
    return Ort::Status(error_msg.c_str(), ORT_FAIL).release();
  }

  return nullptr;
}

}  // namespace onnxruntime::iree
