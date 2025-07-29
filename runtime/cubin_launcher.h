#pragma once

#include <string>
#include <vector>

#include <cuda.h>
#include <vector_types.h>

#define STRINGIFY(x) #x
#define TO_STR(x) STRINGIFY(x)

namespace cubin {

class CubinLoader {
 public:
  CubinLoader() = default;
  CubinLoader(const std::string& cubinFile);

  ~CubinLoader() {
    if (isLoaded_) {
      CUresult res = cuModuleUnload(cuModule_);
      if (res != CUDA_SUCCESS) {
        const char* errMsg;
        cuGetErrorString(res, &errMsg);
        fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg);
      } else {
        fprintf(stdout, "Successfully unloaded %s\n", cubinFile_.c_str());
      }
    }
  }

  CUfunction getKernelFunc(const std::string& name);

 private:
  std::string cubinFile_;
  CUmodule cuModule_;
  bool isLoaded_;
};

class CubinLauncher {
 public:
  CubinLauncher(const std::string& cubinFile) : cubin_(cubinFile) {}

  bool launchKernel(
    const std::string& kernelName,
    const std::vector<void*>& inputs,
    const std::vector<std::size_t>& inputSizes,
    const std::vector<void*>& outputs,
    const std::vector<std::size_t>& outputSizes,
    const int numElements,
    int blockSize);

 private:
  CubinLoader cubin_;
};

}  // namespace cubin