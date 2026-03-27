#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <fstream>
#include <stdexcept>

#include "cubin_launcher.h"

namespace rt {

class CuCtxKeeper final {
 public:
  CuCtxKeeper() = default;
  CuCtxKeeper(const CuCtxKeeper& rhs) = delete;
  CuCtxKeeper& operator=(const CuCtxKeeper& rhs) = delete;
  CuCtxKeeper(CuCtxKeeper&& rhs) = delete;
  CuCtxKeeper& operator=(CuCtxKeeper&& rhs) = delete;

  ~CuCtxKeeper() {
    if (ctx_) {
      CUresult res = cuCtxDestroy(ctx_);
      if (res != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to destroy CUDA context, error code: %d\n", res);
      } else {
        fprintf(stdout, "Successfully destroyed CUDA context.\n");
      }
    }
  }

  void setContext(CUcontext ctx) {
    ctx_ = ctx;
  }

 private:
  CUcontext ctx_;
};

class CudaInitializer {
 public:
  CudaInitializer() {
    CUcontext context = init();
    if (!context) {
      fprintf(stderr, "Failed to init CUDA!!!\n");
      std::terminate();
    }

    ctxKeeper_.setContext(context);
    fprintf(stdout, "Successfully initialized CUDA!\n");
  }

  CUcontext init() {
    CUDA_CHECK_RET_NULL(cuInit(0));

    CUdevice device;
    CUDA_CHECK_RET_NULL(cuDeviceGet(&device, 0));

    CUcontext context;
#if (CUDA_VERSION < 13000)
    CUDA_CHECK_RET_NULL(cuCtxCreate(&context, 0, device));
#else
    CUDA_CHECK_RET_NULL(cuCtxCreate(&context, nullptr, 0, device));
#endif
    return context;
  }

 private:
  CuCtxKeeper ctxKeeper_;
};

static CudaInitializer _cudaInit;


CubinLoader::CubinLoader(const std::string& cubinFile) : cubinFile_(cubinFile) {
  // Read the .cubin file
  std::ifstream ifs(cubinFile, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    fprintf(stderr, "Failed to open cubin file %s\n", cubinFile.c_str());
    isLoaded_ = false;
  }
  
  ifs.seekg(0, ifs.end);
  std::size_t numBytes = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  std::string cubinData(numBytes, '\0');
  ifs.read(cubinData.data(), numBytes);

  // Load the .cubin file into a module
  CUresult res = cuModuleLoadData(&cuModule_, reinterpret_cast<const void*>(cubinData.data()));
  if (res != CUDA_SUCCESS) {
    const char* errMsg;
    cuGetErrorString(res, &errMsg);
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg);
    isLoaded_ = false;
  } else {
    isLoaded_ = true;
    fprintf(stdout, "Successfully loaded cubin %s\n", cubinFile_.c_str());
  }
}

CUfunction CubinLoader::getKernelFunc(const std::string& name) {
  if (!isLoaded_) {
    fprintf(stderr, "cubin file hasn't been loaded yet!\n");
    return nullptr;
  }

  CUfunction kernelFunc;
  CUresult res = cuModuleGetFunction(&kernelFunc, cuModule_, name.c_str());
  if (res != CUDA_SUCCESS) {
    const char* errMsg;
    cuGetErrorString(res, &errMsg);
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg);
    return nullptr;
  }

  return kernelFunc;
}


bool CubinLauncher::launchKernel(
  const std::string& kernelName,
  const std::vector<void*>& inputs,
  const std::vector<std::size_t>& inputSizes,
  const std::vector<void*>& outputs,
  const std::vector<std::size_t>& outputSizes,
  const int numElements,
  int blockSize) {

  // Get kernel function
  CUfunction kernelFunc = cubin_.getKernelFunc(kernelName);
  if (!kernelFunc) {
    return false;
  }

  // Copy input data to device
  std::vector<DevicePtr> devInputs;
  for (int i = 0; i < inputs.size(); i++) {
    CUdeviceptr devPtr;
    CUDA_CHECK_RET_FALSE(cuMemAlloc(&devPtr, inputSizes[i]));
    devInputs.emplace_back(devPtr);
    CUDA_CHECK_RET_FALSE(cuMemcpyHtoD(devInputs[i].get(), inputs[i], inputSizes[i]));
  }

  std::vector<DevicePtr> devOutputs;
  for (int i = 0; i < outputs.size(); i++) {
    CUdeviceptr devPtr;
    CUDA_CHECK_RET_FALSE(cuMemAlloc(&devPtr, outputSizes[i]));
    devOutputs.emplace_back(devPtr);
  }

  // Build kernel args
  // NOTE: MUST keep devInputs and devOutputs unchanged from now on.
  // Otherwise, device pointers may be moved to different locations which leads to 
  // undefined bahavior.
  std::unique_ptr<void*> kernelArgsPtr(new void*[inputs.size() + outputs.size() + 2]);
  void** kernelArgs = kernelArgsPtr.get();
  unsigned argIdx = 0;
  for (int i = 0; i < inputs.size(); i++) {
    kernelArgs[argIdx++] = &devInputs[i].get();
  }

  for (int i = 0; i < outputs.size(); i++) {
    kernelArgs[argIdx++] = &devOutputs[i].get();
  }

  kernelArgs[argIdx++] = const_cast<int*>(&numElements);

  CUdeviceptr devDummy;
  kernelArgs[argIdx++] = &devDummy;

  // Set up kernel launch parameters
  dim3 blockDim(blockSize);
  dim3 gridDim((numElements + blockDim.x - 1) / blockDim.x);

  // Launch the kernel
  CUDA_CHECK_RET_FALSE(cuLaunchKernel(kernelFunc, gridDim.x, 1, 1, blockDim.x, 1, 1, 0, NULL, kernelArgs, NULL));

  // Wait until the kernel launch is done
  CUDA_CHECK_RET_FALSE(cuCtxSynchronize());

  // Copy results back to host
  for (int i = 0; i < outputs.size(); i++) {
    CUDA_CHECK_RET_FALSE(cuMemcpyDtoH(outputs[i], devOutputs[i].get(), outputSizes[i]));
  }

  return true;
}

}  // namespace rt