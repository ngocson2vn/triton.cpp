#include <cuda.h>
#include <vector_types.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <memory>
#include <vector>
#include <fstream>
#include <stdexcept>

#define STRINGIFY(x) #x
#define TO_STR(x) STRINGIFY(x)

// Error checking macro
#define CUDA_CHECK(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    return EXIT_FAILURE; \
  } \
} while (0)

#define CUDA_CHECK_RET_FALSE(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    return false; \
  } \
} while (0)

#define CUDA_CHECK_RET_NULL(cuCall) \
do { \
  CUresult res = cuCall; \
  if (res != CUDA_SUCCESS) { \
    const char* errMsg; \
    cuGetErrorString(res, &errMsg); \
    fprintf(stderr, __FILE__ ":" TO_STR(__LINE__) " CUDA Error: %s\n", errMsg); \
    return nullptr; \
  } \
} while (0)

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

class DevicePtr {
 public:
  // Forbid default constructor
  DevicePtr() = delete;

  // Allow custom constructor
  DevicePtr(CUdeviceptr ptr) : ptr_(ptr) {
    fprintf(stdout, "Allocated device ptr %llu\n", ptr_);
  }

  // Forbid copy constructor
  DevicePtr(const DevicePtr& rhs) = delete;

  // Forbid copy assignment operator
  DevicePtr& operator=(const DevicePtr& rhs) = delete;

  // Allow move constructor
  DevicePtr(DevicePtr&& rhs) noexcept {
    ptr_ = rhs.ptr_;
    rhs.ptr_ = 0;
    fprintf(stdout, "DevicePtr move ctor ptr %llu\n", ptr_);
  }

  // Forbid move assignment operator
  DevicePtr& operator=(DevicePtr&& rhs) = delete;

  ~DevicePtr() {
    if (ptr_) {
      CUresult res = cuMemFree(ptr_);
      if (res != CUDA_SUCCESS) {
        const char* errMsg;
        cuGetErrorString(res, &errMsg);
        fprintf(stderr, "Failed to free device ptr %llu, error: %s\n", ptr_, errMsg);
      } else {
        fprintf(stdout, "Successfully free device ptr %llu\n", ptr_);
      }
    }
  }

  CUdeviceptr& get() {
    return ptr_;
  }

 private:
  CUdeviceptr ptr_;
};

class CubinLoader {
 public:
  CubinLoader() = default;
  CubinLoader(const std::string& cubinFile) : cubinFile_(cubinFile) {
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

  CUfunction getKernelFunc(const std::string& name) {
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

 private:
  std::string cubinFile_;
  CUmodule cuModule_;
  bool isLoaded_;
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
    CUDA_CHECK_RET_NULL(cuCtxCreate(&context, 0, device));

    return context;
  }

 private:
  CuCtxKeeper ctxKeeper_;
};

static CudaInitializer _cudaInit;

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

 private:
  CubinLoader cubin_;
};



int main(int argc, char** argv) {
  if (argc < 4) {
    printf("Usage: %s /path/to/file.cubin KERNEL_NAME BLOCK_SIZE\n", argv[0]);
    return EXIT_FAILURE;
  }

  // Parse arguments
  std::string cubinFile = argv[1];
  std::string kernelName = argv[2];

  int blockSize = -1;
  try {
    blockSize = std::stoi(argv[3]);
  } catch (std::exception& ex) {
    fprintf(stderr, "Failed to parse BLOCK_SIZE, error: %s\n", ex.what());
    return EXIT_FAILURE;
  }

  //
  // Client code
  //

  // Allocate host input
  int numElements = 1024;

  std::unique_ptr<float> h_input1_ptr(new float[numElements]);
  float* h_input1 = h_input1_ptr.get();

  std::unique_ptr<float> h_input2_ptr(new float[numElements]);
  float* h_input2 = h_input2_ptr.get();

  // Initialize input data
  for (int i = 0; i < numElements; i++) {
    h_input1[i] = static_cast<float>(i);
    h_input2[i] = static_cast<float>(i);
  }

  std::vector<void*> inputs;
  inputs.push_back(h_input1);
  inputs.push_back(h_input2);

  std::vector<std::size_t> inputSizes;
  inputSizes.push_back(numElements * sizeof(float));
  inputSizes.push_back(numElements * sizeof(float));

  // Allocate host output
  std::unique_ptr<float> h_output_ptr(new float[numElements]);
  float* h_output = h_output_ptr.get();

  std::vector<void*> outputs;
  std::vector<std::size_t> outputSizes;
  outputs.push_back(h_output);
  outputSizes.push_back(numElements * sizeof(float));

  // An uniform runner API
  CubinLauncher launcher(cubinFile);
  bool ok = launcher.launchKernel(kernelName, inputs, inputSizes, outputs, outputSizes, numElements, blockSize);
  if (!ok) {
    return EXIT_FAILURE;
  }

  // Print output
  printf("=====================================\n");
  for (int i = 0; i < numElements; i++) {
    printf("Output[%d] = %f\n", i, h_output[i]);
  }
  printf("=====================================\n");

  return 0;
}
