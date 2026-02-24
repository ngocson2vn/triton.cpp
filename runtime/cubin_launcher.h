#pragma once

#include <string>
#include <vector>

#include <cuda.h>
#include <vector_types.h>

#include "cuda_utils.h"

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


namespace rt {

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

class DevicePtr {
 public:
  // Allow custom constructor
  DevicePtr(CUdeviceptr ptr) : ptr_(ptr) {
    fprintf(stdout, "Allocated device ptr_ %llu\n", ptr_);
  }

  // Allow move constructor
  DevicePtr(DevicePtr&& rhs) noexcept {
    // By default, ptr_ is initialized to 0
    std::swap(ptr_, rhs.ptr_);
    fprintf(stdout, "DevicePtr move ctor ptr_ %llu rhs.ptr_ %llu\n", ptr_, rhs.ptr_);
  }

  // Forbid default constructor
  DevicePtr() = delete;

  // Forbid copy constructor
  DevicePtr(const DevicePtr& rhs) = delete;

  // Forbid copy assignment operator
  DevicePtr& operator=(const DevicePtr& rhs) = delete;

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
  CUdeviceptr ptr_ = 0;
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

  template <typename ABType, typename DType, int M_TILE, int N_TILE, int K_TILE>
  bool launchMMAKernel(
    const std::string& kernelName,
    const std::vector<void*>& inputs,
    const std::vector<std::size_t>& inputSizes,
    const std::vector<void*>& outputs,
    const std::vector<std::size_t>& outputSizes,
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const uint32_t blockSize) {
    static_assert(M_TILE == 64, "M_TILE must be equal to 64");
    static_assert(N_TILE == 32, "N_TILE must be equal to 64");
    static_assert(K_TILE == 64, "K_TILE must be equal to 64");

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
    CUdeviceptr devPtr;
    CUDA_CHECK_RET_FALSE(cuMemAlloc(&devPtr, outputSizes[0]));
    devOutputs.emplace_back(devPtr);

    // Create tensor maps
    CUtensorMap a_tensor_map{};
    bool status_A = create_tensor_map<ABType, 3>(&a_tensor_map, reinterpret_cast<void*>(devInputs[0].get()), M, K, M_TILE, K_TILE); // BOX_COLS = 64
    if (!status_A) {
      return EXIT_FAILURE;
    }

    CUtensorMap b_tensor_map{};
    bool status_B = create_tensor_map<ABType, 3>(&b_tensor_map, reinterpret_cast<void*>(devInputs[1].get()), N, K, N_TILE, K_TILE); // BOX_COLS = 64
    if (!status_B) {
      return EXIT_FAILURE;
    }

    CUtensorMap d_tensor_map{};
    bool status_D = create_tensor_map<DType, 3>(&d_tensor_map, reinterpret_cast<void*>(devOutputs[0].get()), M, N, M_TILE, N_TILE);
    if (!status_D) {
      return EXIT_FAILURE;
    }

    // Build kernel args
    // .visible .entry triton_dot(
    // 	.param .align 64 .b8 triton_dot_param_0[128],
    // 	.param .u32 triton_dot_param_1,
    // 	.param .u32 triton_dot_param_2,
    // 	.param .u64 triton_dot_param_3,
    // 	.param .u64 triton_dot_param_4,
    // 	.param .align 64 .b8 triton_dot_param_5[128],
    // 	.param .u32 triton_dot_param_6,
    // 	.param .u32 triton_dot_param_7,
    // 	.param .u64 triton_dot_param_8,
    // 	.param .u64 triton_dot_param_9,
    // 	.param .align 64 .b8 triton_dot_param_10[128],
    // 	.param .u32 triton_dot_param_11,
    // 	.param .u32 triton_dot_param_12,
    // 	.param .u64 triton_dot_param_13,
    // 	.param .u64 triton_dot_param_14,
    // 	.param .u64 .ptr .global .align 1 triton_dot_param_15,
    // 	.param .u64 .ptr .global .align 1 triton_dot_param_16
    // )

    // NOTE: MUST keep devInputs and devOutputs unchanged from now on.
    // Otherwise, device pointers may be moved to different locations which leads to 
    // undefined bahavior.
    std::unique_ptr<void*> kernelArgsPtr(new void*[inputs.size() + outputs.size() + 4*3 + 2]);
    void** kernelArgs = kernelArgsPtr.get();
    uint32_t u32_dummy_arg = 0;
    uint64_t u64_dummy_arg = 0;
    unsigned argIdx = 0;

    // A
    kernelArgs[argIdx++] = &a_tensor_map;
    kernelArgs[argIdx++] = &u32_dummy_arg;
    kernelArgs[argIdx++] = &u32_dummy_arg;
    kernelArgs[argIdx++] = &u64_dummy_arg;
    kernelArgs[argIdx++] = &u64_dummy_arg;

    // B
    kernelArgs[argIdx++] = &b_tensor_map;
    kernelArgs[argIdx++] = &u32_dummy_arg;
    kernelArgs[argIdx++] = &u32_dummy_arg;
    kernelArgs[argIdx++] = &u64_dummy_arg;
    kernelArgs[argIdx++] = &u64_dummy_arg;

    // D
    kernelArgs[argIdx++] = &d_tensor_map;
    kernelArgs[argIdx++] = &u32_dummy_arg;
    kernelArgs[argIdx++] = &u32_dummy_arg;
    kernelArgs[argIdx++] = &u64_dummy_arg;
    kernelArgs[argIdx++] = &u64_dummy_arg;

    CUdeviceptr devDummy1;
    kernelArgs[argIdx++] = &devDummy1;
    CUdeviceptr devDummy2;
    kernelArgs[argIdx++] = &devDummy2;

    // Set up kernel launch parameters
    dim3 blockDim(blockSize, 1, 1);
    dim3 gridDim(1, 1, 1);

    // Set shared memory attr
    uint32_t sharedMemBytes = M*K*sizeof(ABType) + N*K*sizeof(ABType) + sizeof(uint64_t);
    CUDA_CHECK_RET_FALSE(cuFuncSetAttribute(kernelFunc, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sharedMemBytes));

    // Launch the kernel
    CUDA_CHECK_RET_FALSE(cuLaunchKernel(kernelFunc, gridDim.x, 1, 1, blockDim.x, 1, 1, sharedMemBytes, NULL, kernelArgs, NULL));

    // Wait until the kernel launch is done
    CUDA_CHECK_RET_FALSE(cuCtxSynchronize());

    // Copy results back to host
    CUDA_CHECK_RET_FALSE(cuMemcpyDtoH(outputs[0], devOutputs[0].get(), outputSizes[0]));

    return true;
  }

 private:
  CubinLoader cubin_;
};

}  // namespace rt