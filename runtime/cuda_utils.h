#include <mutex>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

using half_t = __nv_half;
using bf16_t = __nv_bfloat16;

static std::mutex log_mutex;
#if defined(DEBUG_MODE)
#define LOG_DEBUG(format_str, ...)            \
do {                                          \
  std::lock_guard<std::mutex> lk(log_mutex);  \
  printf("DEBUG_MODE ");                      \
  printf(format_str, ##__VA_ARGS__);          \
} while(0)
#else
#define LOG_DEBUG(format_str, ...)
#endif

#define LOG_INFO(format_str, ...)             \
do {                                          \
  std::lock_guard<std::mutex> lk(log_mutex);  \
  printf("INFO ");                            \
  printf(format_str, ##__VA_ARGS__);          \
} while(0)

#define LOG_ERROR(format_str, ...)            \
do {                                          \
  std::lock_guard<std::mutex> lk(log_mutex);  \
  printf("ERROR ");                           \
  printf(format_str, ##__VA_ARGS__);          \
} while(0)


namespace rt {

template<typename DataType, int swizzleMode = 0>
bool create_tensor_map(CUtensorMap* tensorMap, void* globalAddress, uint32_t ROWS, uint32_t COLS, uint32_t BOX_ROWS, uint32_t BOX_COLS) {
  // tensorRank is the number of dimensions of the array.
  constexpr uint32_t tensorRank = 2;

  // In CUDA, 
  // globalDim[0] is contiguous dimension
  uint64_t globalDim[] = {COLS, ROWS};

  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t globalStrides[] = {sizeof(DataType), COLS * sizeof(DataType)};

  // The boxDim is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t boxDim[] = {BOX_COLS, BOX_ROWS};

  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elementStrides[] = {1, 1};

  CUtensorMapDataType tensorDataType;
  if constexpr(std::is_same<DataType, int>()) {
    tensorDataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32;
  } else if constexpr(std::is_same<DataType, half_t>()) {
    tensorDataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if constexpr(std::is_same<DataType, bf16_t>()) {
    tensorDataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if constexpr(std::is_same<DataType, float>()) {
    tensorDataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else {
    LOG_ERROR("%s:%d: create_tensor_map does not support the provided DataType.\n", __FILE__, __LINE__);
    return false;
  }

  LOG_DEBUG("CUtensorMapDataType: %d\n", tensorDataType);

  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2Promotion;
  if constexpr(swizzleMode == 0) {
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    l2Promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE;
  } else if constexpr(swizzleMode == 1) {
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B;
    l2Promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE;
  } else if constexpr(swizzleMode == 2) {
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
    l2Promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
  } else if constexpr(swizzleMode == 3) {
    LOG_DEBUG("swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B\n");
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    l2Promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  } else {
    LOG_ERROR("%s:%d: Unsupported swizzle mode %d", __FILE__, __LINE__, swizzleMode);
    return false;
  }

  // Create the tensor descriptor.
  CUresult ret = cuTensorMapEncodeTiled(
    tensorMap,                  // CUtensorMap *tensorMap,
    tensorDataType,             // CUtensorMapDataType tensorDataType
    tensorRank,                 // cuuint32_t tensorRank,
    globalAddress,              // void *globalAddress,
    globalDim,                  // const cuuint64_t *globalDim,
    globalStrides + 1,          // const cuuint64_t *globalStrides,
    boxDim,                     // const cuuint32_t *boxDim,
    elementStrides,             // const cuuint32_t *elementStrides,

    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,

    // Swizzling can be used to avoid shared memory bank conflicts.
    swizzle,

    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    l2Promotion,

    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );

  if (ret != CUDA_SUCCESS) {
    LOG_ERROR("%s:%d: Failed to create CUtensorMap object: ROWS=%d COLS=%d BOX_ROWS=%d BOX_COLS=%d\n",
              __FILE__, __LINE__, ROWS, COLS, BOX_ROWS, BOX_COLS);
    const char* errName;
    cuGetErrorName(ret, &errName);
    const char* errMsg;
    cuGetErrorString(ret, &errMsg);
    LOG_ERROR("%s:%d: %s: %s\n", __FILE__, __LINE__, errName, errMsg);
    fflush(stderr);
    return false;
  }

  LOG_DEBUG("Successfully created CUtensorMap object: ROWS=%d COLS=%d BOX_ROWS=%d BOX_COLS=%d\n",
            ROWS, COLS, BOX_ROWS, BOX_COLS);
  return true;
}

}  // namespace rt
