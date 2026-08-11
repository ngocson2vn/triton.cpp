#ifndef TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir {
namespace triton {

namespace NVIDIA {

void populateMemoryOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    const TargetInfo &targetInfo,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit);

void populateBarrierOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit,
                                     NVIDIA::TargetInfo &info);

void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       int computeCapability,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit);

} // namespace NVIDIA
} // namespace triton
} // namespace mlir

#endif
