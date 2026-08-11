#ifndef TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVMDEBUG_PASSES_H
#define TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVMDEBUG_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "nvidia/include/TritonNVIDIAGPUToLLVMDebug/Passes.h.inc"

// std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMDebugPass();

// std::unique_ptr<OperationPass<ModuleOp>>
// createConvertTritonGPUToLLVMDebugPass(int32_t computeCapability);

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMDebugPass(int32_t computeCapability, int32_t ptxVersion);

// std::unique_ptr<OperationPass<ModuleOp>>
// createAllocateSharedMemoryNvPass(int32_t computeCapability, int32_t ptxVersion);

#define GEN_PASS_REGISTRATION
#include "nvidia/include/TritonNVIDIAGPUToLLVMDebug/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
