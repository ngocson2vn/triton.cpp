#include "Dialect/NVGPU/IR/Dialect.h"
#include "TritonNVIDIAGPUToLLVMDebug/Passes.h"
#include "TritonNVIDIAGPUToLLVMDebug/Utility.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "Allocation.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

#include "mlir/IR/AsmState.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONGPUTOLLVMDEBUG
#include "TritonNVIDIAGPUToLLVMDebug/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::NVIDIA;

namespace {

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalDialect<cf::ControlFlowDialect>();
    addLegalDialect<mlir::gpu::GPUDialect>();
    addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();
    addLegalDialect<mlir::arith::ArithDialect>();
    addIllegalDialect<triton::TritonDialect>();
    // addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    // addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    // Warp specialization is lowered later.
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

struct ConvertTritonGPUToLLVMDebug
    : public triton::impl::ConvertTritonGPUToLLVMDebugBase<ConvertTritonGPUToLLVMDebug> {
  using ConvertTritonGPUToLLVMDebugBase::ConvertTritonGPUToLLVMDebugBase;

  ConvertTritonGPUToLLVMDebug(int32_t computeCapability, int32_t ptxVersion)
      : ConvertTritonGPUToLLVMDebugBase({computeCapability, ptxVersion}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    TargetInfo targetInfo(computeCapability, ptxVersion);

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);

    // Lower functions
    TritonLLVMFunctionConversionTarget funcTarget(*context);
    RewritePatternSet funcPatterns(context);
    mlir::triton::populateFuncOpConversionPattern(
        typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    llvm::outs() << "// After applying FuncOpConversion\n";
    auto& funcOp = mod.getBodyRegion().front().front();
    llvm::outs() << funcOp << "\n";
    llvm::outs() << "\n\n";

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(typeConverter);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;

    mlir::triton::NVIDIA::populateMemoryOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);

    populateBarrierOpToLLVMPatterns(typeConverter, patterns, benefit,
                                    targetInfo);

    populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo,
                                      computeCapability, patterns,
                                      axisInfoAnalysis, benefit);

    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, benefit);

    TritonLLVMConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMDebugPass(int32_t computeCapability, int32_t ptxVersion) {
  return std::make_unique<ConvertTritonGPUToLLVMDebug>(computeCapability, ptxVersion);
}

} // namespace triton
} // namespace mlir
