#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "RegisterTritonDialects.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

static LinearLayout getMsgToPackedOffsetLayout(ttg::MemDescType ty) {
  auto ctx = ty.getContext();
  auto kMsg = str_attr("msg");
  auto kBlock = str_attr("block");
  auto shapePerCTA = ttg::getShapePerCTA(ty);
  int rank = shapePerCTA.size();
  auto blockShape = ttng::getTMABlockShape(ty, /*packedSize=*/true);
  auto outDimNames = standardOutDimNames(ctx, rank);
  LinearLayout msgToOffset;
  for (int dim = 0; dim < rank; ++dim) {
    msgToOffset *=
        LinearLayout::strided1D(shapePerCTA[dim] / blockShape[dim],
                                blockShape[dim], kMsg, outDimNames[dim]);
  }
  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());
  for (int i = 0; i < rank; ++i) {
    auto dim = ctaLayout.getCTAOrder()[i];
    msgToOffset *= LinearLayout::identity1D(ctaLayout.getCTASplitNum()[dim],
                                            kBlock, outDimNames[dim]);
  }
  return msgToOffset;
}

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  registerTritonDialects(registry);

  // For translating MLIR module to LLVM IR
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<func::FuncDialect>();

  // Create a ModuleOp
  OpBuilder builder(&context);
  auto loc = UnknownLoc::get(&context);
  ModuleOp mod = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(mod.getBody());

  // Create a RankedTensorType instance
  llvm::SmallVector<int64_t, 2> shape{64, 64};
  mlir::Type elementType = builder.getF16Type();

  // 1. Create the specific encoding attribute
  llvm::SmallVector<unsigned, 2> sizePerThread{2, 2};
  llvm::SmallVector<unsigned, 2> order{1, 0};
  unsigned numWarps = 4;
  unsigned numThreadsPerWarp = 32;
  auto argCTALayout = ttg::CTAEncodingAttr::getDefault(&context, 2);
  mlir::Attribute encoding = ttg::BlockedEncodingAttr::get(&context, shape, sizePerThread, order, numWarps, numThreadsPerWarp, argCTALayout);
  auto argType = mlir::RankedTensorType::get(shape, elementType, encoding);
  llvm::outs() << "argType: ";
  llvm::outs() << argType;
  llvm::outs() << "\n\n";

  // Create FuncOp
  auto retType = builder.getI32Type();
  auto kernelFunc = builder.create<func::FuncOp>(loc, "triton_kernel", builder.getFunctionType({argType}, {retType}));
  Block* kernelBody = kernelFunc.addEntryBlock();
  builder.setInsertionPointToStart(kernelBody);

  Attribute SharedMemorySpace = ttg::SharedMemorySpaceAttr::get(argType.getContext());
  auto CTALayout = ttg::getCTALayout(argType.getEncoding());
  llvm::SmallVector<unsigned> newOrder = {1, 0};
  bool isMMAv5Fp4Padded = false;
  auto newLayout = ttg::NVMMASharedEncodingAttr::get(argType.getContext(), argType.getShape(), newOrder, 
                                                     CTALayout, argType.getElementType(), isMMAv5Fp4Padded);
  auto memDescType = ttg::MemDescType::get(argType.getShape(), argType.getElementType(),
                                           newLayout, SharedMemorySpace);

  llvm::outs() << "memDescType: ";
  llvm::outs() << memDescType;
  llvm::outs() << "\n\n";

  auto msgToOffset = getMsgToPackedOffsetLayout(memDescType);
  llvm::outs() << msgToOffset << "\n\n";

  Value c0 = builder.create<arith::ConstantIntOp>(loc, retType, 0);
  builder.create<func::ReturnOp>(loc, c0);

  // Print the resulting module
  llvm::outs() << "\nMLIR module:\n";
  mod.print(llvm::outs());
  llvm::outs() << "\nAll done\n";
  return 0;
}
