#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinAttributes.h"

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

template <typename T>
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const SmallVector<T>& vec) {
  if (vec.empty()) {
    os << "[]";
    return os;
  }

  os << "[" << vec[0];
  for (int i = 1; i < vec.size(); i++) {
    os << ", " << vec[i];
  }
  os << "]";

  return os;
}

static LinearLayout getMsgToPackedOffsetLayout(ttg::MemDescType ty) {
  llvm::outs() << "--- getMsgToPackedOffsetLayout ---\n";
  auto ctx = ty.getContext();
  auto kMsg = str_attr("msg");
  auto kBlock = str_attr("block");
  auto shapePerCTA = ttg::getShapePerCTA(ty);
  llvm::outs() << "shapePerCTA: " << shapePerCTA << "\n";
  // shapePerCTA: [64, 64]

  int rank = shapePerCTA.size();
  auto blockShape = ttng::getTMABlockShape(ty, /*packedSize=*/true);
  llvm::outs() << "blockShape: " << blockShape << "\n\n";
  // blockShape: [64, 64]

  auto outDimNames = standardOutDimNames(ctx, rank);
  LinearLayout msgToOffset;
  for (int dim = 0; dim < rank; ++dim) {
    // Map a TMA message ID to an offset per dimension. For example,
    // shapePerCTA = [128, 128]
    // blockShape = [64, 64]
    // => The number of TMA messages = 2*2 = 4 (i.e. "msg" has 2 bits)
    // 
    // For dim = 0
    // msg = 0 -> dim0 = 0 * 64 = 0
    // msg = 1 -> dim0 = 1 * 64 = 64
    // 
    // For dim = 1
    // msg = 0 -> dim1 = 0 * 64 = 0
    // msg = 1 -> dim1 = 1 * 64 = 64
    auto layout = LinearLayout::strided1D(shapePerCTA[dim] / blockShape[dim],
                                          blockShape[dim], kMsg, outDimNames[dim]);
    llvm::outs() << outDimNames[dim] << " layout:";
    llvm::outs() << layout << "\n\n";
    msgToOffset *= layout;
  }
  llvm::outs() << "msgToOffset: " << msgToOffset << "\n\n";

  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());
  auto CTAOrder = ctaLayout.getCTAOrder();
  llvm::outs() << "CTAOrder: " << CTAOrder << "\n";
  auto CTASplitNum = ctaLayout.getCTASplitNum();
  llvm::outs() << "CTASplitNum: " << CTASplitNum << "\n\n";

  for (int i = 0; i < rank; ++i) {
    auto dim = CTAOrder[i];
    auto layout = LinearLayout::identity1D(CTASplitNum[dim], kBlock, outDimNames[dim]);
    llvm::outs() << outDimNames[dim] << " layout:";
    llvm::outs() << layout << "\n\n";
    msgToOffset *= layout;
  }

  llvm::outs() << "Final msgToOffset: " << msgToOffset << "\n";
  llvm::outs() << "----------------------------------\n\n";

  return msgToOffset;
}

static LinearLayout
getMsgToUnpackedOffsetLayout(const LinearLayout &packedLayout,
                             ttg::MemDescType ty) {
  auto isFp4Padded =
      cast<ttg::NVMMASharedEncodingAttr>(ty.getEncoding()).getFp4Padded();
  if (!isFp4Padded) {
    return packedLayout;
  }
  auto ctx = ty.getContext();
  auto rank = ty.getRank();
  auto kMsg = str_attr("msg");
  auto kLastDim = str_attr("dim" + Twine(rank - 1));
  // Multiply to offset by 2 in the last dimension
  auto unpackLayout = LinearLayout::zeros1D(1, kMsg, kLastDim, 2);
  return unpackLayout * packedLayout;
}


int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  registerTritonDialects(registry);

  // For translating MLIR module to LLVM IR
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  MLIRContext context(registry);
  MLIRContext* ctx = &context;
  context.loadAllAvailableDialects();
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<func::FuncDialect>();

  // Create a ModuleOp
  OpBuilder builder(ctx);
  auto loc = UnknownLoc::get(ctx);
  ModuleOp mod = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(mod.getBody());

  // 
  // Create a RankedTensorType instance
  // 

  // =======================
  // 1. Tensor shape
  // =======================
  llvm::SmallVector<int64_t, 2> shape{128, 128};

  // =======================
  // 2. Element type
  // =======================
  mlir::Type elementType = builder.getF16Type();

  // =======================
  // 3. Encoding
  // =======================
  unsigned numWarps = 4;
  unsigned numThreadsPerWarp = 32;

  // 2 means rank=2
  auto argCTALayout = ttg::CTAEncodingAttr::getDefault(ctx, 2);

  // Each CTA processes a data tile of blockShape
  llvm::SmallVector<int64_t, 2> blockShape{64, 64};

  // In Triton, sizePerThread does not define the total number of elements a thread holds. 
  // Instead, it defines the size of the contiguous chunks a thread accesses at one time.
  // Since each CTA has 4*32 = 128 threads and blockShape = 64x64, then each thread should be responsible for 64*64/128 = 32 elements.
  // Each thread will simply process 8 separate chunks (32 total elements / 4 elements per chunk = 8 chunks).
  llvm::SmallVector<unsigned, 2> sizePerThread{2, 2};

  // Order: dim=1 -> dim=0
  // [row0][row1]...[rowN]
  llvm::SmallVector<unsigned, 2> order{1, 0};

  mlir::Attribute encoding = ttg::BlockedEncodingAttr::get(ctx, blockShape, sizePerThread, order, numWarps, numThreadsPerWarp, argCTALayout);

  auto tensorType = mlir::RankedTensorType::get(shape, elementType, encoding);
  llvm::outs() << "tensorType: ";
  llvm::outs() << tensorType;
  llvm::outs() << "\n\n";
  // Output:
  // #blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
  // Read explanation in ../refs/BlockedEncoding.md


  // 
  // Create ttg::MemDescType
  // 
  Attribute SharedMemorySpace = ttg::SharedMemorySpaceAttr::get(tensorType.getContext());
  auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
  llvm::SmallVector<unsigned> newOrder = {1, 0};
  bool isMMAv5Fp4Padded = false;
  auto mmaEncoding = ttg::NVMMASharedEncodingAttr::get(tensorType.getContext(), tensorType.getShape(), newOrder, 
                                                       CTALayout, tensorType.getElementType(), isMMAv5Fp4Padded);
  auto smemTy = ttg::MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                                      mmaEncoding, SharedMemorySpace);

  llvm::outs() << "smemTy: " << smemTy << "\n\n";

  auto msgToPackedOffset = getMsgToPackedOffsetLayout(smemTy);
  llvm::outs() << "msgToPackedOffset: " << msgToPackedOffset << "\n\n";

  auto smemLayout = ttg::toLinearLayout(smemTy);
  llvm::outs() << "smemLayout: " << smemLayout << "\n\n";

  auto msgToShared = msgToPackedOffset.invertAndCompose(smemLayout);
  llvm::outs() << "msgToShared: " << msgToShared << "\n\n";

  auto msgToOffset = getMsgToUnpackedOffsetLayout(msgToPackedOffset, smemTy);
  llvm::outs() << "msgToOffset: " << msgToOffset << "\n\n";

  // 
  // Create FuncOp
  // 
  auto retType = builder.getI32Type();
  auto kernelFunc = builder.create<func::FuncOp>(loc, "triton_kernel", builder.getFunctionType({tensorType}, {retType}));
  Block* kernelBody = kernelFunc.addEntryBlock();
  builder.setInsertionPointToStart(kernelBody);

  Value c0 = builder.create<arith::ConstantIntOp>(loc, retType, 0);
  builder.create<func::ReturnOp>(loc, c0);

  // Print the resulting module
  llvm::outs() << "\nMLIR module:\n";
  mod.print(llvm::outs());
  llvm::outs() << "\nAll done\n";
  return 0;
}
