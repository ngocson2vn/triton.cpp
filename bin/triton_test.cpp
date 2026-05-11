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

LinearLayout buildLayoutA(MLIRContext* ctx) {
  // For Layout A from the paper (16x16 tensor):
  // - Logical dimensions: "dim0" (i), "dim1" (j)
  // - Physical dimensions: "register", "thread", "warp"

  // 1. Registers (2x2 registers)
  // Bit 0 of register points to bit 0 of dim1 (j)
  // Bit 1 of register points to bit 0 of dim0 (i)
  auto reg = StringAttr::get(ctx, "register");
  auto thread = StringAttr::get(ctx, "thread");
  auto warp = StringAttr::get(ctx, "warp");
  auto dim1 = StringAttr::get(ctx, "dim1");
  auto dim0 = StringAttr::get(ctx, "dim0");
  LinearLayout regLayout = LinearLayout::identity1D(2, reg, dim1) * LinearLayout::identity1D(2, reg, dim0);

  // 2. Threads (4x8 threads)
  // Bits 0..2 of thread point to bits 1..3 of dim1 (j)
  // Bits 3..4 of thread point to bits 1..2 of dim0 (i)
  LinearLayout threadLayout = LinearLayout::identity1D(8, thread, dim1) * LinearLayout::identity1D(4, thread, dim0);

  // 3. Warps (2x1 warps)
  // Bit 0 of warp points to bit 3 of dim0 (i)
  LinearLayout warpLayout = LinearLayout::identity1D(2, warp, dim0);

  // 4. Combine them using the Product operation (operator*)
  // Multiplying layouts automatically stacks and concatenates their output bases.
  // For dim1: 1 bit (Reg) + 3 bits (Thread) = 4 bits (size 16)
  // For dim0: 1 bit (Reg) + 2 bits (Thread) + 1 bit (Warp) = 4 bits (size 16)
  LinearLayout layoutA = regLayout * threadLayout * warpLayout;

  return layoutA;
}

// A helper function to simulate the F2 matrix-vector multiplication
std::pair<int, int> applyLayout(MLIRContext* ctx, LinearLayout& layout, int v) {
  int v_reg    = v & 0b00000011;
  int v_thread = (v & 0b01111100) >> 2;
  int v_warp   = (v & 0b10000000) >> 7;
  int w_dim1 = 0; // This will hold our column (j)
  int w_dim0 = 0; // This will hold our row (i)

  auto reg = StringAttr::get(ctx, "register");
  auto thread = StringAttr::get(ctx, "thread");
  auto warp = StringAttr::get(ctx, "warp");
  auto dim1 = StringAttr::get(ctx, "dim1");
  auto dim0 = StringAttr::get(ctx, "dim0");

  // 1. Process Register bits
  for (int bit = 0; bit < layout.getInDimSizeLog2(reg); ++bit) {
    if ((v_reg >> bit) & 1) { // If this physical bit is ON
      w_dim1 ^= layout.getBasis(reg, bit, dim1);
      w_dim0 ^= layout.getBasis(reg, bit, dim0);
    }
  }

  // 2. Process Thread bits
  for (int bit = 0; bit < layout.getInDimSizeLog2(thread); ++bit) {
    if ((v_thread >> bit) & 1) { // If this physical bit is ON
      w_dim1 ^= layout.getBasis(thread, bit, dim1);
      w_dim0 ^= layout.getBasis(thread, bit, dim0);
    }
  }

  w_dim0 ^= layout.getBasis(warp, 0, dim0);

  return {w_dim0, w_dim1}; // Return (i, j)
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


  llvm::outs() << "========================================================================================\n";
  llvm::outs() << "LinearLayout\n";
  llvm::outs() << "========================================================================================\n";
  // 1. Define the names of your input (hardware) and output (logical) dimensions
  auto inRowDim = StringAttr::get(ctx, "in_row");
  auto inColDim = StringAttr::get(ctx, "in_col");
  auto outRowDim = StringAttr::get(ctx, "out_row");
  auto outColDim = StringAttr::get(ctx, "out_col");

  // 2. Create a 1D layout for the 16 rows.
  // We use `strided1D` with size=16 and stride=1, meaning each step in `in_row` 
  // moves 1 step in `out_row`.
  LinearLayout rowLayout = LinearLayout::strided1D(
      /*size=*/16, 
      /*stride=*/1, 
      /*inDimName=*/inRowDim, 
      /*outDimName=*/outRowDim
  );

  // 3. Create a 1D layout for the 16 columns.
  // size=16, stride=1
  LinearLayout colLayout = LinearLayout::strided1D(
      /*size=*/16, 
      /*stride=*/1, 
      /*inDimName=*/inColDim, 
      /*outDimName=*/outColDim
  );

  // 4. Multiply them to compute the Direct Sum!
  // This results in a 2D LinearLayout mapping:
  // (in_row=16, in_col=16) -> (out_row=16, out_col=16)
  LinearLayout layout16x16 = rowLayout * colLayout;
  llvm::outs() << "\nlayout16x16: " << layout16x16 << "\n\n";

  auto layoutA = buildLayoutA(ctx);
  llvm::outs() << "layoutA: " << layoutA << "\n\n";

  int v = 0b11010101;
  auto w = applyLayout(ctx, layoutA, v);
  llvm::outs() << "v = " << v << "\n";
  llvm::outs() << "i = " << w.first << "\n";
  llvm::outs() << "j = " << w.second << "\n";

  llvm::outs() << "========================================================================================\n\n";


  //==========================================================================================================
  // MemDescType
  //==========================================================================================================
  // Create a ModuleOp
  OpBuilder builder(ctx);
  auto loc = UnknownLoc::get(ctx);
  ModuleOp mod = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(mod.getBody());

  // Create a RankedTensorType instance
  llvm::SmallVector<int64_t, 2> shape{64, 64};
  mlir::Type elementType = builder.getF16Type();

  // Create the specific encoding attribute
  llvm::SmallVector<unsigned, 2> sizePerThread{2, 2};
  llvm::SmallVector<unsigned, 2> order{1, 0};
  unsigned numWarps = 4;
  unsigned numThreadsPerWarp = 32;
  auto argCTALayout = ttg::CTAEncodingAttr::getDefault(ctx, 2);
  mlir::Attribute encoding = ttg::BlockedEncodingAttr::get(ctx, shape, sizePerThread, order, numWarps, numThreadsPerWarp, argCTALayout);
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
