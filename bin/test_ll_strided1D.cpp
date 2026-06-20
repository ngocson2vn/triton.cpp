#include <bitset>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "triton/Tools/LinearLayout.h"

using namespace mlir;
using namespace mlir::triton;

namespace tt = mlir::triton;


int applyStrided1DLayout(MLIRContext* ctx, LinearLayout& layout, int v) {
  int res = 0;

  auto inDimName = layout.getInDims()[0].first;
  auto outDimName = layout.getOutDims()[0].first;

  // L(x) = x0*L(e0) XOR x1*L(e1) XOR ... XOR xn*L(en)
  // L(ei) are basis vectors
  for (int bit = 0; bit < layout.getInDimSizeLog2(inDimName); ++bit) {
    if ((v >> bit) & 1) { // If this physical bit is ON
      res ^= layout.getBasis(inDimName, bit, outDimName);
    }
  }

  return res; 
}

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  MLIRContext context(registry);
  MLIRContext* ctx = &context;


  llvm::outs() << "========================================================================================\n";
  llvm::outs() << "LinearLayout\n";
  llvm::outs() << "========================================================================================\n";

  auto msg = StringAttr::get(ctx, "msg");
  auto offset = StringAttr::get(ctx, "offset");

  auto layout = LinearLayout::strided1D(8, 2, msg, offset);
  llvm::outs() << "Strided 1D Layout: " << layout << "\n\n";

  llvm::outs() << "Apply layout:\n";
  for (int v = 0; v < layout.getInDimSize(msg); v++) {
    auto w = applyStrided1DLayout(ctx, layout, v);
    llvm::outs() << msg.str() << " = " << v << " -> " << offset.str() << " = " << w << "\n";
  }

  llvm::outs() << "\nAll done\n";
  return 0;
}
