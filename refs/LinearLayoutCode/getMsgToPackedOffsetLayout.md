# getMsgToPackedOffsetLayout
Source code: third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp
```C++
static LinearLayout getMsgToPackedOffsetLayout(ttg::MemDescType ty) {
  auto ctx = ty.getContext();
  auto kMsg = str_attr("msg");
  auto kBlock = str_attr("block");
  auto shapePerCTA = ttg::getShapePerCTA(ty);
  int rank = shapePerCTA.size();
  auto blockShape = ttng::getTMABlockShape(ty, /*packedSize=*/true);
  auto outDimNames = standardOutDimNames(ctx, rank);

  // Build a layout for mapping TMA message IDs to dimensional offsets
  LinearLayout msgToOffset;
  for (int dim = 0; dim < rank; ++dim) {
    // shapePerCTA is the tensor shape
    // One TMA op copies a tile of blockShape
    // 
    // The number of TMA messages
    int32_t size = shapePerCTA[dim] / blockShape[dim];

    // Since one TMA op copies a tile of blockShape, each TMA op must stride over blockShape[dim]
    int32_t stride = blockShape[dim]
    msgToOffset *= LinearLayout::strided1D(size, stride, kMsg, outDimNames[dim]);
  }

  // Build a sublayout for mapping CTA IDs to dimensional offsets
  auto ctaLayout = getCTALayout(ty.getEncoding());
  for (int i = 0; i < rank; ++i) {
    auto dim = ctaLayout.getCTAOrder()[i];
    msgToOffset *= LinearLayout::identity1D(ctaLayout.getCTASplitNum()[dim],
                                            kBlock, outDimNames[dim]);
  }
  return msgToOffset;
}
```
