# Product of 2 LinearLayouts
Implementation: lib/Tools/LinearLayout.cpp
```C++
LinearLayout operator*(LinearLayout inner, LinearLayout outer) {
  // Check that dims common to outer and inner have the same relative order.
  auto inDims = supremum(llvm::to_vector(inner.getInDimNames()),
                         llvm::to_vector(outer.getInDimNames()));
  auto outDims = supremum(llvm::to_vector(inner.getOutDimNames()),
                          llvm::to_vector(outer.getOutDimNames()));

  int level = 1;
  auto level_str = std::getenv("SONY_LOG_LEVEL");
  if (level_str != nullptr) {
    level = std::stoi(level_str);
  }

  auto& sonyOs = getSonyOs(level);
  sonyOs << "--- operator*(LinearLayout inner, LinearLayout outer) ---\n";
  
  sonyOs << "inner: " << inner << "\n\n";

  sonyOs << "outer: " << outer << "\n\n";

  // Get the sizeLog2 of all input and output dimensions we're going to
  // consider, in order.  `inner` is more minor, so its dimensions come
  // first.
  llvm::MapVector<StringAttr, int32_t> inDimSizesLog2;
  llvm::MapVector<StringAttr, int32_t> outDimSizesLog2;
  for (const auto &dim : inDims)
    inDimSizesLog2.insert({dim, 0});
  for (const auto &dim : outDims)
    outDimSizesLog2.insert({dim, 0});
  for (const auto &layout : {inner, outer}) {
    for (StringAttr inDim : layout.getInDimNames()) {
      inDimSizesLog2[inDim] += layout.getInDimSizeLog2(inDim);
    }
    for (StringAttr outDim : layout.getOutDimNames()) {
      outDimSizesLog2[outDim] += layout.getOutDimSizeLog2(outDim);
    }
  }

  sonyOs << "inDimSizesLog2:\n";
  for (const auto& it : inDimSizesLog2) {
    sonyOs << "  - " << it.first << ": " << it.second << "\n";
  }

  sonyOs << "outDimSizesLog2:\n";
  for (const auto& it : outDimSizesLog2) {
    sonyOs << "  - " << it.first << ": " << it.second << "\n";
  }
  sonyOs << "\n";

  BasesT allBases;
  for (auto [inDimName, inDimSizeLog2] : inDimSizesLog2) {
    std::vector<std::vector<int32_t>> &inDimBases = allBases[inDimName];

    // Fill with zeros.
    inDimBases = std::vector<std::vector<int32_t>>(
        inDimSizeLog2, std::vector<int32_t>(outDimSizesLog2.size(), 0));

    for (auto [outDimIdx, outDimNameAndSize] : llvm::enumerate(outDimSizesLog2)) {
      auto [outDimName, outDimSize] = outDimNameAndSize;
      if (inner.hasInDim(inDimName) && inner.hasOutDim(outDimName)) {
        for (int i = 0; i < inner.getInDimSizeLog2(inDimName); i++) {
          inDimBases[i][outDimIdx] = inner.getBasis(inDimName, i, outDimName);
        }
      }
      if (outer.hasInDim(inDimName) && outer.hasOutDim(outDimName)) {
        int offset = inner.hasInDim(inDimName)   ? inner.getInDimSizeLog2(inDimName)   : 0;
        int shift  = inner.hasOutDim(outDimName) ? inner.getOutDimSizeLog2(outDimName) : 0;
        for (int i = 0; i < outer.getInDimSizeLog2(inDimName); i++) {
          inDimBases[offset + i][outDimIdx] = outer.getBasis(inDimName, i, outDimName) << shift;
        }
      }
    }
  }

  llvm::SmallVector<std::pair<StringAttr, int32_t>> outDimSizes;
  for (auto [outDim, sizeLog2] : outDimSizesLog2) {
    outDimSizes.push_back({outDim, 1 << sizeLog2});
  }
  auto result = LinearLayout(std::move(allBases), outDimSizes,
                      inner.isSurjective() && outer.isSurjective());
  
  sonyOs << "result: " << result << "\n";
  sonyOs << "---------------------------------------------------------\n\n";

  return result;
}
```
