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

    // Since one TMA op copies a tile of blockShape, each TMA op must stride over blockShape[dim] per dim
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

Debug logs:
```
--- getMsgToPackedOffsetLayout ---
shapePerCTA: [128, 128]
blockShape: [64, 64]

"dim0" layout:
 - msg=1 -> (64)
where out dims are: [dim0 (size 128)]

"dim1" layout:
 - msg=1 -> (64)
where out dims are: [dim1 (size 128)]

msgToOffset: 
 - msg=1 -> (64, 0)
   msg=2 -> (0, 64)
where out dims are: [dim0 (size 128), dim1 (size 128)]

CTAOrder: [0, 1]
CTASplitNum: [1, 1]

"dim0" layout:
 - block is a size 1 dimension
where out dims are: [dim0 (size 1)]

"dim1" layout:
 - block is a size 1 dimension
where out dims are: [dim1 (size 1)]

Final msgToOffset: 
 - msg=1 -> (64, 0)
   msg=2 -> (0, 64)
 - block is a size 1 dimension
where out dims are: [dim0 (size 128), dim1 (size 128)]
----------------------------------
```

About the final msgToOffset: <br/>
Since `block` is a size 1 dimension, its sole value is 0 => it requires zero bit.

Since the length of the standard basis of `msg` is 2 => it requires 2 bits => its values are [0, 1, 2, 3].

The basis of `msg` vector space is: $e_0 = (1, 0), \ e_1 = (0, 1)$. <br/>
$
\begin{aligned}
\text{msgToOffset}(e_0) = (64, 0) \\
\text{msgToOffset}(e_1) = (0, 64)
\end{aligned}
$

For every vector $u=(u_0, u_1) \in \text{msg}$: <br/>
$\text{msgToOffset}(u) = u_0 * \text{msgToOffset}(e_0) + u_1 * \text{msgToOffset}(e_1)$

$
\begin{aligned}
&\text{msg} = 0 \Rightarrow u = (0, 0):\ \text{msgToOffset}(u) = 0 * (64, 0) + 0 * (0, 64) = (0, 0) \\
&\text{msg} = 1 \Rightarrow u = (1, 0):\ \text{msgToOffset}(u) = 1 * (64, 0) + 0 * (0, 64) = (64, 0) \\
&\text{msg} = 2 \Rightarrow u = (0, 1):\ \text{msgToOffset}(u) = 0 * (64, 0) + 1 * (0, 64) = (0, 64) \\
&\text{msg} = 3 \Rightarrow u = (1, 1):\ \text{msgToOffset}(u) = 1 * (64, 0) + 1 * (0, 64) = (64, 64)
\end{aligned}
$