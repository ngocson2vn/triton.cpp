# AsyncTMACopyGlobalToLocalOpConversion
Source: third_party/nvidia/lib/TritonNVIDIAGPUToLLVMDebug/LoadStoreOpToLLVM.cpp:1305
```C++
struct AsyncTMACopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<
          triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs() << "\n=== AsyncTMACopyGlobalToLocalOpConversion ===\n";
    llvm::outs() << op << "\n\n";

    if (op.getCache() != triton::CacheModifier::NONE)
      return op.emitError("cache modifiers not supported yet");
    if (op.getEvict() != triton::EvictionPolicy::NORMAL)
      return op.emitError("eviction policy not supported yet");
    if (op.getIsVolatile())
      return op.emitError("volatile not supported yet");

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    ttg::MemDescType dstTy = op.getResult().getType();
    llvm::outs() << "dstTy: " << dstTy << "\n\n";

    Type llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto barrierMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getBarrier(),
        typeConverter->convertType(op.getBarrier().getType().getElementType()),
        rewriter);
    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getResult(), llvmElemTy, rewriter);
    Value dstBase = dstMemObj.getShmemAffineBase(loc, rewriter, dstTy);
    auto voidTy = void_ty(op->getContext());
    auto id = getThreadId(rewriter, loc);

    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = ttg::lookupNumWarps(op);
    int warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    Value warpID = nvgpu::WarpIdOp::create(rewriter, loc);
    Value pred = adaptor.getPred();
    // Select just one thread for the TMA copy. This also helps the compiler to
    // figure out that the op is uniform.
    pred = b.and_(pred, LLVM::NVIDIA::createElectPredicate(loc, rewriter));

    auto smemTy = op.getResult().getType();
    Attribute encoding = smemTy.getEncoding();
    auto mmaEncoding = dyn_cast_or_null<NVMMASharedEncodingAttr>(encoding);

    auto shapePerCTA = ttg::getShapePerCTA(smemTy);
    int rank = op.getCoord().size();

    // Mapping 1D TMA message IDs -> 2D tensor coordinates
    // L1: msg -> coordinates
    auto msgToPackedOffset = getMsgToPackedOffsetLayout(smemTy);

    // Mapping 1D shared memory offsets -> 2D tensor coordinates
    // L2: shared_memory_offset -> coordinates
    auto smemLayout = ttg::toLinearLayout(smemTy);

    // L1: msg -> coordinates
    // L2^{-1}: coordinates -> shared_memory_offset
    // L2^{-1}(L1): msg -> shared_memory_offset
    // invert: find the inverse of smemLayout
    // compose: smemLayout^{-1}(msgToPackedOffset)
    auto msgToShared = msgToPackedOffset.invertAndCompose(smemLayout);

    auto msgToOffset = getMsgToUnpackedOffsetLayout(msgToPackedOffset, smemTy);

    auto ctx = op.getContext();
    auto kMsg = str_attr("msg");
    auto kBlock = str_attr("block");
    const auto numCopies = msgToOffset.getInDimSize(kMsg);
    auto zero = b.i32_val(0);
    auto ctaId = nvgpu::ClusterCTAIdOp::create(rewriter, loc);

    // The bounding box inner dimension must be less than or equal to the
    // swizzle size.
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    // We clamp the block size and the codegen will emit multiple copy
    // operations.
    for (int copyIdx = 0; copyIdx < numCopies; copyIdx += numWarps) {
      int numWarpsToCopy = std::min(numCopies - copyIdx, numWarps);
      if (numWarpsToCopy == 1)
        warpID = b.i32_val(0);
      Value boxPred =
          b.and_(pred, b.icmp_ult(id, b.i32_val(numWarpsToCopy * warpSize)));
      ::mlir::triton::PTXBuilder ptxBuilderTMA;
      Type elemPtrTy = ptr_ty(rewriter.getContext(), 3);
      Value copyIdxVal = b.add(warpID, b.i32_val(copyIdx));
      Value shMemOffset =
          applyLinearLayout(loc, rewriter, msgToShared,
                            {{kMsg, copyIdxVal}, {kBlock, zero}})[0]
              .second;
      Value shMemPtr = b.gep(elemPtrTy, llvmElemTy, dstBase, shMemOffset);
      SmallVector<PTXBuilder::Operand *> operands = {
          ptxBuilderTMA.newOperand(boxPred, "b"),
          ptxBuilderTMA.newOperand(shMemPtr, "r"),
          ptxBuilderTMA.newOperand(adaptor.getDesc(), "l")};
      std::string tmaInst =
          "@$0 cp.async.bulk.tensor." + std::to_string(rank) +
          "d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {";

      auto offsets = applyLinearLayout(loc, rewriter, msgToOffset,
                                       {{kMsg, copyIdxVal}, {kBlock, ctaId}});
      int operandIdx = 3;
      for (int i = 0; i < rank; i++) {
        Value coord = adaptor.getCoord()[rank - i - 1];
        if (i < offsets.size())
          coord = b.add(coord, offsets[offsets.size() - i - 1].second);
        operands.push_back(ptxBuilderTMA.newOperand(coord, "r"));
        tmaInst += "$" + std::to_string(operandIdx++);
        if (i != rank - 1)
          tmaInst += ", ";
      }
      operands.push_back(
          ptxBuilderTMA.newOperand(barrierMemObj.getBase(), "r"));
      tmaInst += "}], [$" + std::to_string(operandIdx++) + "];";

      llvm::outs() << tmaInst << "\n\n";

      auto &tma = *ptxBuilderTMA.create(tmaInst);
      tma(operands, /*onlyAttachMLIRArgs=*/true);
      ptxBuilderTMA.launch(rewriter, loc, voidTy);
    }
    rewriter.eraseOp(op);
    return success();
  }
};
```
