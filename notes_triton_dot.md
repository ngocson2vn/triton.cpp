# Lowering Triton Dot op
```C++
bin/triton_compiler.cpp
  -> std::string targetStr = std::string("cuda:").append(std::to_string(capability));
  -> pm.addPass(mlir::triton::createConvertTritonToTritonGPU({targetStr, 4, 32, 1}));

  -> pm.addPass(mlir::triton::gpu::createTritonGPUAccelerateMatmul());

  -> pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(capability, ptxVersion));
  
  -> pm.run(module.get())
    // 1. Adds DotOperandEncodingAttr
    -> ConvertTritonToTritonGPU::runOnOperation() (lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:802)
      -> populateTritonPatterns(typeConverter, patterns, numCTAs);
        -> patterns.insert<..., TritonDotPattern, ...>(typeConverter, context);
      -> mlir::applyPartialConversion(...)
        -> TritonDotPattern::matchAndRewrite(...) (lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:215)

    // 2. Transforms tt.dot -> ttng.tc_gen5_mma
    -> TritonGPUAccelerateMatmulPass::runOnOperation() (lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:991)
      -> patterns.add<BlockedToMMAv5, ScaledBlockedToMMAv5>(context, computeCapability, benefitMMAv5);
      -> mlir::applyPatternsGreedily(...)
        -> BlockedToMMAv5::matchAndRewrite(...) (lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:529)

    // 3. Converts ttng.tc_gen5_mma to Inline PTX Assembly (tcgen05.mma)
    -> ConvertTritonGPUToLLVM::::runOnOperation() (third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:80)
      -> mlir::triton::NVIDIA::populateTCGen5MMAOpToLLVMPattern(typeConverter, patterns, benefit);
        -> patterns.add<TCGen5MMAOpConversion, TCGen5MMAScaledOpConversion, TCGen5CommitOpConversion>(typeConverter, benefit);
      -> mlir::applyPartialConversion(...)
        -> TCGen5MMAOpConversion::matchAndRewrite(...) (third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp:656)
```
