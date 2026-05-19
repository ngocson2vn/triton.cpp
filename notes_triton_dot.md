# Lowering Triton Dot op
```C++
bin/triton_compiler.cpp
  -> std::string targetStr = std::string("cuda:").append(std::to_string(capability));
  -> pm.addPass(mlir::triton::createConvertTritonToTritonGPU({targetStr, 4, 32, 1}));

  -> pm.addPass(mlir::triton::gpu::createTritonGPUAccelerateMatmul());

  -> pm.addPass(mlir::triton::nvidia_gpu::createTritonNvidiaGPUTMALoweringPass());

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

    // 3. Transforms tt.descriptor_load -> ttng.async_tma_copy_global_to_local
    -> TritonNvidiaGPUTMALoweringPass::runOnOperation() (lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:196)
      -> mlir::applyPatternsGreedily(...)
        -> TMALoadLowering::matchAndRewrite(...) (lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:67)

    // 4. Converts ttng.tc_gen5_mma to Inline PTX Assembly (tcgen05.mma)
    -> ConvertTritonGPUToLLVM::::runOnOperation() (third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp:80)
      -> mlir::triton::NVIDIA::populateTCGen5MMAOpToLLVMPattern(typeConverter, patterns, benefit);
        -> patterns.add<TCGen5MMAOpConversion, TCGen5MMAScaledOpConversion, TCGen5CommitOpConversion>(typeConverter, benefit);
      -> mlir::applyPartialConversion(...)
        -> TCGen5MMAOpConversion::matchAndRewrite(...) (third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp:656)
```

## 1. Adds DotOperandEncodingAttr
lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:215
<br/>

Output IR:
```MLIR
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    // ...
    %4 = ttg.convert_layout %0 : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked3}>>
    %5 = ttg.convert_layout %3 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked3}>>
    %6 = ttg.convert_layout %cst : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #blocked3>
    %7 = tt.dot %4, %5, %6, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked3}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked3}>> -> tensor<64x32xf32, #blocked3>
    // ...
  }
}
```


## 2. Transforms tt.dot -> ttng.tc_gen5_mma
lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:529

Output IR:
```MLIR
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #blocked1>
    %1 = ttg.local_alloc %0 : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %2 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #blocked1>
    %3 = tt.trans %2 {order = array<i32: 1, 0>} : tensor<32x64xf16, #blocked1> -> tensor<64x32xf16, #blocked2>
    %4 = ttg.local_alloc %3 : (tensor<64x32xf16, #blocked2>) -> !ttg.memdesc<64x32xf16, #shared1, #smem>
    %5 = ttg.convert_layout %cst : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #linear>
    %result, %token = ttng.tmem_alloc %5 : (tensor<64x32xf32, #linear>) -> (!ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %6 = ttng.tc_gen5_mma %1, %4, %result[%token], %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x32xf16, #shared1, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0, %token_1 = ttng.tmem_load %result[%6] : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %7 = ttg.convert_layout %result_0 : tensor<64x32xf32, #linear> -> tensor<64x32xf32, #blocked>
    %8 = ttg.convert_layout %7 : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #blocked3>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %8 : !tt.tensordesc<tensor<64x32xf32>>, tensor<64x32xf32, #blocked3>
    tt.return
  }
}
```

## 3. Lowering ttng.async_tma_copy_global_to_local
third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp:1310
```MLIR
"ttng.async_tma_copy_global_to_local"(%2, %27, %27, %44, %36, %25) <{cache = 1 : i32, evict = 1 : i32, isVolatile = false}> : (!tt.tensordesc<tensor<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, i32, i32, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, i1) -> ()
```

```C++
auto smemTy = op.getResult().getType();
// smemTy: !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>

auto msgToPackedOffset = getMsgToPackedOffsetLayout(smemTy);
```

Debug:
```
--- operator*(LinearLayout inner, LinearLayout outer) ---
inner: 
(empty layout)

outer: 
 - msg is a size 1 dimension
where out dims are: [dim0 (size 64)]

inDims:
  - "msg"
outDims:
  - "dim0"

inDimSizesLog2:
  - "msg": 0
outDimSizesLog2:
  - "dim0": 6

result: 
 - msg is a size 1 dimension
where out dims are: [dim0 (size 64)]
---------------------------------------------------------

--- operator*(LinearLayout inner, LinearLayout outer) ---
inner: 
 - msg is a size 1 dimension
where out dims are: [dim0 (size 64)]

outer: 
 - msg is a size 1 dimension
where out dims are: [dim1 (size 64)]

inDims:
  - "msg"
outDims:
  - "dim0"
  - "dim1"

inDimSizesLog2:
  - "msg": 0
outDimSizesLog2:
  - "dim0": 6
  - "dim1": 6

result: 
 - msg is a size 1 dimension
where out dims are: [dim0 (size 64), dim1 (size 64)]
---------------------------------------------------------

--- operator*(LinearLayout inner, LinearLayout outer) ---
inner: 
 - msg is a size 1 dimension
where out dims are: [dim0 (size 64), dim1 (size 64)]

outer: 
 - block is a size 1 dimension
where out dims are: [dim0 (size 1)]

inDims:
  - "msg"
  - "block"
outDims:
  - "dim0"
  - "dim1"

inDimSizesLog2:
  - "msg": 0
  - "block": 0
outDimSizesLog2:
  - "dim0": 6
  - "dim1": 6

result: 
 - msg is a size 1 dimension
 - block is a size 1 dimension
where out dims are: [dim0 (size 64), dim1 (size 64)]
---------------------------------------------------------

--- operator*(LinearLayout inner, LinearLayout outer) ---
inner: 
 - msg is a size 1 dimension
 - block is a size 1 dimension
where out dims are: [dim0 (size 64), dim1 (size 64)]

outer: 
 - block is a size 1 dimension
where out dims are: [dim1 (size 1)]

inDims:
  - "msg"
  - "block"
outDims:
  - "dim0"
  - "dim1"

inDimSizesLog2:
  - "msg": 0
  - "block": 0
outDimSizesLog2:
  - "dim0": 6
  - "dim1": 6

result: 
 - msg is a size 1 dimension
 - block is a size 1 dimension
where out dims are: [dim0 (size 64), dim1 (size 64)]
---------------------------------------------------------
```
