# LocalAllocOpConversion
lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp
```C++
smemObj:
  - base -> global_smem
  - baseElemType = llvmElemTy
  - offsets -> [llvm.mlir.constant(0 : i32), llvm.mlir.constant(0 : i32)]
```