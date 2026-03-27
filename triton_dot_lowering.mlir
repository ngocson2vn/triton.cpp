// -----// IR Dump After ConvertTritonToTritonGPU (convert-triton-to-tritongpu) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #blocked1>
    %1 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #blocked1>
    %2 = tt.trans %1 {order = array<i32: 1, 0>} : tensor<32x64xf16, #blocked1> -> tensor<64x32xf16, #blocked2>
    %3 = ttg.convert_layout %2 : tensor<64x32xf16, #blocked2> -> tensor<64x32xf16, #blocked>
    %4 = ttg.convert_layout %0 : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked3}>>
    %5 = ttg.convert_layout %3 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked3}>>
    %6 = ttg.convert_layout %cst : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #blocked3>
    %7 = tt.dot %4, %5, %6, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked3}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked3}>> -> tensor<64x32xf32, #blocked3>
    %8 = ttg.convert_layout %7 : tensor<64x32xf32, #blocked3> -> tensor<64x32xf32, #blocked>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %8 : !tt.tensordesc<tensor<64x32xf32>>, tensor<64x32xf32, #blocked>
    tt.return
  }
}


// -----// IR Dump After TritonGPUCoalesce (tritongpu-coalesce) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #blocked1>
    %1 = ttg.convert_layout %0 : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %2 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #blocked1>
    %3 = ttg.convert_layout %2 : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #blocked2>
    %4 = tt.trans %3 {order = array<i32: 1, 0>} : tensor<32x64xf16, #blocked2> -> tensor<64x32xf16, #blocked3>
    %5 = ttg.convert_layout %4 : tensor<64x32xf16, #blocked3> -> tensor<64x32xf16, #blocked>
    %6 = ttg.convert_layout %1 : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked4}>>
    %7 = ttg.convert_layout %5 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked4}>>
    %8 = ttg.convert_layout %cst : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #blocked4>
    %9 = tt.dot %6, %7, %8, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked4}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked4}>> -> tensor<64x32xf32, #blocked4>
    %10 = ttg.convert_layout %9 : tensor<64x32xf32, #blocked4> -> tensor<64x32xf32, #blocked>
    %11 = ttg.convert_layout %10 : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #blocked5>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %11 : !tt.tensordesc<tensor<64x32xf32>>, tensor<64x32xf32, #blocked5>
    tt.return
  }
}


// -----// IR Dump After TritonGPURemoveLayoutConversions (tritongpu-remove-layout-conversions) //----- //
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #blocked1>
    %1 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #blocked1>
    %2 = tt.trans %1 {order = array<i32: 1, 0>} : tensor<32x64xf16, #blocked1> -> tensor<64x32xf16, #blocked2>
    %3 = ttg.convert_layout %0 : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %4 = ttg.convert_layout %2 : tensor<64x32xf16, #blocked2> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %5 = tt.dot %3, %4, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x32xf32, #blocked>
    %6 = ttg.convert_layout %5 : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #blocked3>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %6 : !tt.tensordesc<tensor<64x32xf32>>, tensor<64x32xf32, #blocked3>
    tt.return
  }
}


// -----// IR Dump After TritonGPUAccelerateMatmul (tritongpu-accelerate-matmul) //----- //
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


// -----// IR Dump After TritonGPURemoveLayoutConversions (tritongpu-remove-layout-conversions) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #blocked>
    %1 = ttg.local_alloc %0 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %2 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #blocked>
    %3 = tt.trans %2 {order = array<i32: 1, 0>} : tensor<32x64xf16, #blocked> -> tensor<64x32xf16, #blocked1>
    %4 = ttg.local_alloc %3 : (tensor<64x32xf16, #blocked1>) -> !ttg.memdesc<64x32xf16, #shared1, #smem>
    %result, %token = ttng.tmem_alloc %cst : (tensor<64x32xf32, #linear>) -> (!ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %5 = ttng.tc_gen5_mma %1, %4, %result[%token], %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x32xf16, #shared1, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0, %token_1 = ttng.tmem_load %result[%5] : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %6 = ttg.convert_layout %result_0 : tensor<64x32xf32, #linear> -> tensor<64x32xf32, #blocked2>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %6 : !tt.tensordesc<tensor<64x32xf32>>, tensor<64x32xf32, #blocked2>
    tt.return
  }
}


// -----// IR Dump After TritonGPUOptimizeDotOperands (tritongpu-optimize-dot-operands) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #blocked>
    %1 = ttg.local_alloc %0 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %2 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #blocked>
    %3 = ttg.local_alloc %2 : (tensor<32x64xf16, #blocked>) -> !ttg.memdesc<32x64xf16, #shared, #smem>
    %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem> -> !ttg.memdesc<64x32xf16, #shared1, #smem>
    %result, %token = ttng.tmem_alloc %cst : (tensor<64x32xf32, #linear>) -> (!ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %5 = ttng.tc_gen5_mma %1, %4, %result[%token], %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x32xf16, #shared1, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0, %token_1 = ttng.tmem_load %result[%5] : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %6 = ttg.convert_layout %result_0 : tensor<64x32xf32, #linear> -> tensor<64x32xf32, #blocked1>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %6 : !tt.tensordesc<tensor<64x32xf32>>, tensor<64x32xf32, #blocked1>
    tt.return
  }
}


// -----// IR Dump After TritonGPURemoveLayoutConversions (tritongpu-remove-layout-conversions) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16, #blocked>
    %1 = ttg.local_alloc %0 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %2 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #blocked>
    %3 = ttg.local_alloc %2 : (tensor<32x64xf16, #blocked>) -> !ttg.memdesc<32x64xf16, #shared, #smem>
    %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem> -> !ttg.memdesc<64x32xf16, #shared1, #smem>
    %result, %token = ttng.tmem_alloc %cst : (tensor<64x32xf32, #linear>) -> (!ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %5 = ttng.tc_gen5_mma %1, %4, %result[%token], %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x32xf16, #shared1, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0, %token_1 = ttng.tmem_load %result[%5] : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %6 = ttg.convert_layout %result_0 : tensor<64x32xf32, #linear> -> tensor<64x32xf32, #blocked1>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %6 : !tt.tensordesc<tensor<64x32xf32>>, tensor<64x32xf32, #blocked1>
    tt.return
  }
}


// -----// IR Dump After TritonNvidiaGPUOptimizeDescriptorEncodingPass (triton-nvidia-optimize-descriptor-encoding) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked>
    %1 = ttg.local_alloc %0 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %2 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16, #shared>> -> tensor<32x64xf16, #blocked>
    %3 = ttg.local_alloc %2 : (tensor<32x64xf16, #blocked>) -> !ttg.memdesc<32x64xf16, #shared, #smem>
    %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem> -> !ttg.memdesc<64x32xf16, #shared2, #smem>
    %result, %token = ttng.tmem_alloc %cst : (tensor<64x32xf32, #linear>) -> (!ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %5 = ttng.tc_gen5_mma %1, %4, %result[%token], %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x32xf16, #shared2, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0, %token_1 = ttng.tmem_load %result[%5] : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %6 = ttg.convert_layout %result_0 : tensor<64x32xf32, #linear> -> tensor<64x32xf32, #blocked1>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %6 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, tensor<64x32xf32, #blocked1>
    tt.return
  }
}


// -----// IR Dump After TritonGPUHoistTMEMAlloc (tritongpu-hoist-tmem-alloc) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked>
    %1 = ttg.local_alloc %0 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %2 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16, #shared>> -> tensor<32x64xf16, #blocked>
    %3 = ttg.local_alloc %2 : (tensor<32x64xf16, #blocked>) -> !ttg.memdesc<32x64xf16, #shared, #smem>
    %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem> -> !ttg.memdesc<64x32xf16, #shared2, #smem>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %true_0 = arith.constant true
    %5 = ttng.tmem_store %cst, %result[%token], %true_0 : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %6 = ttng.tc_gen5_mma %1, %4, %result[%5], %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x32xf16, #shared2, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_1, %token_2 = ttng.tmem_load %result[%6] : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %7 = ttg.convert_layout %result_1 : tensor<64x32xf32, #linear> -> tensor<64x32xf32, #blocked1>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %7 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, tensor<64x32xf32, #blocked1>
    tt.return
  }
}


// -----// IR Dump After TritonNvidiaGPUPromoteLHSToTMemPass (tritongpu-promote-lhs-to-tmem) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked>
    %1 = ttg.local_alloc %0 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %2 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16, #shared>> -> tensor<32x64xf16, #blocked>
    %3 = ttg.local_alloc %2 : (tensor<32x64xf16, #blocked>) -> !ttg.memdesc<32x64xf16, #shared, #smem>
    %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem> -> !ttg.memdesc<64x32xf16, #shared2, #smem>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %5 = ttng.tmem_store %cst, %result[%token], %true : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %6 = ttng.tc_gen5_mma %1, %4, %result[%5], %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x32xf16, #shared2, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0, %token_1 = ttng.tmem_load %result[%6] : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %7 = ttg.convert_layout %result_0 : tensor<64x32xf32, #linear> -> tensor<64x32xf32, #blocked1>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %7 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, tensor<64x32xf32, #blocked1>
    tt.return
  }
}


// -----// IR Dump After SCCP (sccp) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked>
    %1 = ttg.local_alloc %0 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %2 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16, #shared>> -> tensor<32x64xf16, #blocked>
    %3 = ttg.local_alloc %2 : (tensor<32x64xf16, #blocked>) -> !ttg.memdesc<32x64xf16, #shared, #smem>
    %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem> -> !ttg.memdesc<64x32xf16, #shared2, #smem>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %5 = ttng.tmem_store %cst, %result[%token], %true : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %6 = ttng.tc_gen5_mma %1, %4, %result[%5], %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x32xf16, #shared2, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0, %token_1 = ttng.tmem_load %result[%6] : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %7 = ttg.convert_layout %result_0 : tensor<64x32xf32, #linear> -> tensor<64x32xf32, #blocked1>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %7 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, tensor<64x32xf32, #blocked1>
    tt.return
  }
}


// -----// IR Dump After TritonGPUAutomaticWarpSpecialization (tritongpu-automatic-warp-specialization) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked>
    %1 = ttg.local_alloc %0 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %2 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16, #shared>> -> tensor<32x64xf16, #blocked>
    %3 = ttg.local_alloc %2 : (tensor<32x64xf16, #blocked>) -> !ttg.memdesc<32x64xf16, #shared, #smem>
    %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem> -> !ttg.memdesc<64x32xf16, #shared2, #smem>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %5 = ttng.tmem_store %cst, %result[%token], %true : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %6 = ttng.tc_gen5_mma %1, %4, %result[%5], %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x32xf16, #shared2, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0, %token_1 = ttng.tmem_load %result[%6] : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %7 = ttg.convert_layout %result_0 : tensor<64x32xf32, #linear> -> tensor<64x32xf32, #blocked1>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %7 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, tensor<64x32xf32, #blocked1>
    tt.return
  }
}


// -----// IR Dump After TritonGPUHoistTMEMAlloc (tritongpu-hoist-tmem-alloc) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked>
    %1 = ttg.local_alloc %0 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %2 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16, #shared>> -> tensor<32x64xf16, #blocked>
    %3 = ttg.local_alloc %2 : (tensor<32x64xf16, #blocked>) -> !ttg.memdesc<32x64xf16, #shared, #smem>
    %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem> -> !ttg.memdesc<64x32xf16, #shared2, #smem>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %true_0 = arith.constant true
    %5 = ttng.tmem_store %cst, %result[%token], %true_0 : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %6 = ttng.tc_gen5_mma %1, %4, %result[%5], %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x32xf16, #shared2, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_1, %token_2 = ttng.tmem_load %result[%6] : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %7 = ttg.convert_layout %result_1 : tensor<64x32xf32, #linear> -> tensor<64x32xf32, #blocked1>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %7 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, tensor<64x32xf32, #blocked1>
    tt.return
  }
}


// -----// IR Dump After TritonNvidiaGPURemoveTMEMTokensPass (triton-nvidia-gpu-remove-tmem-tokens) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %0 = ub.poison : !ttg.async.token
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %1 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked>
    %2 = ttg.local_alloc %1 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %3 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16, #shared>> -> tensor<32x64xf16, #blocked>
    %4 = ttg.local_alloc %3 : (tensor<32x64xf16, #blocked>) -> !ttg.memdesc<32x64xf16, #shared, #smem>
    %5 = ttg.memdesc_trans %4 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem> -> !ttg.memdesc<64x32xf16, #shared2, #smem>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %true_0 = arith.constant true
    ttng.tmem_store %cst, %result, %true_0 : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %2, %5, %result, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x32xf16, #shared2, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_1 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %6 = ttg.convert_layout %result_1 : tensor<64x32xf32, #linear> -> tensor<64x32xf32, #blocked1>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %6 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, tensor<64x32xf32, #blocked1>
    tt.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked>
    %1 = ttg.local_alloc %0 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %2 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16, #shared>> -> tensor<32x64xf16, #blocked>
    %3 = ttg.local_alloc %2 : (tensor<32x64xf16, #blocked>) -> !ttg.memdesc<32x64xf16, #shared, #smem>
    %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem> -> !ttg.memdesc<64x32xf16, #shared2, #smem>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %result, %true : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %1, %4, %result, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x32xf16, #shared2, #smem>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %5 = ttg.convert_layout %result_0 : tensor<64x32xf32, #linear> -> tensor<64x32xf32, #blocked1>
    tt.descriptor_store %arg10[%c0_i32, %c0_i32], %5 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, tensor<64x32xf32, #blocked1>
    tt.return
  }
}


// -----// IR Dump After TritonNvidiaGPUTMALoweringPass (triton-nvidia-tma-lowering) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %1, 8192, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %1, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %3, 4096, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg5[%c0_i32, %c0_i32] %2, %3, %true : !tt.tensordesc<tensor<32x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %3, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %3 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %4 = ttg.memdesc_trans %2 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %result, %true : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %0, %4, %result, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %5 = ttg.convert_layout %result_0 : tensor<64x32xf32, #linear> -> tensor<64x32xf32, #blocked>
    %6 = ttg.local_alloc %5 : (tensor<64x32xf32, #blocked>) -> !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %arg10[%c0_i32, %c0_i32] %6 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}


// -----// IR Dump After TritonGPURemoveLayoutConversions (tritongpu-remove-layout-conversions) //----- //
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %1, 8192, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %1, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %3, 4096, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg5[%c0_i32, %c0_i32] %2, %3, %true : !tt.tensordesc<tensor<32x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %3, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %3 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %4 = ttg.memdesc_trans %2 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %result, %true : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %0, %4, %result, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %5 = ttg.local_alloc %result_0 : (tensor<64x32xf32, #linear>) -> !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %arg10[%c0_i32, %c0_i32] %5 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}


// -----// IR Dump After TritonGPUReorderInstructions (tritongpu-reorder-instructions) //----- //
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %1, 8192, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %1, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %3 = ttg.memdesc_trans %2 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %4, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %4, 4096, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg5[%c0_i32, %c0_i32] %2, %4, %true : !tt.tensordesc<tensor<32x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %4, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %4 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %result, %true : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %0, %3, %result, %true, %true : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_0 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %5 = ttg.local_alloc %result_0 : (tensor<64x32xf32, #linear>) -> !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %arg10[%c0_i32, %c0_i32] %5 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}


// -----// IR Dump After TritonNvidiaGPUMMALoweringPass (triton-nvidia-mma-lowering) //----- //
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %1, 8192, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %1, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %3 = ttg.memdesc_trans %2 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %4, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %4, 4096, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg5[%c0_i32, %c0_i32] %2, %4, %true : !tt.tensordesc<tensor<32x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %4, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %4 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %result, %true : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.tc_gen5_mma %0, %3, %result, %true, %true, %5[%true] {is_async} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.wait_barrier %5, %c0_i32, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %5 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result_0 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %6 = ttg.local_alloc %result_0 : (tensor<64x32xf32, #linear>) -> !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %arg10[%c0_i32, %c0_i32] %6 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}


// -----// IR Dump After SCCP (sccp) //----- //
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %1, 8192, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %1, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %3 = ttg.memdesc_trans %2 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %4, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %4, 4096, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg5[%c0_i32, %c0_i32] %2, %4, %true : !tt.tensordesc<tensor<32x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %4, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %4 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %result, %true : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.tc_gen5_mma %0, %3, %result, %true, %true, %5[%true] {is_async} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.wait_barrier %5, %c0_i32, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %5 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result_0 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %6 = ttg.local_alloc %result_0 : (tensor<64x32xf32, #linear>) -> !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %arg10[%c0_i32, %c0_i32] %6 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}


// -----// IR Dump After TritonGPUAllocateWarpGroups (tritongpu-allocate-warp-groups) //----- //
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %1, 8192, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %1, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %3 = ttg.memdesc_trans %2 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %4, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %4, 4096, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg5[%c0_i32, %c0_i32] %2, %4, %true : !tt.tensordesc<tensor<32x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %4, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %4 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %result, %true : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.tc_gen5_mma %0, %3, %result, %true, %true, %5[%true] {is_async} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.wait_barrier %5, %c0_i32, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %5 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result_0 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %6 = ttg.local_alloc %result_0 : (tensor<64x32xf32, #linear>) -> !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %arg10[%c0_i32, %c0_i32] %6 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}


// -----// IR Dump After AllocateSharedMemory (allocate-shared-memory) //----- //
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 12296 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %1, 8192, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %1, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %2 = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %3 = ttg.memdesc_trans %2 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>
    %4 = ttg.local_alloc {allocation.offset = 12288 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %4, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %4, 4096, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg5[%c0_i32, %c0_i32] %2, %4, %true : !tt.tensordesc<tensor<32x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %4, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %4 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %result, %true : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %5 = ttg.local_alloc {allocation.offset = 12288 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.tc_gen5_mma %0, %3, %result, %true, %true, %5[%true] {is_async} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.wait_barrier %5, %c0_i32, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %5 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result_0 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %6 = ttg.local_alloc %result_0 {allocation.offset = 0 : i32} : (tensor<64x32xf32, #linear>) -> !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %arg10[%c0_i32, %c0_i32] %6 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}


// -----// IR Dump After TritonTensorMemoryAllocationPass (triton-tensor-memory-allocation) //----- //
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 12296 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 32 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %1, 8192, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %1, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %2 = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %3 = ttg.memdesc_trans %2 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>
    %4 = ttg.local_alloc {allocation.offset = 12288 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %4, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %4, 4096, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg5[%c0_i32, %c0_i32] %2, %4, %true : !tt.tensordesc<tensor<32x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %4, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %4 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %result, %true : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %5 = ttg.local_alloc {allocation.offset = 12288 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.tc_gen5_mma %0, %3, %result, %true, %true, %5[%true] {is_async} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.wait_barrier %5, %c0_i32, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %5 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result_0 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %6 = ttg.local_alloc %result_0 {allocation.offset = 0 : i32} : (tensor<64x32xf32, #linear>) -> !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %arg10[%c0_i32, %c0_i32] %6 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}


// -----// IR Dump After TritonGPUGlobalScratchAllocationPass (tritongpu-global-scratch-memory-allocation) //----- //
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 12296 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 32 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %1, 8192, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %1, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %2 = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %3 = ttg.memdesc_trans %2 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>
    %4 = ttg.local_alloc {allocation.offset = 12288 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %4, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %4, 4096, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg5[%c0_i32, %c0_i32] %2, %4, %true : !tt.tensordesc<tensor<32x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %4, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %4 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %result, %true : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %5 = ttg.local_alloc {allocation.offset = 12288 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.tc_gen5_mma %0, %3, %result, %true, %true, %5[%true] {is_async} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.wait_barrier %5, %c0_i32, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %5 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result_0 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %6 = ttg.local_alloc %result_0 {allocation.offset = 0 : i32} : (tensor<64x32xf32, #linear>) -> !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %arg10[%c0_i32, %c0_i32] %6 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}


// -----// IR Dump After TritonGPUProxyFenceInsertion (triton-nvidia-gpu-proxy-fence-insertion) //----- //
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 12296 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 32 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #shared1>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #linear>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %1, 8192, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %1, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %2 = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    %3 = ttg.memdesc_trans %2 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>
    %4 = ttg.local_alloc {allocation.offset = 12288 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %4, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.barrier_expect %4, 4096, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_global_to_local %arg5[%c0_i32, %c0_i32] %2, %4, %true : !tt.tensordesc<tensor<32x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %4, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %4 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %result, %true : tensor<64x32xf32, #linear> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>
    %5 = ttg.local_alloc {allocation.offset = 12288 : i32} : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.tc_gen5_mma %0, %3, %result, %true, %true, %5[%true] {is_async} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x32xf16, #shared3, #smem, mutable>, !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.wait_barrier %5, %c0_i32, %true : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.inval_barrier %5 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    %result_0 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #linear>
    %6 = ttg.local_alloc %result_0 {allocation.offset = 0 : i32} : (tensor<64x32xf32, #linear>) -> !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %arg10[%c0_i32, %c0_i32] %6 : !tt.tensordesc<tensor<64x32xf32, #shared1>>, !ttg.memdesc<64x32xf32, #shared1, #smem>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}


// -----// IR Dump After ConvertTritonGPUToLLVM (convert-triton-gpu-to-llvm) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 12296 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 32 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr<1>, %arg16: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %0 = builtin.unrealized_conversion_cast %arg10 : !llvm.ptr to !tt.tensordesc<tensor<64x32xf32, #shared>>
    %1 = builtin.unrealized_conversion_cast %arg5 : !llvm.ptr to !tt.tensordesc<tensor<32x64xf16, #shared1>>
    %2 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !tt.tensordesc<tensor<64x64xf16, #shared1>>
    %3 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %4 = llvm.bitcast %3 : f32 to f32
    %5 = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %7 = llvm.insertvalue %4, %6[1] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %8 = llvm.insertvalue %4, %7[2] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %9 = llvm.insertvalue %4, %8[3] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %10 = llvm.insertvalue %4, %9[4] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %11 = llvm.insertvalue %4, %10[5] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %12 = llvm.insertvalue %4, %11[6] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %13 = llvm.insertvalue %4, %12[7] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %14 = llvm.insertvalue %4, %13[8] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %15 = llvm.insertvalue %4, %14[9] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %16 = llvm.insertvalue %4, %15[10] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %17 = llvm.insertvalue %4, %16[11] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %18 = llvm.insertvalue %4, %17[12] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %19 = llvm.insertvalue %4, %18[13] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %20 = llvm.insertvalue %4, %19[14] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %21 = llvm.insertvalue %4, %20[15] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %22 = llvm.mlir.constant(true) : i1
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = llvm.mlir.constant(0 : i32) : i32
    %25 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %26 = llvm.getelementptr %25[%24] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %27 = llvm.mlir.constant(0 : i32) : i32
    %28 = llvm.mlir.undef : !llvm.struct<(ptr<3>, i32, i32)>
    %29 = llvm.insertvalue %26, %28[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %30 = llvm.insertvalue %27, %29[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %31 = llvm.insertvalue %27, %30[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    %32 = llvm.mlir.constant(8192 : i32) : i32
    %33 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %34 = llvm.getelementptr %33[%32] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %35 = llvm.mlir.constant(0 : i32) : i32
    %36 = llvm.mlir.undef : !llvm.struct<(ptr<3>, i32)>
    %37 = llvm.insertvalue %34, %36[0] : !llvm.struct<(ptr<3>, i32)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(ptr<3>, i32)> 
    %39 = llvm.extractvalue %38[0] : !llvm.struct<(ptr<3>, i32)> 
    %40 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<3>, i32)> 
    %41 = nvvm.read.ptx.sreg.tid.x : i32
    %42 = llvm.mlir.constant(127 : i32) : i32
    %43 = llvm.and %41, %42 : i32
    %44 = llvm.mlir.constant(0 : i32) : i32
    %45 = llvm.icmp "eq" %43, %44 : i32
    %46 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %45, %39 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %47 = llvm.extractvalue %38[0] : !llvm.struct<(ptr<3>, i32)> 
    %48 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<3>, i32)> 
    %49 = nvvm.read.ptx.sreg.tid.x : i32
    %50 = llvm.mlir.constant(127 : i32) : i32
    %51 = llvm.and %49, %50 : i32
    %52 = llvm.mlir.constant(0 : i32) : i32
    %53 = llvm.icmp "eq" %51, %52 : i32
    %54 = llvm.and %53, %22 : i1
    %55 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 8192;", "b,r" %54, %47 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %56 = llvm.extractvalue %38[0] : !llvm.struct<(ptr<3>, i32)> 
    %57 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<3>, i32)> 
    %58 = llvm.extractvalue %31[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %59 = llvm.extractvalue %31[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %60 = llvm.extractvalue %31[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    %61 = llvm.mlir.constant(0 : i32) : i32
    %62 = llvm.getelementptr %58[%61] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %63 = nvvm.read.ptx.sreg.tid.x : i32
    %64 = llvm.mlir.constant(127 : i32) : i32
    %65 = llvm.and %63, %64 : i32
    %66 = nvg.warp_id
    %67 = nvvm.elect.sync -> i1
    %68 = llvm.and %22, %67 : i1
    %69 = llvm.mlir.constant(0 : i32) : i32
    %70 = llvm.mlir.constant(0 : i32) : i32
    %71 = nvg.cluster_id
    %72 = llvm.mlir.constant(0 : i32) : i32
    %73 = llvm.mlir.constant(32 : i32) : i32
    %74 = llvm.icmp "ult" %65, %73 : i32
    %75 = llvm.and %68, %74 : i1
    %76 = llvm.mlir.constant(0 : i32) : i32
    %77 = llvm.add %72, %76 : i32
    %78 = llvm.mlir.constant(0 : i32) : i32
    %79 = llvm.mlir.constant(0 : i32) : i32
    %80 = llvm.mlir.constant(0 : i32) : i32
    %81 = llvm.shl %77, %80 : i32
    %82 = llvm.or %79, %81 : i32
    %83 = llvm.mlir.constant(0 : i32) : i32
    %84 = llvm.mlir.constant(0 : i32) : i32
    %85 = llvm.mlir.constant(0 : i32) : i32
    %86 = llvm.or disjoint %84, %85 : i32
    %87 = llvm.xor %78, %86 : i32
    %88 = llvm.mlir.constant(0 : i32) : i32
    %89 = llvm.mlir.constant(0 : i32) : i32
    %90 = llvm.mlir.constant(0 : i32) : i32
    %91 = llvm.or disjoint %89, %90 : i32
    %92 = llvm.xor %78, %91 : i32
    %93 = llvm.getelementptr %62[%87] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %94 = llvm.mlir.constant(0 : i32) : i32
    %95 = llvm.mlir.constant(0 : i32) : i32
    %96 = llvm.mlir.constant(0 : i32) : i32
    %97 = llvm.shl %77, %96 : i32
    %98 = llvm.or %95, %97 : i32
    %99 = llvm.mlir.constant(0 : i32) : i32
    %100 = llvm.shl %70, %99 : i32
    %101 = llvm.or %98, %100 : i32
    %102 = llvm.mlir.constant(0 : i32) : i32
    %103 = llvm.mlir.constant(0 : i32) : i32
    %104 = llvm.mlir.constant(0 : i32) : i32
    %105 = llvm.or disjoint %103, %104 : i32
    %106 = llvm.xor %94, %105 : i32
    %107 = llvm.mlir.constant(0 : i32) : i32
    %108 = llvm.mlir.constant(0 : i32) : i32
    %109 = llvm.mlir.constant(0 : i32) : i32
    %110 = llvm.or disjoint %108, %109 : i32
    %111 = llvm.xor %94, %110 : i32
    %112 = llvm.add %23, %111 : i32
    %113 = llvm.add %23, %106 : i32
    %114 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %75, %93, %arg0, %112, %113, %56 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %115 = llvm.extractvalue %38[0] : !llvm.struct<(ptr<3>, i32)> 
    %116 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<3>, i32)> 
    %117 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %115, %23 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %118 = llvm.extractvalue %38[0] : !llvm.struct<(ptr<3>, i32)> 
    %119 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<3>, i32)> 
    %120 = nvvm.read.ptx.sreg.tid.x : i32
    %121 = llvm.mlir.constant(127 : i32) : i32
    %122 = llvm.and %120, %121 : i32
    %123 = llvm.mlir.constant(0 : i32) : i32
    %124 = llvm.icmp "eq" %122, %123 : i32
    %125 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %124, %118 : (i1, !llvm.ptr<3>) -> !llvm.void
    %126 = llvm.mlir.constant(8192 : i32) : i32
    %127 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %128 = llvm.getelementptr %127[%126] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %129 = llvm.mlir.constant(0 : i32) : i32
    %130 = llvm.mlir.undef : !llvm.struct<(ptr<3>, i32, i32)>
    %131 = llvm.insertvalue %128, %130[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %132 = llvm.insertvalue %129, %131[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %133 = llvm.insertvalue %129, %132[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    %134 = llvm.extractvalue %133[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %135 = llvm.extractvalue %133[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %136 = llvm.extractvalue %133[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    %137 = llvm.mlir.undef : !llvm.struct<(ptr<3>, i32, i32)>
    %138 = llvm.insertvalue %134, %137[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %139 = llvm.insertvalue %136, %138[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %140 = llvm.insertvalue %135, %139[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    %141 = llvm.mlir.constant(12288 : i32) : i32
    %142 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %143 = llvm.getelementptr %142[%141] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %144 = llvm.mlir.constant(0 : i32) : i32
    %145 = llvm.mlir.undef : !llvm.struct<(ptr<3>, i32)>
    %146 = llvm.insertvalue %143, %145[0] : !llvm.struct<(ptr<3>, i32)> 
    %147 = llvm.insertvalue %144, %146[1] : !llvm.struct<(ptr<3>, i32)> 
    %148 = llvm.extractvalue %147[0] : !llvm.struct<(ptr<3>, i32)> 
    %149 = llvm.extractvalue %147[1] : !llvm.struct<(ptr<3>, i32)> 
    %150 = nvvm.read.ptx.sreg.tid.x : i32
    %151 = llvm.mlir.constant(127 : i32) : i32
    %152 = llvm.and %150, %151 : i32
    %153 = llvm.mlir.constant(0 : i32) : i32
    %154 = llvm.icmp "eq" %152, %153 : i32
    %155 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %154, %148 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %156 = llvm.extractvalue %147[0] : !llvm.struct<(ptr<3>, i32)> 
    %157 = llvm.extractvalue %147[1] : !llvm.struct<(ptr<3>, i32)> 
    %158 = nvvm.read.ptx.sreg.tid.x : i32
    %159 = llvm.mlir.constant(127 : i32) : i32
    %160 = llvm.and %158, %159 : i32
    %161 = llvm.mlir.constant(0 : i32) : i32
    %162 = llvm.icmp "eq" %160, %161 : i32
    %163 = llvm.and %162, %22 : i1
    %164 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 4096;", "b,r" %163, %156 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
    nvvm.barrier0
    %165 = llvm.extractvalue %147[0] : !llvm.struct<(ptr<3>, i32)> 
    %166 = llvm.extractvalue %147[1] : !llvm.struct<(ptr<3>, i32)> 
    %167 = llvm.extractvalue %133[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %168 = llvm.extractvalue %133[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %169 = llvm.extractvalue %133[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    %170 = llvm.mlir.constant(0 : i32) : i32
    %171 = llvm.getelementptr %167[%170] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %172 = nvvm.read.ptx.sreg.tid.x : i32
    %173 = llvm.mlir.constant(127 : i32) : i32
    %174 = llvm.and %172, %173 : i32
    %175 = nvg.warp_id
    %176 = nvvm.elect.sync -> i1
    %177 = llvm.and %22, %176 : i1
    %178 = llvm.mlir.constant(0 : i32) : i32
    %179 = llvm.mlir.constant(0 : i32) : i32
    %180 = nvg.cluster_id
    %181 = llvm.mlir.constant(0 : i32) : i32
    %182 = llvm.mlir.constant(32 : i32) : i32
    %183 = llvm.icmp "ult" %174, %182 : i32
    %184 = llvm.and %177, %183 : i1
    %185 = llvm.mlir.constant(0 : i32) : i32
    %186 = llvm.add %181, %185 : i32
    %187 = llvm.mlir.constant(0 : i32) : i32
    %188 = llvm.mlir.constant(0 : i32) : i32
    %189 = llvm.mlir.constant(0 : i32) : i32
    %190 = llvm.shl %186, %189 : i32
    %191 = llvm.or %188, %190 : i32
    %192 = llvm.mlir.constant(0 : i32) : i32
    %193 = llvm.mlir.constant(0 : i32) : i32
    %194 = llvm.mlir.constant(0 : i32) : i32
    %195 = llvm.or disjoint %193, %194 : i32
    %196 = llvm.xor %187, %195 : i32
    %197 = llvm.mlir.constant(0 : i32) : i32
    %198 = llvm.mlir.constant(0 : i32) : i32
    %199 = llvm.mlir.constant(0 : i32) : i32
    %200 = llvm.or disjoint %198, %199 : i32
    %201 = llvm.xor %187, %200 : i32
    %202 = llvm.getelementptr %171[%196] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %203 = llvm.mlir.constant(0 : i32) : i32
    %204 = llvm.mlir.constant(0 : i32) : i32
    %205 = llvm.mlir.constant(0 : i32) : i32
    %206 = llvm.shl %186, %205 : i32
    %207 = llvm.or %204, %206 : i32
    %208 = llvm.mlir.constant(0 : i32) : i32
    %209 = llvm.shl %179, %208 : i32
    %210 = llvm.or %207, %209 : i32
    %211 = llvm.mlir.constant(0 : i32) : i32
    %212 = llvm.mlir.constant(0 : i32) : i32
    %213 = llvm.mlir.constant(0 : i32) : i32
    %214 = llvm.or disjoint %212, %213 : i32
    %215 = llvm.xor %203, %214 : i32
    %216 = llvm.mlir.constant(0 : i32) : i32
    %217 = llvm.mlir.constant(0 : i32) : i32
    %218 = llvm.mlir.constant(0 : i32) : i32
    %219 = llvm.or disjoint %217, %218 : i32
    %220 = llvm.xor %203, %219 : i32
    %221 = llvm.add %23, %220 : i32
    %222 = llvm.add %23, %215 : i32
    %223 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %184, %202, %arg5, %221, %222, %165 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %224 = llvm.extractvalue %147[0] : !llvm.struct<(ptr<3>, i32)> 
    %225 = llvm.extractvalue %147[1] : !llvm.struct<(ptr<3>, i32)> 
    %226 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %224, %23 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %227 = llvm.extractvalue %147[0] : !llvm.struct<(ptr<3>, i32)> 
    %228 = llvm.extractvalue %147[1] : !llvm.struct<(ptr<3>, i32)> 
    %229 = nvvm.read.ptx.sreg.tid.x : i32
    %230 = llvm.mlir.constant(127 : i32) : i32
    %231 = llvm.and %229, %230 : i32
    %232 = llvm.mlir.constant(0 : i32) : i32
    %233 = llvm.icmp "eq" %231, %232 : i32
    %234 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %233, %227 : (i1, !llvm.ptr<3>) -> !llvm.void
    %235 = nvg.tensor_memory_base
    %236 = llvm.ptrtoint %235 : !llvm.ptr<6> to i32
    %237 = llvm.mlir.constant(0 : i32) : i32
    %238 = llvm.add %236, %237 : i32
    %239 = llvm.inttoptr %238 : i32 to !llvm.ptr<3>
    %240 = llvm.extractvalue %21[0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %241 = llvm.extractvalue %21[1] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %242 = llvm.extractvalue %21[2] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %243 = llvm.extractvalue %21[3] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %244 = llvm.extractvalue %21[4] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %245 = llvm.extractvalue %21[5] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %246 = llvm.extractvalue %21[6] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %247 = llvm.extractvalue %21[7] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %248 = llvm.extractvalue %21[8] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %249 = llvm.extractvalue %21[9] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %250 = llvm.extractvalue %21[10] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %251 = llvm.extractvalue %21[11] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %252 = llvm.extractvalue %21[12] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %253 = llvm.extractvalue %21[13] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %254 = llvm.extractvalue %21[14] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %255 = llvm.extractvalue %21[15] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %256 = llvm.ptrtoint %239 : !llvm.ptr<3> to i32
    %257 = nvg.warp_id
    %258 = llvm.mlir.constant(3 : i32) : i32
    %259 = llvm.and %257, %258 : i32
    %260 = llvm.mlir.constant(21 : i32) : i32
    %261 = llvm.shl %259, %260 : i32
    %262 = llvm.add %256, %261 : i32
    %263 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.st.sync.aligned.16x32bx2.x16.b32 [$1 + 0], 16, {$2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17};", "b,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r" %22, %262, %240, %241, %242, %243, %244, %245, %246, %247, %248, %249, %250, %251, %252, %253, %254, %255 : (i1, i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.void
    nvvm.tcgen05.wait <store>
    nvvm.barrier0
    %264 = llvm.mlir.constant(12288 : i32) : i32
    %265 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %266 = llvm.getelementptr %265[%264] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %267 = llvm.mlir.constant(0 : i32) : i32
    %268 = llvm.mlir.undef : !llvm.struct<(ptr<3>, i32)>
    %269 = llvm.insertvalue %266, %268[0] : !llvm.struct<(ptr<3>, i32)> 
    %270 = llvm.insertvalue %267, %269[1] : !llvm.struct<(ptr<3>, i32)> 
    nvvm.barrier0
    %271 = llvm.extractvalue %270[0] : !llvm.struct<(ptr<3>, i32)> 
    %272 = llvm.extractvalue %270[1] : !llvm.struct<(ptr<3>, i32)> 
    %273 = nvvm.read.ptx.sreg.tid.x : i32
    %274 = llvm.mlir.constant(127 : i32) : i32
    %275 = llvm.and %273, %274 : i32
    %276 = llvm.mlir.constant(0 : i32) : i32
    %277 = llvm.icmp "eq" %275, %276 : i32
    %278 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %277, %271 : (i1, !llvm.ptr<3>) -> !llvm.void
    %279 = llvm.ptrtoint %239 : !llvm.ptr<3> to i32
    %280 = nvg.warp_id
    %281 = llvm.mlir.constant(0 : i32) : i32
    %282 = llvm.icmp "eq" %280, %281 : i32
    %283 = llvm.and %22, %282 : i1
    llvm.cond_br %283, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %284 = nvvm.elect.sync -> i1
    %285 = llvm.extractvalue %31[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %286 = llvm.extractvalue %31[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %287 = llvm.extractvalue %31[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    %288 = llvm.mlir.constant(0 : i32) : i32
    %289 = llvm.getelementptr %285[%288] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %290 = llvm.extractvalue %140[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %291 = llvm.extractvalue %140[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %292 = llvm.extractvalue %140[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    %293 = llvm.mlir.constant(0 : i32) : i32
    %294 = llvm.getelementptr %290[%293] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %295 = llvm.ptrtoint %289 : !llvm.ptr<3> to i32
    %296 = llvm.mlir.constant(4 : i32) : i32
    %297 = llvm.lshr %295, %296 : i32
    %298 = llvm.mlir.constant(16383 : i32) : i32
    %299 = llvm.and %297, %298 : i32
    %300 = llvm.zext %299 : i32 to i64
    %301 = llvm.ptrtoint %294 : !llvm.ptr<3> to i32
    %302 = llvm.mlir.constant(4 : i32) : i32
    %303 = llvm.lshr %301, %302 : i32
    %304 = llvm.mlir.constant(16383 : i32) : i32
    %305 = llvm.and %303, %304 : i32
    %306 = llvm.zext %305 : i32 to i64
    %307 = llvm.mlir.constant(4611756662049472512 : i64) : i64
    %308 = llvm.add %307, %300 : i64
    %309 = llvm.mlir.constant(4611756662049472512 : i64) : i64
    %310 = llvm.add %309, %306 : i64
    %311 = llvm.mlir.constant(67633168 : i32) : i32
    %312 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %279, %308, %310, %311, %22, %284 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %313 = llvm.mlir.constant(true) : i1
    %314 = llvm.mlir.constant(4611756662049472514 : i64) : i64
    %315 = llvm.add %314, %300 : i64
    %316 = llvm.mlir.constant(4611756662049472514 : i64) : i64
    %317 = llvm.add %316, %306 : i64
    %318 = llvm.mlir.constant(67633168 : i32) : i32
    %319 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %279, %315, %317, %318, %313, %284 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %320 = llvm.mlir.constant(true) : i1
    %321 = llvm.mlir.constant(4611756662049472516 : i64) : i64
    %322 = llvm.add %321, %300 : i64
    %323 = llvm.mlir.constant(4611756662049472516 : i64) : i64
    %324 = llvm.add %323, %306 : i64
    %325 = llvm.mlir.constant(67633168 : i32) : i32
    %326 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %279, %322, %324, %325, %320, %284 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %327 = llvm.mlir.constant(true) : i1
    %328 = llvm.mlir.constant(4611756662049472518 : i64) : i64
    %329 = llvm.add %328, %300 : i64
    %330 = llvm.mlir.constant(4611756662049472518 : i64) : i64
    %331 = llvm.add %330, %306 : i64
    %332 = llvm.mlir.constant(67633168 : i32) : i32
    %333 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %279, %329, %331, %332, %327, %284 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %334 = llvm.mlir.constant(true) : i1
    %335 = llvm.and %22, %284 : i1
    %336 = llvm.extractvalue %270[0] : !llvm.struct<(ptr<3>, i32)> 
    %337 = llvm.extractvalue %270[1] : !llvm.struct<(ptr<3>, i32)> 
    %338 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$1];", "b,l" %335, %336 : (i1, !llvm.ptr<3>) -> !llvm.void
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    nvvm.barrier0
    %339 = llvm.extractvalue %270[0] : !llvm.struct<(ptr<3>, i32)> 
    %340 = llvm.extractvalue %270[1] : !llvm.struct<(ptr<3>, i32)> 
    %341 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %339, %23 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %342 = llvm.extractvalue %270[0] : !llvm.struct<(ptr<3>, i32)> 
    %343 = llvm.extractvalue %270[1] : !llvm.struct<(ptr<3>, i32)> 
    %344 = nvvm.read.ptx.sreg.tid.x : i32
    %345 = llvm.mlir.constant(127 : i32) : i32
    %346 = llvm.and %344, %345 : i32
    %347 = llvm.mlir.constant(0 : i32) : i32
    %348 = llvm.icmp "eq" %346, %347 : i32
    %349 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %348, %342 : (i1, !llvm.ptr<3>) -> !llvm.void
    %350 = llvm.mlir.constant(true) : i1
    %351 = llvm.ptrtoint %239 : !llvm.ptr<3> to i32
    %352 = nvg.warp_id
    %353 = llvm.mlir.constant(3 : i32) : i32
    %354 = llvm.and %352, %353 : i32
    %355 = llvm.mlir.constant(21 : i32) : i32
    %356 = llvm.shl %354, %355 : i32
    %357 = llvm.add %351, %356 : i32
    %358 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15}, [$16 + 0], 16;", "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r" %357 : (i32) -> !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    %359 = llvm.extractvalue %358[0] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %360 = llvm.bitcast %359 : i32 to f32
    %361 = llvm.extractvalue %358[1] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %362 = llvm.bitcast %361 : i32 to f32
    %363 = llvm.extractvalue %358[2] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %364 = llvm.bitcast %363 : i32 to f32
    %365 = llvm.extractvalue %358[3] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %366 = llvm.bitcast %365 : i32 to f32
    %367 = llvm.extractvalue %358[4] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %368 = llvm.bitcast %367 : i32 to f32
    %369 = llvm.extractvalue %358[5] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %370 = llvm.bitcast %369 : i32 to f32
    %371 = llvm.extractvalue %358[6] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %372 = llvm.bitcast %371 : i32 to f32
    %373 = llvm.extractvalue %358[7] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %374 = llvm.bitcast %373 : i32 to f32
    %375 = llvm.extractvalue %358[8] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %376 = llvm.bitcast %375 : i32 to f32
    %377 = llvm.extractvalue %358[9] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %378 = llvm.bitcast %377 : i32 to f32
    %379 = llvm.extractvalue %358[10] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %380 = llvm.bitcast %379 : i32 to f32
    %381 = llvm.extractvalue %358[11] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %382 = llvm.bitcast %381 : i32 to f32
    %383 = llvm.extractvalue %358[12] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %384 = llvm.bitcast %383 : i32 to f32
    %385 = llvm.extractvalue %358[13] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %386 = llvm.bitcast %385 : i32 to f32
    %387 = llvm.extractvalue %358[14] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %388 = llvm.bitcast %387 : i32 to f32
    %389 = llvm.extractvalue %358[15] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %390 = llvm.bitcast %389 : i32 to f32
    %391 = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
    %392 = llvm.insertvalue %360, %391[0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %393 = llvm.insertvalue %362, %392[1] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %394 = llvm.insertvalue %364, %393[2] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %395 = llvm.insertvalue %366, %394[3] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %396 = llvm.insertvalue %368, %395[4] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %397 = llvm.insertvalue %370, %396[5] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %398 = llvm.insertvalue %372, %397[6] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %399 = llvm.insertvalue %374, %398[7] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %400 = llvm.insertvalue %376, %399[8] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %401 = llvm.insertvalue %378, %400[9] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %402 = llvm.insertvalue %380, %401[10] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %403 = llvm.insertvalue %382, %402[11] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %404 = llvm.insertvalue %384, %403[12] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %405 = llvm.insertvalue %386, %404[13] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %406 = llvm.insertvalue %388, %405[14] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %407 = llvm.insertvalue %390, %406[15] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    nvvm.tcgen05.wait <load>
    %408 = llvm.mlir.constant(0 : i32) : i32
    %409 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %410 = llvm.getelementptr %409[%408] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %411 = llvm.mlir.constant(0 : i32) : i32
    %412 = llvm.extractvalue %407[0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %413 = llvm.extractvalue %407[1] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %414 = llvm.extractvalue %407[2] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %415 = llvm.extractvalue %407[3] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %416 = llvm.extractvalue %407[4] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %417 = llvm.extractvalue %407[5] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %418 = llvm.extractvalue %407[6] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %419 = llvm.extractvalue %407[7] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %420 = llvm.extractvalue %407[8] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %421 = llvm.extractvalue %407[9] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %422 = llvm.extractvalue %407[10] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %423 = llvm.extractvalue %407[11] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %424 = llvm.extractvalue %407[12] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %425 = llvm.extractvalue %407[13] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %426 = llvm.extractvalue %407[14] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %427 = llvm.extractvalue %407[15] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %428 = llvm.mlir.constant(0 : i32) : i32
    %429 = nvvm.read.ptx.sreg.tid.x : i32
    %430 = llvm.mlir.constant(127 : i32) : i32
    %431 = llvm.and %429, %430 : i32
    %432 = llvm.mlir.constant(32 : i32) : i32
    %433 = llvm.urem %431, %432 : i32
    %434 = llvm.udiv %431, %432 : i32
    %435 = llvm.mlir.constant(0 : i32) : i32
    %436 = llvm.mlir.constant(0 : i32) : i32
    %437 = llvm.mlir.constant(0 : i32) : i32
    %438 = llvm.mlir.constant(0 : i32) : i32
    %439 = llvm.shl %433, %438 : i32
    %440 = llvm.or %437, %439 : i32
    %441 = llvm.mlir.constant(5 : i32) : i32
    %442 = llvm.shl %434, %441 : i32
    %443 = llvm.or %440, %442 : i32
    %444 = llvm.mlir.constant(15 : i32) : i32
    %445 = llvm.and %443, %444 : i32
    %446 = llvm.mlir.constant(7 : i32) : i32
    %447 = llvm.shl %445, %446 : i32
    %448 = llvm.mlir.constant(96 : i32) : i32
    %449 = llvm.and %443, %448 : i32
    %450 = llvm.mlir.constant(6 : i32) : i32
    %451 = llvm.shl %449, %450 : i32
    %452 = llvm.mlir.constant(7 : i32) : i32
    %453 = llvm.and %443, %452 : i32
    %454 = llvm.mlir.constant(4 : i32) : i32
    %455 = llvm.shl %453, %454 : i32
    %456 = llvm.mlir.constant(0 : i32) : i32
    %457 = llvm.mlir.constant(16 : i32) : i32
    %458 = llvm.and %443, %457 : i32
    %459 = llvm.icmp "eq" %458, %456 : i32
    %460 = llvm.mlir.constant(64 : i32) : i32
    %461 = llvm.select %459, %456, %460 : i1, i32
    %462 = llvm.xor %451, %455 : i32
    %463 = llvm.xor %462, %461 : i32
    %464 = llvm.or disjoint %447, %463 : i32
    %465 = llvm.xor %436, %464 : i32
    %466 = llvm.mlir.constant(4 : i32) : i32
    %467 = llvm.mul %428, %466 : i32
    %468 = llvm.xor %465, %467 : i32
    %469 = llvm.mlir.constant(0 : i32) : i32
    %470 = llvm.xor %468, %469 : i32
    %471 = llvm.mlir.constant(0 : i32) : i32
    %472 = llvm.add %470, %471 : i32
    %473 = llvm.getelementptr inbounds %410[%472] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %474 = llvm.mlir.undef : vector<4xf32>
    %475 = llvm.mlir.constant(0 : i32) : i32
    %476 = llvm.insertelement %412, %474[%475 : i32] : vector<4xf32>
    %477 = llvm.mlir.constant(1 : i32) : i32
    %478 = llvm.insertelement %413, %476[%477 : i32] : vector<4xf32>
    %479 = llvm.mlir.constant(2 : i32) : i32
    %480 = llvm.insertelement %414, %478[%479 : i32] : vector<4xf32>
    %481 = llvm.mlir.constant(3 : i32) : i32
    %482 = llvm.insertelement %415, %480[%481 : i32] : vector<4xf32>
    %483 = llvm.mlir.constant(true) : i1
    %484 = llvm.mlir.constant(0 : i32) : i32
    %485 = llvm.extractelement %482[%484 : i32] : vector<4xf32>
    %486 = llvm.mlir.constant(1 : i32) : i32
    %487 = llvm.extractelement %482[%486 : i32] : vector<4xf32>
    %488 = llvm.mlir.constant(2 : i32) : i32
    %489 = llvm.extractelement %482[%488 : i32] : vector<4xf32>
    %490 = llvm.mlir.constant(3 : i32) : i32
    %491 = llvm.extractelement %482[%490 : i32] : vector<4xf32>
    %492 = llvm.bitcast %485 : f32 to i32
    %493 = llvm.bitcast %487 : f32 to i32
    %494 = llvm.bitcast %489 : f32 to i32
    %495 = llvm.bitcast %491 : f32 to i32
    %496 = llvm.mlir.undef : vector<4xi32>
    %497 = llvm.mlir.constant(0 : i32) : i32
    %498 = llvm.insertelement %492, %496[%497 : i32] : vector<4xi32>
    %499 = llvm.mlir.constant(1 : i32) : i32
    %500 = llvm.insertelement %493, %498[%499 : i32] : vector<4xi32>
    %501 = llvm.mlir.constant(2 : i32) : i32
    %502 = llvm.insertelement %494, %500[%501 : i32] : vector<4xi32>
    %503 = llvm.mlir.constant(3 : i32) : i32
    %504 = llvm.insertelement %495, %502[%503 : i32] : vector<4xi32>
    llvm.store %504, %473 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %505 = llvm.mlir.constant(16 : i32) : i32
    %506 = llvm.xor %468, %505 : i32
    %507 = llvm.mlir.constant(0 : i32) : i32
    %508 = llvm.add %506, %507 : i32
    %509 = llvm.getelementptr inbounds %410[%508] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %510 = llvm.mlir.undef : vector<4xf32>
    %511 = llvm.mlir.constant(0 : i32) : i32
    %512 = llvm.insertelement %416, %510[%511 : i32] : vector<4xf32>
    %513 = llvm.mlir.constant(1 : i32) : i32
    %514 = llvm.insertelement %417, %512[%513 : i32] : vector<4xf32>
    %515 = llvm.mlir.constant(2 : i32) : i32
    %516 = llvm.insertelement %418, %514[%515 : i32] : vector<4xf32>
    %517 = llvm.mlir.constant(3 : i32) : i32
    %518 = llvm.insertelement %419, %516[%517 : i32] : vector<4xf32>
    %519 = llvm.mlir.constant(true) : i1
    %520 = llvm.mlir.constant(0 : i32) : i32
    %521 = llvm.extractelement %518[%520 : i32] : vector<4xf32>
    %522 = llvm.mlir.constant(1 : i32) : i32
    %523 = llvm.extractelement %518[%522 : i32] : vector<4xf32>
    %524 = llvm.mlir.constant(2 : i32) : i32
    %525 = llvm.extractelement %518[%524 : i32] : vector<4xf32>
    %526 = llvm.mlir.constant(3 : i32) : i32
    %527 = llvm.extractelement %518[%526 : i32] : vector<4xf32>
    %528 = llvm.bitcast %521 : f32 to i32
    %529 = llvm.bitcast %523 : f32 to i32
    %530 = llvm.bitcast %525 : f32 to i32
    %531 = llvm.bitcast %527 : f32 to i32
    %532 = llvm.mlir.undef : vector<4xi32>
    %533 = llvm.mlir.constant(0 : i32) : i32
    %534 = llvm.insertelement %528, %532[%533 : i32] : vector<4xi32>
    %535 = llvm.mlir.constant(1 : i32) : i32
    %536 = llvm.insertelement %529, %534[%535 : i32] : vector<4xi32>
    %537 = llvm.mlir.constant(2 : i32) : i32
    %538 = llvm.insertelement %530, %536[%537 : i32] : vector<4xi32>
    %539 = llvm.mlir.constant(3 : i32) : i32
    %540 = llvm.insertelement %531, %538[%539 : i32] : vector<4xi32>
    llvm.store %540, %509 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %541 = llvm.mlir.constant(32 : i32) : i32
    %542 = llvm.xor %468, %541 : i32
    %543 = llvm.mlir.constant(0 : i32) : i32
    %544 = llvm.add %542, %543 : i32
    %545 = llvm.getelementptr inbounds %410[%544] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %546 = llvm.mlir.undef : vector<4xf32>
    %547 = llvm.mlir.constant(0 : i32) : i32
    %548 = llvm.insertelement %420, %546[%547 : i32] : vector<4xf32>
    %549 = llvm.mlir.constant(1 : i32) : i32
    %550 = llvm.insertelement %421, %548[%549 : i32] : vector<4xf32>
    %551 = llvm.mlir.constant(2 : i32) : i32
    %552 = llvm.insertelement %422, %550[%551 : i32] : vector<4xf32>
    %553 = llvm.mlir.constant(3 : i32) : i32
    %554 = llvm.insertelement %423, %552[%553 : i32] : vector<4xf32>
    %555 = llvm.mlir.constant(true) : i1
    %556 = llvm.mlir.constant(0 : i32) : i32
    %557 = llvm.extractelement %554[%556 : i32] : vector<4xf32>
    %558 = llvm.mlir.constant(1 : i32) : i32
    %559 = llvm.extractelement %554[%558 : i32] : vector<4xf32>
    %560 = llvm.mlir.constant(2 : i32) : i32
    %561 = llvm.extractelement %554[%560 : i32] : vector<4xf32>
    %562 = llvm.mlir.constant(3 : i32) : i32
    %563 = llvm.extractelement %554[%562 : i32] : vector<4xf32>
    %564 = llvm.bitcast %557 : f32 to i32
    %565 = llvm.bitcast %559 : f32 to i32
    %566 = llvm.bitcast %561 : f32 to i32
    %567 = llvm.bitcast %563 : f32 to i32
    %568 = llvm.mlir.undef : vector<4xi32>
    %569 = llvm.mlir.constant(0 : i32) : i32
    %570 = llvm.insertelement %564, %568[%569 : i32] : vector<4xi32>
    %571 = llvm.mlir.constant(1 : i32) : i32
    %572 = llvm.insertelement %565, %570[%571 : i32] : vector<4xi32>
    %573 = llvm.mlir.constant(2 : i32) : i32
    %574 = llvm.insertelement %566, %572[%573 : i32] : vector<4xi32>
    %575 = llvm.mlir.constant(3 : i32) : i32
    %576 = llvm.insertelement %567, %574[%575 : i32] : vector<4xi32>
    llvm.store %576, %545 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %577 = llvm.mlir.constant(48 : i32) : i32
    %578 = llvm.xor %468, %577 : i32
    %579 = llvm.mlir.constant(0 : i32) : i32
    %580 = llvm.add %578, %579 : i32
    %581 = llvm.getelementptr inbounds %410[%580] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %582 = llvm.mlir.undef : vector<4xf32>
    %583 = llvm.mlir.constant(0 : i32) : i32
    %584 = llvm.insertelement %424, %582[%583 : i32] : vector<4xf32>
    %585 = llvm.mlir.constant(1 : i32) : i32
    %586 = llvm.insertelement %425, %584[%585 : i32] : vector<4xf32>
    %587 = llvm.mlir.constant(2 : i32) : i32
    %588 = llvm.insertelement %426, %586[%587 : i32] : vector<4xf32>
    %589 = llvm.mlir.constant(3 : i32) : i32
    %590 = llvm.insertelement %427, %588[%589 : i32] : vector<4xf32>
    %591 = llvm.mlir.constant(true) : i1
    %592 = llvm.mlir.constant(0 : i32) : i32
    %593 = llvm.extractelement %590[%592 : i32] : vector<4xf32>
    %594 = llvm.mlir.constant(1 : i32) : i32
    %595 = llvm.extractelement %590[%594 : i32] : vector<4xf32>
    %596 = llvm.mlir.constant(2 : i32) : i32
    %597 = llvm.extractelement %590[%596 : i32] : vector<4xf32>
    %598 = llvm.mlir.constant(3 : i32) : i32
    %599 = llvm.extractelement %590[%598 : i32] : vector<4xf32>
    %600 = llvm.bitcast %593 : f32 to i32
    %601 = llvm.bitcast %595 : f32 to i32
    %602 = llvm.bitcast %597 : f32 to i32
    %603 = llvm.bitcast %599 : f32 to i32
    %604 = llvm.mlir.undef : vector<4xi32>
    %605 = llvm.mlir.constant(0 : i32) : i32
    %606 = llvm.insertelement %600, %604[%605 : i32] : vector<4xi32>
    %607 = llvm.mlir.constant(1 : i32) : i32
    %608 = llvm.insertelement %601, %606[%607 : i32] : vector<4xi32>
    %609 = llvm.mlir.constant(2 : i32) : i32
    %610 = llvm.insertelement %602, %608[%609 : i32] : vector<4xi32>
    %611 = llvm.mlir.constant(3 : i32) : i32
    %612 = llvm.insertelement %603, %610[%611 : i32] : vector<4xi32>
    llvm.store %612, %581 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %613 = llvm.mlir.undef : !llvm.struct<(ptr<3>, i32, i32)>
    %614 = llvm.insertvalue %410, %613[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %615 = llvm.insertvalue %411, %614[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %616 = llvm.insertvalue %411, %615[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
    nvvm.barrier0
    %617 = llvm.extractvalue %616[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %618 = llvm.extractvalue %616[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %619 = llvm.extractvalue %616[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    %620 = llvm.mlir.constant(0 : i32) : i32
    %621 = llvm.getelementptr %617[%620] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %622 = nvvm.read.ptx.sreg.tid.x : i32
    %623 = llvm.mlir.constant(127 : i32) : i32
    %624 = llvm.and %622, %623 : i32
    %625 = nvvm.elect.sync -> i1
    %626 = nvg.warp_id
    %627 = llvm.mlir.constant(0 : i32) : i32
    %628 = llvm.mlir.constant(0 : i32) : i32
    %629 = nvg.cluster_id
    %630 = llvm.mlir.constant(0 : i32) : i32
    %631 = llvm.mlir.constant(32 : i32) : i32
    %632 = llvm.icmp "ult" %624, %631 : i32
    %633 = llvm.and %625, %632 : i1
    %634 = llvm.mlir.constant(0 : i32) : i32
    %635 = llvm.add %630, %634 : i32
    %636 = llvm.mlir.constant(0 : i32) : i32
    %637 = llvm.mlir.constant(0 : i32) : i32
    %638 = llvm.mlir.constant(0 : i32) : i32
    %639 = llvm.shl %635, %638 : i32
    %640 = llvm.or %637, %639 : i32
    %641 = llvm.mlir.constant(0 : i32) : i32
    %642 = llvm.mlir.constant(0 : i32) : i32
    %643 = llvm.mlir.constant(0 : i32) : i32
    %644 = llvm.or disjoint %642, %643 : i32
    %645 = llvm.xor %636, %644 : i32
    %646 = llvm.mlir.constant(0 : i32) : i32
    %647 = llvm.mlir.constant(0 : i32) : i32
    %648 = llvm.mlir.constant(0 : i32) : i32
    %649 = llvm.or disjoint %647, %648 : i32
    %650 = llvm.xor %636, %649 : i32
    %651 = llvm.getelementptr %621[%645] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %652 = llvm.mlir.constant(0 : i32) : i32
    %653 = llvm.mlir.constant(0 : i32) : i32
    %654 = llvm.mlir.constant(0 : i32) : i32
    %655 = llvm.shl %635, %654 : i32
    %656 = llvm.or %653, %655 : i32
    %657 = llvm.mlir.constant(0 : i32) : i32
    %658 = llvm.shl %628, %657 : i32
    %659 = llvm.or %656, %658 : i32
    %660 = llvm.mlir.constant(0 : i32) : i32
    %661 = llvm.mlir.constant(0 : i32) : i32
    %662 = llvm.mlir.constant(0 : i32) : i32
    %663 = llvm.or disjoint %661, %662 : i32
    %664 = llvm.xor %652, %663 : i32
    %665 = llvm.mlir.constant(0 : i32) : i32
    %666 = llvm.mlir.constant(0 : i32) : i32
    %667 = llvm.mlir.constant(0 : i32) : i32
    %668 = llvm.or disjoint %666, %667 : i32
    %669 = llvm.xor %652, %668 : i32
    %670 = llvm.add %23, %669 : i32
    %671 = llvm.add %23, %664 : i32
    %672 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$1, {$2, $3}], [$4];", "b,l,r,r,r" %633, %arg10, %670, %671, %651 : (i1, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.cp.async.bulk.commit.group
    nvvm.cp.async.bulk.wait_group 0 {read}
    nvvm.barrier0
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 12296 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 32 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr<1>, %arg16: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %0 = llvm.mlir.constant(48 : i32) : i32
    %1 = llvm.mlir.undef : vector<4xi32>
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.undef : vector<4xf32>
    %5 = llvm.mlir.constant(64 : i32) : i32
    %6 = llvm.mlir.constant(16 : i32) : i32
    %7 = llvm.mlir.constant(6 : i32) : i32
    %8 = llvm.mlir.constant(96 : i32) : i32
    %9 = llvm.mlir.constant(7 : i32) : i32
    %10 = llvm.mlir.constant(15 : i32) : i32
    %11 = llvm.mlir.constant(5 : i32) : i32
    %12 = llvm.mlir.constant(4611756662049472518 : i64) : i64
    %13 = llvm.mlir.constant(4611756662049472516 : i64) : i64
    %14 = llvm.mlir.constant(4611756662049472514 : i64) : i64
    %15 = llvm.mlir.constant(67633168 : i32) : i32
    %16 = llvm.mlir.constant(4611756662049472512 : i64) : i64
    %17 = llvm.mlir.constant(16383 : i32) : i32
    %18 = llvm.mlir.constant(4 : i32) : i32
    %19 = llvm.mlir.constant(21 : i32) : i32
    %20 = llvm.mlir.constant(3 : i32) : i32
    %21 = llvm.mlir.constant(32 : i32) : i32
    %22 = llvm.mlir.constant(127 : i32) : i32
    %23 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %24 = llvm.mlir.constant(0 : i32) : i32
    %25 = llvm.mlir.constant(true) : i1
    %26 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %27 = llvm.getelementptr %23[8192] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %28 = nvvm.read.ptx.sreg.tid.x : i32
    %29 = llvm.and %28, %22 : i32
    %30 = llvm.icmp "eq" %29, %24 : i32
    %31 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %30, %27 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %32 = nvvm.read.ptx.sreg.tid.x : i32
    %33 = llvm.and %32, %22 : i32
    %34 = llvm.icmp "eq" %33, %24 : i32
    %35 = llvm.and %34, %25 : i1
    %36 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 8192;", "b,r" %35, %27 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %37 = nvvm.read.ptx.sreg.tid.x : i32
    %38 = llvm.and %37, %22 : i32
    %39 = nvvm.elect.sync -> i1
    %40 = llvm.and %25, %39 : i1
    %41 = llvm.icmp "ult" %38, %21 : i32
    %42 = llvm.and %40, %41 : i1
    %43 = llvm.xor %24, %24 : i32
    %44 = llvm.getelementptr %23[%43] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %45 = llvm.xor %24, %24 : i32
    %46 = llvm.xor %24, %24 : i32
    %47 = llvm.add %46, %24 : i32
    %48 = llvm.add %45, %24 : i32
    %49 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %42, %44, %arg0, %47, %48, %27 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %50 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %27, %24 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %51 = nvvm.read.ptx.sreg.tid.x : i32
    %52 = llvm.and %51, %22 : i32
    %53 = llvm.icmp "eq" %52, %24 : i32
    %54 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %53, %27 : (i1, !llvm.ptr<3>) -> !llvm.void
    %55 = llvm.getelementptr %23[8192] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %56 = llvm.getelementptr %23[12288] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %57 = nvvm.read.ptx.sreg.tid.x : i32
    %58 = llvm.and %57, %22 : i32
    %59 = llvm.icmp "eq" %58, %24 : i32
    %60 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %59, %56 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %61 = nvvm.read.ptx.sreg.tid.x : i32
    %62 = llvm.and %61, %22 : i32
    %63 = llvm.icmp "eq" %62, %24 : i32
    %64 = llvm.and %63, %25 : i1
    %65 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 4096;", "b,r" %64, %56 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
    nvvm.barrier0
    %66 = nvvm.read.ptx.sreg.tid.x : i32
    %67 = llvm.and %66, %22 : i32
    %68 = nvvm.elect.sync -> i1
    %69 = llvm.and %25, %68 : i1
    %70 = llvm.icmp "ult" %67, %21 : i32
    %71 = llvm.and %69, %70 : i1
    %72 = llvm.xor %24, %24 : i32
    %73 = llvm.getelementptr %55[%72] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %74 = llvm.xor %24, %24 : i32
    %75 = llvm.xor %24, %24 : i32
    %76 = llvm.add %75, %24 : i32
    %77 = llvm.add %74, %24 : i32
    %78 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %71, %73, %arg5, %76, %77, %56 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %79 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %56, %24 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %80 = nvvm.read.ptx.sreg.tid.x : i32
    %81 = llvm.and %80, %22 : i32
    %82 = llvm.icmp "eq" %81, %24 : i32
    %83 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %82, %56 : (i1, !llvm.ptr<3>) -> !llvm.void
    %84 = nvg.tensor_memory_base
    %85 = llvm.ptrtoint %84 : !llvm.ptr<6> to i32
    %86 = llvm.add %85, %24 : i32
    %87 = llvm.inttoptr %86 : i32 to !llvm.ptr<3>
    %88 = llvm.ptrtoint %87 : !llvm.ptr<3> to i32
    %89 = nvg.warp_id
    %90 = llvm.and %89, %20 : i32
    %91 = llvm.shl %90, %19 : i32
    %92 = llvm.add %88, %91 : i32
    %93 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.st.sync.aligned.16x32bx2.x16.b32 [$1 + 0], 16, {$2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17};", "b,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r" %25, %92, %26, %26, %26, %26, %26, %26, %26, %26, %26, %26, %26, %26, %26, %26, %26, %26 : (i1, i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.void
    nvvm.tcgen05.wait <store>
    nvvm.barrier0
    %94 = llvm.getelementptr %23[12288] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    nvvm.barrier0
    %95 = nvvm.read.ptx.sreg.tid.x : i32
    %96 = llvm.and %95, %22 : i32
    %97 = llvm.icmp "eq" %96, %24 : i32
    %98 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %97, %94 : (i1, !llvm.ptr<3>) -> !llvm.void
    %99 = llvm.ptrtoint %87 : !llvm.ptr<3> to i32
    %100 = nvg.warp_id
    %101 = llvm.icmp "eq" %100, %24 : i32
    %102 = llvm.and %25, %101 : i1
    llvm.cond_br %102, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %103 = nvvm.elect.sync -> i1
    %104 = llvm.ptrtoint %23 : !llvm.ptr<3> to i32
    %105 = llvm.lshr %104, %18 : i32
    %106 = llvm.and %105, %17 : i32
    %107 = llvm.zext %106 : i32 to i64
    %108 = llvm.ptrtoint %55 : !llvm.ptr<3> to i32
    %109 = llvm.lshr %108, %18 : i32
    %110 = llvm.and %109, %17 : i32
    %111 = llvm.zext %110 : i32 to i64
    %112 = llvm.add %107, %16 : i64
    %113 = llvm.add %111, %16 : i64
    %114 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %99, %112, %113, %15, %25, %103 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %115 = llvm.add %107, %14 : i64
    %116 = llvm.add %111, %14 : i64
    %117 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %99, %115, %116, %15, %25, %103 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %118 = llvm.add %107, %13 : i64
    %119 = llvm.add %111, %13 : i64
    %120 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %99, %118, %119, %15, %25, %103 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %121 = llvm.add %107, %12 : i64
    %122 = llvm.add %111, %12 : i64
    %123 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %99, %121, %122, %15, %25, %103 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %124 = llvm.and %25, %103 : i1
    %125 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$1];", "b,l" %124, %94 : (i1, !llvm.ptr<3>) -> !llvm.void
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    nvvm.barrier0
    %126 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %94, %24 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %127 = nvvm.read.ptx.sreg.tid.x : i32
    %128 = llvm.and %127, %22 : i32
    %129 = llvm.icmp "eq" %128, %24 : i32
    %130 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %129, %94 : (i1, !llvm.ptr<3>) -> !llvm.void
    %131 = llvm.ptrtoint %87 : !llvm.ptr<3> to i32
    %132 = nvg.warp_id
    %133 = llvm.and %132, %20 : i32
    %134 = llvm.shl %133, %19 : i32
    %135 = llvm.add %131, %134 : i32
    %136 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15}, [$16 + 0], 16;", "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r" %135 : (i32) -> !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    %137 = llvm.extractvalue %136[0] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %138 = llvm.bitcast %137 : i32 to f32
    %139 = llvm.extractvalue %136[1] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %140 = llvm.bitcast %139 : i32 to f32
    %141 = llvm.extractvalue %136[2] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %142 = llvm.bitcast %141 : i32 to f32
    %143 = llvm.extractvalue %136[3] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %144 = llvm.bitcast %143 : i32 to f32
    %145 = llvm.extractvalue %136[4] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %146 = llvm.bitcast %145 : i32 to f32
    %147 = llvm.extractvalue %136[5] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %148 = llvm.bitcast %147 : i32 to f32
    %149 = llvm.extractvalue %136[6] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %150 = llvm.bitcast %149 : i32 to f32
    %151 = llvm.extractvalue %136[7] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %152 = llvm.bitcast %151 : i32 to f32
    %153 = llvm.extractvalue %136[8] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %154 = llvm.bitcast %153 : i32 to f32
    %155 = llvm.extractvalue %136[9] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %156 = llvm.bitcast %155 : i32 to f32
    %157 = llvm.extractvalue %136[10] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %158 = llvm.bitcast %157 : i32 to f32
    %159 = llvm.extractvalue %136[11] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %160 = llvm.bitcast %159 : i32 to f32
    %161 = llvm.extractvalue %136[12] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %162 = llvm.bitcast %161 : i32 to f32
    %163 = llvm.extractvalue %136[13] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %164 = llvm.bitcast %163 : i32 to f32
    %165 = llvm.extractvalue %136[14] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %166 = llvm.bitcast %165 : i32 to f32
    %167 = llvm.extractvalue %136[15] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %168 = llvm.bitcast %167 : i32 to f32
    nvvm.tcgen05.wait <load>
    %169 = nvvm.read.ptx.sreg.tid.x : i32
    %170 = llvm.and %169, %22 : i32
    %171 = llvm.urem %170, %21 : i32
    %172 = llvm.udiv %170, %21 : i32
    %173 = llvm.shl %171, %24 : i32
    %174 = llvm.or %24, %173 : i32
    %175 = llvm.shl %172, %11 : i32
    %176 = llvm.or %174, %175 : i32
    %177 = llvm.and %176, %10 : i32
    %178 = llvm.shl %177, %9 : i32
    %179 = llvm.and %176, %8 : i32
    %180 = llvm.shl %179, %7 : i32
    %181 = llvm.and %176, %9 : i32
    %182 = llvm.shl %181, %18 : i32
    %183 = llvm.and %176, %6 : i32
    %184 = llvm.icmp "eq" %183, %24 : i32
    %185 = llvm.select %184, %24, %5 : i1, i32
    %186 = llvm.xor %180, %182 : i32
    %187 = llvm.xor %186, %185 : i32
    %188 = llvm.or disjoint %178, %187 : i32
    %189 = llvm.xor %24, %188 : i32
    %190 = llvm.mul %24, %18 : i32
    %191 = llvm.xor %189, %190 : i32
    %192 = llvm.xor %191, %24 : i32
    %193 = llvm.add %192, %24 : i32
    %194 = llvm.getelementptr inbounds %23[%193] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %195 = llvm.insertelement %138, %4[%24 : i32] : vector<4xf32>
    %196 = llvm.insertelement %140, %195[%3 : i32] : vector<4xf32>
    %197 = llvm.insertelement %142, %196[%2 : i32] : vector<4xf32>
    %198 = llvm.insertelement %144, %197[%20 : i32] : vector<4xf32>
    %199 = llvm.extractelement %198[%24 : i32] : vector<4xf32>
    %200 = llvm.extractelement %198[%3 : i32] : vector<4xf32>
    %201 = llvm.extractelement %198[%2 : i32] : vector<4xf32>
    %202 = llvm.extractelement %198[%20 : i32] : vector<4xf32>
    %203 = llvm.bitcast %199 : f32 to i32
    %204 = llvm.bitcast %200 : f32 to i32
    %205 = llvm.bitcast %201 : f32 to i32
    %206 = llvm.bitcast %202 : f32 to i32
    %207 = llvm.insertelement %203, %1[%24 : i32] : vector<4xi32>
    %208 = llvm.insertelement %204, %207[%3 : i32] : vector<4xi32>
    %209 = llvm.insertelement %205, %208[%2 : i32] : vector<4xi32>
    %210 = llvm.insertelement %206, %209[%20 : i32] : vector<4xi32>
    llvm.store %210, %194 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %211 = llvm.xor %191, %6 : i32
    %212 = llvm.add %211, %24 : i32
    %213 = llvm.getelementptr inbounds %23[%212] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %214 = llvm.insertelement %146, %4[%24 : i32] : vector<4xf32>
    %215 = llvm.insertelement %148, %214[%3 : i32] : vector<4xf32>
    %216 = llvm.insertelement %150, %215[%2 : i32] : vector<4xf32>
    %217 = llvm.insertelement %152, %216[%20 : i32] : vector<4xf32>
    %218 = llvm.extractelement %217[%24 : i32] : vector<4xf32>
    %219 = llvm.extractelement %217[%3 : i32] : vector<4xf32>
    %220 = llvm.extractelement %217[%2 : i32] : vector<4xf32>
    %221 = llvm.extractelement %217[%20 : i32] : vector<4xf32>
    %222 = llvm.bitcast %218 : f32 to i32
    %223 = llvm.bitcast %219 : f32 to i32
    %224 = llvm.bitcast %220 : f32 to i32
    %225 = llvm.bitcast %221 : f32 to i32
    %226 = llvm.insertelement %222, %1[%24 : i32] : vector<4xi32>
    %227 = llvm.insertelement %223, %226[%3 : i32] : vector<4xi32>
    %228 = llvm.insertelement %224, %227[%2 : i32] : vector<4xi32>
    %229 = llvm.insertelement %225, %228[%20 : i32] : vector<4xi32>
    llvm.store %229, %213 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %230 = llvm.xor %191, %21 : i32
    %231 = llvm.add %230, %24 : i32
    %232 = llvm.getelementptr inbounds %23[%231] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %233 = llvm.insertelement %154, %4[%24 : i32] : vector<4xf32>
    %234 = llvm.insertelement %156, %233[%3 : i32] : vector<4xf32>
    %235 = llvm.insertelement %158, %234[%2 : i32] : vector<4xf32>
    %236 = llvm.insertelement %160, %235[%20 : i32] : vector<4xf32>
    %237 = llvm.extractelement %236[%24 : i32] : vector<4xf32>
    %238 = llvm.extractelement %236[%3 : i32] : vector<4xf32>
    %239 = llvm.extractelement %236[%2 : i32] : vector<4xf32>
    %240 = llvm.extractelement %236[%20 : i32] : vector<4xf32>
    %241 = llvm.bitcast %237 : f32 to i32
    %242 = llvm.bitcast %238 : f32 to i32
    %243 = llvm.bitcast %239 : f32 to i32
    %244 = llvm.bitcast %240 : f32 to i32
    %245 = llvm.insertelement %241, %1[%24 : i32] : vector<4xi32>
    %246 = llvm.insertelement %242, %245[%3 : i32] : vector<4xi32>
    %247 = llvm.insertelement %243, %246[%2 : i32] : vector<4xi32>
    %248 = llvm.insertelement %244, %247[%20 : i32] : vector<4xi32>
    llvm.store %248, %232 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %249 = llvm.xor %191, %0 : i32
    %250 = llvm.add %249, %24 : i32
    %251 = llvm.getelementptr inbounds %23[%250] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %252 = llvm.insertelement %162, %4[%24 : i32] : vector<4xf32>
    %253 = llvm.insertelement %164, %252[%3 : i32] : vector<4xf32>
    %254 = llvm.insertelement %166, %253[%2 : i32] : vector<4xf32>
    %255 = llvm.insertelement %168, %254[%20 : i32] : vector<4xf32>
    %256 = llvm.extractelement %255[%24 : i32] : vector<4xf32>
    %257 = llvm.extractelement %255[%3 : i32] : vector<4xf32>
    %258 = llvm.extractelement %255[%2 : i32] : vector<4xf32>
    %259 = llvm.extractelement %255[%20 : i32] : vector<4xf32>
    %260 = llvm.bitcast %256 : f32 to i32
    %261 = llvm.bitcast %257 : f32 to i32
    %262 = llvm.bitcast %258 : f32 to i32
    %263 = llvm.bitcast %259 : f32 to i32
    %264 = llvm.insertelement %260, %1[%24 : i32] : vector<4xi32>
    %265 = llvm.insertelement %261, %264[%3 : i32] : vector<4xi32>
    %266 = llvm.insertelement %262, %265[%2 : i32] : vector<4xi32>
    %267 = llvm.insertelement %263, %266[%20 : i32] : vector<4xi32>
    llvm.store %267, %251 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
    nvvm.barrier0
    %268 = nvvm.read.ptx.sreg.tid.x : i32
    %269 = llvm.and %268, %22 : i32
    %270 = nvvm.elect.sync -> i1
    %271 = llvm.icmp "ult" %269, %21 : i32
    %272 = llvm.and %270, %271 : i1
    %273 = llvm.xor %24, %24 : i32
    %274 = llvm.getelementptr %23[%273] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %275 = llvm.xor %24, %24 : i32
    %276 = llvm.xor %24, %24 : i32
    %277 = llvm.add %276, %24 : i32
    %278 = llvm.add %275, %24 : i32
    %279 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$1, {$2, $3}], [$4];", "b,l,r,r,r" %272, %arg10, %277, %278, %274 : (i1, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.cp.async.bulk.commit.group
    nvvm.cp.async.bulk.wait_group 0 {read}
    nvvm.barrier0
    llvm.return
  }
}


// -----// IR Dump After CSE (cse) //----- //
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 12296 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 32 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr<1>, %arg16: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %0 = llvm.mlir.constant(48 : i32) : i32
    %1 = llvm.mlir.undef : vector<4xi32>
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.undef : vector<4xf32>
    %5 = llvm.mlir.constant(64 : i32) : i32
    %6 = llvm.mlir.constant(16 : i32) : i32
    %7 = llvm.mlir.constant(6 : i32) : i32
    %8 = llvm.mlir.constant(96 : i32) : i32
    %9 = llvm.mlir.constant(7 : i32) : i32
    %10 = llvm.mlir.constant(15 : i32) : i32
    %11 = llvm.mlir.constant(5 : i32) : i32
    %12 = llvm.mlir.constant(4611756662049472518 : i64) : i64
    %13 = llvm.mlir.constant(4611756662049472516 : i64) : i64
    %14 = llvm.mlir.constant(4611756662049472514 : i64) : i64
    %15 = llvm.mlir.constant(67633168 : i32) : i32
    %16 = llvm.mlir.constant(4611756662049472512 : i64) : i64
    %17 = llvm.mlir.constant(16383 : i32) : i32
    %18 = llvm.mlir.constant(4 : i32) : i32
    %19 = llvm.mlir.constant(21 : i32) : i32
    %20 = llvm.mlir.constant(3 : i32) : i32
    %21 = llvm.mlir.constant(32 : i32) : i32
    %22 = llvm.mlir.constant(127 : i32) : i32
    %23 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %24 = llvm.mlir.constant(0 : i32) : i32
    %25 = llvm.mlir.constant(true) : i1
    %26 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %27 = llvm.getelementptr %23[8192] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %28 = nvvm.read.ptx.sreg.tid.x : i32
    %29 = llvm.and %28, %22 : i32
    %30 = llvm.icmp "eq" %29, %24 : i32
    %31 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %30, %27 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %32 = llvm.and %30, %25 : i1
    %33 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 8192;", "b,r" %32, %27 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %34 = nvvm.elect.sync -> i1
    %35 = llvm.and %25, %34 : i1
    %36 = llvm.icmp "ult" %29, %21 : i32
    %37 = llvm.and %35, %36 : i1
    %38 = llvm.xor %24, %24 : i32
    %39 = llvm.getelementptr %23[%38] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %40 = llvm.add %38, %24 : i32
    %41 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %37, %39, %arg0, %40, %40, %27 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %42 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %27, %24 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %43 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %30, %27 : (i1, !llvm.ptr<3>) -> !llvm.void
    %44 = llvm.getelementptr %23[12288] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %45 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %30, %44 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %46 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 4096;", "b,r" %32, %44 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
    nvvm.barrier0
    %47 = nvvm.elect.sync -> i1
    %48 = llvm.and %25, %47 : i1
    %49 = llvm.and %48, %36 : i1
    %50 = llvm.getelementptr %27[%38] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %51 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %49, %50, %arg5, %40, %40, %44 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %52 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %44, %24 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %53 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %30, %44 : (i1, !llvm.ptr<3>) -> !llvm.void
    %54 = nvg.tensor_memory_base
    %55 = llvm.ptrtoint %54 : !llvm.ptr<6> to i32
    %56 = llvm.add %55, %24 : i32
    %57 = llvm.inttoptr %56 : i32 to !llvm.ptr<3>
    %58 = llvm.ptrtoint %57 : !llvm.ptr<3> to i32
    %59 = nvg.warp_id
    %60 = llvm.and %59, %20 : i32
    %61 = llvm.shl %60, %19 : i32
    %62 = llvm.add %58, %61 : i32
    %63 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.st.sync.aligned.16x32bx2.x16.b32 [$1 + 0], 16, {$2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17};", "b,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r" %25, %62, %26, %26, %26, %26, %26, %26, %26, %26, %26, %26, %26, %26, %26, %26, %26, %26 : (i1, i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.void
    nvvm.tcgen05.wait <store>
    nvvm.barrier0
    nvvm.barrier0
    %64 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %30, %44 : (i1, !llvm.ptr<3>) -> !llvm.void
    %65 = llvm.icmp "eq" %59, %24 : i32
    %66 = llvm.and %25, %65 : i1
    llvm.cond_br %66, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %67 = nvvm.elect.sync -> i1
    %68 = llvm.ptrtoint %23 : !llvm.ptr<3> to i32
    %69 = llvm.lshr %68, %18 : i32
    %70 = llvm.and %69, %17 : i32
    %71 = llvm.zext %70 : i32 to i64
    %72 = llvm.ptrtoint %27 : !llvm.ptr<3> to i32
    %73 = llvm.lshr %72, %18 : i32
    %74 = llvm.and %73, %17 : i32
    %75 = llvm.zext %74 : i32 to i64
    %76 = llvm.add %71, %16 : i64
    %77 = llvm.add %75, %16 : i64
    %78 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %58, %76, %77, %15, %25, %67 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %79 = llvm.add %71, %14 : i64
    %80 = llvm.add %75, %14 : i64
    %81 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %58, %79, %80, %15, %25, %67 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %82 = llvm.add %71, %13 : i64
    %83 = llvm.add %75, %13 : i64
    %84 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %58, %82, %83, %15, %25, %67 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %85 = llvm.add %71, %12 : i64
    %86 = llvm.add %75, %12 : i64
    %87 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %58, %85, %86, %15, %25, %67 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %88 = llvm.and %25, %67 : i1
    %89 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$1];", "b,l" %88, %44 : (i1, !llvm.ptr<3>) -> !llvm.void
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    nvvm.barrier0
    %90 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %44, %24 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %91 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %30, %44 : (i1, !llvm.ptr<3>) -> !llvm.void
    %92 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15}, [$16 + 0], 16;", "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r" %62 : (i32) -> !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    %93 = llvm.extractvalue %92[0] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %94 = llvm.bitcast %93 : i32 to f32
    %95 = llvm.extractvalue %92[1] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %96 = llvm.bitcast %95 : i32 to f32
    %97 = llvm.extractvalue %92[2] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %98 = llvm.bitcast %97 : i32 to f32
    %99 = llvm.extractvalue %92[3] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %100 = llvm.bitcast %99 : i32 to f32
    %101 = llvm.extractvalue %92[4] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %102 = llvm.bitcast %101 : i32 to f32
    %103 = llvm.extractvalue %92[5] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %104 = llvm.bitcast %103 : i32 to f32
    %105 = llvm.extractvalue %92[6] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %106 = llvm.bitcast %105 : i32 to f32
    %107 = llvm.extractvalue %92[7] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %108 = llvm.bitcast %107 : i32 to f32
    %109 = llvm.extractvalue %92[8] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %110 = llvm.bitcast %109 : i32 to f32
    %111 = llvm.extractvalue %92[9] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %112 = llvm.bitcast %111 : i32 to f32
    %113 = llvm.extractvalue %92[10] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %114 = llvm.bitcast %113 : i32 to f32
    %115 = llvm.extractvalue %92[11] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %116 = llvm.bitcast %115 : i32 to f32
    %117 = llvm.extractvalue %92[12] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %118 = llvm.bitcast %117 : i32 to f32
    %119 = llvm.extractvalue %92[13] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %120 = llvm.bitcast %119 : i32 to f32
    %121 = llvm.extractvalue %92[14] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %122 = llvm.bitcast %121 : i32 to f32
    %123 = llvm.extractvalue %92[15] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %124 = llvm.bitcast %123 : i32 to f32
    nvvm.tcgen05.wait <load>
    %125 = llvm.urem %29, %21 : i32
    %126 = llvm.udiv %29, %21 : i32
    %127 = llvm.shl %125, %24 : i32
    %128 = llvm.or %24, %127 : i32
    %129 = llvm.shl %126, %11 : i32
    %130 = llvm.or %128, %129 : i32
    %131 = llvm.and %130, %10 : i32
    %132 = llvm.shl %131, %9 : i32
    %133 = llvm.and %130, %8 : i32
    %134 = llvm.shl %133, %7 : i32
    %135 = llvm.and %130, %9 : i32
    %136 = llvm.shl %135, %18 : i32
    %137 = llvm.and %130, %6 : i32
    %138 = llvm.icmp "eq" %137, %24 : i32
    %139 = llvm.select %138, %24, %5 : i1, i32
    %140 = llvm.xor %134, %136 : i32
    %141 = llvm.xor %140, %139 : i32
    %142 = llvm.or disjoint %132, %141 : i32
    %143 = llvm.xor %24, %142 : i32
    %144 = llvm.mul %24, %18 : i32
    %145 = llvm.xor %143, %144 : i32
    %146 = llvm.xor %145, %24 : i32
    %147 = llvm.add %146, %24 : i32
    %148 = llvm.getelementptr inbounds %23[%147] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %149 = llvm.insertelement %94, %4[%24 : i32] : vector<4xf32>
    %150 = llvm.insertelement %96, %149[%3 : i32] : vector<4xf32>
    %151 = llvm.insertelement %98, %150[%2 : i32] : vector<4xf32>
    %152 = llvm.insertelement %100, %151[%20 : i32] : vector<4xf32>
    %153 = llvm.extractelement %152[%24 : i32] : vector<4xf32>
    %154 = llvm.extractelement %152[%3 : i32] : vector<4xf32>
    %155 = llvm.extractelement %152[%2 : i32] : vector<4xf32>
    %156 = llvm.extractelement %152[%20 : i32] : vector<4xf32>
    %157 = llvm.bitcast %153 : f32 to i32
    %158 = llvm.bitcast %154 : f32 to i32
    %159 = llvm.bitcast %155 : f32 to i32
    %160 = llvm.bitcast %156 : f32 to i32
    %161 = llvm.insertelement %157, %1[%24 : i32] : vector<4xi32>
    %162 = llvm.insertelement %158, %161[%3 : i32] : vector<4xi32>
    %163 = llvm.insertelement %159, %162[%2 : i32] : vector<4xi32>
    %164 = llvm.insertelement %160, %163[%20 : i32] : vector<4xi32>
    llvm.store %164, %148 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %165 = llvm.xor %145, %6 : i32
    %166 = llvm.add %165, %24 : i32
    %167 = llvm.getelementptr inbounds %23[%166] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %168 = llvm.insertelement %102, %4[%24 : i32] : vector<4xf32>
    %169 = llvm.insertelement %104, %168[%3 : i32] : vector<4xf32>
    %170 = llvm.insertelement %106, %169[%2 : i32] : vector<4xf32>
    %171 = llvm.insertelement %108, %170[%20 : i32] : vector<4xf32>
    %172 = llvm.extractelement %171[%24 : i32] : vector<4xf32>
    %173 = llvm.extractelement %171[%3 : i32] : vector<4xf32>
    %174 = llvm.extractelement %171[%2 : i32] : vector<4xf32>
    %175 = llvm.extractelement %171[%20 : i32] : vector<4xf32>
    %176 = llvm.bitcast %172 : f32 to i32
    %177 = llvm.bitcast %173 : f32 to i32
    %178 = llvm.bitcast %174 : f32 to i32
    %179 = llvm.bitcast %175 : f32 to i32
    %180 = llvm.insertelement %176, %1[%24 : i32] : vector<4xi32>
    %181 = llvm.insertelement %177, %180[%3 : i32] : vector<4xi32>
    %182 = llvm.insertelement %178, %181[%2 : i32] : vector<4xi32>
    %183 = llvm.insertelement %179, %182[%20 : i32] : vector<4xi32>
    llvm.store %183, %167 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %184 = llvm.xor %145, %21 : i32
    %185 = llvm.add %184, %24 : i32
    %186 = llvm.getelementptr inbounds %23[%185] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %187 = llvm.insertelement %110, %4[%24 : i32] : vector<4xf32>
    %188 = llvm.insertelement %112, %187[%3 : i32] : vector<4xf32>
    %189 = llvm.insertelement %114, %188[%2 : i32] : vector<4xf32>
    %190 = llvm.insertelement %116, %189[%20 : i32] : vector<4xf32>
    %191 = llvm.extractelement %190[%24 : i32] : vector<4xf32>
    %192 = llvm.extractelement %190[%3 : i32] : vector<4xf32>
    %193 = llvm.extractelement %190[%2 : i32] : vector<4xf32>
    %194 = llvm.extractelement %190[%20 : i32] : vector<4xf32>
    %195 = llvm.bitcast %191 : f32 to i32
    %196 = llvm.bitcast %192 : f32 to i32
    %197 = llvm.bitcast %193 : f32 to i32
    %198 = llvm.bitcast %194 : f32 to i32
    %199 = llvm.insertelement %195, %1[%24 : i32] : vector<4xi32>
    %200 = llvm.insertelement %196, %199[%3 : i32] : vector<4xi32>
    %201 = llvm.insertelement %197, %200[%2 : i32] : vector<4xi32>
    %202 = llvm.insertelement %198, %201[%20 : i32] : vector<4xi32>
    llvm.store %202, %186 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %203 = llvm.xor %145, %0 : i32
    %204 = llvm.add %203, %24 : i32
    %205 = llvm.getelementptr inbounds %23[%204] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %206 = llvm.insertelement %118, %4[%24 : i32] : vector<4xf32>
    %207 = llvm.insertelement %120, %206[%3 : i32] : vector<4xf32>
    %208 = llvm.insertelement %122, %207[%2 : i32] : vector<4xf32>
    %209 = llvm.insertelement %124, %208[%20 : i32] : vector<4xf32>
    %210 = llvm.extractelement %209[%24 : i32] : vector<4xf32>
    %211 = llvm.extractelement %209[%3 : i32] : vector<4xf32>
    %212 = llvm.extractelement %209[%2 : i32] : vector<4xf32>
    %213 = llvm.extractelement %209[%20 : i32] : vector<4xf32>
    %214 = llvm.bitcast %210 : f32 to i32
    %215 = llvm.bitcast %211 : f32 to i32
    %216 = llvm.bitcast %212 : f32 to i32
    %217 = llvm.bitcast %213 : f32 to i32
    %218 = llvm.insertelement %214, %1[%24 : i32] : vector<4xi32>
    %219 = llvm.insertelement %215, %218[%3 : i32] : vector<4xi32>
    %220 = llvm.insertelement %216, %219[%2 : i32] : vector<4xi32>
    %221 = llvm.insertelement %217, %220[%20 : i32] : vector<4xi32>
    llvm.store %221, %205 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
    nvvm.barrier0
    %222 = nvvm.elect.sync -> i1
    %223 = llvm.and %222, %36 : i1
    %224 = llvm.getelementptr %23[%38] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %225 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$1, {$2, $3}], [$4];", "b,l,r,r,r" %223, %arg10, %40, %40, %224 : (i1, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.cp.async.bulk.commit.group
    nvvm.cp.async.bulk.wait_group 0 {read}
    nvvm.barrier0
    llvm.return
  }
}


// -----// IR Dump After ConvertNVGPUToLLVM (convert-nv-gpu-to-llvm) //----- //
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 12296 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 32 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr<1>, %arg16: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    %1 = llvm.mlir.constant(32 : i32) : i32
    %2 = llvm.icmp "ult" %0, %1 : i32
    %3 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %4 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], 32;", "b,r" %2, %3 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %5 = llvm.load %3 : !llvm.ptr<3> -> i32
    nvvm.barrier0
    %6 = llvm.inttoptr %5 : i32 to !llvm.ptr<6>
    %7 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", "b" %2 : (i1) -> !llvm.void
    %8 = llvm.mlir.constant(-1 : i32) : i32
    %9 = llvm.mlir.constant(31 : i32) : i32
    %10 = llvm.mlir.constant(48 : i32) : i32
    %11 = llvm.mlir.undef : vector<4xi32>
    %12 = llvm.mlir.constant(2 : i32) : i32
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.mlir.undef : vector<4xf32>
    %15 = llvm.mlir.constant(64 : i32) : i32
    %16 = llvm.mlir.constant(16 : i32) : i32
    %17 = llvm.mlir.constant(6 : i32) : i32
    %18 = llvm.mlir.constant(96 : i32) : i32
    %19 = llvm.mlir.constant(7 : i32) : i32
    %20 = llvm.mlir.constant(15 : i32) : i32
    %21 = llvm.mlir.constant(5 : i32) : i32
    %22 = llvm.mlir.constant(4611756662049472518 : i64) : i64
    %23 = llvm.mlir.constant(4611756662049472516 : i64) : i64
    %24 = llvm.mlir.constant(4611756662049472514 : i64) : i64
    %25 = llvm.mlir.constant(67633168 : i32) : i32
    %26 = llvm.mlir.constant(4611756662049472512 : i64) : i64
    %27 = llvm.mlir.constant(16383 : i32) : i32
    %28 = llvm.mlir.constant(4 : i32) : i32
    %29 = llvm.mlir.constant(21 : i32) : i32
    %30 = llvm.mlir.constant(3 : i32) : i32
    %31 = llvm.mlir.constant(32 : i32) : i32
    %32 = llvm.mlir.constant(127 : i32) : i32
    %33 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %34 = llvm.mlir.constant(0 : i32) : i32
    %35 = llvm.mlir.constant(true) : i1
    %36 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %37 = llvm.getelementptr %33[8192] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %38 = nvvm.read.ptx.sreg.tid.x : i32
    %39 = llvm.and %38, %32 : i32
    %40 = llvm.icmp "eq" %39, %34 : i32
    %41 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %40, %37 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %42 = llvm.and %40, %35 : i1
    %43 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 8192;", "b,r" %42, %37 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %44 = nvvm.elect.sync -> i1
    %45 = llvm.and %35, %44 : i1
    %46 = llvm.icmp "ult" %39, %31 : i32
    %47 = llvm.and %45, %46 : i1
    %48 = llvm.xor %34, %34 : i32
    %49 = llvm.getelementptr %33[%48] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %50 = llvm.add %48, %34 : i32
    %51 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %47, %49, %arg0, %50, %50, %37 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %52 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %37, %34 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %53 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %40, %37 : (i1, !llvm.ptr<3>) -> !llvm.void
    %54 = llvm.getelementptr %33[12288] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %55 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %40, %54 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %56 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 4096;", "b,r" %42, %54 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
    nvvm.barrier0
    %57 = nvvm.elect.sync -> i1
    %58 = llvm.and %35, %57 : i1
    %59 = llvm.and %58, %46 : i1
    %60 = llvm.getelementptr %37[%48] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %61 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %59, %60, %arg5, %50, %50, %54 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %62 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %54, %34 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %63 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %40, %54 : (i1, !llvm.ptr<3>) -> !llvm.void
    %64 = llvm.ptrtoint %6 : !llvm.ptr<6> to i32
    %65 = llvm.add %64, %34 : i32
    %66 = llvm.inttoptr %65 : i32 to !llvm.ptr<3>
    %67 = llvm.ptrtoint %66 : !llvm.ptr<3> to i32
    %68 = nvvm.read.ptx.sreg.tid.x : i32
    %69 = llvm.udiv %68, %31 : i32
    %70 = nvvm.shfl.sync  idx %8, %69, %34, %9 : i32 -> i32
    %71 = llvm.and %70, %30 : i32
    %72 = llvm.shl %71, %29 : i32
    %73 = llvm.add %67, %72 : i32
    %74 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.st.sync.aligned.16x32bx2.x16.b32 [$1 + 0], 16, {$2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17};", "b,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r" %35, %73, %36, %36, %36, %36, %36, %36, %36, %36, %36, %36, %36, %36, %36, %36, %36, %36 : (i1, i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.void
    nvvm.tcgen05.wait <store>
    nvvm.barrier0
    nvvm.barrier0
    %75 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %40, %54 : (i1, !llvm.ptr<3>) -> !llvm.void
    %76 = llvm.icmp "eq" %70, %34 : i32
    %77 = llvm.and %35, %76 : i1
    llvm.cond_br %77, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %78 = nvvm.elect.sync -> i1
    %79 = llvm.ptrtoint %33 : !llvm.ptr<3> to i32
    %80 = llvm.lshr %79, %28 : i32
    %81 = llvm.and %80, %27 : i32
    %82 = llvm.zext %81 : i32 to i64
    %83 = llvm.ptrtoint %37 : !llvm.ptr<3> to i32
    %84 = llvm.lshr %83, %28 : i32
    %85 = llvm.and %84, %27 : i32
    %86 = llvm.zext %85 : i32 to i64
    %87 = llvm.add %82, %26 : i64
    %88 = llvm.add %86, %26 : i64
    %89 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %67, %87, %88, %25, %35, %78 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %90 = llvm.add %82, %24 : i64
    %91 = llvm.add %86, %24 : i64
    %92 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %67, %90, %91, %25, %35, %78 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %93 = llvm.add %82, %23 : i64
    %94 = llvm.add %86, %23 : i64
    %95 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %67, %93, %94, %25, %35, %78 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %96 = llvm.add %82, %22 : i64
    %97 = llvm.add %86, %22 : i64
    %98 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %67, %96, %97, %25, %35, %78 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %99 = llvm.and %35, %78 : i1
    %100 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$1];", "b,l" %99, %54 : (i1, !llvm.ptr<3>) -> !llvm.void
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    nvvm.barrier0
    %101 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %54, %34 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %102 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %40, %54 : (i1, !llvm.ptr<3>) -> !llvm.void
    %103 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15}, [$16 + 0], 16;", "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r" %73 : (i32) -> !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    %104 = llvm.extractvalue %103[0] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %105 = llvm.bitcast %104 : i32 to f32
    %106 = llvm.extractvalue %103[1] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %107 = llvm.bitcast %106 : i32 to f32
    %108 = llvm.extractvalue %103[2] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %109 = llvm.bitcast %108 : i32 to f32
    %110 = llvm.extractvalue %103[3] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %111 = llvm.bitcast %110 : i32 to f32
    %112 = llvm.extractvalue %103[4] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %113 = llvm.bitcast %112 : i32 to f32
    %114 = llvm.extractvalue %103[5] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %115 = llvm.bitcast %114 : i32 to f32
    %116 = llvm.extractvalue %103[6] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %117 = llvm.bitcast %116 : i32 to f32
    %118 = llvm.extractvalue %103[7] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %119 = llvm.bitcast %118 : i32 to f32
    %120 = llvm.extractvalue %103[8] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %121 = llvm.bitcast %120 : i32 to f32
    %122 = llvm.extractvalue %103[9] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %123 = llvm.bitcast %122 : i32 to f32
    %124 = llvm.extractvalue %103[10] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %125 = llvm.bitcast %124 : i32 to f32
    %126 = llvm.extractvalue %103[11] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %127 = llvm.bitcast %126 : i32 to f32
    %128 = llvm.extractvalue %103[12] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %129 = llvm.bitcast %128 : i32 to f32
    %130 = llvm.extractvalue %103[13] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %131 = llvm.bitcast %130 : i32 to f32
    %132 = llvm.extractvalue %103[14] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %133 = llvm.bitcast %132 : i32 to f32
    %134 = llvm.extractvalue %103[15] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %135 = llvm.bitcast %134 : i32 to f32
    nvvm.tcgen05.wait <load>
    %136 = llvm.urem %39, %31 : i32
    %137 = llvm.udiv %39, %31 : i32
    %138 = llvm.shl %136, %34 : i32
    %139 = llvm.or %34, %138 : i32
    %140 = llvm.shl %137, %21 : i32
    %141 = llvm.or %139, %140 : i32
    %142 = llvm.and %141, %20 : i32
    %143 = llvm.shl %142, %19 : i32
    %144 = llvm.and %141, %18 : i32
    %145 = llvm.shl %144, %17 : i32
    %146 = llvm.and %141, %19 : i32
    %147 = llvm.shl %146, %28 : i32
    %148 = llvm.and %141, %16 : i32
    %149 = llvm.icmp "eq" %148, %34 : i32
    %150 = llvm.select %149, %34, %15 : i1, i32
    %151 = llvm.xor %145, %147 : i32
    %152 = llvm.xor %151, %150 : i32
    %153 = llvm.or disjoint %143, %152 : i32
    %154 = llvm.xor %34, %153 : i32
    %155 = llvm.mul %34, %28 : i32
    %156 = llvm.xor %154, %155 : i32
    %157 = llvm.xor %156, %34 : i32
    %158 = llvm.add %157, %34 : i32
    %159 = llvm.getelementptr inbounds %33[%158] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %160 = llvm.insertelement %105, %14[%34 : i32] : vector<4xf32>
    %161 = llvm.insertelement %107, %160[%13 : i32] : vector<4xf32>
    %162 = llvm.insertelement %109, %161[%12 : i32] : vector<4xf32>
    %163 = llvm.insertelement %111, %162[%30 : i32] : vector<4xf32>
    %164 = llvm.extractelement %163[%34 : i32] : vector<4xf32>
    %165 = llvm.extractelement %163[%13 : i32] : vector<4xf32>
    %166 = llvm.extractelement %163[%12 : i32] : vector<4xf32>
    %167 = llvm.extractelement %163[%30 : i32] : vector<4xf32>
    %168 = llvm.bitcast %164 : f32 to i32
    %169 = llvm.bitcast %165 : f32 to i32
    %170 = llvm.bitcast %166 : f32 to i32
    %171 = llvm.bitcast %167 : f32 to i32
    %172 = llvm.insertelement %168, %11[%34 : i32] : vector<4xi32>
    %173 = llvm.insertelement %169, %172[%13 : i32] : vector<4xi32>
    %174 = llvm.insertelement %170, %173[%12 : i32] : vector<4xi32>
    %175 = llvm.insertelement %171, %174[%30 : i32] : vector<4xi32>
    llvm.store %175, %159 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %176 = llvm.xor %156, %16 : i32
    %177 = llvm.add %176, %34 : i32
    %178 = llvm.getelementptr inbounds %33[%177] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %179 = llvm.insertelement %113, %14[%34 : i32] : vector<4xf32>
    %180 = llvm.insertelement %115, %179[%13 : i32] : vector<4xf32>
    %181 = llvm.insertelement %117, %180[%12 : i32] : vector<4xf32>
    %182 = llvm.insertelement %119, %181[%30 : i32] : vector<4xf32>
    %183 = llvm.extractelement %182[%34 : i32] : vector<4xf32>
    %184 = llvm.extractelement %182[%13 : i32] : vector<4xf32>
    %185 = llvm.extractelement %182[%12 : i32] : vector<4xf32>
    %186 = llvm.extractelement %182[%30 : i32] : vector<4xf32>
    %187 = llvm.bitcast %183 : f32 to i32
    %188 = llvm.bitcast %184 : f32 to i32
    %189 = llvm.bitcast %185 : f32 to i32
    %190 = llvm.bitcast %186 : f32 to i32
    %191 = llvm.insertelement %187, %11[%34 : i32] : vector<4xi32>
    %192 = llvm.insertelement %188, %191[%13 : i32] : vector<4xi32>
    %193 = llvm.insertelement %189, %192[%12 : i32] : vector<4xi32>
    %194 = llvm.insertelement %190, %193[%30 : i32] : vector<4xi32>
    llvm.store %194, %178 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %195 = llvm.xor %156, %31 : i32
    %196 = llvm.add %195, %34 : i32
    %197 = llvm.getelementptr inbounds %33[%196] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %198 = llvm.insertelement %121, %14[%34 : i32] : vector<4xf32>
    %199 = llvm.insertelement %123, %198[%13 : i32] : vector<4xf32>
    %200 = llvm.insertelement %125, %199[%12 : i32] : vector<4xf32>
    %201 = llvm.insertelement %127, %200[%30 : i32] : vector<4xf32>
    %202 = llvm.extractelement %201[%34 : i32] : vector<4xf32>
    %203 = llvm.extractelement %201[%13 : i32] : vector<4xf32>
    %204 = llvm.extractelement %201[%12 : i32] : vector<4xf32>
    %205 = llvm.extractelement %201[%30 : i32] : vector<4xf32>
    %206 = llvm.bitcast %202 : f32 to i32
    %207 = llvm.bitcast %203 : f32 to i32
    %208 = llvm.bitcast %204 : f32 to i32
    %209 = llvm.bitcast %205 : f32 to i32
    %210 = llvm.insertelement %206, %11[%34 : i32] : vector<4xi32>
    %211 = llvm.insertelement %207, %210[%13 : i32] : vector<4xi32>
    %212 = llvm.insertelement %208, %211[%12 : i32] : vector<4xi32>
    %213 = llvm.insertelement %209, %212[%30 : i32] : vector<4xi32>
    llvm.store %213, %197 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %214 = llvm.xor %156, %10 : i32
    %215 = llvm.add %214, %34 : i32
    %216 = llvm.getelementptr inbounds %33[%215] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %217 = llvm.insertelement %129, %14[%34 : i32] : vector<4xf32>
    %218 = llvm.insertelement %131, %217[%13 : i32] : vector<4xf32>
    %219 = llvm.insertelement %133, %218[%12 : i32] : vector<4xf32>
    %220 = llvm.insertelement %135, %219[%30 : i32] : vector<4xf32>
    %221 = llvm.extractelement %220[%34 : i32] : vector<4xf32>
    %222 = llvm.extractelement %220[%13 : i32] : vector<4xf32>
    %223 = llvm.extractelement %220[%12 : i32] : vector<4xf32>
    %224 = llvm.extractelement %220[%30 : i32] : vector<4xf32>
    %225 = llvm.bitcast %221 : f32 to i32
    %226 = llvm.bitcast %222 : f32 to i32
    %227 = llvm.bitcast %223 : f32 to i32
    %228 = llvm.bitcast %224 : f32 to i32
    %229 = llvm.insertelement %225, %11[%34 : i32] : vector<4xi32>
    %230 = llvm.insertelement %226, %229[%13 : i32] : vector<4xi32>
    %231 = llvm.insertelement %227, %230[%12 : i32] : vector<4xi32>
    %232 = llvm.insertelement %228, %231[%30 : i32] : vector<4xi32>
    llvm.store %232, %216 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
    nvvm.barrier0
    %233 = nvvm.elect.sync -> i1
    %234 = llvm.and %233, %46 : i1
    %235 = llvm.getelementptr %33[%48] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %236 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$1, {$2, $3}], [$4];", "b,l,r,r,r" %234, %arg10, %50, %50, %235 : (i1, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.cp.async.bulk.commit.group
    nvvm.cp.async.bulk.wait_group 0 {read}
    nvvm.barrier0
    nvvm.barrier0
    %237 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.dealloc.cta_group::1.sync.aligned.b32 $1, 32;", "b,r" %2, %6 : (i1, !llvm.ptr<6>) -> !llvm.void
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 12296 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 32 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr<1>, %arg16: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(127 : i32) : i32
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(21 : i32) : i32
    %6 = llvm.mlir.constant(4 : i32) : i32
    %7 = llvm.mlir.constant(16383 : i32) : i32
    %8 = llvm.mlir.constant(4611756662049472512 : i64) : i64
    %9 = llvm.mlir.constant(67633168 : i32) : i32
    %10 = llvm.mlir.constant(4611756662049472514 : i64) : i64
    %11 = llvm.mlir.constant(4611756662049472516 : i64) : i64
    %12 = llvm.mlir.constant(4611756662049472518 : i64) : i64
    %13 = llvm.mlir.constant(5 : i32) : i32
    %14 = llvm.mlir.constant(15 : i32) : i32
    %15 = llvm.mlir.constant(7 : i32) : i32
    %16 = llvm.mlir.constant(96 : i32) : i32
    %17 = llvm.mlir.constant(6 : i32) : i32
    %18 = llvm.mlir.constant(16 : i32) : i32
    %19 = llvm.mlir.constant(64 : i32) : i32
    %20 = llvm.mlir.undef : vector<4xf32>
    %21 = llvm.mlir.constant(1 : i32) : i32
    %22 = llvm.mlir.constant(2 : i32) : i32
    %23 = llvm.mlir.undef : vector<4xi32>
    %24 = llvm.mlir.constant(48 : i32) : i32
    %25 = llvm.mlir.constant(31 : i32) : i32
    %26 = llvm.mlir.constant(-1 : i32) : i32
    %27 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %28 = llvm.mlir.constant(32 : i32) : i32
    %29 = nvvm.read.ptx.sreg.tid.x : i32
    %30 = llvm.icmp "ult" %29, %28 : i32
    %31 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], 32;", "b,r" %30, %27 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %32 = llvm.load %27 : !llvm.ptr<3> -> i32
    nvvm.barrier0
    %33 = llvm.inttoptr %32 : i32 to !llvm.ptr<6>
    %34 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", "b" %30 : (i1) -> !llvm.void
    %35 = llvm.getelementptr %27[8192] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %36 = nvvm.read.ptx.sreg.tid.x : i32
    %37 = llvm.and %36, %3 : i32
    %38 = llvm.icmp "eq" %37, %2 : i32
    %39 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %38, %35 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %40 = llvm.and %38, %1 : i1
    %41 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 8192;", "b,r" %40, %35 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %42 = nvvm.elect.sync -> i1
    %43 = llvm.and %1, %42 : i1
    %44 = llvm.icmp "ult" %37, %28 : i32
    %45 = llvm.and %43, %44 : i1
    %46 = llvm.xor %2, %2 : i32
    %47 = llvm.getelementptr %27[%46] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %48 = llvm.add %46, %2 : i32
    %49 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %45, %47, %arg0, %48, %48, %35 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %50 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %35, %2 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %51 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %38, %35 : (i1, !llvm.ptr<3>) -> !llvm.void
    %52 = llvm.getelementptr %27[12288] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %53 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %38, %52 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %54 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 4096;", "b,r" %40, %52 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
    nvvm.barrier0
    %55 = nvvm.elect.sync -> i1
    %56 = llvm.and %1, %55 : i1
    %57 = llvm.and %56, %44 : i1
    %58 = llvm.getelementptr %35[%46] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %59 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %57, %58, %arg5, %48, %48, %52 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %60 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %52, %2 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %61 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %38, %52 : (i1, !llvm.ptr<3>) -> !llvm.void
    %62 = llvm.ptrtoint %33 : !llvm.ptr<6> to i32
    %63 = llvm.add %62, %2 : i32
    %64 = llvm.inttoptr %63 : i32 to !llvm.ptr<3>
    %65 = llvm.ptrtoint %64 : !llvm.ptr<3> to i32
    %66 = nvvm.read.ptx.sreg.tid.x : i32
    %67 = llvm.udiv %66, %28 : i32
    %68 = nvvm.shfl.sync  idx %26, %67, %2, %25 : i32 -> i32
    %69 = llvm.and %68, %4 : i32
    %70 = llvm.shl %69, %5 : i32
    %71 = llvm.add %65, %70 : i32
    %72 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.st.sync.aligned.16x32bx2.x16.b32 [$1 + 0], 16, {$2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17};", "b,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r" %1, %71, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0 : (i1, i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.void
    nvvm.tcgen05.wait <store>
    nvvm.barrier0
    nvvm.barrier0
    %73 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %38, %52 : (i1, !llvm.ptr<3>) -> !llvm.void
    %74 = llvm.icmp "eq" %68, %2 : i32
    %75 = llvm.and %1, %74 : i1
    llvm.cond_br %75, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %76 = nvvm.elect.sync -> i1
    %77 = llvm.ptrtoint %27 : !llvm.ptr<3> to i32
    %78 = llvm.lshr %77, %6 : i32
    %79 = llvm.and %78, %7 : i32
    %80 = llvm.zext %79 : i32 to i64
    %81 = llvm.ptrtoint %35 : !llvm.ptr<3> to i32
    %82 = llvm.lshr %81, %6 : i32
    %83 = llvm.and %82, %7 : i32
    %84 = llvm.zext %83 : i32 to i64
    %85 = llvm.add %80, %8 : i64
    %86 = llvm.add %84, %8 : i64
    %87 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %65, %85, %86, %9, %1, %76 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %88 = llvm.add %80, %10 : i64
    %89 = llvm.add %84, %10 : i64
    %90 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %65, %88, %89, %9, %1, %76 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %91 = llvm.add %80, %11 : i64
    %92 = llvm.add %84, %11 : i64
    %93 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %65, %91, %92, %9, %1, %76 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %94 = llvm.add %80, %12 : i64
    %95 = llvm.add %84, %12 : i64
    %96 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %65, %94, %95, %9, %1, %76 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %97 = llvm.and %1, %76 : i1
    %98 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$1];", "b,l" %97, %52 : (i1, !llvm.ptr<3>) -> !llvm.void
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    nvvm.barrier0
    %99 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %52, %2 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %100 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %38, %52 : (i1, !llvm.ptr<3>) -> !llvm.void
    %101 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15}, [$16 + 0], 16;", "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r" %71 : (i32) -> !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    %102 = llvm.extractvalue %101[0] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %103 = llvm.bitcast %102 : i32 to f32
    %104 = llvm.extractvalue %101[1] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %105 = llvm.bitcast %104 : i32 to f32
    %106 = llvm.extractvalue %101[2] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %107 = llvm.bitcast %106 : i32 to f32
    %108 = llvm.extractvalue %101[3] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %109 = llvm.bitcast %108 : i32 to f32
    %110 = llvm.extractvalue %101[4] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %111 = llvm.bitcast %110 : i32 to f32
    %112 = llvm.extractvalue %101[5] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %113 = llvm.bitcast %112 : i32 to f32
    %114 = llvm.extractvalue %101[6] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %115 = llvm.bitcast %114 : i32 to f32
    %116 = llvm.extractvalue %101[7] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %117 = llvm.bitcast %116 : i32 to f32
    %118 = llvm.extractvalue %101[8] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %119 = llvm.bitcast %118 : i32 to f32
    %120 = llvm.extractvalue %101[9] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %121 = llvm.bitcast %120 : i32 to f32
    %122 = llvm.extractvalue %101[10] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %123 = llvm.bitcast %122 : i32 to f32
    %124 = llvm.extractvalue %101[11] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %125 = llvm.bitcast %124 : i32 to f32
    %126 = llvm.extractvalue %101[12] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %127 = llvm.bitcast %126 : i32 to f32
    %128 = llvm.extractvalue %101[13] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %129 = llvm.bitcast %128 : i32 to f32
    %130 = llvm.extractvalue %101[14] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %131 = llvm.bitcast %130 : i32 to f32
    %132 = llvm.extractvalue %101[15] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %133 = llvm.bitcast %132 : i32 to f32
    nvvm.tcgen05.wait <load>
    %134 = llvm.urem %37, %28 : i32
    %135 = llvm.udiv %37, %28 : i32
    %136 = llvm.shl %134, %2 : i32
    %137 = llvm.or %2, %136 : i32
    %138 = llvm.shl %135, %13 : i32
    %139 = llvm.or %137, %138 : i32
    %140 = llvm.and %139, %14 : i32
    %141 = llvm.shl %140, %15 : i32
    %142 = llvm.and %139, %16 : i32
    %143 = llvm.shl %142, %17 : i32
    %144 = llvm.and %139, %15 : i32
    %145 = llvm.shl %144, %6 : i32
    %146 = llvm.and %139, %18 : i32
    %147 = llvm.icmp "eq" %146, %2 : i32
    %148 = llvm.select %147, %2, %19 : i1, i32
    %149 = llvm.xor %143, %145 : i32
    %150 = llvm.xor %149, %148 : i32
    %151 = llvm.or disjoint %141, %150 : i32
    %152 = llvm.xor %2, %151 : i32
    %153 = llvm.mul %2, %6 : i32
    %154 = llvm.xor %152, %153 : i32
    %155 = llvm.xor %154, %2 : i32
    %156 = llvm.add %155, %2 : i32
    %157 = llvm.getelementptr inbounds %27[%156] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %158 = llvm.insertelement %103, %20[%2 : i32] : vector<4xf32>
    %159 = llvm.insertelement %105, %158[%21 : i32] : vector<4xf32>
    %160 = llvm.insertelement %107, %159[%22 : i32] : vector<4xf32>
    %161 = llvm.insertelement %109, %160[%4 : i32] : vector<4xf32>
    %162 = llvm.extractelement %161[%2 : i32] : vector<4xf32>
    %163 = llvm.extractelement %161[%21 : i32] : vector<4xf32>
    %164 = llvm.extractelement %161[%22 : i32] : vector<4xf32>
    %165 = llvm.extractelement %161[%4 : i32] : vector<4xf32>
    %166 = llvm.bitcast %162 : f32 to i32
    %167 = llvm.bitcast %163 : f32 to i32
    %168 = llvm.bitcast %164 : f32 to i32
    %169 = llvm.bitcast %165 : f32 to i32
    %170 = llvm.insertelement %166, %23[%2 : i32] : vector<4xi32>
    %171 = llvm.insertelement %167, %170[%21 : i32] : vector<4xi32>
    %172 = llvm.insertelement %168, %171[%22 : i32] : vector<4xi32>
    %173 = llvm.insertelement %169, %172[%4 : i32] : vector<4xi32>
    llvm.store %173, %157 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %174 = llvm.xor %154, %18 : i32
    %175 = llvm.add %174, %2 : i32
    %176 = llvm.getelementptr inbounds %27[%175] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %177 = llvm.insertelement %111, %20[%2 : i32] : vector<4xf32>
    %178 = llvm.insertelement %113, %177[%21 : i32] : vector<4xf32>
    %179 = llvm.insertelement %115, %178[%22 : i32] : vector<4xf32>
    %180 = llvm.insertelement %117, %179[%4 : i32] : vector<4xf32>
    %181 = llvm.extractelement %180[%2 : i32] : vector<4xf32>
    %182 = llvm.extractelement %180[%21 : i32] : vector<4xf32>
    %183 = llvm.extractelement %180[%22 : i32] : vector<4xf32>
    %184 = llvm.extractelement %180[%4 : i32] : vector<4xf32>
    %185 = llvm.bitcast %181 : f32 to i32
    %186 = llvm.bitcast %182 : f32 to i32
    %187 = llvm.bitcast %183 : f32 to i32
    %188 = llvm.bitcast %184 : f32 to i32
    %189 = llvm.insertelement %185, %23[%2 : i32] : vector<4xi32>
    %190 = llvm.insertelement %186, %189[%21 : i32] : vector<4xi32>
    %191 = llvm.insertelement %187, %190[%22 : i32] : vector<4xi32>
    %192 = llvm.insertelement %188, %191[%4 : i32] : vector<4xi32>
    llvm.store %192, %176 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %193 = llvm.xor %154, %28 : i32
    %194 = llvm.add %193, %2 : i32
    %195 = llvm.getelementptr inbounds %27[%194] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %196 = llvm.insertelement %119, %20[%2 : i32] : vector<4xf32>
    %197 = llvm.insertelement %121, %196[%21 : i32] : vector<4xf32>
    %198 = llvm.insertelement %123, %197[%22 : i32] : vector<4xf32>
    %199 = llvm.insertelement %125, %198[%4 : i32] : vector<4xf32>
    %200 = llvm.extractelement %199[%2 : i32] : vector<4xf32>
    %201 = llvm.extractelement %199[%21 : i32] : vector<4xf32>
    %202 = llvm.extractelement %199[%22 : i32] : vector<4xf32>
    %203 = llvm.extractelement %199[%4 : i32] : vector<4xf32>
    %204 = llvm.bitcast %200 : f32 to i32
    %205 = llvm.bitcast %201 : f32 to i32
    %206 = llvm.bitcast %202 : f32 to i32
    %207 = llvm.bitcast %203 : f32 to i32
    %208 = llvm.insertelement %204, %23[%2 : i32] : vector<4xi32>
    %209 = llvm.insertelement %205, %208[%21 : i32] : vector<4xi32>
    %210 = llvm.insertelement %206, %209[%22 : i32] : vector<4xi32>
    %211 = llvm.insertelement %207, %210[%4 : i32] : vector<4xi32>
    llvm.store %211, %195 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %212 = llvm.xor %154, %24 : i32
    %213 = llvm.add %212, %2 : i32
    %214 = llvm.getelementptr inbounds %27[%213] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %215 = llvm.insertelement %127, %20[%2 : i32] : vector<4xf32>
    %216 = llvm.insertelement %129, %215[%21 : i32] : vector<4xf32>
    %217 = llvm.insertelement %131, %216[%22 : i32] : vector<4xf32>
    %218 = llvm.insertelement %133, %217[%4 : i32] : vector<4xf32>
    %219 = llvm.extractelement %218[%2 : i32] : vector<4xf32>
    %220 = llvm.extractelement %218[%21 : i32] : vector<4xf32>
    %221 = llvm.extractelement %218[%22 : i32] : vector<4xf32>
    %222 = llvm.extractelement %218[%4 : i32] : vector<4xf32>
    %223 = llvm.bitcast %219 : f32 to i32
    %224 = llvm.bitcast %220 : f32 to i32
    %225 = llvm.bitcast %221 : f32 to i32
    %226 = llvm.bitcast %222 : f32 to i32
    %227 = llvm.insertelement %223, %23[%2 : i32] : vector<4xi32>
    %228 = llvm.insertelement %224, %227[%21 : i32] : vector<4xi32>
    %229 = llvm.insertelement %225, %228[%22 : i32] : vector<4xi32>
    %230 = llvm.insertelement %226, %229[%4 : i32] : vector<4xi32>
    llvm.store %230, %214 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
    nvvm.barrier0
    %231 = nvvm.elect.sync -> i1
    %232 = llvm.and %231, %44 : i1
    %233 = llvm.getelementptr %27[%46] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %234 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$1, {$2, $3}], [$4];", "b,l,r,r,r" %232, %arg10, %48, %48, %233 : (i1, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.cp.async.bulk.commit.group
    nvvm.cp.async.bulk.wait_group 0 {read}
    nvvm.barrier0
    nvvm.barrier0
    %235 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.dealloc.cta_group::1.sync.aligned.b32 $1, 32;", "b,r" %30, %33 : (i1, !llvm.ptr<6>) -> !llvm.void
    llvm.return
  }
}


// -----// IR Dump After CSE (cse) //----- //
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 12296 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 32 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr<1>, %arg16: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(127 : i32) : i32
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(21 : i32) : i32
    %6 = llvm.mlir.constant(4 : i32) : i32
    %7 = llvm.mlir.constant(16383 : i32) : i32
    %8 = llvm.mlir.constant(4611756662049472512 : i64) : i64
    %9 = llvm.mlir.constant(67633168 : i32) : i32
    %10 = llvm.mlir.constant(4611756662049472514 : i64) : i64
    %11 = llvm.mlir.constant(4611756662049472516 : i64) : i64
    %12 = llvm.mlir.constant(4611756662049472518 : i64) : i64
    %13 = llvm.mlir.constant(5 : i32) : i32
    %14 = llvm.mlir.constant(15 : i32) : i32
    %15 = llvm.mlir.constant(7 : i32) : i32
    %16 = llvm.mlir.constant(96 : i32) : i32
    %17 = llvm.mlir.constant(6 : i32) : i32
    %18 = llvm.mlir.constant(16 : i32) : i32
    %19 = llvm.mlir.constant(64 : i32) : i32
    %20 = llvm.mlir.undef : vector<4xf32>
    %21 = llvm.mlir.constant(1 : i32) : i32
    %22 = llvm.mlir.constant(2 : i32) : i32
    %23 = llvm.mlir.undef : vector<4xi32>
    %24 = llvm.mlir.constant(48 : i32) : i32
    %25 = llvm.mlir.constant(31 : i32) : i32
    %26 = llvm.mlir.constant(-1 : i32) : i32
    %27 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %28 = llvm.mlir.constant(32 : i32) : i32
    %29 = nvvm.read.ptx.sreg.tid.x : i32
    %30 = llvm.icmp "ult" %29, %28 : i32
    %31 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], 32;", "b,r" %30, %27 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %32 = llvm.load %27 : !llvm.ptr<3> -> i32
    nvvm.barrier0
    %33 = llvm.inttoptr %32 : i32 to !llvm.ptr<6>
    %34 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", "b" %30 : (i1) -> !llvm.void
    %35 = llvm.getelementptr %27[8192] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %36 = llvm.and %29, %3 : i32
    %37 = llvm.icmp "eq" %36, %2 : i32
    %38 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %37, %35 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %39 = llvm.and %37, %1 : i1
    %40 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 8192;", "b,r" %39, %35 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %41 = nvvm.elect.sync -> i1
    %42 = llvm.and %1, %41 : i1
    %43 = llvm.icmp "ult" %36, %28 : i32
    %44 = llvm.and %42, %43 : i1
    %45 = llvm.xor %2, %2 : i32
    %46 = llvm.getelementptr %27[%45] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %47 = llvm.add %45, %2 : i32
    %48 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %44, %46, %arg0, %47, %47, %35 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %49 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %35, %2 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %50 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %37, %35 : (i1, !llvm.ptr<3>) -> !llvm.void
    %51 = llvm.getelementptr %27[12288] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %52 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %37, %51 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %53 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 4096;", "b,r" %39, %51 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
    nvvm.barrier0
    %54 = nvvm.elect.sync -> i1
    %55 = llvm.and %1, %54 : i1
    %56 = llvm.and %55, %43 : i1
    %57 = llvm.getelementptr %35[%45] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %58 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %56, %57, %arg5, %47, %47, %51 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %59 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %51, %2 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %60 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %37, %51 : (i1, !llvm.ptr<3>) -> !llvm.void
    %61 = llvm.ptrtoint %33 : !llvm.ptr<6> to i32
    %62 = llvm.add %61, %2 : i32
    %63 = llvm.inttoptr %62 : i32 to !llvm.ptr<3>
    %64 = llvm.ptrtoint %63 : !llvm.ptr<3> to i32
    %65 = llvm.udiv %29, %28 : i32
    %66 = nvvm.shfl.sync  idx %26, %65, %2, %25 : i32 -> i32
    %67 = llvm.and %66, %4 : i32
    %68 = llvm.shl %67, %5 : i32
    %69 = llvm.add %64, %68 : i32
    %70 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.st.sync.aligned.16x32bx2.x16.b32 [$1 + 0], 16, {$2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17};", "b,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r" %1, %69, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0 : (i1, i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.void
    nvvm.tcgen05.wait <store>
    nvvm.barrier0
    nvvm.barrier0
    %71 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %37, %51 : (i1, !llvm.ptr<3>) -> !llvm.void
    %72 = llvm.icmp "eq" %66, %2 : i32
    %73 = llvm.and %1, %72 : i1
    llvm.cond_br %73, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %74 = nvvm.elect.sync -> i1
    %75 = llvm.ptrtoint %27 : !llvm.ptr<3> to i32
    %76 = llvm.lshr %75, %6 : i32
    %77 = llvm.and %76, %7 : i32
    %78 = llvm.zext %77 : i32 to i64
    %79 = llvm.ptrtoint %35 : !llvm.ptr<3> to i32
    %80 = llvm.lshr %79, %6 : i32
    %81 = llvm.and %80, %7 : i32
    %82 = llvm.zext %81 : i32 to i64
    %83 = llvm.add %78, %8 : i64
    %84 = llvm.add %82, %8 : i64
    %85 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %64, %83, %84, %9, %1, %74 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %86 = llvm.add %78, %10 : i64
    %87 = llvm.add %82, %10 : i64
    %88 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %64, %86, %87, %9, %1, %74 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %89 = llvm.add %78, %11 : i64
    %90 = llvm.add %82, %11 : i64
    %91 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %64, %89, %90, %9, %1, %74 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %92 = llvm.add %78, %12 : i64
    %93 = llvm.add %82, %12 : i64
    %94 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %64, %92, %93, %9, %1, %74 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %95 = llvm.and %1, %74 : i1
    %96 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$1];", "b,l" %95, %51 : (i1, !llvm.ptr<3>) -> !llvm.void
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    nvvm.barrier0
    %97 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %51, %2 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %98 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %37, %51 : (i1, !llvm.ptr<3>) -> !llvm.void
    %99 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15}, [$16 + 0], 16;", "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r" %69 : (i32) -> !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    %100 = llvm.extractvalue %99[0] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %101 = llvm.bitcast %100 : i32 to f32
    %102 = llvm.extractvalue %99[1] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %103 = llvm.bitcast %102 : i32 to f32
    %104 = llvm.extractvalue %99[2] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %105 = llvm.bitcast %104 : i32 to f32
    %106 = llvm.extractvalue %99[3] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %107 = llvm.bitcast %106 : i32 to f32
    %108 = llvm.extractvalue %99[4] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %109 = llvm.bitcast %108 : i32 to f32
    %110 = llvm.extractvalue %99[5] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %111 = llvm.bitcast %110 : i32 to f32
    %112 = llvm.extractvalue %99[6] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %113 = llvm.bitcast %112 : i32 to f32
    %114 = llvm.extractvalue %99[7] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %115 = llvm.bitcast %114 : i32 to f32
    %116 = llvm.extractvalue %99[8] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %117 = llvm.bitcast %116 : i32 to f32
    %118 = llvm.extractvalue %99[9] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %119 = llvm.bitcast %118 : i32 to f32
    %120 = llvm.extractvalue %99[10] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %121 = llvm.bitcast %120 : i32 to f32
    %122 = llvm.extractvalue %99[11] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %123 = llvm.bitcast %122 : i32 to f32
    %124 = llvm.extractvalue %99[12] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %125 = llvm.bitcast %124 : i32 to f32
    %126 = llvm.extractvalue %99[13] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %127 = llvm.bitcast %126 : i32 to f32
    %128 = llvm.extractvalue %99[14] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %129 = llvm.bitcast %128 : i32 to f32
    %130 = llvm.extractvalue %99[15] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %131 = llvm.bitcast %130 : i32 to f32
    nvvm.tcgen05.wait <load>
    %132 = llvm.urem %36, %28 : i32
    %133 = llvm.udiv %36, %28 : i32
    %134 = llvm.shl %132, %2 : i32
    %135 = llvm.or %2, %134 : i32
    %136 = llvm.shl %133, %13 : i32
    %137 = llvm.or %135, %136 : i32
    %138 = llvm.and %137, %14 : i32
    %139 = llvm.shl %138, %15 : i32
    %140 = llvm.and %137, %16 : i32
    %141 = llvm.shl %140, %17 : i32
    %142 = llvm.and %137, %15 : i32
    %143 = llvm.shl %142, %6 : i32
    %144 = llvm.and %137, %18 : i32
    %145 = llvm.icmp "eq" %144, %2 : i32
    %146 = llvm.select %145, %2, %19 : i1, i32
    %147 = llvm.xor %141, %143 : i32
    %148 = llvm.xor %147, %146 : i32
    %149 = llvm.or disjoint %139, %148 : i32
    %150 = llvm.xor %2, %149 : i32
    %151 = llvm.mul %2, %6 : i32
    %152 = llvm.xor %150, %151 : i32
    %153 = llvm.xor %152, %2 : i32
    %154 = llvm.add %153, %2 : i32
    %155 = llvm.getelementptr inbounds %27[%154] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %156 = llvm.insertelement %101, %20[%2 : i32] : vector<4xf32>
    %157 = llvm.insertelement %103, %156[%21 : i32] : vector<4xf32>
    %158 = llvm.insertelement %105, %157[%22 : i32] : vector<4xf32>
    %159 = llvm.insertelement %107, %158[%4 : i32] : vector<4xf32>
    %160 = llvm.extractelement %159[%2 : i32] : vector<4xf32>
    %161 = llvm.extractelement %159[%21 : i32] : vector<4xf32>
    %162 = llvm.extractelement %159[%22 : i32] : vector<4xf32>
    %163 = llvm.extractelement %159[%4 : i32] : vector<4xf32>
    %164 = llvm.bitcast %160 : f32 to i32
    %165 = llvm.bitcast %161 : f32 to i32
    %166 = llvm.bitcast %162 : f32 to i32
    %167 = llvm.bitcast %163 : f32 to i32
    %168 = llvm.insertelement %164, %23[%2 : i32] : vector<4xi32>
    %169 = llvm.insertelement %165, %168[%21 : i32] : vector<4xi32>
    %170 = llvm.insertelement %166, %169[%22 : i32] : vector<4xi32>
    %171 = llvm.insertelement %167, %170[%4 : i32] : vector<4xi32>
    llvm.store %171, %155 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %172 = llvm.xor %152, %18 : i32
    %173 = llvm.add %172, %2 : i32
    %174 = llvm.getelementptr inbounds %27[%173] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %175 = llvm.insertelement %109, %20[%2 : i32] : vector<4xf32>
    %176 = llvm.insertelement %111, %175[%21 : i32] : vector<4xf32>
    %177 = llvm.insertelement %113, %176[%22 : i32] : vector<4xf32>
    %178 = llvm.insertelement %115, %177[%4 : i32] : vector<4xf32>
    %179 = llvm.extractelement %178[%2 : i32] : vector<4xf32>
    %180 = llvm.extractelement %178[%21 : i32] : vector<4xf32>
    %181 = llvm.extractelement %178[%22 : i32] : vector<4xf32>
    %182 = llvm.extractelement %178[%4 : i32] : vector<4xf32>
    %183 = llvm.bitcast %179 : f32 to i32
    %184 = llvm.bitcast %180 : f32 to i32
    %185 = llvm.bitcast %181 : f32 to i32
    %186 = llvm.bitcast %182 : f32 to i32
    %187 = llvm.insertelement %183, %23[%2 : i32] : vector<4xi32>
    %188 = llvm.insertelement %184, %187[%21 : i32] : vector<4xi32>
    %189 = llvm.insertelement %185, %188[%22 : i32] : vector<4xi32>
    %190 = llvm.insertelement %186, %189[%4 : i32] : vector<4xi32>
    llvm.store %190, %174 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %191 = llvm.xor %152, %28 : i32
    %192 = llvm.add %191, %2 : i32
    %193 = llvm.getelementptr inbounds %27[%192] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %194 = llvm.insertelement %117, %20[%2 : i32] : vector<4xf32>
    %195 = llvm.insertelement %119, %194[%21 : i32] : vector<4xf32>
    %196 = llvm.insertelement %121, %195[%22 : i32] : vector<4xf32>
    %197 = llvm.insertelement %123, %196[%4 : i32] : vector<4xf32>
    %198 = llvm.extractelement %197[%2 : i32] : vector<4xf32>
    %199 = llvm.extractelement %197[%21 : i32] : vector<4xf32>
    %200 = llvm.extractelement %197[%22 : i32] : vector<4xf32>
    %201 = llvm.extractelement %197[%4 : i32] : vector<4xf32>
    %202 = llvm.bitcast %198 : f32 to i32
    %203 = llvm.bitcast %199 : f32 to i32
    %204 = llvm.bitcast %200 : f32 to i32
    %205 = llvm.bitcast %201 : f32 to i32
    %206 = llvm.insertelement %202, %23[%2 : i32] : vector<4xi32>
    %207 = llvm.insertelement %203, %206[%21 : i32] : vector<4xi32>
    %208 = llvm.insertelement %204, %207[%22 : i32] : vector<4xi32>
    %209 = llvm.insertelement %205, %208[%4 : i32] : vector<4xi32>
    llvm.store %209, %193 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %210 = llvm.xor %152, %24 : i32
    %211 = llvm.add %210, %2 : i32
    %212 = llvm.getelementptr inbounds %27[%211] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %213 = llvm.insertelement %125, %20[%2 : i32] : vector<4xf32>
    %214 = llvm.insertelement %127, %213[%21 : i32] : vector<4xf32>
    %215 = llvm.insertelement %129, %214[%22 : i32] : vector<4xf32>
    %216 = llvm.insertelement %131, %215[%4 : i32] : vector<4xf32>
    %217 = llvm.extractelement %216[%2 : i32] : vector<4xf32>
    %218 = llvm.extractelement %216[%21 : i32] : vector<4xf32>
    %219 = llvm.extractelement %216[%22 : i32] : vector<4xf32>
    %220 = llvm.extractelement %216[%4 : i32] : vector<4xf32>
    %221 = llvm.bitcast %217 : f32 to i32
    %222 = llvm.bitcast %218 : f32 to i32
    %223 = llvm.bitcast %219 : f32 to i32
    %224 = llvm.bitcast %220 : f32 to i32
    %225 = llvm.insertelement %221, %23[%2 : i32] : vector<4xi32>
    %226 = llvm.insertelement %222, %225[%21 : i32] : vector<4xi32>
    %227 = llvm.insertelement %223, %226[%22 : i32] : vector<4xi32>
    %228 = llvm.insertelement %224, %227[%4 : i32] : vector<4xi32>
    llvm.store %228, %212 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
    nvvm.barrier0
    %229 = nvvm.elect.sync -> i1
    %230 = llvm.and %229, %43 : i1
    %231 = llvm.getelementptr %27[%45] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %232 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$1, {$2, $3}], [$4];", "b,l,r,r,r" %230, %arg10, %47, %47, %231 : (i1, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.cp.async.bulk.commit.group
    nvvm.cp.async.bulk.wait_group 0 {read}
    nvvm.barrier0
    nvvm.barrier0
    %233 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.dealloc.cta_group::1.sync.aligned.b32 $1, 32;", "b,r" %30, %33 : (i1, !llvm.ptr<6>) -> !llvm.void
    llvm.return
  }
}


// -----// IR Dump After ConvertNVVMToLLVMPass (convert-nvvm-to-llvm) //----- //
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 12296 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 32 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr<1>, %arg16: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(127 : i32) : i32
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(21 : i32) : i32
    %6 = llvm.mlir.constant(4 : i32) : i32
    %7 = llvm.mlir.constant(16383 : i32) : i32
    %8 = llvm.mlir.constant(4611756662049472512 : i64) : i64
    %9 = llvm.mlir.constant(67633168 : i32) : i32
    %10 = llvm.mlir.constant(4611756662049472514 : i64) : i64
    %11 = llvm.mlir.constant(4611756662049472516 : i64) : i64
    %12 = llvm.mlir.constant(4611756662049472518 : i64) : i64
    %13 = llvm.mlir.constant(5 : i32) : i32
    %14 = llvm.mlir.constant(15 : i32) : i32
    %15 = llvm.mlir.constant(7 : i32) : i32
    %16 = llvm.mlir.constant(96 : i32) : i32
    %17 = llvm.mlir.constant(6 : i32) : i32
    %18 = llvm.mlir.constant(16 : i32) : i32
    %19 = llvm.mlir.constant(64 : i32) : i32
    %20 = llvm.mlir.undef : vector<4xf32>
    %21 = llvm.mlir.constant(1 : i32) : i32
    %22 = llvm.mlir.constant(2 : i32) : i32
    %23 = llvm.mlir.undef : vector<4xi32>
    %24 = llvm.mlir.constant(48 : i32) : i32
    %25 = llvm.mlir.constant(31 : i32) : i32
    %26 = llvm.mlir.constant(-1 : i32) : i32
    %27 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %28 = llvm.mlir.constant(32 : i32) : i32
    %29 = nvvm.read.ptx.sreg.tid.x : i32
    %30 = llvm.icmp "ult" %29, %28 : i32
    %31 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], 32;", "b,r" %30, %27 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %32 = llvm.load %27 : !llvm.ptr<3> -> i32
    nvvm.barrier0
    %33 = llvm.inttoptr %32 : i32 to !llvm.ptr<6>
    %34 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", "b" %30 : (i1) -> !llvm.void
    %35 = llvm.getelementptr %27[8192] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %36 = llvm.and %29, %3 : i32
    %37 = llvm.icmp "eq" %36, %2 : i32
    %38 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %37, %35 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %39 = llvm.and %37, %1 : i1
    %40 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 8192;", "b,r" %39, %35 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %41 = nvvm.elect.sync -> i1
    %42 = llvm.and %1, %41 : i1
    %43 = llvm.icmp "ult" %36, %28 : i32
    %44 = llvm.and %42, %43 : i1
    %45 = llvm.xor %2, %2 : i32
    %46 = llvm.getelementptr %27[%45] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %47 = llvm.add %45, %2 : i32
    %48 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %44, %46, %arg0, %47, %47, %35 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %49 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %35, %2 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %50 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %37, %35 : (i1, !llvm.ptr<3>) -> !llvm.void
    %51 = llvm.getelementptr %27[12288] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %52 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %37, %51 : (i1, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %53 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 4096;", "b,r" %39, %51 : (i1, !llvm.ptr<3>) -> !llvm.void
    llvm.inline_asm has_side_effects asm_dialect = att "fence.proxy.async.shared::cta;", ""  : () -> ()
    nvvm.barrier0
    %54 = nvvm.elect.sync -> i1
    %55 = llvm.and %1, %54 : i1
    %56 = llvm.and %55, %43 : i1
    %57 = llvm.getelementptr %35[%45] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %58 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %56, %57, %arg5, %47, %47, %51 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.barrier0
    %59 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %51, %2 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %60 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %37, %51 : (i1, !llvm.ptr<3>) -> !llvm.void
    %61 = llvm.ptrtoint %33 : !llvm.ptr<6> to i32
    %62 = llvm.add %61, %2 : i32
    %63 = llvm.inttoptr %62 : i32 to !llvm.ptr<3>
    %64 = llvm.ptrtoint %63 : !llvm.ptr<3> to i32
    %65 = llvm.udiv %29, %28 : i32
    %66 = nvvm.shfl.sync  idx %26, %65, %2, %25 : i32 -> i32
    %67 = llvm.and %66, %4 : i32
    %68 = llvm.shl %67, %5 : i32
    %69 = llvm.add %64, %68 : i32
    %70 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.st.sync.aligned.16x32bx2.x16.b32 [$1 + 0], 16, {$2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17};", "b,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r" %1, %69, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0 : (i1, i32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.void
    nvvm.tcgen05.wait <store>
    nvvm.barrier0
    nvvm.barrier0
    %71 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %37, %51 : (i1, !llvm.ptr<3>) -> !llvm.void
    %72 = llvm.icmp "eq" %66, %2 : i32
    %73 = llvm.and %1, %72 : i1
    llvm.cond_br %73, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %74 = nvvm.elect.sync -> i1
    %75 = llvm.ptrtoint %27 : !llvm.ptr<3> to i32
    %76 = llvm.lshr %75, %6 : i32
    %77 = llvm.and %76, %7 : i32
    %78 = llvm.zext %77 : i32 to i64
    %79 = llvm.ptrtoint %35 : !llvm.ptr<3> to i32
    %80 = llvm.lshr %79, %6 : i32
    %81 = llvm.and %80, %7 : i32
    %82 = llvm.zext %81 : i32 to i64
    %83 = llvm.add %78, %8 : i64
    %84 = llvm.add %82, %8 : i64
    %85 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %64, %83, %84, %9, %1, %74 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %86 = llvm.add %78, %10 : i64
    %87 = llvm.add %82, %10 : i64
    %88 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %64, %86, %87, %9, %1, %74 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %89 = llvm.add %78, %11 : i64
    %90 = llvm.add %82, %11 : i64
    %91 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %64, %89, %90, %9, %1, %74 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %92 = llvm.add %78, %12 : i64
    %93 = llvm.add %82, %12 : i64
    %94 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %64, %92, %93, %9, %1, %74 : (i32, i64, i64, i32, i1, i1) -> !llvm.void
    %95 = llvm.and %1, %74 : i1
    %96 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$1];", "b,l" %95, %51 : (i1, !llvm.ptr<3>) -> !llvm.void
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    nvvm.barrier0
    %97 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %51, %2 : (!llvm.ptr<3>, i32) -> !llvm.void
    nvvm.barrier0
    %98 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %37, %51 : (i1, !llvm.ptr<3>) -> !llvm.void
    %99 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15}, [$16 + 0], 16;", "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r" %69 : (i32) -> !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    %100 = llvm.extractvalue %99[0] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %101 = llvm.bitcast %100 : i32 to f32
    %102 = llvm.extractvalue %99[1] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %103 = llvm.bitcast %102 : i32 to f32
    %104 = llvm.extractvalue %99[2] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %105 = llvm.bitcast %104 : i32 to f32
    %106 = llvm.extractvalue %99[3] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %107 = llvm.bitcast %106 : i32 to f32
    %108 = llvm.extractvalue %99[4] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %109 = llvm.bitcast %108 : i32 to f32
    %110 = llvm.extractvalue %99[5] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %111 = llvm.bitcast %110 : i32 to f32
    %112 = llvm.extractvalue %99[6] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %113 = llvm.bitcast %112 : i32 to f32
    %114 = llvm.extractvalue %99[7] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %115 = llvm.bitcast %114 : i32 to f32
    %116 = llvm.extractvalue %99[8] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %117 = llvm.bitcast %116 : i32 to f32
    %118 = llvm.extractvalue %99[9] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %119 = llvm.bitcast %118 : i32 to f32
    %120 = llvm.extractvalue %99[10] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %121 = llvm.bitcast %120 : i32 to f32
    %122 = llvm.extractvalue %99[11] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %123 = llvm.bitcast %122 : i32 to f32
    %124 = llvm.extractvalue %99[12] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %125 = llvm.bitcast %124 : i32 to f32
    %126 = llvm.extractvalue %99[13] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %127 = llvm.bitcast %126 : i32 to f32
    %128 = llvm.extractvalue %99[14] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %129 = llvm.bitcast %128 : i32 to f32
    %130 = llvm.extractvalue %99[15] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> 
    %131 = llvm.bitcast %130 : i32 to f32
    nvvm.tcgen05.wait <load>
    %132 = llvm.urem %36, %28 : i32
    %133 = llvm.udiv %36, %28 : i32
    %134 = llvm.shl %132, %2 : i32
    %135 = llvm.or %2, %134 : i32
    %136 = llvm.shl %133, %13 : i32
    %137 = llvm.or %135, %136 : i32
    %138 = llvm.and %137, %14 : i32
    %139 = llvm.shl %138, %15 : i32
    %140 = llvm.and %137, %16 : i32
    %141 = llvm.shl %140, %17 : i32
    %142 = llvm.and %137, %15 : i32
    %143 = llvm.shl %142, %6 : i32
    %144 = llvm.and %137, %18 : i32
    %145 = llvm.icmp "eq" %144, %2 : i32
    %146 = llvm.select %145, %2, %19 : i1, i32
    %147 = llvm.xor %141, %143 : i32
    %148 = llvm.xor %147, %146 : i32
    %149 = llvm.or disjoint %139, %148 : i32
    %150 = llvm.xor %2, %149 : i32
    %151 = llvm.mul %2, %6 : i32
    %152 = llvm.xor %150, %151 : i32
    %153 = llvm.xor %152, %2 : i32
    %154 = llvm.add %153, %2 : i32
    %155 = llvm.getelementptr inbounds %27[%154] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %156 = llvm.insertelement %101, %20[%2 : i32] : vector<4xf32>
    %157 = llvm.insertelement %103, %156[%21 : i32] : vector<4xf32>
    %158 = llvm.insertelement %105, %157[%22 : i32] : vector<4xf32>
    %159 = llvm.insertelement %107, %158[%4 : i32] : vector<4xf32>
    %160 = llvm.extractelement %159[%2 : i32] : vector<4xf32>
    %161 = llvm.extractelement %159[%21 : i32] : vector<4xf32>
    %162 = llvm.extractelement %159[%22 : i32] : vector<4xf32>
    %163 = llvm.extractelement %159[%4 : i32] : vector<4xf32>
    %164 = llvm.bitcast %160 : f32 to i32
    %165 = llvm.bitcast %161 : f32 to i32
    %166 = llvm.bitcast %162 : f32 to i32
    %167 = llvm.bitcast %163 : f32 to i32
    %168 = llvm.insertelement %164, %23[%2 : i32] : vector<4xi32>
    %169 = llvm.insertelement %165, %168[%21 : i32] : vector<4xi32>
    %170 = llvm.insertelement %166, %169[%22 : i32] : vector<4xi32>
    %171 = llvm.insertelement %167, %170[%4 : i32] : vector<4xi32>
    llvm.store %171, %155 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %172 = llvm.xor %152, %18 : i32
    %173 = llvm.add %172, %2 : i32
    %174 = llvm.getelementptr inbounds %27[%173] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %175 = llvm.insertelement %109, %20[%2 : i32] : vector<4xf32>
    %176 = llvm.insertelement %111, %175[%21 : i32] : vector<4xf32>
    %177 = llvm.insertelement %113, %176[%22 : i32] : vector<4xf32>
    %178 = llvm.insertelement %115, %177[%4 : i32] : vector<4xf32>
    %179 = llvm.extractelement %178[%2 : i32] : vector<4xf32>
    %180 = llvm.extractelement %178[%21 : i32] : vector<4xf32>
    %181 = llvm.extractelement %178[%22 : i32] : vector<4xf32>
    %182 = llvm.extractelement %178[%4 : i32] : vector<4xf32>
    %183 = llvm.bitcast %179 : f32 to i32
    %184 = llvm.bitcast %180 : f32 to i32
    %185 = llvm.bitcast %181 : f32 to i32
    %186 = llvm.bitcast %182 : f32 to i32
    %187 = llvm.insertelement %183, %23[%2 : i32] : vector<4xi32>
    %188 = llvm.insertelement %184, %187[%21 : i32] : vector<4xi32>
    %189 = llvm.insertelement %185, %188[%22 : i32] : vector<4xi32>
    %190 = llvm.insertelement %186, %189[%4 : i32] : vector<4xi32>
    llvm.store %190, %174 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %191 = llvm.xor %152, %28 : i32
    %192 = llvm.add %191, %2 : i32
    %193 = llvm.getelementptr inbounds %27[%192] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %194 = llvm.insertelement %117, %20[%2 : i32] : vector<4xf32>
    %195 = llvm.insertelement %119, %194[%21 : i32] : vector<4xf32>
    %196 = llvm.insertelement %121, %195[%22 : i32] : vector<4xf32>
    %197 = llvm.insertelement %123, %196[%4 : i32] : vector<4xf32>
    %198 = llvm.extractelement %197[%2 : i32] : vector<4xf32>
    %199 = llvm.extractelement %197[%21 : i32] : vector<4xf32>
    %200 = llvm.extractelement %197[%22 : i32] : vector<4xf32>
    %201 = llvm.extractelement %197[%4 : i32] : vector<4xf32>
    %202 = llvm.bitcast %198 : f32 to i32
    %203 = llvm.bitcast %199 : f32 to i32
    %204 = llvm.bitcast %200 : f32 to i32
    %205 = llvm.bitcast %201 : f32 to i32
    %206 = llvm.insertelement %202, %23[%2 : i32] : vector<4xi32>
    %207 = llvm.insertelement %203, %206[%21 : i32] : vector<4xi32>
    %208 = llvm.insertelement %204, %207[%22 : i32] : vector<4xi32>
    %209 = llvm.insertelement %205, %208[%4 : i32] : vector<4xi32>
    llvm.store %209, %193 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    %210 = llvm.xor %152, %24 : i32
    %211 = llvm.add %210, %2 : i32
    %212 = llvm.getelementptr inbounds %27[%211] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %213 = llvm.insertelement %125, %20[%2 : i32] : vector<4xf32>
    %214 = llvm.insertelement %127, %213[%21 : i32] : vector<4xf32>
    %215 = llvm.insertelement %129, %214[%22 : i32] : vector<4xf32>
    %216 = llvm.insertelement %131, %215[%4 : i32] : vector<4xf32>
    %217 = llvm.extractelement %216[%2 : i32] : vector<4xf32>
    %218 = llvm.extractelement %216[%21 : i32] : vector<4xf32>
    %219 = llvm.extractelement %216[%22 : i32] : vector<4xf32>
    %220 = llvm.extractelement %216[%4 : i32] : vector<4xf32>
    %221 = llvm.bitcast %217 : f32 to i32
    %222 = llvm.bitcast %218 : f32 to i32
    %223 = llvm.bitcast %219 : f32 to i32
    %224 = llvm.bitcast %220 : f32 to i32
    %225 = llvm.insertelement %221, %23[%2 : i32] : vector<4xi32>
    %226 = llvm.insertelement %222, %225[%21 : i32] : vector<4xi32>
    %227 = llvm.insertelement %223, %226[%22 : i32] : vector<4xi32>
    %228 = llvm.insertelement %224, %227[%4 : i32] : vector<4xi32>
    llvm.store %228, %212 {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<3>
    llvm.inline_asm has_side_effects asm_dialect = att "fence.proxy.async.shared::cta;", ""  : () -> ()
    nvvm.barrier0
    %229 = nvvm.elect.sync -> i1
    %230 = llvm.and %229, %43 : i1
    %231 = llvm.getelementptr %27[%45] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %232 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$1, {$2, $3}], [$4];", "b,l,r,r,r" %230, %arg10, %47, %47, %231 : (i1, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    nvvm.cp.async.bulk.commit.group
    nvvm.cp.async.bulk.wait_group 0 {read}
    nvvm.barrier0
    nvvm.barrier0
    %233 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.dealloc.cta_group::1.sync.aligned.b32 $1, 32;", "b,r" %30, %33 : (i1, !llvm.ptr<6>) -> !llvm.void
    llvm.return
  }
}


// =================================================
// LLVM IR
// =================================================
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"

@global_smem = external addrspace(3) global [0 x i8], align 16

; Function Attrs: nounwind
define ptx_kernel void @triton_dot(ptr byval([128 x i8]) align 64 "nvvm.grid_constant" %0, i32 %1, i32 %2, i64 %3, i64 %4, ptr byval([128 x i8]) align 64 "nvvm.grid_constant" %5, i32 %6, i32 %7, i64 %8, i64 %9, ptr byval([128 x i8]) align 64 "nvvm.grid_constant" %10, i32 %11, i32 %12, i64 %13, i64 %14, ptr addrspace(1) readnone captures(none) %15, ptr addrspace(1) readnone captures(none) %16) local_unnamed_addr #0 {
  %18 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %19 = icmp samesign ult i32 %18, 32
  tail call void asm sideeffect "@$0 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], 32;", "b,r"(i1 %19, ptr addrspace(3) @global_smem) #6
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %20 = load i32, ptr addrspace(3) @global_smem, align 16
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  tail call void asm sideeffect "@$0 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", "b"(i1 %19) #6
  %21 = and i32 %18, 127
  %22 = icmp eq i32 %21, 0
  tail call void asm sideeffect "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r"(i1 %22, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192)) #6
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  tail call void asm sideeffect "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 8192;", "b,r"(i1 %22, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192)) #6
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %23 = tail call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1)
  %24 = extractvalue { i32, i1 } %23, 1
  %25 = icmp samesign ult i32 %21, 32
  %26 = and i1 %25, %24
  call void asm sideeffect "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r"(i1 %26, ptr addrspace(3) @global_smem, ptr nonnull %0, i32 0, i32 0, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192)) #6
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  call void asm sideeffect "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r"(ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), i32 0) #6
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  call void asm sideeffect "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r"(i1 %22, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192)) #6
  call void asm sideeffect "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r"(i1 %22, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288)) #6
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  call void asm sideeffect "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 4096;", "b,r"(i1 %22, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288)) #6
  call void asm sideeffect "fence.proxy.async.shared::cta;", ""() #6
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %27 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1)
  %28 = extractvalue { i32, i1 } %27, 1
  %29 = and i1 %25, %28
  call void asm sideeffect "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r"(i1 %29, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192), ptr nonnull %5, i32 0, i32 0, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288)) #6
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  call void asm sideeffect "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r"(ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288), i32 0) #6
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  call void asm sideeffect "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r"(i1 %22, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288)) #6
  %30 = lshr i32 %18, 5
  %31 = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %30, i32 0, i32 31)
  %32 = shl i32 %31, 21
  %33 = and i32 %32, 6291456
  %34 = add i32 %33, %20
  call void asm sideeffect "@$0 tcgen05.st.sync.aligned.16x32bx2.x16.b32 [$1 + 0], 16, {$2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17};", "b,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r"(i1 true, i32 %34, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00) #6
  call void @llvm.nvvm.tcgen05.wait.st()
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  call void asm sideeffect "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r"(i1 %22, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288)) #6
  %35 = icmp eq i32 %31, 0
  br i1 %35, label %36, label %53

36:                                               ; preds = %17
  %37 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1)
  %38 = extractvalue { i32, i1 } %37, 1
  %39 = lshr exact i32 ptrtoint (ptr addrspace(3) @global_smem to i32), 4
  %40 = and i32 %39, 16383
  %41 = zext nneg i32 %40 to i64
  %42 = lshr exact i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192) to i32), 4
  %43 = and i32 %42, 16383
  %44 = zext nneg i32 %43 to i64
  %45 = or disjoint i64 %41, 4611756662049472512
  %46 = or disjoint i64 %44, 4611756662049472512
  call void asm sideeffect "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b"(i32 %20, i64 %45, i64 %46, i32 67633168, i1 true, i1 %38) #6
  %47 = add nuw nsw i64 %41, 4611756662049472514
  %48 = add nuw nsw i64 %44, 4611756662049472514
  call void asm sideeffect "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b"(i32 %20, i64 %47, i64 %48, i32 67633168, i1 true, i1 %38) #6
  %49 = add nuw nsw i64 %41, 4611756662049472516
  %50 = add nuw nsw i64 %44, 4611756662049472516
  call void asm sideeffect "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b"(i32 %20, i64 %49, i64 %50, i32 67633168, i1 true, i1 %38) #6
  %51 = add nuw nsw i64 %41, 4611756662049472518
  %52 = add nuw nsw i64 %44, 4611756662049472518
  call void asm sideeffect "@$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b"(i32 %20, i64 %51, i64 %52, i32 67633168, i1 true, i1 %38) #6
  call void asm sideeffect "@$0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$1];", "b,l"(i1 %38, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288)) #6
  br label %53

53:                                               ; preds = %36, %17
  %54 = inttoptr i32 %20 to ptr addrspace(6)
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  call void asm sideeffect "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r"(ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288), i32 0) #6
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  call void asm sideeffect "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r"(i1 %22, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 12288)) #6
  %55 = call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } asm sideeffect "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15}, [$16 + 0], 16;", "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r"(i32 %34) #6
  %56 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 0
  %57 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 1
  %58 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 2
  %59 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 3
  %60 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 4
  %61 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 5
  %62 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 6
  %63 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 7
  %64 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 8
  %65 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 9
  %66 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 10
  %67 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 11
  %68 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 12
  %69 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 13
  %70 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 14
  %71 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %55, 15
  call void @llvm.nvvm.tcgen05.wait.ld()
  %72 = shl nuw nsw i32 %18, 7
  %73 = and i32 %72, 1920
  %74 = shl nuw nsw i32 %18, 6
  %75 = and i32 %74, 6144
  %76 = shl nuw nsw i32 %18, 4
  %77 = and i32 %76, 112
  %78 = shl nuw nsw i32 %18, 2
  %79 = and i32 %78, 64
  %80 = or disjoint i32 %75, %77
  %81 = xor i32 %80, %79
  %82 = or disjoint i32 %81, %73
  %83 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %82
  %84 = insertelement <4 x i32> poison, i32 %56, i64 0
  %85 = insertelement <4 x i32> %84, i32 %57, i64 1
  %86 = insertelement <4 x i32> %85, i32 %58, i64 2
  %87 = insertelement <4 x i32> %86, i32 %59, i64 3
  store <4 x i32> %87, ptr addrspace(3) %83, align 16
  %88 = xor i32 %82, 16
  %89 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %88
  %90 = insertelement <4 x i32> poison, i32 %60, i64 0
  %91 = insertelement <4 x i32> %90, i32 %61, i64 1
  %92 = insertelement <4 x i32> %91, i32 %62, i64 2
  %93 = insertelement <4 x i32> %92, i32 %63, i64 3
  store <4 x i32> %93, ptr addrspace(3) %89, align 16
  %94 = xor i32 %82, 32
  %95 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %94
  %96 = insertelement <4 x i32> poison, i32 %64, i64 0
  %97 = insertelement <4 x i32> %96, i32 %65, i64 1
  %98 = insertelement <4 x i32> %97, i32 %66, i64 2
  %99 = insertelement <4 x i32> %98, i32 %67, i64 3
  store <4 x i32> %99, ptr addrspace(3) %95, align 16
  %100 = xor i32 %82, 48
  %101 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %100
  %102 = insertelement <4 x i32> poison, i32 %68, i64 0
  %103 = insertelement <4 x i32> %102, i32 %69, i64 1
  %104 = insertelement <4 x i32> %103, i32 %70, i64 2
  %105 = insertelement <4 x i32> %104, i32 %71, i64 3
  store <4 x i32> %105, ptr addrspace(3) %101, align 16
  call void asm sideeffect "fence.proxy.async.shared::cta;", ""() #6
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %106 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1)
  %107 = extractvalue { i32, i1 } %106, 1
  %108 = and i1 %25, %107
  call void asm sideeffect "@$0 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$1, {$2, $3}], [$4];", "b,l,r,r,r"(i1 %108, ptr nonnull %10, i32 0, i32 0, ptr addrspace(3) @global_smem) #6
  call void @llvm.nvvm.cp.async.bulk.commit.group()
  call void @llvm.nvvm.cp.async.bulk.wait.group.read(i32 0)
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  call void asm sideeffect "@$0 tcgen05.dealloc.cta_group::1.sync.aligned.b32 $1, 32;", "b,r"(i1 %19, ptr addrspace(6) %54) #6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare { i32, i1 } @llvm.nvvm.elect.sync(i32) #3

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.idx.i32(i32, i32, i32, i32) #4

; Function Attrs: convergent nounwind memory(inaccessiblemem: readwrite)
declare void @llvm.nvvm.tcgen05.wait.st() #5

; Function Attrs: convergent nounwind memory(inaccessiblemem: readwrite)
declare void @llvm.nvvm.tcgen05.wait.ld() #5

; Function Attrs: nounwind
declare void @llvm.nvvm.cp.async.bulk.commit.group() #6

; Function Attrs: nounwind
declare void @llvm.nvvm.cp.async.bulk.wait.group.read(i32 immarg) #6

attributes #0 = { nounwind "nvvm.reqntid"="128" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { convergent mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #4 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #5 = { convergent nounwind memory(inaccessiblemem: readwrite) }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
!2 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}


// =================================================
// PTX
// =================================================
//
// Generated by LLVM NVPTX Back-End
//

.version 8.6
.target sm_100a
.address_size 64

	// .globl	triton_dot              // -- Begin function triton_dot
.extern .shared .align 16 .b8 global_smem[];
                                        // @triton_dot
.visible .entry triton_dot(
	.param .align 64 .b8 triton_dot_param_0[128],
	.param .u32 triton_dot_param_1,
	.param .u32 triton_dot_param_2,
	.param .u64 triton_dot_param_3,
	.param .u64 triton_dot_param_4,
	.param .align 64 .b8 triton_dot_param_5[128],
	.param .u32 triton_dot_param_6,
	.param .u32 triton_dot_param_7,
	.param .u64 triton_dot_param_8,
	.param .u64 triton_dot_param_9,
	.param .align 64 .b8 triton_dot_param_10[128],
	.param .u32 triton_dot_param_11,
	.param .u32 triton_dot_param_12,
	.param .u64 triton_dot_param_13,
	.param .u64 triton_dot_param_14,
	.param .u64 .ptr .global .align 1 triton_dot_param_15,
	.param .u64 .ptr .global .align 1 triton_dot_param_16
)
.reqntid 128
{
	.reg .pred 	%p<13>;
	.reg .b32 	%r<54>;
	.reg .b64 	%rd<18>;

// %bb.0:
	mov.b64 	%rd3, triton_dot_param_0;
	mov.b64 	%rd4, triton_dot_param_10;
	cvta.param.u64 	%rd17, %rd4;
	mov.b64 	%rd5, triton_dot_param_5;
	cvta.param.u64 	%rd2, %rd5;
	cvta.param.u64 	%rd1, %rd3;
	mov.u32 	%r1, %tid.x;
	setp.lt.u32 	%p1, %r1, 32;
	mov.b32 	%r3, global_smem;
	// begin inline asm
	@%p1 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r3], 32;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r34, [global_smem];
	bar.sync 	0;
	// begin inline asm
	@%p1 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
	// end inline asm
	and.b32 	%r2, %r1, 127;
	setp.eq.b32 	%p2, %r2, 0;
	add.s32 	%r4, %r3, 8192;
	// begin inline asm
	@%p2 mbarrier.init.shared::cta.b64 [%r4], 1;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p2 mbarrier.arrive.expect_tx.shared.b64 _, [%r4], 8192;
	// end inline asm
	bar.sync 	0;
	elect.sync 	%r5|%p5, -1;
	setp.lt.u32 	%p6, %r2, 32;
	and.pred 	%p3, %p6, %p5;
	mov.b32 	%r16, 0;
	// begin inline asm
	@%p3 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r3], [%rd1, {%r16, %r16}], [%r4];
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	
{
	.reg .pred complete;
	waitLoop:
	mbarrier.try_wait.parity.shared.b64 complete, [%r4], %r16;
	@!complete bra.uni waitLoop;
}

	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p2 mbarrier.inval.shared::cta.b64 [%r4];
	// end inline asm
	add.s32 	%r15, %r3, 12288;
	// begin inline asm
	@%p2 mbarrier.init.shared::cta.b64 [%r15], 1;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p2 mbarrier.arrive.expect_tx.shared.b64 _, [%r15], 4096;
	// end inline asm
	// begin inline asm
	fence.proxy.async.shared::cta;
	// end inline asm
	bar.sync 	0;
	elect.sync 	%r6|%p7, -1;
	and.pred 	%p4, %p6, %p7;
	// begin inline asm
	@%p4 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r4], [%rd2, {%r16, %r16}], [%r15];
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	
{
	.reg .pred complete;
	waitLoop:
	mbarrier.try_wait.parity.shared.b64 complete, [%r15], %r16;
	@!complete bra.uni waitLoop;
}

	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p2 mbarrier.inval.shared::cta.b64 [%r15];
	// end inline asm
	shr.u32 	%r7, %r1, 5;
	shfl.sync.idx.b32 	%r8, %r7, 0, 31, -1;
	shl.b32 	%r9, %r8, 21;
	and.b32 	%r10, %r9, 6291456;
	add.s32 	%r33, %r10, %r34;
	mov.pred 	%p9, -1;
	// begin inline asm
	@%p9 tcgen05.st.sync.aligned.16x32bx2.x16.b32 [%r33 + 0], 16, {%r16, %r16, %r16, %r16, %r16, %r16, %r16, %r16, %r16, %r16, %r16, %r16, %r16, %r16, %r16, %r16};
	// end inline asm
	tcgen05.wait::st.sync.aligned;
	bar.sync 	0;
	bar.sync 	0;
	// begin inline asm
	@%p2 mbarrier.init.shared::cta.b64 [%r15], 1;
	// end inline asm
	setp.ne.b32 	%p8, %r8, 0;
	@%p8 bra 	$L__BB0_2;
// %bb.1:
	elect.sync 	%r12|%p10, -1;
	bfe.u32 	%r13, %r3, 4, 14;
	cvt.u64.u32 	%rd15, %r13;
	bfe.u32 	%r14, %r4, 4, 14;
	cvt.u64.u32 	%rd16, %r14;
	or.b64 	%rd6, %rd15, 4611756662049472512;
	or.b64 	%rd7, %rd16, 4611756662049472512;
	mov.b32 	%r11, 67633168;
	// begin inline asm
	@%p10 tcgen05.mma.cta_group::1.kind::f16 [ %r34 + 0 ], %rd6, %rd7, %r11, %p9;
	// end inline asm
	add.s64 	%rd8, %rd15, 4611756662049472514;
	add.s64 	%rd9, %rd16, 4611756662049472514;
	// begin inline asm
	@%p10 tcgen05.mma.cta_group::1.kind::f16 [ %r34 + 0 ], %rd8, %rd9, %r11, %p9;
	// end inline asm
	add.s64 	%rd10, %rd15, 4611756662049472516;
	add.s64 	%rd11, %rd16, 4611756662049472516;
	// begin inline asm
	@%p10 tcgen05.mma.cta_group::1.kind::f16 [ %r34 + 0 ], %rd10, %rd11, %r11, %p9;
	// end inline asm
	add.s64 	%rd12, %rd15, 4611756662049472518;
	add.s64 	%rd13, %rd16, 4611756662049472518;
	// begin inline asm
	@%p10 tcgen05.mma.cta_group::1.kind::f16 [ %r34 + 0 ], %rd12, %rd13, %r11, %p9;
	// end inline asm
	cvt.u64.u32 	%rd14, %r15;
	// begin inline asm
	@%p10 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%rd14];
	// end inline asm
$L__BB0_2:
	bar.sync 	0;
	// begin inline asm
	
{
	.reg .pred complete;
	waitLoop:
	mbarrier.try_wait.parity.shared.b64 complete, [%r15], %r16;
	@!complete bra.uni waitLoop;
}

	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p2 mbarrier.inval.shared::cta.b64 [%r15];
	// end inline asm
	// begin inline asm
	tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {%r17, %r18, %r19, %r20, %r21, %r22, %r23, %r24, %r25, %r26, %r27, %r28, %r29, %r30, %r31, %r32}, [%r33 + 0], 16;
	// end inline asm
	tcgen05.wait::ld.sync.aligned;
	shl.b32 	%r35, %r1, 7;
	and.b32 	%r36, %r35, 1920;
	shl.b32 	%r37, %r1, 6;
	and.b32 	%r38, %r37, 6144;
	shl.b32 	%r39, %r1, 4;
	and.b32 	%r40, %r39, 112;
	shl.b32 	%r41, %r1, 2;
	and.b32 	%r42, %r41, 64;
	or.b32 	%r43, %r38, %r40;
	xor.b32 	%r44, %r43, %r42;
	or.b32 	%r45, %r44, %r36;
	add.s32 	%r46, %r3, %r45;
	st.shared.v4.b32 	[%r46], {%r17, %r18, %r19, %r20};
	xor.b32 	%r47, %r45, 16;
	add.s32 	%r48, %r3, %r47;
	st.shared.v4.b32 	[%r48], {%r21, %r22, %r23, %r24};
	xor.b32 	%r49, %r45, 32;
	add.s32 	%r50, %r3, %r49;
	st.shared.v4.b32 	[%r50], {%r25, %r26, %r27, %r28};
	xor.b32 	%r51, %r45, 48;
	add.s32 	%r52, %r3, %r51;
	st.shared.v4.b32 	[%r52], {%r29, %r30, %r31, %r32};
	// begin inline asm
	fence.proxy.async.shared::cta;
	// end inline asm
	bar.sync 	0;
	elect.sync 	%r53|%p12, -1;
	and.pred 	%p11, %p6, %p12;
	// begin inline asm
	@%p11 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd17, {%r16, %r16}], [%r3];
	// end inline asm
	cp.async.bulk.commit_group;
	cp.async.bulk.wait_group.read 	0;
	bar.sync 	0;
	bar.sync 	0;
	// begin inline asm
	@%p1 tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r34, 32;
	// end inline asm
	ret;
                                        // -- End function
}

