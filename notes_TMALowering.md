# TMA Lowering
```MLIR
// => BEGIN lowerTMALoad
tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
  %c0_i32 = arith.constant 0 : i32
  %true = arith.constant true
  %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>>
  %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %1 = ttg.local_alloc %0 : (tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
  %2 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %3 = ttg.local_alloc %2 : (tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
  %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> -> !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>
  %result = ttng.tmem_alloc : () -> !ttg.memdesc<64x32xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>, #ttng.tensor_memory, mutable>
  ttng.tmem_store %cst, %result, %true : tensor<64x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>> -> !ttg.memdesc<64x32xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>, #ttng.tensor_memory, mutable>
  ttng.tc_gen5_mma %1, %4, %result, %true, %true : !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<64x32xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>, #ttng.tensor_memory, mutable>
  %result_0 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>>
  %5 = ttg.convert_layout %result_0 : tensor<64x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>> -> tensor<64x32xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %6 = ttg.local_alloc %5 : (tensor<64x32xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x32xf32, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>, #ttg.shared_memory>
  ttng.fence_async_shared {bCluster = false}
  ttng.async_tma_copy_local_to_global %arg10[%c0_i32, %c0_i32] %6 : !tt.tensordesc<tensor<64x32xf32, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>>>, !ttg.memdesc<64x32xf32, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>, #ttg.shared_memory>
  ttng.async_tma_store_wait {pendings = 0 : i32}
  tt.return
}

op: %4 = tt.descriptor_load %arg5[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>

// <= END lowerTMALoad
tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
  %c0_i32 = arith.constant 0 : i32
  %true = arith.constant true
  %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>>
  %0 = tt.descriptor_load %arg0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %1 = ttg.local_alloc %0 : (tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
  %2 = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
  %3 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
  ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
  %true_0 = arith.constant true
  ttng.barrier_expect %3, 4096, %true_0 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
  ttng.async_tma_copy_global_to_local %arg5[%c0_i32, %c0_i32] %2, %3, %true_0 : !tt.tensordesc<tensor<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
  %c0_i32_1 = arith.constant 0 : i32
  ttng.wait_barrier %3, %c0_i32_1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
  ttng.inval_barrier %3 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
  %4 = ttg.memdesc_trans %2 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
  %result = ttng.tmem_alloc : () -> !ttg.memdesc<64x32xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>, #ttng.tensor_memory, mutable>
  ttng.tmem_store %cst, %result, %true : tensor<64x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>> -> !ttg.memdesc<64x32xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>, #ttng.tensor_memory, mutable>
  ttng.tc_gen5_mma %1, %4, %result, %true, %true : !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x32xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>, #ttng.tensor_memory, mutable>
  %result_2 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>>
  %5 = ttg.convert_layout %result_2 : tensor<64x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>> -> tensor<64x32xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %6 = ttg.local_alloc %5 : (tensor<64x32xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x32xf32, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>, #ttg.shared_memory>
  ttng.fence_async_shared {bCluster = false}
  ttng.async_tma_copy_local_to_global %arg10[%c0_i32, %c0_i32] %6 : !tt.tensordesc<tensor<64x32xf32, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>>>, !ttg.memdesc<64x32xf32, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>, #ttg.shared_memory>
  ttng.async_tma_store_wait {pendings = 0 : i32}
  tt.return
}

// <= END lowerTMALoad
tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<64x32xf32, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64) attributes {noinline = false} {
  %c0_i32 = arith.constant 0 : i32
  %true = arith.constant true
  %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>>
  
  %0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
  %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
  ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
  %true_0 = arith.constant true
  ttng.barrier_expect %1, 8192, %true_0 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
  ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true_0 : !tt.tensordesc<tensor<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
  %c0_i32_1 = arith.constant 0 : i32
  ttng.wait_barrier %1, %c0_i32_1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
  ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>

  %2 = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
  %3 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
  ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
  %true_2 = arith.constant true
  ttng.barrier_expect %3, 4096, %true_2 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
  ttng.async_tma_copy_global_to_local %arg5[%c0_i32, %c0_i32] %2, %3, %true_2 : !tt.tensordesc<tensor<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
  %c0_i32_3 = arith.constant 0 : i32
  ttng.wait_barrier %3, %c0_i32_3 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
  ttng.inval_barrier %3 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>

  %4 = ttg.memdesc_trans %2 {order = array<i32: 1, 0>} : !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory, mutable>
  %result = ttng.tmem_alloc : () -> !ttg.memdesc<64x32xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>, #ttng.tensor_memory, mutable>
  ttng.tmem_store %cst, %result, %true : tensor<64x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>> -> !ttg.memdesc<64x32xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>, #ttng.tensor_memory, mutable>
  ttng.tc_gen5_mma %0, %4, %result, %true, %true : !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x32xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>, #ttng.tensor_memory, mutable>
  %result_4 = ttng.tmem_load %result : !ttg.memdesc<64x32xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>>
  %5 = ttg.convert_layout %result_4 : tensor<64x32xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = []}>> -> tensor<64x32xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %6 = ttg.local_alloc %5 : (tensor<64x32xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x32xf32, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>, #ttg.shared_memory>
  ttng.fence_async_shared {bCluster = false}
  ttng.async_tma_copy_local_to_global %arg10[%c0_i32, %c0_i32] %6 : !tt.tensordesc<tensor<64x32xf32, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>>>, !ttg.memdesc<64x32xf32, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>, #ttg.shared_memory>
  ttng.async_tma_store_wait {pendings = 0 : i32}
  tt.return
}
```