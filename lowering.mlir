// -----// IR Dump After ConvertTritonToTritonGPU (convert-triton-to-tritongpu) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<128xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<128xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    %13 = arith.addf %9, %12 : tensor<128xf32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %15, %13, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----// IR Dump After TritonGPUCoalesce (tritongpu-coalesce) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<128xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<128xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %9 = ttg.convert_layout %8 : tensor<128x!tt.ptr<f32>, #blocked> -> tensor<128x!tt.ptr<f32>, #blocked>
    %10 = ttg.convert_layout %6 : tensor<128xi1, #blocked> -> tensor<128xi1, #blocked>
    %11 = tt.load %9, %10 : tensor<128x!tt.ptr<f32>, #blocked>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %13 = tt.addptr %12, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %14 = ttg.convert_layout %13 : tensor<128x!tt.ptr<f32>, #blocked> -> tensor<128x!tt.ptr<f32>, #blocked>
    %15 = ttg.convert_layout %6 : tensor<128xi1, #blocked> -> tensor<128xi1, #blocked>
    %16 = tt.load %14, %15 : tensor<128x!tt.ptr<f32>, #blocked>
    %17 = arith.addf %11, %16 : tensor<128xf32, #blocked>
    %18 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %19 = tt.addptr %18, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %20 = ttg.convert_layout %19 : tensor<128x!tt.ptr<f32>, #blocked> -> tensor<128x!tt.ptr<f32>, #blocked>
    %21 = ttg.convert_layout %17 : tensor<128xf32, #blocked> -> tensor<128xf32, #blocked>
    %22 = ttg.convert_layout %6 : tensor<128xi1, #blocked> -> tensor<128xi1, #blocked>
    tt.store %20, %21, %22 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----// IR Dump After TritonGPURemoveLayoutConversions (tritongpu-remove-layout-conversions) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<128xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<128xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    %13 = arith.addf %9, %12 : tensor<128xf32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %15, %13, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----// IR Dump After SCCP (sccp) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<128xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<128xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    %13 = arith.addf %9, %12 : tensor<128xf32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %15, %13, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----// IR Dump After TritonGPUAllocateWarpGroups (tritongpu-allocate-warp-groups) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<128xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<128xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    %13 = arith.addf %9, %12 : tensor<128xf32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %15, %13, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----// IR Dump After AllocateSharedMemory (allocate-shared-memory) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<128xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<128xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    %13 = arith.addf %9, %12 : tensor<128xf32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %15, %13, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----// IR Dump After TritonTensorMemoryAllocationPass (triton-tensor-memory-allocation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<128xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<128xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    %13 = arith.addf %9, %12 : tensor<128xf32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %15, %13, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----// IR Dump After TritonGPUGlobalScratchAllocationPass (tritongpu-global-scratch-memory-allocation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<128xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<128xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    %13 = arith.addf %9, %12 : tensor<128xf32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %15, %13, %6 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----// IR Dump After ConvertTritonGPUToLLVM (convert-triton-gpu-to-llvm) //----- //
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @add_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: i32, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %0 = builtin.unrealized_conversion_cast %arg2 : !llvm.ptr<1> to !tt.ptr<f32>
    %1 = builtin.unrealized_conversion_cast %arg1 : !llvm.ptr<1> to !tt.ptr<f32>
    %2 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<1> to !tt.ptr<f32>
    %3 = llvm.mlir.constant(128 : i32) : i32
    %4 = nvvm.read.ptx.sreg.ctaid.x : i32
    %5 = llvm.mul %4, %3 : i32
    %6 = llvm.mlir.constant(0 : index) : i32
    %7 = nvvm.read.ptx.sreg.tid.x : i32
    %8 = llvm.mlir.constant(127 : i32) : i32
    %9 = llvm.and %7, %8 : i32
    %10 = llvm.mlir.constant(32 : i32) : i32
    %11 = llvm.urem %9, %10 : i32
    %12 = llvm.udiv %9, %10 : i32
    %13 = llvm.mlir.constant(0 : i32) : i32
    %14 = nvg.cluster_id
    %15 = llvm.mlir.constant(0 : i32) : i32
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.constant(0 : i32) : i32
    %18 = llvm.mlir.constant(0 : i32) : i32
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.shl %11, %19 : i32
    %21 = llvm.or %18, %20 : i32
    %22 = llvm.mlir.constant(5 : i32) : i32
    %23 = llvm.shl %12, %22 : i32
    %24 = llvm.or %21, %23 : i32
    %25 = llvm.mlir.constant(7 : i32) : i32
    %26 = llvm.shl %13, %25 : i32
    %27 = llvm.or %24, %26 : i32
    %28 = llvm.mlir.constant(127 : i32) : i32
    %29 = llvm.and %27, %28 : i32
    %30 = llvm.mlir.constant(0 : i32) : i32
    %31 = llvm.lshr %29, %30 : i32
    %32 = llvm.mlir.constant(0 : i32) : i32
    %33 = llvm.mlir.constant(0 : i32) : i32
    %34 = llvm.or disjoint %31, %33 : i32
    %35 = llvm.xor %17, %34 : i32
    %36 = llvm.mlir.constant(0 : i32) : i32
    %37 = llvm.xor %35, %36 : i32
    %38 = llvm.add %37, %6 : i32
    %39 = llvm.mlir.undef : !llvm.struct<(i32)>
    %40 = llvm.insertvalue %38, %39[0] : !llvm.struct<(i32)> 
    %41 = llvm.bitcast %5 : i32 to i32
    %42 = llvm.mlir.undef : !llvm.struct<(i32)>
    %43 = llvm.insertvalue %41, %42[0] : !llvm.struct<(i32)> 
    %44 = llvm.extractvalue %43[0] : !llvm.struct<(i32)> 
    %45 = llvm.extractvalue %40[0] : !llvm.struct<(i32)> 
    %46 = llvm.add %44, %45 : i32
    %47 = llvm.mlir.undef : !llvm.struct<(i32)>
    %48 = llvm.insertvalue %46, %47[0] : !llvm.struct<(i32)> 
    %49 = llvm.bitcast %arg3 : i32 to i32
    %50 = llvm.mlir.undef : !llvm.struct<(i32)>
    %51 = llvm.insertvalue %49, %50[0] : !llvm.struct<(i32)> 
    %52 = llvm.extractvalue %48[0] : !llvm.struct<(i32)> 
    %53 = llvm.extractvalue %51[0] : !llvm.struct<(i32)> 
    %54 = llvm.icmp "slt" %52, %53 : i32
    %55 = llvm.mlir.undef : !llvm.struct<(i1)>
    %56 = llvm.insertvalue %54, %55[0] : !llvm.struct<(i1)> 
    %57 = llvm.bitcast %arg0 : !llvm.ptr<1> to !llvm.ptr<1>
    %58 = llvm.mlir.undef : !llvm.struct<(ptr<1>)>
    %59 = llvm.insertvalue %57, %58[0] : !llvm.struct<(ptr<1>)> 
    %60 = llvm.extractvalue %59[0] : !llvm.struct<(ptr<1>)> 
    %61 = llvm.extractvalue %48[0] : !llvm.struct<(i32)> 
    %62 = llvm.getelementptr %60[%61] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %63 = llvm.mlir.undef : !llvm.struct<(ptr<1>)>
    %64 = llvm.insertvalue %62, %63[0] : !llvm.struct<(ptr<1>)> 
    %65 = llvm.extractvalue %64[0] : !llvm.struct<(ptr<1>)> 
    %66 = llvm.extractvalue %56[0] : !llvm.struct<(i1)> 
    %67 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b" %65, %66 : (!llvm.ptr<1>, i1) -> i32
    %68 = llvm.bitcast %67 : i32 to vector<1xf32>
    %69 = llvm.mlir.constant(0 : index) : i32
    %70 = llvm.extractelement %68[%69 : i32] : vector<1xf32>
    %71 = llvm.mlir.undef : !llvm.struct<(f32)>
    %72 = llvm.insertvalue %70, %71[0] : !llvm.struct<(f32)> 
    %73 = llvm.bitcast %arg1 : !llvm.ptr<1> to !llvm.ptr<1>
    %74 = llvm.mlir.undef : !llvm.struct<(ptr<1>)>
    %75 = llvm.insertvalue %73, %74[0] : !llvm.struct<(ptr<1>)> 
    %76 = llvm.extractvalue %75[0] : !llvm.struct<(ptr<1>)> 
    %77 = llvm.extractvalue %48[0] : !llvm.struct<(i32)> 
    %78 = llvm.getelementptr %76[%77] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %79 = llvm.mlir.undef : !llvm.struct<(ptr<1>)>
    %80 = llvm.insertvalue %78, %79[0] : !llvm.struct<(ptr<1>)> 
    %81 = llvm.extractvalue %80[0] : !llvm.struct<(ptr<1>)> 
    %82 = llvm.extractvalue %56[0] : !llvm.struct<(i1)> 
    %83 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b" %81, %82 : (!llvm.ptr<1>, i1) -> i32
    %84 = llvm.bitcast %83 : i32 to vector<1xf32>
    %85 = llvm.mlir.constant(0 : index) : i32
    %86 = llvm.extractelement %84[%85 : i32] : vector<1xf32>
    %87 = llvm.mlir.undef : !llvm.struct<(f32)>
    %88 = llvm.insertvalue %86, %87[0] : !llvm.struct<(f32)> 
    %89 = llvm.extractvalue %72[0] : !llvm.struct<(f32)> 
    %90 = llvm.extractvalue %88[0] : !llvm.struct<(f32)> 
    %91 = llvm.fadd %89, %90 : f32
    %92 = llvm.mlir.undef : !llvm.struct<(f32)>
    %93 = llvm.insertvalue %91, %92[0] : !llvm.struct<(f32)> 
    %94 = llvm.bitcast %arg2 : !llvm.ptr<1> to !llvm.ptr<1>
    %95 = llvm.mlir.undef : !llvm.struct<(ptr<1>)>
    %96 = llvm.insertvalue %94, %95[0] : !llvm.struct<(ptr<1>)> 
    %97 = llvm.extractvalue %96[0] : !llvm.struct<(ptr<1>)> 
    %98 = llvm.extractvalue %48[0] : !llvm.struct<(i32)> 
    %99 = llvm.getelementptr %97[%98] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %100 = llvm.mlir.undef : !llvm.struct<(ptr<1>)>
    %101 = llvm.insertvalue %99, %100[0] : !llvm.struct<(ptr<1>)> 
    %102 = llvm.extractvalue %101[0] : !llvm.struct<(ptr<1>)> 
    %103 = llvm.extractvalue %93[0] : !llvm.struct<(f32)> 
    %104 = llvm.extractvalue %56[0] : !llvm.struct<(i1)> 
    %105 = llvm.mlir.constant(0 : i32) : i32
    %106 = nvvm.read.ptx.sreg.tid.x : i32
    %107 = llvm.mlir.constant(127 : i32) : i32
    %108 = llvm.and %106, %107 : i32
    %109 = llvm.mlir.constant(32 : i32) : i32
    %110 = llvm.urem %108, %109 : i32
    %111 = llvm.udiv %108, %109 : i32
    %112 = llvm.mlir.undef : vector<1xf32>
    %113 = llvm.bitcast %103 : f32 to f32
    %114 = llvm.mlir.constant(0 : i32) : i32
    %115 = llvm.insertelement %113, %112[%114 : i32] : vector<1xf32>
    %116 = llvm.bitcast %115 : vector<1xf32> to i32
    %117 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b" %116, %102, %104 : (i32, !llvm.ptr<1>, i1) -> !llvm.void
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @add_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: i32, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %0 = llvm.mlir.undef : vector<1xf32>
    %1 = llvm.mlir.constant(5 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(32 : i32) : i32
    %4 = llvm.mlir.constant(127 : i32) : i32
    %5 = llvm.mlir.constant(0 : index) : i32
    %6 = llvm.mlir.constant(128 : i32) : i32
    %7 = nvvm.read.ptx.sreg.ctaid.x : i32
    %8 = llvm.mul %7, %6 : i32
    %9 = nvvm.read.ptx.sreg.tid.x : i32
    %10 = llvm.and %9, %4 : i32
    %11 = llvm.urem %10, %3 : i32
    %12 = llvm.udiv %10, %3 : i32
    %13 = llvm.shl %11, %2 : i32
    %14 = llvm.or %2, %13 : i32
    %15 = llvm.shl %12, %1 : i32
    %16 = llvm.or %14, %15 : i32
    %17 = llvm.or %16, %2 : i32
    %18 = llvm.and %17, %4 : i32
    %19 = llvm.lshr %18, %2 : i32
    %20 = llvm.or disjoint %19, %2 : i32
    %21 = llvm.xor %2, %20 : i32
    %22 = llvm.xor %21, %2 : i32
    %23 = llvm.add %22, %5 : i32
    %24 = llvm.add %8, %23 : i32
    %25 = llvm.icmp "slt" %24, %arg3 : i32
    %26 = llvm.getelementptr %arg0[%24] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %27 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b" %26, %25 : (!llvm.ptr<1>, i1) -> i32
    %28 = llvm.bitcast %27 : i32 to vector<1xf32>
    %29 = llvm.extractelement %28[%5 : i32] : vector<1xf32>
    %30 = llvm.getelementptr %arg1[%24] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %31 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b" %30, %25 : (!llvm.ptr<1>, i1) -> i32
    %32 = llvm.bitcast %31 : i32 to vector<1xf32>
    %33 = llvm.extractelement %32[%5 : i32] : vector<1xf32>
    %34 = llvm.fadd %29, %33 : f32
    %35 = llvm.getelementptr %arg2[%24] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %36 = llvm.insertelement %34, %0[%2 : i32] : vector<1xf32>
    %37 = llvm.bitcast %36 : vector<1xf32> to i32
    %38 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b" %37, %35, %25 : (i32, !llvm.ptr<1>, i1) -> !llvm.void
    llvm.return
  }
}


