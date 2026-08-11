// -----// IR Dump After SCCP (sccp) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}


// -----// IR Dump After TritonGPUAllocateWarpGroups (tritongpu-allocate-warp-groups) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}


// -----// IR Dump After AllocateSharedMemory (allocate-shared-memory) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8200 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<64x64xf16, #shared>>) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}


// -----// IR Dump After ConvertTritonGPUToLLVMDebug (convert-triton-gpu-to-llvm-debug) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8200 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !tt.tensordesc<tensor<64x64xf16, #shared>>
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %3 = llvm.getelementptr %2[%1] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.undef : !llvm.struct<(ptr<3>, i32, i32)>
    %6 = llvm.insertvalue %3, %5[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %7 = llvm.insertvalue %4, %6[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %8 = llvm.insertvalue %4, %7[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    %9 = llvm.mlir.constant(8192 : i32) : i32
    %10 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %11 = llvm.getelementptr %10[%9] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %12 = llvm.mlir.constant(0 : i32) : i32
    %13 = llvm.mlir.undef : !llvm.struct<(ptr<3>, i32)>
    %14 = llvm.insertvalue %11, %13[0] : !llvm.struct<(ptr<3>, i32)> 
    %15 = llvm.insertvalue %12, %14[1] : !llvm.struct<(ptr<3>, i32)> 
    %16 = llvm.extractvalue %15[0] : !llvm.struct<(ptr<3>, i32)> 
    %17 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<3>, i32)> 
    %18 = llvm.extractvalue %8[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %19 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %20 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    %21 = llvm.mlir.constant(0 : i32) : i32
    %22 = llvm.getelementptr %18[%21] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %thread_id_x = gpu.thread_id  x
    %23 = arith.index_cast %thread_id_x : index to i32
    %24 = llvm.mlir.constant(127 : i32) : i32
    %25 = llvm.and %23, %24 : i32
    %26 = nvg.warp_id
    %27 = nvvm.elect.sync -> i1
    %28 = llvm.and %true, %27 : i1
    %29 = llvm.mlir.constant(0 : i32) : i32
    %30 = nvg.cluster_id
    %31 = llvm.mlir.constant(0 : i32) : i32
    %32 = llvm.mlir.constant(32 : i32) : i32
    %33 = llvm.icmp "ult" %25, %32 : i32
    %34 = llvm.and %28, %33 : i1
    %35 = llvm.mlir.constant(0 : i32) : i32
    %36 = llvm.add %31, %35 : i32
    %37 = llvm.mlir.constant(0 : i32) : i32
    %38 = llvm.mlir.constant(0 : i32) : i32
    %39 = llvm.mlir.constant(0 : i32) : i32
    %40 = llvm.shl %36, %39 : i32
    %41 = llvm.or %38, %40 : i32
    %42 = llvm.mlir.constant(0 : i32) : i32
    %43 = llvm.mlir.constant(0 : i32) : i32
    %44 = llvm.mlir.constant(0 : i32) : i32
    %45 = llvm.or disjoint %43, %44 : i32
    %46 = llvm.xor %37, %45 : i32
    %47 = llvm.mlir.constant(0 : i32) : i32
    %48 = llvm.mlir.constant(0 : i32) : i32
    %49 = llvm.mlir.constant(0 : i32) : i32
    %50 = llvm.or disjoint %48, %49 : i32
    %51 = llvm.xor %37, %50 : i32
    %52 = llvm.getelementptr %22[%46] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %53 = llvm.mlir.constant(0 : i32) : i32
    %54 = llvm.mlir.constant(0 : i32) : i32
    %55 = llvm.mlir.constant(0 : i32) : i32
    %56 = llvm.shl %36, %55 : i32
    %57 = llvm.or %54, %56 : i32
    %58 = llvm.mlir.constant(0 : i32) : i32
    %59 = llvm.shl %30, %58 : i32
    %60 = llvm.or %57, %59 : i32
    %61 = llvm.mlir.constant(0 : i32) : i32
    %62 = llvm.mlir.constant(0 : i32) : i32
    %63 = llvm.mlir.constant(0 : i32) : i32
    %64 = llvm.or disjoint %62, %63 : i32
    %65 = llvm.xor %53, %64 : i32
    %66 = llvm.mlir.constant(0 : i32) : i32
    %67 = llvm.mlir.constant(0 : i32) : i32
    %68 = llvm.mlir.constant(0 : i32) : i32
    %69 = llvm.or disjoint %67, %68 : i32
    %70 = llvm.xor %53, %69 : i32
    %71 = llvm.add %c0_i32, %70 : i32
    %72 = llvm.add %c0_i32, %65 : i32
    %73 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %34, %52, %arg0, %71, %72, %16 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8200 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = llvm.mlir.constant(32 : i32) : i32
    %1 = llvm.mlir.constant(127 : i32) : i32
    %2 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %3 = llvm.mlir.constant(0 : i32) : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %4 = llvm.getelementptr %2[8192] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %thread_id_x = gpu.thread_id  x
    %5 = arith.index_cast %thread_id_x : index to i32
    %6 = llvm.and %5, %1 : i32
    %7 = nvvm.elect.sync -> i1
    %8 = llvm.and %true, %7 : i1
    %9 = llvm.icmp "ult" %6, %0 : i32
    %10 = llvm.and %8, %9 : i1
    %11 = llvm.xor %3, %3 : i32
    %12 = llvm.getelementptr %2[%11] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %13 = llvm.xor %3, %3 : i32
    %14 = llvm.xor %3, %3 : i32
    %15 = llvm.add %14, %c0_i32 : i32
    %16 = llvm.add %13, %c0_i32 : i32
    %17 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %10, %12, %arg0, %15, %16, %4 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    llvm.return
  }
}


// -----// IR Dump After CSE (cse) //----- //
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8200 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = llvm.mlir.constant(32 : i32) : i32
    %1 = llvm.mlir.constant(127 : i32) : i32
    %2 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %3 = llvm.mlir.constant(0 : i32) : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %4 = llvm.getelementptr %2[8192] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %thread_id_x = gpu.thread_id  x
    %5 = arith.index_cast %thread_id_x : index to i32
    %6 = llvm.and %5, %1 : i32
    %7 = nvvm.elect.sync -> i1
    %8 = llvm.and %true, %7 : i1
    %9 = llvm.icmp "ult" %6, %0 : i32
    %10 = llvm.and %8, %9 : i1
    %11 = llvm.xor %3, %3 : i32
    %12 = llvm.getelementptr %2[%11] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %13 = llvm.add %11, %c0_i32 : i32
    %14 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %10, %12, %arg0, %13, %13, %4 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    llvm.return
  }
}


