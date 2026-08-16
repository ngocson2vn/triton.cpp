// -----// IR Dump After SCCP (sccp) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    tt.return
  }
}


// -----// IR Dump After TritonGPUAllocateWarpGroups (tritongpu-allocate-warp-groups) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    tt.return
  }
}


// -----// IR Dump After AllocateSharedMemory (allocate-shared-memory) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 16392 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    tt.return
  }
}


// -----// IR Dump After ConvertTritonGPUToLLVMDebug (convert-triton-gpu-to-llvm-debug) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 16392 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 32>} {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !tt.tensordesc<tensor<128x64xf16, #shared>>
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
    %9 = llvm.mlir.constant(16384 : i32) : i32
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
    %24 = llvm.mlir.constant(31 : i32) : i32
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
    %43 = llvm.mlir.constant(1 : i32) : i32
    %44 = llvm.and %41, %43 : i32
    %45 = llvm.icmp "eq" %44, %42 : i32
    %46 = llvm.mlir.constant(4096 : i32) : i32
    %47 = llvm.select %45, %42, %46 : i1, i32
    %48 = llvm.mlir.constant(0 : i32) : i32
    %49 = llvm.or disjoint %47, %48 : i32
    %50 = llvm.xor %37, %49 : i32
    %51 = llvm.mlir.constant(0 : i32) : i32
    %52 = llvm.mlir.constant(0 : i32) : i32
    %53 = llvm.mlir.constant(0 : i32) : i32
    %54 = llvm.or disjoint %52, %53 : i32
    %55 = llvm.xor %37, %54 : i32
    %56 = llvm.getelementptr %22[%50] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %57 = llvm.mlir.constant(0 : i32) : i32
    %58 = llvm.mlir.constant(0 : i32) : i32
    %59 = llvm.mlir.constant(0 : i32) : i32
    %60 = llvm.shl %36, %59 : i32
    %61 = llvm.or %58, %60 : i32
    %62 = llvm.mlir.constant(1 : i32) : i32
    %63 = llvm.shl %30, %62 : i32
    %64 = llvm.or %61, %63 : i32
    %65 = llvm.mlir.constant(0 : i32) : i32
    %66 = llvm.mlir.constant(1 : i32) : i32
    %67 = llvm.and %64, %66 : i32
    %68 = llvm.icmp "eq" %67, %65 : i32
    %69 = llvm.mlir.constant(64 : i32) : i32
    %70 = llvm.select %68, %65, %69 : i1, i32
    %71 = llvm.mlir.constant(0 : i32) : i32
    %72 = llvm.or disjoint %70, %71 : i32
    %73 = llvm.xor %57, %72 : i32
    %74 = llvm.mlir.constant(0 : i32) : i32
    %75 = llvm.mlir.constant(0 : i32) : i32
    %76 = llvm.mlir.constant(0 : i32) : i32
    %77 = llvm.or disjoint %75, %76 : i32
    %78 = llvm.xor %57, %77 : i32
    %79 = llvm.add %c0_i32, %78 : i32
    %80 = llvm.add %c0_i32, %73 : i32
    %81 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %34, %56, %arg0, %79, %80, %16 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %82 = llvm.mlir.constant(0 : i32) : i32
    %83 = llvm.mlir.constant(32 : i32) : i32
    %84 = llvm.icmp "ult" %25, %83 : i32
    %85 = llvm.and %28, %84 : i1
    %86 = llvm.mlir.constant(1 : i32) : i32
    %87 = llvm.add %82, %86 : i32
    %88 = llvm.mlir.constant(0 : i32) : i32
    %89 = llvm.mlir.constant(0 : i32) : i32
    %90 = llvm.mlir.constant(0 : i32) : i32
    %91 = llvm.shl %87, %90 : i32
    %92 = llvm.or %89, %91 : i32
    %93 = llvm.mlir.constant(0 : i32) : i32
    %94 = llvm.mlir.constant(1 : i32) : i32
    %95 = llvm.and %92, %94 : i32
    %96 = llvm.icmp "eq" %95, %93 : i32
    %97 = llvm.mlir.constant(4096 : i32) : i32
    %98 = llvm.select %96, %93, %97 : i1, i32
    %99 = llvm.mlir.constant(0 : i32) : i32
    %100 = llvm.or disjoint %98, %99 : i32
    %101 = llvm.xor %88, %100 : i32
    %102 = llvm.mlir.constant(0 : i32) : i32
    %103 = llvm.mlir.constant(0 : i32) : i32
    %104 = llvm.mlir.constant(0 : i32) : i32
    %105 = llvm.or disjoint %103, %104 : i32
    %106 = llvm.xor %88, %105 : i32
    %107 = llvm.getelementptr %22[%101] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %108 = llvm.mlir.constant(0 : i32) : i32
    %109 = llvm.mlir.constant(0 : i32) : i32
    %110 = llvm.mlir.constant(0 : i32) : i32
    %111 = llvm.shl %87, %110 : i32
    %112 = llvm.or %109, %111 : i32
    %113 = llvm.mlir.constant(1 : i32) : i32
    %114 = llvm.shl %30, %113 : i32
    %115 = llvm.or %112, %114 : i32
    %116 = llvm.mlir.constant(0 : i32) : i32
    %117 = llvm.mlir.constant(1 : i32) : i32
    %118 = llvm.and %115, %117 : i32
    %119 = llvm.icmp "eq" %118, %116 : i32
    %120 = llvm.mlir.constant(64 : i32) : i32
    %121 = llvm.select %119, %116, %120 : i1, i32
    %122 = llvm.mlir.constant(0 : i32) : i32
    %123 = llvm.or disjoint %121, %122 : i32
    %124 = llvm.xor %108, %123 : i32
    %125 = llvm.mlir.constant(0 : i32) : i32
    %126 = llvm.mlir.constant(0 : i32) : i32
    %127 = llvm.mlir.constant(0 : i32) : i32
    %128 = llvm.or disjoint %126, %127 : i32
    %129 = llvm.xor %108, %128 : i32
    %130 = llvm.add %c0_i32, %129 : i32
    %131 = llvm.add %c0_i32, %124 : i32
    %132 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %85, %107, %arg0, %130, %131, %16 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 16392 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 32>} {
    %0 = llvm.mlir.constant(64 : i32) : i32
    %1 = llvm.mlir.constant(4096 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(32 : i32) : i32
    %4 = llvm.mlir.constant(31 : i32) : i32
    %5 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %6 = llvm.mlir.constant(0 : i32) : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %7 = llvm.getelementptr %5[16384] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %thread_id_x = gpu.thread_id  x
    %8 = arith.index_cast %thread_id_x : index to i32
    %9 = llvm.and %8, %4 : i32
    %10 = nvvm.elect.sync -> i1
    %11 = llvm.and %true, %10 : i1
    %12 = nvg.cluster_id
    %13 = llvm.icmp "ult" %9, %3 : i32
    %14 = llvm.and %11, %13 : i1
    %15 = llvm.add %6, %6 : i32
    %16 = llvm.shl %15, %6 : i32
    %17 = llvm.or %6, %16 : i32
    %18 = llvm.and %17, %2 : i32
    %19 = llvm.icmp "eq" %18, %6 : i32
    %20 = llvm.select %19, %6, %1 : i1, i32
    %21 = llvm.or disjoint %20, %6 : i32
    %22 = llvm.xor %6, %21 : i32
    %23 = llvm.getelementptr %5[%22] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %24 = llvm.shl %15, %6 : i32
    %25 = llvm.or %6, %24 : i32
    %26 = llvm.shl %12, %2 : i32
    %27 = llvm.or %25, %26 : i32
    %28 = llvm.and %27, %2 : i32
    %29 = llvm.icmp "eq" %28, %6 : i32
    %30 = llvm.select %29, %6, %0 : i1, i32
    %31 = llvm.or disjoint %30, %6 : i32
    %32 = llvm.xor %6, %31 : i32
    %33 = llvm.xor %6, %6 : i32
    %34 = llvm.add %33, %c0_i32 : i32
    %35 = llvm.add %32, %c0_i32 : i32
    %36 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %14, %23, %arg0, %34, %35, %7 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %37 = llvm.icmp "ult" %9, %3 : i32
    %38 = llvm.and %11, %37 : i1
    %39 = llvm.add %6, %2 : i32
    %40 = llvm.shl %39, %6 : i32
    %41 = llvm.or %6, %40 : i32
    %42 = llvm.and %41, %2 : i32
    %43 = llvm.icmp "eq" %42, %6 : i32
    %44 = llvm.select %43, %6, %1 : i1, i32
    %45 = llvm.or disjoint %44, %6 : i32
    %46 = llvm.xor %6, %45 : i32
    %47 = llvm.getelementptr %5[%46] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %48 = llvm.shl %39, %6 : i32
    %49 = llvm.or %6, %48 : i32
    %50 = llvm.shl %12, %2 : i32
    %51 = llvm.or %49, %50 : i32
    %52 = llvm.and %51, %2 : i32
    %53 = llvm.icmp "eq" %52, %6 : i32
    %54 = llvm.select %53, %6, %0 : i1, i32
    %55 = llvm.or disjoint %54, %6 : i32
    %56 = llvm.xor %6, %55 : i32
    %57 = llvm.xor %6, %6 : i32
    %58 = llvm.add %57, %c0_i32 : i32
    %59 = llvm.add %56, %c0_i32 : i32
    %60 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %38, %47, %arg0, %58, %59, %7 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    llvm.return
  }
}


// -----// IR Dump After CSE (cse) //----- //
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 16392 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 32>} {
    %0 = llvm.mlir.constant(64 : i32) : i32
    %1 = llvm.mlir.constant(4096 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(32 : i32) : i32
    %4 = llvm.mlir.constant(31 : i32) : i32
    %5 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %6 = llvm.mlir.constant(0 : i32) : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %7 = llvm.getelementptr %5[16384] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %thread_id_x = gpu.thread_id  x
    %8 = arith.index_cast %thread_id_x : index to i32
    %9 = llvm.and %8, %4 : i32
    %10 = nvvm.elect.sync -> i1
    %11 = llvm.and %true, %10 : i1
    %12 = nvg.cluster_id
    %13 = llvm.icmp "ult" %9, %3 : i32
    %14 = llvm.and %11, %13 : i1
    %15 = llvm.add %6, %6 : i32
    %16 = llvm.shl %15, %6 : i32
    %17 = llvm.or %6, %16 : i32
    %18 = llvm.and %17, %2 : i32
    %19 = llvm.icmp "eq" %18, %6 : i32
    %20 = llvm.select %19, %6, %1 : i1, i32
    %21 = llvm.or disjoint %20, %6 : i32
    %22 = llvm.xor %6, %21 : i32
    %23 = llvm.getelementptr %5[%22] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %24 = llvm.shl %12, %2 : i32
    %25 = llvm.or %17, %24 : i32
    %26 = llvm.and %25, %2 : i32
    %27 = llvm.icmp "eq" %26, %6 : i32
    %28 = llvm.select %27, %6, %0 : i1, i32
    %29 = llvm.or disjoint %28, %6 : i32
    %30 = llvm.xor %6, %29 : i32
    %31 = llvm.xor %6, %6 : i32
    %32 = llvm.add %31, %c0_i32 : i32
    %33 = llvm.add %30, %c0_i32 : i32
    %34 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %14, %23, %arg0, %32, %33, %7 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %35 = llvm.add %6, %2 : i32
    %36 = llvm.shl %35, %6 : i32
    %37 = llvm.or %6, %36 : i32
    %38 = llvm.and %37, %2 : i32
    %39 = llvm.icmp "eq" %38, %6 : i32
    %40 = llvm.select %39, %6, %1 : i1, i32
    %41 = llvm.or disjoint %40, %6 : i32
    %42 = llvm.xor %6, %41 : i32
    %43 = llvm.getelementptr %5[%42] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %44 = llvm.or %37, %24 : i32
    %45 = llvm.and %44, %2 : i32
    %46 = llvm.icmp "eq" %45, %6 : i32
    %47 = llvm.select %46, %6, %0 : i1, i32
    %48 = llvm.or disjoint %47, %6 : i32
    %49 = llvm.xor %6, %48 : i32
    %50 = llvm.add %49, %c0_i32 : i32
    %51 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %14, %43, %arg0, %32, %50, %7 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    llvm.return
  }
}


