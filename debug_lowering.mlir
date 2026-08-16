// -----// IR Dump After SCCP (sccp) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<256x128xf16, #shared>>) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<256x128xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    tt.return
  }
}


// -----// IR Dump After TritonGPUAllocateWarpGroups (tritongpu-allocate-warp-groups) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<256x128xf16, #shared>>) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<256x128xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    tt.return
  }
}


// -----// IR Dump After AllocateSharedMemory (allocate-shared-memory) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<256x128xf16, #shared>>) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<256x128xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    tt.return
  }
}


// -----// IR Dump After ConvertTritonGPUToLLVMDebug (convert-triton-gpu-to-llvm-debug) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !tt.tensordesc<tensor<256x128xf16, #shared>>
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
    %9 = llvm.mlir.constant(65536 : i32) : i32
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
    %31 = llvm.mlir.constant(128 : i32) : i32
    %32 = llvm.icmp "ult" %25, %31 : i32
    %33 = llvm.and %28, %32 : i1
    %34 = llvm.mlir.constant(0 : i32) : i32
    %35 = llvm.add %26, %34 : i32
    %36 = llvm.mlir.constant(0 : i32) : i32
    %37 = llvm.mlir.constant(0 : i32) : i32
    %38 = llvm.mlir.constant(0 : i32) : i32
    %39 = llvm.shl %35, %38 : i32
    %40 = llvm.or %37, %39 : i32
    %41 = llvm.mlir.constant(7 : i32) : i32
    %42 = llvm.and %40, %41 : i32
    %43 = llvm.mlir.constant(12 : i32) : i32
    %44 = llvm.shl %42, %43 : i32
    %45 = llvm.mlir.constant(0 : i32) : i32
    %46 = llvm.mlir.constant(0 : i32) : i32
    %47 = llvm.or disjoint %44, %46 : i32
    %48 = llvm.xor %36, %47 : i32
    %49 = llvm.mlir.constant(0 : i32) : i32
    %50 = llvm.mlir.constant(0 : i32) : i32
    %51 = llvm.mlir.constant(0 : i32) : i32
    %52 = llvm.or disjoint %50, %51 : i32
    %53 = llvm.xor %36, %52 : i32
    %54 = llvm.getelementptr %22[%48] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %55 = llvm.mlir.constant(0 : i32) : i32
    %56 = llvm.mlir.constant(0 : i32) : i32
    %57 = llvm.mlir.constant(0 : i32) : i32
    %58 = llvm.shl %35, %57 : i32
    %59 = llvm.or %56, %58 : i32
    %60 = llvm.mlir.constant(3 : i32) : i32
    %61 = llvm.shl %30, %60 : i32
    %62 = llvm.or %59, %61 : i32
    %63 = llvm.mlir.constant(3 : i32) : i32
    %64 = llvm.and %62, %63 : i32
    %65 = llvm.mlir.constant(6 : i32) : i32
    %66 = llvm.shl %64, %65 : i32
    %67 = llvm.mlir.constant(0 : i32) : i32
    %68 = llvm.mlir.constant(0 : i32) : i32
    %69 = llvm.or disjoint %66, %68 : i32
    %70 = llvm.xor %55, %69 : i32
    %71 = llvm.mlir.constant(0 : i32) : i32
    %72 = llvm.mlir.constant(4 : i32) : i32
    %73 = llvm.and %62, %72 : i32
    %74 = llvm.icmp "eq" %73, %71 : i32
    %75 = llvm.mlir.constant(64 : i32) : i32
    %76 = llvm.select %74, %71, %75 : i1, i32
    %77 = llvm.mlir.constant(0 : i32) : i32
    %78 = llvm.or disjoint %76, %77 : i32
    %79 = llvm.xor %55, %78 : i32
    %80 = llvm.add %c0_i32, %79 : i32
    %81 = llvm.add %c0_i32, %70 : i32
    %82 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %33, %54, %arg0, %80, %81, %16 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %83 = llvm.mlir.constant(128 : i32) : i32
    %84 = llvm.icmp "ult" %25, %83 : i32
    %85 = llvm.and %28, %84 : i1
    %86 = llvm.mlir.constant(4 : i32) : i32
    %87 = llvm.add %26, %86 : i32
    %88 = llvm.mlir.constant(0 : i32) : i32
    %89 = llvm.mlir.constant(0 : i32) : i32
    %90 = llvm.mlir.constant(0 : i32) : i32
    %91 = llvm.shl %87, %90 : i32
    %92 = llvm.or %89, %91 : i32
    %93 = llvm.mlir.constant(7 : i32) : i32
    %94 = llvm.and %92, %93 : i32
    %95 = llvm.mlir.constant(12 : i32) : i32
    %96 = llvm.shl %94, %95 : i32
    %97 = llvm.mlir.constant(0 : i32) : i32
    %98 = llvm.mlir.constant(0 : i32) : i32
    %99 = llvm.or disjoint %96, %98 : i32
    %100 = llvm.xor %88, %99 : i32
    %101 = llvm.mlir.constant(0 : i32) : i32
    %102 = llvm.mlir.constant(0 : i32) : i32
    %103 = llvm.mlir.constant(0 : i32) : i32
    %104 = llvm.or disjoint %102, %103 : i32
    %105 = llvm.xor %88, %104 : i32
    %106 = llvm.getelementptr %22[%100] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %107 = llvm.mlir.constant(0 : i32) : i32
    %108 = llvm.mlir.constant(0 : i32) : i32
    %109 = llvm.mlir.constant(0 : i32) : i32
    %110 = llvm.shl %87, %109 : i32
    %111 = llvm.or %108, %110 : i32
    %112 = llvm.mlir.constant(3 : i32) : i32
    %113 = llvm.shl %30, %112 : i32
    %114 = llvm.or %111, %113 : i32
    %115 = llvm.mlir.constant(3 : i32) : i32
    %116 = llvm.and %114, %115 : i32
    %117 = llvm.mlir.constant(6 : i32) : i32
    %118 = llvm.shl %116, %117 : i32
    %119 = llvm.mlir.constant(0 : i32) : i32
    %120 = llvm.mlir.constant(0 : i32) : i32
    %121 = llvm.or disjoint %118, %120 : i32
    %122 = llvm.xor %107, %121 : i32
    %123 = llvm.mlir.constant(0 : i32) : i32
    %124 = llvm.mlir.constant(4 : i32) : i32
    %125 = llvm.and %114, %124 : i32
    %126 = llvm.icmp "eq" %125, %123 : i32
    %127 = llvm.mlir.constant(64 : i32) : i32
    %128 = llvm.select %126, %123, %127 : i1, i32
    %129 = llvm.mlir.constant(0 : i32) : i32
    %130 = llvm.or disjoint %128, %129 : i32
    %131 = llvm.xor %107, %130 : i32
    %132 = llvm.add %c0_i32, %131 : i32
    %133 = llvm.add %c0_i32, %122 : i32
    %134 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %85, %106, %arg0, %132, %133, %16 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = llvm.mlir.constant(64 : i32) : i32
    %1 = llvm.mlir.constant(4 : i32) : i32
    %2 = llvm.mlir.constant(6 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.constant(12 : i32) : i32
    %5 = llvm.mlir.constant(7 : i32) : i32
    %6 = llvm.mlir.constant(128 : i32) : i32
    %7 = llvm.mlir.constant(127 : i32) : i32
    %8 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %9 = llvm.mlir.constant(0 : i32) : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %10 = llvm.getelementptr %8[65536] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %thread_id_x = gpu.thread_id  x
    %11 = arith.index_cast %thread_id_x : index to i32
    %12 = llvm.and %11, %7 : i32
    %13 = nvg.warp_id
    %14 = nvvm.elect.sync -> i1
    %15 = llvm.and %true, %14 : i1
    %16 = nvg.cluster_id
    %17 = llvm.icmp "ult" %12, %6 : i32
    %18 = llvm.and %15, %17 : i1
    %19 = llvm.add %13, %9 : i32
    %20 = llvm.shl %19, %9 : i32
    %21 = llvm.or %9, %20 : i32
    %22 = llvm.and %21, %5 : i32
    %23 = llvm.shl %22, %4 : i32
    %24 = llvm.or disjoint %23, %9 : i32
    %25 = llvm.xor %9, %24 : i32
    %26 = llvm.getelementptr %8[%25] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %27 = llvm.shl %19, %9 : i32
    %28 = llvm.or %9, %27 : i32
    %29 = llvm.shl %16, %3 : i32
    %30 = llvm.or %28, %29 : i32
    %31 = llvm.and %30, %3 : i32
    %32 = llvm.shl %31, %2 : i32
    %33 = llvm.or disjoint %32, %9 : i32
    %34 = llvm.xor %9, %33 : i32
    %35 = llvm.and %30, %1 : i32
    %36 = llvm.icmp "eq" %35, %9 : i32
    %37 = llvm.select %36, %9, %0 : i1, i32
    %38 = llvm.or disjoint %37, %9 : i32
    %39 = llvm.xor %9, %38 : i32
    %40 = llvm.add %39, %c0_i32 : i32
    %41 = llvm.add %34, %c0_i32 : i32
    %42 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %18, %26, %arg0, %40, %41, %10 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %43 = llvm.icmp "ult" %12, %6 : i32
    %44 = llvm.and %15, %43 : i1
    %45 = llvm.add %13, %1 : i32
    %46 = llvm.shl %45, %9 : i32
    %47 = llvm.or %9, %46 : i32
    %48 = llvm.and %47, %5 : i32
    %49 = llvm.shl %48, %4 : i32
    %50 = llvm.or disjoint %49, %9 : i32
    %51 = llvm.xor %9, %50 : i32
    %52 = llvm.getelementptr %8[%51] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %53 = llvm.shl %45, %9 : i32
    %54 = llvm.or %9, %53 : i32
    %55 = llvm.shl %16, %3 : i32
    %56 = llvm.or %54, %55 : i32
    %57 = llvm.and %56, %3 : i32
    %58 = llvm.shl %57, %2 : i32
    %59 = llvm.or disjoint %58, %9 : i32
    %60 = llvm.xor %9, %59 : i32
    %61 = llvm.and %56, %1 : i32
    %62 = llvm.icmp "eq" %61, %9 : i32
    %63 = llvm.select %62, %9, %0 : i1, i32
    %64 = llvm.or disjoint %63, %9 : i32
    %65 = llvm.xor %9, %64 : i32
    %66 = llvm.add %65, %c0_i32 : i32
    %67 = llvm.add %60, %c0_i32 : i32
    %68 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %44, %52, %arg0, %66, %67, %10 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    llvm.return
  }
}


// -----// IR Dump After CSE (cse) //----- //
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = llvm.mlir.constant(64 : i32) : i32
    %1 = llvm.mlir.constant(4 : i32) : i32
    %2 = llvm.mlir.constant(6 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.constant(12 : i32) : i32
    %5 = llvm.mlir.constant(7 : i32) : i32
    %6 = llvm.mlir.constant(128 : i32) : i32
    %7 = llvm.mlir.constant(127 : i32) : i32
    %8 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %9 = llvm.mlir.constant(0 : i32) : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %10 = llvm.getelementptr %8[65536] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %thread_id_x = gpu.thread_id  x
    %11 = arith.index_cast %thread_id_x : index to i32
    %12 = llvm.and %11, %7 : i32
    %13 = nvg.warp_id
    %14 = nvvm.elect.sync -> i1
    %15 = llvm.and %true, %14 : i1
    %16 = nvg.cluster_id
    %17 = llvm.icmp "ult" %12, %6 : i32
    %18 = llvm.and %15, %17 : i1
    %19 = llvm.add %13, %9 : i32
    %20 = llvm.shl %19, %9 : i32
    %21 = llvm.or %9, %20 : i32
    %22 = llvm.and %21, %5 : i32
    %23 = llvm.shl %22, %4 : i32
    %24 = llvm.or disjoint %23, %9 : i32
    %25 = llvm.xor %9, %24 : i32
    %26 = llvm.getelementptr %8[%25] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %27 = llvm.shl %16, %3 : i32
    %28 = llvm.or %21, %27 : i32
    %29 = llvm.and %28, %3 : i32
    %30 = llvm.shl %29, %2 : i32
    %31 = llvm.or disjoint %30, %9 : i32
    %32 = llvm.xor %9, %31 : i32
    %33 = llvm.and %28, %1 : i32
    %34 = llvm.icmp "eq" %33, %9 : i32
    %35 = llvm.select %34, %9, %0 : i1, i32
    %36 = llvm.or disjoint %35, %9 : i32
    %37 = llvm.xor %9, %36 : i32
    %38 = llvm.add %37, %c0_i32 : i32
    %39 = llvm.add %32, %c0_i32 : i32
    %40 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %18, %26, %arg0, %38, %39, %10 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %41 = llvm.add %13, %1 : i32
    %42 = llvm.shl %41, %9 : i32
    %43 = llvm.or %9, %42 : i32
    %44 = llvm.and %43, %5 : i32
    %45 = llvm.shl %44, %4 : i32
    %46 = llvm.or disjoint %45, %9 : i32
    %47 = llvm.xor %9, %46 : i32
    %48 = llvm.getelementptr %8[%47] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %49 = llvm.or %43, %27 : i32
    %50 = llvm.and %49, %3 : i32
    %51 = llvm.shl %50, %2 : i32
    %52 = llvm.or disjoint %51, %9 : i32
    %53 = llvm.xor %9, %52 : i32
    %54 = llvm.and %49, %1 : i32
    %55 = llvm.icmp "eq" %54, %9 : i32
    %56 = llvm.select %55, %9, %0 : i1, i32
    %57 = llvm.or disjoint %56, %9 : i32
    %58 = llvm.xor %9, %57 : i32
    %59 = llvm.add %58, %c0_i32 : i32
    %60 = llvm.add %53, %c0_i32 : i32
    %61 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %18, %48, %arg0, %59, %60, %10 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    llvm.return
  }
}


