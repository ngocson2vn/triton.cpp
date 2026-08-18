// -----// IR Dump After SCCP (sccp) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.barrier_expect %1, 16384, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %1, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}


// -----// IR Dump After TritonGPUAllocateWarpGroups (tritongpu-allocate-warp-groups) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.barrier_expect %1, 16384, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %1, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}


// -----// IR Dump After AllocateSharedMemory (allocate-shared-memory) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 16392 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  tt.func public @triton_dot(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.barrier_expect %1, 16384, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !tt.tensordesc<tensor<128x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    ttng.wait_barrier %1, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}


// -----// IR Dump After ConvertTritonGPUToLLVMDebug (convert-triton-gpu-to-llvm-debug) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 16392 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 32>} {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !tt.tensordesc<tensor<128x64xf16, #shared>>
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
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
    %thread_id_x = gpu.thread_id  x
    %18 = arith.index_cast %thread_id_x : index to i32
    %19 = llvm.mlir.constant(31 : i32) : i32
    %20 = llvm.and %18, %19 : i32
    %21 = llvm.mlir.constant(0 : i32) : i32
    %22 = llvm.icmp "eq" %20, %21 : i32
    %23 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %22, %16 : (i1, !llvm.ptr<3>) -> !llvm.void
    %24 = llvm.extractvalue %15[0] : !llvm.struct<(ptr<3>, i32)> 
    %25 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<3>, i32)> 
    %thread_id_x_0 = gpu.thread_id  x
    %26 = arith.index_cast %thread_id_x_0 : index to i32
    %27 = llvm.mlir.constant(31 : i32) : i32
    %28 = llvm.and %26, %27 : i32
    %29 = llvm.mlir.constant(0 : i32) : i32
    %30 = llvm.icmp "eq" %28, %29 : i32
    %31 = llvm.and %30, %true : i1
    %32 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 16384;", "b,r" %31, %24 : (i1, !llvm.ptr<3>) -> !llvm.void
    %33 = llvm.extractvalue %15[0] : !llvm.struct<(ptr<3>, i32)> 
    %34 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<3>, i32)> 
    %35 = llvm.extractvalue %8[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %36 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %37 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    %38 = llvm.mlir.constant(0 : i32) : i32
    %39 = llvm.getelementptr %35[%38] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %thread_id_x_1 = gpu.thread_id  x
    %40 = arith.index_cast %thread_id_x_1 : index to i32
    %41 = llvm.mlir.constant(31 : i32) : i32
    %42 = llvm.and %40, %41 : i32
    %43 = nvg.warp_id
    %44 = nvvm.elect.sync -> i1
    %45 = llvm.and %true, %44 : i1
    %46 = llvm.mlir.constant(0 : i32) : i32
    %47 = nvg.cluster_id
    %48 = llvm.mlir.constant(0 : i32) : i32
    %49 = llvm.mlir.constant(32 : i32) : i32
    %50 = llvm.icmp "ult" %42, %49 : i32
    %51 = llvm.and %45, %50 : i1
    %52 = llvm.mlir.constant(0 : i32) : i32
    %53 = llvm.add %48, %52 : i32
    %54 = llvm.mlir.constant(0 : i32) : i32
    %55 = llvm.mlir.constant(0 : i32) : i32
    %56 = llvm.mlir.constant(0 : i32) : i32
    %57 = llvm.shl %53, %56 : i32
    %58 = llvm.or %55, %57 : i32
    %59 = llvm.mlir.constant(0 : i32) : i32
    %60 = llvm.mlir.constant(1 : i32) : i32
    %61 = llvm.and %58, %60 : i32
    %62 = llvm.icmp "eq" %61, %59 : i32
    %63 = llvm.mlir.constant(4096 : i32) : i32
    %64 = llvm.select %62, %59, %63 : i1, i32
    %65 = llvm.mlir.constant(0 : i32) : i32
    %66 = llvm.or disjoint %64, %65 : i32
    %67 = llvm.xor %54, %66 : i32
    %68 = llvm.mlir.constant(0 : i32) : i32
    %69 = llvm.mlir.constant(0 : i32) : i32
    %70 = llvm.mlir.constant(0 : i32) : i32
    %71 = llvm.or disjoint %69, %70 : i32
    %72 = llvm.xor %54, %71 : i32
    %73 = llvm.getelementptr %39[%67] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %74 = llvm.mlir.constant(0 : i32) : i32
    %75 = llvm.mlir.constant(0 : i32) : i32
    %76 = llvm.mlir.constant(0 : i32) : i32
    %77 = llvm.shl %53, %76 : i32
    %78 = llvm.or %75, %77 : i32
    %79 = llvm.mlir.constant(1 : i32) : i32
    %80 = llvm.shl %47, %79 : i32
    %81 = llvm.or %78, %80 : i32
    %82 = llvm.mlir.constant(0 : i32) : i32
    %83 = llvm.mlir.constant(1 : i32) : i32
    %84 = llvm.and %81, %83 : i32
    %85 = llvm.icmp "eq" %84, %82 : i32
    %86 = llvm.mlir.constant(64 : i32) : i32
    %87 = llvm.select %85, %82, %86 : i1, i32
    %88 = llvm.mlir.constant(0 : i32) : i32
    %89 = llvm.or disjoint %87, %88 : i32
    %90 = llvm.xor %74, %89 : i32
    %91 = llvm.mlir.constant(0 : i32) : i32
    %92 = llvm.mlir.constant(0 : i32) : i32
    %93 = llvm.mlir.constant(0 : i32) : i32
    %94 = llvm.or disjoint %92, %93 : i32
    %95 = llvm.xor %74, %94 : i32
    %96 = llvm.add %c0_i32, %95 : i32
    %97 = llvm.add %c0_i32, %90 : i32
    %98 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %51, %73, %arg0, %96, %97, %33 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %99 = llvm.mlir.constant(0 : i32) : i32
    %100 = llvm.mlir.constant(32 : i32) : i32
    %101 = llvm.icmp "ult" %42, %100 : i32
    %102 = llvm.and %45, %101 : i1
    %103 = llvm.mlir.constant(1 : i32) : i32
    %104 = llvm.add %99, %103 : i32
    %105 = llvm.mlir.constant(0 : i32) : i32
    %106 = llvm.mlir.constant(0 : i32) : i32
    %107 = llvm.mlir.constant(0 : i32) : i32
    %108 = llvm.shl %104, %107 : i32
    %109 = llvm.or %106, %108 : i32
    %110 = llvm.mlir.constant(0 : i32) : i32
    %111 = llvm.mlir.constant(1 : i32) : i32
    %112 = llvm.and %109, %111 : i32
    %113 = llvm.icmp "eq" %112, %110 : i32
    %114 = llvm.mlir.constant(4096 : i32) : i32
    %115 = llvm.select %113, %110, %114 : i1, i32
    %116 = llvm.mlir.constant(0 : i32) : i32
    %117 = llvm.or disjoint %115, %116 : i32
    %118 = llvm.xor %105, %117 : i32
    %119 = llvm.mlir.constant(0 : i32) : i32
    %120 = llvm.mlir.constant(0 : i32) : i32
    %121 = llvm.mlir.constant(0 : i32) : i32
    %122 = llvm.or disjoint %120, %121 : i32
    %123 = llvm.xor %105, %122 : i32
    %124 = llvm.getelementptr %39[%118] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %125 = llvm.mlir.constant(0 : i32) : i32
    %126 = llvm.mlir.constant(0 : i32) : i32
    %127 = llvm.mlir.constant(0 : i32) : i32
    %128 = llvm.shl %104, %127 : i32
    %129 = llvm.or %126, %128 : i32
    %130 = llvm.mlir.constant(1 : i32) : i32
    %131 = llvm.shl %47, %130 : i32
    %132 = llvm.or %129, %131 : i32
    %133 = llvm.mlir.constant(0 : i32) : i32
    %134 = llvm.mlir.constant(1 : i32) : i32
    %135 = llvm.and %132, %134 : i32
    %136 = llvm.icmp "eq" %135, %133 : i32
    %137 = llvm.mlir.constant(64 : i32) : i32
    %138 = llvm.select %136, %133, %137 : i1, i32
    %139 = llvm.mlir.constant(0 : i32) : i32
    %140 = llvm.or disjoint %138, %139 : i32
    %141 = llvm.xor %125, %140 : i32
    %142 = llvm.mlir.constant(0 : i32) : i32
    %143 = llvm.mlir.constant(0 : i32) : i32
    %144 = llvm.mlir.constant(0 : i32) : i32
    %145 = llvm.or disjoint %143, %144 : i32
    %146 = llvm.xor %125, %145 : i32
    %147 = llvm.add %c0_i32, %146 : i32
    %148 = llvm.add %c0_i32, %141 : i32
    %149 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %102, %124, %arg0, %147, %148, %33 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %150 = llvm.extractvalue %15[0] : !llvm.struct<(ptr<3>, i32)> 
    %151 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<3>, i32)> 
    %152 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %150, %c0_i32 : (!llvm.ptr<3>, i32) -> !llvm.void
    %153 = llvm.extractvalue %15[0] : !llvm.struct<(ptr<3>, i32)> 
    %154 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<3>, i32)> 
    %thread_id_x_2 = gpu.thread_id  x
    %155 = arith.index_cast %thread_id_x_2 : index to i32
    %156 = llvm.mlir.constant(31 : i32) : i32
    %157 = llvm.and %155, %156 : i32
    %158 = llvm.mlir.constant(0 : i32) : i32
    %159 = llvm.icmp "eq" %157, %158 : i32
    %160 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %159, %153 : (i1, !llvm.ptr<3>) -> !llvm.void
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
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %7 = llvm.getelementptr %5[16384] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %thread_id_x = gpu.thread_id  x
    %8 = arith.index_cast %thread_id_x : index to i32
    %9 = llvm.and %8, %4 : i32
    %10 = llvm.icmp "eq" %9, %6 : i32
    %11 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %10, %7 : (i1, !llvm.ptr<3>) -> !llvm.void
    %thread_id_x_0 = gpu.thread_id  x
    %12 = arith.index_cast %thread_id_x_0 : index to i32
    %13 = llvm.and %12, %4 : i32
    %14 = llvm.icmp "eq" %13, %6 : i32
    %15 = llvm.and %14, %true : i1
    %16 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 16384;", "b,r" %15, %7 : (i1, !llvm.ptr<3>) -> !llvm.void
    %thread_id_x_1 = gpu.thread_id  x
    %17 = arith.index_cast %thread_id_x_1 : index to i32
    %18 = llvm.and %17, %4 : i32
    %19 = nvvm.elect.sync -> i1
    %20 = llvm.and %true, %19 : i1
    %21 = nvg.cluster_id
    %22 = llvm.icmp "ult" %18, %3 : i32
    %23 = llvm.and %20, %22 : i1
    %24 = llvm.add %6, %6 : i32
    %25 = llvm.shl %24, %6 : i32
    %26 = llvm.or %6, %25 : i32
    %27 = llvm.and %26, %2 : i32
    %28 = llvm.icmp "eq" %27, %6 : i32
    %29 = llvm.select %28, %6, %1 : i1, i32
    %30 = llvm.or disjoint %29, %6 : i32
    %31 = llvm.xor %6, %30 : i32
    %32 = llvm.getelementptr %5[%31] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %33 = llvm.shl %24, %6 : i32
    %34 = llvm.or %6, %33 : i32
    %35 = llvm.shl %21, %2 : i32
    %36 = llvm.or %34, %35 : i32
    %37 = llvm.and %36, %2 : i32
    %38 = llvm.icmp "eq" %37, %6 : i32
    %39 = llvm.select %38, %6, %0 : i1, i32
    %40 = llvm.or disjoint %39, %6 : i32
    %41 = llvm.xor %6, %40 : i32
    %42 = llvm.xor %6, %6 : i32
    %43 = llvm.add %42, %c0_i32 : i32
    %44 = llvm.add %41, %c0_i32 : i32
    %45 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %23, %32, %arg0, %43, %44, %7 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %46 = llvm.icmp "ult" %18, %3 : i32
    %47 = llvm.and %20, %46 : i1
    %48 = llvm.add %6, %2 : i32
    %49 = llvm.shl %48, %6 : i32
    %50 = llvm.or %6, %49 : i32
    %51 = llvm.and %50, %2 : i32
    %52 = llvm.icmp "eq" %51, %6 : i32
    %53 = llvm.select %52, %6, %1 : i1, i32
    %54 = llvm.or disjoint %53, %6 : i32
    %55 = llvm.xor %6, %54 : i32
    %56 = llvm.getelementptr %5[%55] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %57 = llvm.shl %48, %6 : i32
    %58 = llvm.or %6, %57 : i32
    %59 = llvm.shl %21, %2 : i32
    %60 = llvm.or %58, %59 : i32
    %61 = llvm.and %60, %2 : i32
    %62 = llvm.icmp "eq" %61, %6 : i32
    %63 = llvm.select %62, %6, %0 : i1, i32
    %64 = llvm.or disjoint %63, %6 : i32
    %65 = llvm.xor %6, %64 : i32
    %66 = llvm.xor %6, %6 : i32
    %67 = llvm.add %66, %c0_i32 : i32
    %68 = llvm.add %65, %c0_i32 : i32
    %69 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %47, %56, %arg0, %67, %68, %7 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %70 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %7, %c0_i32 : (!llvm.ptr<3>, i32) -> !llvm.void
    %thread_id_x_2 = gpu.thread_id  x
    %71 = arith.index_cast %thread_id_x_2 : index to i32
    %72 = llvm.and %71, %4 : i32
    %73 = llvm.icmp "eq" %72, %6 : i32
    %74 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %73, %7 : (i1, !llvm.ptr<3>) -> !llvm.void
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
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %7 = llvm.getelementptr %5[16384] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %thread_id_x = gpu.thread_id  x
    %8 = arith.index_cast %thread_id_x : index to i32
    %9 = llvm.and %8, %4 : i32
    %10 = llvm.icmp "eq" %9, %6 : i32
    %11 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %10, %7 : (i1, !llvm.ptr<3>) -> !llvm.void
    %12 = llvm.and %10, %true : i1
    %13 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 16384;", "b,r" %12, %7 : (i1, !llvm.ptr<3>) -> !llvm.void
    %14 = nvvm.elect.sync -> i1
    %15 = llvm.and %true, %14 : i1
    %16 = nvg.cluster_id
    %17 = llvm.icmp "ult" %9, %3 : i32
    %18 = llvm.and %15, %17 : i1
    %19 = llvm.add %6, %6 : i32
    %20 = llvm.shl %19, %6 : i32
    %21 = llvm.or %6, %20 : i32
    %22 = llvm.and %21, %2 : i32
    %23 = llvm.icmp "eq" %22, %6 : i32
    %24 = llvm.select %23, %6, %1 : i1, i32
    %25 = llvm.or disjoint %24, %6 : i32
    %26 = llvm.xor %6, %25 : i32
    %27 = llvm.getelementptr %5[%26] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %28 = llvm.shl %16, %2 : i32
    %29 = llvm.or %21, %28 : i32
    %30 = llvm.and %29, %2 : i32
    %31 = llvm.icmp "eq" %30, %6 : i32
    %32 = llvm.select %31, %6, %0 : i1, i32
    %33 = llvm.or disjoint %32, %6 : i32
    %34 = llvm.xor %6, %33 : i32
    %35 = llvm.xor %6, %6 : i32
    %36 = llvm.add %35, %c0_i32 : i32
    %37 = llvm.add %34, %c0_i32 : i32
    %38 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %18, %27, %arg0, %36, %37, %7 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %39 = llvm.add %6, %2 : i32
    %40 = llvm.shl %39, %6 : i32
    %41 = llvm.or %6, %40 : i32
    %42 = llvm.and %41, %2 : i32
    %43 = llvm.icmp "eq" %42, %6 : i32
    %44 = llvm.select %43, %6, %1 : i1, i32
    %45 = llvm.or disjoint %44, %6 : i32
    %46 = llvm.xor %6, %45 : i32
    %47 = llvm.getelementptr %5[%46] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %48 = llvm.or %41, %28 : i32
    %49 = llvm.and %48, %2 : i32
    %50 = llvm.icmp "eq" %49, %6 : i32
    %51 = llvm.select %50, %6, %0 : i1, i32
    %52 = llvm.or disjoint %51, %6 : i32
    %53 = llvm.xor %6, %52 : i32
    %54 = llvm.add %53, %c0_i32 : i32
    %55 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %18, %47, %arg0, %36, %54, %7 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
    %56 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r" %7, %c0_i32 : (!llvm.ptr<3>, i32) -> !llvm.void
    %57 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r" %10, %7 : (i1, !llvm.ptr<3>) -> !llvm.void
    llvm.return
  }
}


