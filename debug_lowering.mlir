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


// -----// IR Dump After TritonNvidiaGpuLoweringDebugPass (triton-nvidia-gpu-lowering-debug) //----- //
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8200 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @triton_dot(%arg0: !llvm.ptr {llvm.align = 64 : i32, llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !tt.tensordesc<tensor<64x64xf16, #shared>>
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %5 = llvm.getelementptr %4[%3] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.undef : !llvm.struct<(ptr<3>, i32, i32)>
    %8 = llvm.insertvalue %5, %7[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %9 = llvm.insertvalue %6, %8[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %10 = llvm.insertvalue %6, %9[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    %11 = llvm.mlir.constant(8192 : i32) : i32
    %12 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %13 = llvm.getelementptr %12[%11] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    %14 = llvm.mlir.constant(0 : i32) : i32
    %15 = llvm.mlir.undef : !llvm.struct<(ptr<3>, i32)>
    %16 = llvm.insertvalue %13, %15[0] : !llvm.struct<(ptr<3>, i32)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(ptr<3>, i32)> 
    %18 = llvm.extractvalue %17[0] : !llvm.struct<(ptr<3>, i32)> 
    %19 = llvm.extractvalue %17[1] : !llvm.struct<(ptr<3>, i32)> 
    %20 = llvm.extractvalue %10[0] : !llvm.struct<(ptr<3>, i32, i32)> 
    %21 = llvm.extractvalue %10[1] : !llvm.struct<(ptr<3>, i32, i32)> 
    %22 = llvm.extractvalue %10[2] : !llvm.struct<(ptr<3>, i32, i32)> 
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = llvm.getelementptr %20[%23] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %25 = nvvm.read.ptx.sreg.tid.x : i32
    %26 = llvm.mlir.constant(127 : i32) : i32
    %27 = llvm.and %25, %26 : i32
    %28 = nvg.warp_id
    %29 = nvvm.elect.sync -> i1
    %30 = llvm.and %1, %29 : i1
    %31 = llvm.mlir.constant(0 : i32) : i32
    %32 = nvg.cluster_id
    %33 = llvm.mlir.constant(0 : i32) : i32
    %34 = llvm.mlir.constant(32 : i32) : i32
    %35 = llvm.icmp "ult" %27, %34 : i32
    %36 = llvm.and %30, %35 : i1
    %37 = llvm.mlir.constant(0 : i32) : i32
    %38 = llvm.add %33, %37 : i32
    %39 = llvm.mlir.constant(0 : i32) : i32
    %40 = llvm.mlir.constant(0 : i32) : i32
    %41 = llvm.mlir.constant(0 : i32) : i32
    %42 = llvm.shl %38, %41 : i32
    %43 = llvm.or %40, %42 : i32
    %44 = llvm.mlir.constant(0 : i32) : i32
    %45 = llvm.mlir.constant(0 : i32) : i32
    %46 = llvm.mlir.constant(0 : i32) : i32
    %47 = llvm.or disjoint %45, %46 : i32
    %48 = llvm.xor %39, %47 : i32
    %49 = llvm.mlir.constant(0 : i32) : i32
    %50 = llvm.mlir.constant(0 : i32) : i32
    %51 = llvm.mlir.constant(0 : i32) : i32
    %52 = llvm.or disjoint %50, %51 : i32
    %53 = llvm.xor %39, %52 : i32
    %54 = llvm.getelementptr %24[%48] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %55 = llvm.mlir.constant(0 : i32) : i32
    %56 = llvm.mlir.constant(0 : i32) : i32
    %57 = llvm.mlir.constant(0 : i32) : i32
    %58 = llvm.shl %38, %57 : i32
    %59 = llvm.or %56, %58 : i32
    %60 = llvm.mlir.constant(0 : i32) : i32
    %61 = llvm.shl %32, %60 : i32
    %62 = llvm.or %59, %61 : i32
    %63 = llvm.mlir.constant(0 : i32) : i32
    %64 = llvm.mlir.constant(0 : i32) : i32
    %65 = llvm.mlir.constant(0 : i32) : i32
    %66 = llvm.or disjoint %64, %65 : i32
    %67 = llvm.xor %55, %66 : i32
    %68 = llvm.mlir.constant(0 : i32) : i32
    %69 = llvm.mlir.constant(0 : i32) : i32
    %70 = llvm.mlir.constant(0 : i32) : i32
    %71 = llvm.or disjoint %69, %70 : i32
    %72 = llvm.xor %55, %71 : i32
    %73 = llvm.add %2, %72 : i32
    %74 = llvm.add %2, %67 : i32
    %75 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %36, %54, %arg0, %73, %74, %18 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
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
    %4 = llvm.mlir.constant(true) : i1
    %5 = llvm.getelementptr %2[8192] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %6 = nvvm.read.ptx.sreg.tid.x : i32
    %7 = llvm.and %6, %1 : i32
    %8 = nvvm.elect.sync -> i1
    %9 = llvm.and %4, %8 : i1
    %10 = llvm.icmp "ult" %7, %0 : i32
    %11 = llvm.and %9, %10 : i1
    %12 = llvm.xor %3, %3 : i32
    %13 = llvm.getelementptr %2[%12] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %14 = llvm.xor %3, %3 : i32
    %15 = llvm.xor %3, %3 : i32
    %16 = llvm.add %15, %3 : i32
    %17 = llvm.add %14, %3 : i32
    %18 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %11, %13, %arg0, %16, %17, %5 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
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
    %4 = llvm.mlir.constant(true) : i1
    %5 = llvm.getelementptr %2[8192] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %6 = nvvm.read.ptx.sreg.tid.x : i32
    %7 = llvm.and %6, %1 : i32
    %8 = nvvm.elect.sync -> i1
    %9 = llvm.and %4, %8 : i1
    %10 = llvm.icmp "ult" %7, %0 : i32
    %11 = llvm.and %9, %10 : i1
    %12 = llvm.xor %3, %3 : i32
    %13 = llvm.getelementptr %2[%12] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    %14 = llvm.add %12, %3 : i32
    %15 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" %11, %13, %arg0, %14, %14, %5 : (i1, !llvm.ptr<3>, !llvm.ptr, i32, i32, !llvm.ptr<3>) -> !llvm.void
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
define ptx_kernel void @triton_dot(ptr byval([128 x i8]) align 64 "nvvm.grid_constant" %0, ptr addrspace(1) readnone captures(none) %1, ptr addrspace(1) readnone captures(none) %2) local_unnamed_addr #0 {
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %5 = and i32 %4, 96
  %6 = tail call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1)
  %7 = extractvalue { i32, i1 } %6, 1
  %8 = icmp eq i32 %5, 0
  %9 = and i1 %8, %7
  call void asm sideeffect "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r"(i1 %9, ptr addrspace(3) @global_smem, ptr nonnull %0, i32 0, i32 0, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 8192)) #3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare { i32, i1 } @llvm.nvvm.elect.sync(i32) #2

attributes #0 = { nounwind "nvvm.reqntid"="128" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #3 = { nounwind }

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
	.param .u64 .ptr .global .align 1 triton_dot_param_1,
	.param .u64 .ptr .global .align 1 triton_dot_param_2
)
.reqntid 128
{
	.reg .pred 	%p<4>;
	.reg .b32 	%r<7>;
	.reg .b64 	%rd<3>;

// %bb.0:
	mov.b64 	%rd2, triton_dot_param_0;
	cvta.param.u64 	%rd1, %rd2;
	mov.u32 	%r4, %tid.x;
	and.b32 	%r5, %r4, 96;
	elect.sync 	%r6|%p2, -1;
	setp.eq.b32 	%p3, %r5, 0;
	and.pred 	%p1, %p3, %p2;
	mov.b32 	%r1, global_smem;
	add.s32 	%r3, %r1, 8192;
	mov.b32 	%r2, 0;
	// begin inline asm
	@%p1 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r1], [%rd1, {%r2, %r2}], [%r3];
	// end inline asm
	ret;
                                        // -- End function
}

