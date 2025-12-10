```MLIR
// %r133 = K + 63
// %p56 = (threadIdx.x == 0).
// %r1 = ceil(K / 64)
// %r6 = ceil(K / 64) - 4
// %r447 is initialized to 0

// %r73 = address of shared memory array global_smem

// Initialize 5 mbarrier objects
// %r67 = %r73 + 81920
// %r68 = %r67 + 8
// %r69 = %r68 + 8
// %r70 = %r69 + 8
// %r71 = %r70 + 8

// %p7 = is the leader thread and in first warp
// %p18 = (K + 63 > 63)
// %p21 = (threadIdx.x < 32).

// %rd3 = address of tensor map A
// %rd2 = address of tensor map B
// %r129 = pid_m (block_m)
// %r402 = block_m * 64


// 
// Prefetch 4 tiles along K dimension
// 
// %r402 = block_m * 64
// %r79 = block_n * 64

// ==============
// %r428 = 0
// ==============
// A-tile
@%p7 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r73], [%rd3, {%r428, %r402}], [%r67];

// B-tile
@%p8 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r77], [%rd2, {%r428, %r79}], [%r67];

// ==============
// %r83 = 64
// ==============
// A-tile
@%p10 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r82], [%rd3, {%r83, %r402}], [%r68];

// B-tile
@%p11 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r86], [%rd2, {%r83, %r79}], [%r68];

// ==============
// %r92 = 128
// ==============
// A-tile
@%p13 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r91], [%rd3, {%r92, %r402}], [%r69];

// B-tile
@%p14 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r95], [%rd2, {%r92, %r79}], [%r69];

// ==============
// %r101 = 192
// ==============
// A-tile
@%p16 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r100], [%rd3, {%r101, %r402}], [%r70];

// B-tile
@%p17 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r104], [%rd2, {%r101, %r79}], [%r70];

// 
// Enter Main Loop
// 
// $L__BB0_3: Main Loop
// 
// %r77 = shared + 40960

// %r430 = 3
// %r359 = %r430 + 1 = 4
```