# BlockedEncoding
```MLIR
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
```

This text string is the **MLIR assembly format** of the exact `BlockedEncodingAttr` we constructed using C++ earlier. When Triton compiles your Python code into its Intermediate Representation (IR), it prints the hardware layout rules in this human-readable format.

While the C++ code used scalar values (like `numWarps = 4` and `numThreadsPerWarp = 32`), Triton mathematically maps those scalars onto the $N$-dimensional shape of your tensor.

Here is the breakdown of what each parameter means in this spatial representation:

### 1. `sizePerThread = [2, 2]`

As discussed, this defines the contiguous chunk of elements each thread holds in its registers.

* It covers **2 rows** and **2 columns**.

### 2. `order = [1, 0]`

This dictates the memory layout contiguity.

* `1` (columns) is the fastest-varying/contiguous dimension.
* `0` (rows) is the slowest-varying dimension.
* This represents a standard **row-major** layout.

### 3. `threadsPerWarp = [1, 32]`

In C++, we simply stated `numThreadsPerWarp = 32`. Here, Triton has decided *how* to arrange those 32 threads geometrically across the 2D tensor.

* `[1, 32]` means the warp is arranged as **1 row** and **32 columns**.
* **Why this shape?** Triton deliberately aligns the threads along the contiguous memory dimension. Since `order = [1, 0]` tells us the columns are contiguous in memory, placing all 32 threads in a single row (`[1, 32]`) ensures that when these threads fetch data, they read 32 adjacent elements perfectly in sequence. This guarantees 100% memory coalescing.

### 4. `warpsPerCTA = [4, 1]`

In C++, we defined `numWarps = 4`. Just like the threads, Triton arranges these 4 warps geometrically.

* `[4, 1]` means the 4 warps are stacked as **4 rows** and **1 column**.
* **Why this shape?** Because the warp itself is extremely wide horizontally (`32` columns), Triton stacks the warps vertically to build a more balanced, square-like tile, which is optimal for tasks like matrix multiplication and reducing shared memory bank conflicts.

---

### The Big Picture: Calculating the "Rep" Shape

By multiplying these arrays together, we can see exactly how much of the tensor the entire Thread Block (CTA) processes in a single step (one replication, or "rep"):

* **Row capacity per rep:** $2 \text{ (elements/thread)} \times 1 \text{ (thread/warp)} \times 4 \text{ (warps/CTA)} = 8 \text{ rows}$
* **Column capacity per rep:** $2 \text{ (elements/thread)} \times 32 \text{ (threads/warp)} \times 1 \text{ (warp/CTA)} = 64 \text{ columns}$

In one synchronized step, the CTA processes an **$8 \times 64$** tile of data.

Recall from the C++ snippet that the total `blockShape` is $64 \times 64$. Because the CTA processes $8 \times 64$ elements at a time, it simply needs to repeat this workload **8 times** along the row axis ($64 / 8 = 8$) to finish the entire $64 \times 64$ block. This perfectly matches the "8 reps per thread" math we calculated earlier!
