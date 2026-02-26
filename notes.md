# Dialects
## ttg - TritonGPU_Dialect
```C++
// include/triton/Dialect/TritonGPU/IR/TritonGPUDialect.td
cppNamespace = "::mlir::triton::gpu"
```

## ttng - TritonNvidiaGPU_Dialect
```C++
// include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td
cppNamespace = "::mlir::triton::nvidia_gpu"
```

## nvg - NVGPU_Dialect
This dialect is created by NVIDIA team, not by Triton team
```C++
// third_party/nvidia/include/Dialect/NVGPU/IR/NVGPUDialect.td
cppNamespace = "::mlir::triton::nvgpu"
```

## ttng vs nvg
In the Triton compiler infrastructure, both the **`ttng` (TritonNvidiaGPU)** dialect and the **`nvg` (NVGPU)** dialect serve critical but completely distinct roles within the MLIR-based progressive lowering pipeline.

Although both are specific to NVIDIA hardware, they operate at different levels of abstraction. The `ttng` dialect is a high-level representation used for tensor and block-level optimizations, while the `nvg` dialect acts as a low-level bridge to actual PTX instructions.

Here is why both are necessary:

### 1. Different Levels of Abstraction

* **`ttng` (TritonNvidiaGPU) is Tensor/Block-oriented:** This dialect extends the generic `ttg` (TritonGPU) dialect with NVIDIA-specific macro concepts. It introduces operations like `ttng.async_tma_copy_global_to_local`, `ttng.warp_group_dot`, and `ttng.tc_gen5_mma`. Operations here still understand Triton’s high-level type system (e.g., `tensor` with `#blocked` or `#shared` layouts, and `tensordesc`). It exists so that high-level compiler passes—such as software pipelining, TMA materialization, and Warp Specialization—can perform global reasoning about tile-level data movement.
* **`nvg` (NVGPU) is Thread/Instruction-oriented:** This dialect maps almost directly to actual PTX instructions (like `wgmma.mma_async`, `ld.acquire`, `mbarrier`, and `wgmma_wait_group`). It discards high-level tensor layouts and instead operates on low-level types: LLVM pointers, exact register counts, scalar synchronizations, and fine-grained memory semantics.

### 2. The Progressive Lowering Pipeline

MLIR compilers rely on **progressive lowering** to systematically translate code from human-readable abstractions down to machine code. The compilation flow looks roughly like this:
`tt` (Triton IR) ➔ `ttg` (Triton GPU IR) ➔ **`ttng`** (NVIDIA Tensor IR) ➔ **`nvg`** (NVIDIA Thread/PTX IR) ➔ `LLVM IR / NVVM`

The absolute key reason for creating the nvg dialect is to give the compiler a way to mathematically understand and optimize bare-metal PTX instructions using MLIR, rather than treating them as opaque text strings.

If the `nvg` dialect did not exist, the compiler would be forced to lower the high-level `ttng` operations directly into opaque strings of inline PTX assembly or complex LLVM intrinsics. By having `nvg` as an intermediate step, the compiler can still use MLIR's infrastructure to perform low-level peephole optimizations, analyze memory effects, and manage register footprints *before* generating the final LLVM IR.

Instruction scheduling is a standard compiler concept, and while true, it might not fully capture why Triton developers specifically felt the need to build and maintain an entirely separate dialect just for NVIDIA GPUs.

To give you a much stronger, hardcore compiler-engineering reason, we need to look at the sheer nightmare of **generating code for H100 Tensor Cores without a dedicated abstraction layer.**

The most practical, undeniable benefit of the `nvg` dialect is that it prevents a **combinatorial explosion of inline assembly strings** and solves the problem of **dynamic register packing**.

Here is the concrete example:

#### The Nightmare: Variable Register Counts for `wgmma`

When a Hopper GPU executes a `wgmma.mma_async` instruction, it doesn't just take a single "tensor" or a single "pointer." If the matrix data is stored in registers, PTX requires you to pass the *exact* number of 32-bit registers explicitly in the assembly string.

Depending on the tile size (M, N, K dimensions) and the data type (FP8, FP16, BF16, TF32), a single matrix multiplication might require 4, 8, 16, or 32 physical registers.

If Triton went directly from high-level `ttng` tensors to LLVM inline assembly (skipping `nvg`), the compiler would have to manually hardcode hundreds of different assembly string templates.

**Without `nvg` (Direct to Inline Assembly):**
The high-level MLIR passes would have to contain incredibly brittle, ugly C++ logic to generate exact strings based on tensor shapes:

```cpp
// Hypothetical nightmare C++ code inside the high-level compiler
if (tile_shape == "64x64x16" && dtype == "fp16") {
    // Requires exactly 4 registers for A, 4 for B
    emit(llvm.inline_asm "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16 %0, %1, %2, %3, %4, %5, %6, %7;");
} else if (tile_shape == "64x64x32" && dtype == "fp8") {
    // Requires exactly 8 registers for A, 8 for B
    emit(llvm.inline_asm "wgmma.mma_async... %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15;"); 
} 
// ... imagine hundreds of these if-statements ...

```

This tightly couples high-level tensor loop logic with the lowest-level string formatting. If NVIDIA releases a new data type or tile size, the high-level compiler logic breaks.

#### The Solution: `nvg` as the "PTX Generation Engine"

The `nvg` dialect completely decouples this process. Instead of forcing high-level passes to care about PTX string formatting and register counting, `nvg` uses MLIR's `!llvm.struct` type to package an arbitrary number of registers together.

**With `nvg`:**
The high-level passes simply lower a tensor into a generic struct of registers and hand it to `nvg`:

```mlir
// The nvg dialect takes a generic struct. It doesn't care if it holds 4 or 16 registers.
%result = nvgpu.wgmma.mma_async %matrixA_struct, %matrixB_struct : (!llvm.struct<...>, !llvm.struct<...>)

```

Then, at the very end of the compilation pipeline, a dedicated pass (`NVGPUToLLVMPass.cpp`) looks at the `nvgpu.wgmma` operation. It counts how many elements are in the struct, and it *programmatically generates* the correct PTX assembly string on the fly.

#### Why this is a Massive Practical Benefit

1. **Separation of Concerns:** The developers writing high-level optimizations (like loop unrolling or pipelining in `ttng`) never have to look at or write PTX assembly strings. They just emit `nvg` operations.
2. **Maintainability:** When Blackwell (Compute 100) introduces new matrix instructions, Triton developers only have to update the final `nvg -> LLVM` string generator, rather than tearing apart the entire upper-level compiler.
3. **Type Legalization:** MLIR's type system handles packing and unpacking MLIR `vector` types into LLVM `struct` types automatically, feeding them perfectly into the `nvg` ops.

In short, `nvg` is fundamentally necessary to keep the compiler's source code **sane, modular, and mathematically sound when dealing with hardware instructions that require highly variable register inputs**.
<br/>

### Why not just bake those exact same PTX abstractions directly into `ttng`?
In compiler engineering, creating a "mega-dialect" that handles everything from high-level tensors down to low-level hardware instructions is a well-known anti-pattern.

We keep `ttng` strictly separated from PTX instructions for three fundamental reasons: **Preservation of Geometry, Phase Ordering, and Pass Complexity.**

#### 1. The Semantic Gap: Tensors vs. Registers

The `ttng` dialect's superpower is that it understands **geometry**. When an operation is in `ttng`, the compiler knows it is looking at a `tensor<128x128xf16>` with a specific memory layout (e.g., `#blocked` or `#shared`).

PTX instructions, however, have no concept of geometry. A `wgmma` or a `ldmatrix` instruction only understands a flat list of 32-bit registers and memory addresses.

If we let `ttng` deal with PTX instructions, the compiler would have to prematurely "shatter" that beautiful 128x128 tensor into dozens of individual registers. Once you shatter a tensor into raw registers, **you lose the geometric information forever**. You can no longer easily answer questions like, "Are these memory accesses contiguous?" or "Can I safely prefetch this tile?"

#### 2. High-Level Optimizations Need to Happen *Before* Shattering

Many of Triton's most critical performance optimizations—like **Software Pipelining** (overlapping math and memory across loop iterations) and **Warp Specialization** (dedicating some warps to memory fetching and others to math)—are vastly easier to perform when the data is still represented as a unified tensor.

If `ttng` contained both tensor operations and PTX operations, an optimization pass like Software Pipelining would have a nightmare trying to figure out what to do. It would have to juggle moving high-level tensors around while simultaneously trying to accurately pipeline fragmented structs of raw PTX registers.

By enforcing a strict boundary:

1. We run all high-level optimizations on `ttng` (while everything is still an easily mathematically manipulatable tensor).
2. Once those high-level optimizations are finished, we **lower** the code down to `nvg`.
3. Now that it is in `nvg` (PTX/register level), we run a completely different set of low-level optimizations (like register allocation, peephole optimizations, and barrier synchronization).

#### 3. Avoiding Compiler Code Bloat (Separation of Concerns)

If we merged `nvg` into `ttng`, the C++ codebase for the `ttng` dialect would become massive and unmaintainable. Every time NVIDIA released a new GPU architecture with new PTX opcodes (like the shift from Ampere to Hopper, or Hopper to Blackwell), developers would have to touch the exact same files that handle Triton's core tensor logic.

By isolating PTX instructions into `nvg`, the Triton team ensures that:

* **`ttng`** acts as the high-level, architecture-aware tensor planner.
* **`nvg`** acts as the dumb but precise "PTX assembly formatting engine."

In short, we don't let `ttng` deal with PTX instructions because a good compiler separates the *strategy* (how do we move these massive blocks of data?) from the *tactics* (exactly which hardware registers do we put them in?).


### 3. Why a custom `nvg` instead of upstream MLIR's `nvgpu`?

You might wonder why Triton maintains its own `nvg` dialect in `third_party/nvidia` instead of just using the standard `nvgpu` dialect that already exists in upstream LLVM/MLIR.

The primary reason is **development velocity**. Triton is designed to extract maximum performance from cutting-edge NVIDIA architectures (like Hopper and Blackwell). Features such as Tensor Memory Accelerator (TMA) instructions, specialized cache eviction modifiers, and advanced thread-block synchronization primitives are needed in Triton immediately upon hardware release. Upstream MLIR moves much slower and requires rigorous standardization. By maintaining its own `nvg` dialect, Triton engineers can rapidly model new PTX instructions and quickly iterate without waiting months for upstream MLIR merges.


# Conversion vs Transforms
In the Triton compiler (which is built on top of the MLIR framework), the absolute difference between the **`Conversion`** and **`Transforms`** directories stems from standard MLIR compiler design principles:

### 1. `Conversion` (Inter-dialect Translation)

The `Conversion` directory is responsible for **lowering** or translating code from one dialect (abstraction level) to another.

* **What it does:** It takes operations defined in one dialect and converts them into operations of a lower-level or different dialect. This often involves changing the types of the variables and the fundamental semantics of the program.
* **Example (`lib/Conversion/`):** Inside this directory, you will find passes like `TritonToTritonGPU` (which converts hardware-agnostic Triton IR into GPU-specific Triton IR) and `TritonGPUToLLVM` (which translates TritonGPU IR down into the LLVM IR dialect so it can eventually be compiled to machine code like PTX).

### 2. `Transforms` (Intra-dialect Optimization)

The `Transforms` directories are responsible for **optimizing** or restructuring code *within the same dialect*.

* **What it does:** It takes an existing IR, analyzes it, and mutates it to make it run faster or use memory more efficiently, without fundamentally changing the dialect or the abstraction level.
* **Example (`lib/Dialect/TritonNvidiaGPU/Transforms`):** Inside this directory, you will find passes that are strictly scoped to the `TritonNvidiaGPU` dialect. Examples include memory coalescing, software pipelining, inserting async memory fences (like `cp.async.commit_group`), or optimizing Tensor Memory Accelerator (TMA) usage. The input is `TritonNvidiaGPU` IR, and the output is just a more highly-optimized version of `TritonNvidiaGPU` IR.

### Summary

* **`Transforms`** = **Optimization** (Same dialect in $\rightarrow$ Optimized same dialect out).
* **`Conversion`** = **Lowering/Translation** (Higher-level dialect in $\rightarrow$ Lower-level dialect out).

# The `ConvertTritonGPUToLLVM` pass
I found that the pass `ConvertTritonGPUToLLVM` which is a third-party pass in https://github.com/triton-lang/triton/blob/main/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp#L78 applies multiple patterns from the Conversion's sub-directory https://github.com/triton-lang/triton/tree/main/lib/Conversion/TritonGPUToLLVM. I feel that this is a not modular design because the third-party code is tightly-coupled with the Triton's core code. For a sane, modular design, I think the third-party code should be responsible for itself only.

Your observation about the `ConvertTritonGPUToLLVM` pass in the NVIDIA third-party directory utilizing patterns from the core `lib/Conversion/TritonGPUToLLVM` directory is entirely accurate. However, in the context of the **MLIR (Multi-Level Intermediate Representation)** framework—upon which Triton is built—this is actually considered a highly modular, standard, and desirable design pattern.

Here is why this architecture is used and why it is actually an example of good separation of concerns:

### 1. Composition over Duplication

Converting `TritonGPU` IR to `LLVM` IR is a massive task. A large percentage of this conversion is **hardware-agnostic**. For example:

* Converting basic control flow (like `scf.for` or `scf.if`).
* Calculating standard pointer arithmetic.
* Lowering basic arithmetic operations (add, multiply) on standard registers.

If the NVIDIA backend were entirely isolated and responsible for *everything*, it would have to reimplement thousands of lines of code to handle these standard conversions. AMD (ROCm) and Intel (XPU) backends would have to copy-paste that exact same code into their respective third-party directories. By having a core `TritonGPUToLLVM` library, Triton provides a repository of generic conversion patterns that any backend can reuse.

### 2. MLIR’s Pattern-Based Framework

MLIR achieves modularity through **Rewrite Patterns**. In a typical MLIR conversion pass, you create a `RewritePatternSet` (a bucket of rules on how to convert specific operations).

The design you are seeing works like this:

1. The NVIDIA-specific pass creates an empty `RewritePatternSet`.
2. It calls a function from the core Triton library (e.g., `populateTritonGPUToLLVMPatterns`) to fill the bucket with all the generic, hardware-agnostic lowering rules.
3. It then adds its **own NVIDIA-specific patterns** to the bucket (e.g., lowering matrix multiply to NVIDIA's specific `mma.sync` PTX instructions, or utilizing the NVVM dialect).
4. If there is a conflict, MLIR allows target-specific patterns to have a higher "benefit" (priority) so they override the generic ones.

### 3. True Separation of Concerns

Rather than tight coupling, this represents a classic dependency hierarchy:

* **Core Triton** knows *nothing* about NVIDIA, AMD, or Intel. It only knows about generic LLVM.
* **Third-Party Backends** depend on Core Triton. The NVIDIA backend knows about both NVIDIA hardware *and* generic LLVM.

If the core Triton code started calling into the `third_party/nvidia` directory, *that* would be a violation of modularity and a circular dependency. But a specific backend leveraging a generic core library to avoid reinventing the wheel is standard software engineering.

### Summary

In MLIR compilers, a "Pass" is often just a driver that collects conversion rules (patterns) from various libraries. The NVIDIA `ConvertTritonGPUToLLVM` pass is acting as a composer: it takes the generic Triton $\rightarrow$ LLVM rules from the core, mixes them with NVIDIA-specific NVVM rules from the third-party directory, and feeds the combined set to the MLIR dialect conversion engine. This keeps the core clean of hardware specifics while preventing third-party vendors from having to maintain boilerplate code.

## Why don't Triton team separate concerns like this?
- Create a hardware-agnostic pass for converting ttg to llvm.
- Create a NVIDIA-specific pass for converting ttng to llvm.

Your proposed design—running a generic pass first, followed by a vendor-specific pass—is a very logical way to think about standard software pipelines. In fact, for many optimization steps, compilers *do* work exactly like this.

However, when it comes to **Dialect Conversion** (specifically lowering to LLVM) in the MLIR framework, separating this into two distinct sequential *passes* introduces severe technical hurdles.

Here is why the Triton team, and MLIR compiler engineers in general, avoid splitting lowering conversions into multiple passes, and instead combine the patterns into one pass:

### 1. The "Type Conversion" Problem

The biggest hurdle is how MLIR handles types during conversion.
When converting from Triton dialects to the LLVM dialect, you aren't just changing the operations (e.g., `add` to `llvm.add`); you are fundamentally changing the **types of the data**. High-level Triton constructs like "distributed block tensors" are entirely erased and replaced with low-level LLVM pointers, structs, and vectors.

If you split this into two passes:

* **Pass 1 (Generic `ttg` $\rightarrow$ LLVM):** This pass converts all generic Triton ops and fundamentally changes their data types to LLVM types.
* **Pass 2 (NVIDIA `ttng` $\rightarrow$ LLVM):** When this pass tries to run, it encounters a broken IR. The NVIDIA-specific operations (which were skipped in Pass 1) are expecting to consume and produce Triton tensors. But because Pass 1 already ran, the surrounding code is now using LLVM pointers and structs.

To make a two-pass system work, you would have to write complex, fragile "materialization" code to constantly bridge the gap between lowered LLVM types and unlowered Triton types in the intermediate state between Pass 1 and Pass 2.

### 2. Operations Don't Exist in a Vacuum

In Triton IR, generic `ttg` operations and vendor-specific `ttng` operations are heavily intertwined.

For example, a generic `ttg.load` might feed data directly into an NVIDIA-specific `ttng.mma` (Matrix Multiply-Accumulate) operation.
If you try to lower the `ttg.load` in Pass 1, the compiler doesn't know what to do with the output because the `ttng.mma` operation hasn't been lowered yet and doesn't understand LLVM types. MLIR's dialect conversion framework is designed to succeed or fail as a whole to guarantee the IR remains strictly valid. Leaving "half-lowered" IR lying around between passes is considered an anti-pattern in MLIR.

### 3. Modularity at the "Pattern" Level, Not the "Pass" Level

In standard software, we often use **functions** or **classes** as our unit of modularity. In MLIR dialect conversion, the unit of modularity is the **Rewrite Pattern**, not the Pass.

The Triton team *did* separate the concerns, exactly as you suggest, but they did it at the MLIR idiomatic level:

* **Module A (Hardware-Agnostic):** Defines `Pattern_LowerGenericLoad`, `Pattern_LowerGenericAdd`, etc.
* **Module B (NVIDIA-Specific):** Defines `Pattern_LowerNvidiaMMA`, `Pattern_LowerTMA`, etc.

The Pass itself (`ConvertTritonGPUToLLVM`) is essentially just an empty execution shell. It says to the MLIR engine: *"Here is a bucket containing Module A's patterns, and Module B's patterns. Please look at the whole program and apply these rules simultaneously so the types line up perfectly."*

### Summary

Splitting the lowering into two sequential passes would create a "chicken-and-egg" problem with data types, resulting in a fractured, half-lowered intermediate representation. By separating the logic into distinct libraries of **Patterns**, but executing them together in a single **Pass**, Triton achieves both clean code modularity and mathematical compiler safety.

## Hardware-Agnostic and NVIDIA-Specific patterns
>If you try to lower the ttg.load in Pass 1, the compiler doesn't know what to do with the output because the ttng.mma operation hasn't been lowered yet and doesn't understand LLVM types.

This problem also exists when Hardware-Agnostic patterns are applied first, then NVIDIA-Specific patterns are applied next, right?

That is an incredibly sharp observation. You are entirely correct that if the patterns were applied strictly sequentially—all generic patterns running to completion, followed by all NVIDIA-specific patterns—we would hit the exact same broken type problem.

However, the reason this problem doesn't happen is that **MLIR does not apply the patterns sequentially.**

Here is the secret to how MLIR's Dialect Conversion framework solves this:

### 1. The Single "Bucket" Approach

When the pass runs, it doesn't say "Run Module A, then run Module B." Instead, it dumps all the patterns from both the hardware-agnostic library and the NVIDIA-specific library into a single, combined bucket called a `RewritePatternSet`.

### 2. Simultaneous, Worklist-Driven Application

Once all the patterns are in the bucket, the MLIR conversion driver takes over. It walks through the operations in the code and looks into the bucket to find the best-matching pattern for each operation, regardless of where that pattern originally came from.

* It sees a `ttg.load` $\rightarrow$ it grabs the generic lowering pattern.
* It sees a `ttng.mma` right next to it $\rightarrow$ it grabs the NVIDIA-specific lowering pattern.

It applies these transformations dynamically as it traverses the graph. It is a cooperative, unified process rather than a two-step sequential one.

### 3. The Magic of the `TypeConverter`

Even with simultaneous application, MLIR still has to deal with the moment in time when `ttg.load` is converted to an LLVM pointer, but `ttng.mma` hasn't been processed yet (perhaps it's next in the worklist).

To prevent the compiler from crashing during this intermediate state, MLIR uses a tool called the `TypeConverter`.

* When the generic pattern lowers `ttg.load`, the `TypeConverter` automatically inserts a temporary operation called an **`unrealized_conversion_cast`**.
* This cast acts as a temporary bridge, essentially saying: *"Pretend this new LLVM pointer is still a Triton tensor for a microsecond so the rest of the code doesn't break."*
* Moments later, the MLIR driver reaches the `ttng.mma` operation, applies the NVIDIA-specific pattern from the bucket, and updates its expected inputs to LLVM pointers.
* Because both sides now understand LLVM pointers, MLIR "folds" (deletes) the temporary cast because it is no longer needed.

### Why Two Passes Fail but One Pass Succeeds

If you use **two separate passes**, Pass 1 must finish completely, and MLIR enforces strict rules that the IR must be 100% legal and free of unresolved types before moving to Pass 2. It will see the mismatched types and crash.

If you use **one combined pass** (with patterns from multiple libraries), MLIR treats the entire conversion as a single transaction. It allows temporary type mismatches (using casts) while it works through the graph, as long as everything is perfectly resolved by the time the single pass finishes.
