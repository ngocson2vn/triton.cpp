<!-- TOC START -->
- [LinearLayout](#linearlayout)
    - [1. High-Level Concept](#1-high-level-concept)
    - [2. The Linearity Property (XOR Trick)](#2-the-linearity-property-xor-trick)
    - [3. Purpose within Triton](#3-purpose-within-triton)
    - [Summary](#summary)
<!-- TOC END -->

# LinearLayout
In Triton compiler's source code, the class `LinearLayout` is defined here https://github.com/triton-lang/triton/blob/main/include/triton/Tools/LinearLayout.h

Based on the source code and documentation from the Triton compiler, the `LinearLayout` class is a core abstraction used to model how tensors are stored in hardware registers.

Here is an explanation of what the `LinearLayout` class is and how it works:

### 1. High-Level Concept
At its core, a `LinearLayout` (LL) is a mathematical function that maps a **"hardware location"** to a **"logical tensor index"**.

When working with GPUs, multidimensional tensors are typically distributed across multiple threads and warps. For example, if you have a 2D tensor $T$, a `LinearLayout` $L$ defines exactly which element of the tensor is stored in a given thread and warp:
$$L(\text{thread\_id}, \text{warp\_id}) = (x, y)$$
In this scenario, a register belonging to thread `t` within warp `w` contains the value $T[x,y]$.

### 2. The Linearity Property (XOR Trick)
The defining feature of this layout is that the mapping function is **linear over bitwise XOR** (which effectively models arithmetic in the Galois field of 2, or $GF(2)$). 

Because of this linearity constraint, the compiler does not need to store an enormous lookup table mapping every single thread and warp to a tensor index. Instead, it only needs to define the layout at certain special "power-of-2" points—known as **basis vectors**.

If you know the mapping for the bases (e.g., $L(1,0)$, $L(2,0)$, $L(0,1)$, $L(0,2)$), you can compute any other point in the layout by XOR-ing the bases together using the **linearity rule**:
$$L(t_1 \oplus t_2, w_1 \oplus w_2) = L(t_1, w_1) \oplus L(t_2, w_2)$$

*Example given in the source comments:*
If you want to find the tensor coordinate for thread 0, warp 3, you just break down 3 into powers of 2 (2 and 1) and XOR their known mappings:
$$L(0, 3) = L(0, 2 \oplus 1) = L(0,2) \oplus L(0,1)$$

### 3. Purpose within Triton
Triton performs numerous complex optimizations like reshaping, transposing, and broadcasting memory layout structures down to the register level. The `LinearLayout` class facilitates this by providing a robust mathematical framework. 

Looking at the class declaration, it provides several operations to manipulate these mappings:
* **`reshapeIns`, `reshapeOuts`, `flattenIns`, `flattenOuts`**: Functions to modify the shape of the input dimensions (hardware locations) or output dimensions (tensor logical layout).
* **`transposeIns`, `transposeOuts`**: Swap dimensions around without changing the underlying memory structure.
* **`concatIns`, `concatOuts`**: Concatenates two layouts by their dimensions.
* **Math operations (Direct Sums, etc.)**: You can multiply two distinct layouts (e.g., one mapping rows, one mapping columns) to get a multi-dimensional direct sum representing the combined 2D layout.
* **Math properties**: Methods like `isSurjective()`, `isInjective()`, and `isInvertible()` to check if the mapping covers the entire tensor space or if there are duplicate mappings (broadcasting).

### Summary
The `LinearLayout` is an elegant, highly-optimized representation of data placement in GPU memory and registers. By representing GPU data distribution as linear maps over $GF(2)$, Triton can quickly and scalably reason about complex tensor reshaping, tile layouts, and register allocations purely using basic linear algebra.
