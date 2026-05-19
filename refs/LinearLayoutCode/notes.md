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

# The operator*
The operator* is implemented as follows:
```C++
LinearLayout operator*(LinearLayout inner, LinearLayout outer) {
  // Check that dims common to outer and inner have the same relative order.
  auto inDims = supremum(llvm::to_vector(inner.getInDimNames()),
                         llvm::to_vector(outer.getInDimNames()));
  auto outDims = supremum(llvm::to_vector(inner.getOutDimNames()),
                          llvm::to_vector(outer.getOutDimNames()));

  llvm::outs() << "--- operator*(LinearLayout inner, LinearLayout outer) ---\n";
  
  llvm::outs() << "inner: " << inner << "\n\n";

  llvm::outs() << "outer: " << outer << "\n\n";

  llvm::outs() << "inDims:\n";
  for (const auto& e : inDims) {
    llvm::outs() << "  - " << e << "\n";
  }

  llvm::outs() << "outDims:\n";
  for (const auto& e : outDims) {
    llvm::outs() << "  - " << e << "\n";
  }
  llvm::outs() << "\n";

  // Get the sizeLog2 of all input and output dimensions we're going to
  // consider, in order.  `inner` is more minor, so its dimensions come
  // first.
  llvm::MapVector<StringAttr, int32_t> inDimSizesLog2;
  llvm::MapVector<StringAttr, int32_t> outDimSizesLog2;
  for (const auto &dim : inDims)
    inDimSizesLog2.insert({dim, 0});
  for (const auto &dim : outDims)
    outDimSizesLog2.insert({dim, 0});
  for (const auto &layout : {inner, outer}) {
    for (StringAttr inDim : layout.getInDimNames()) {
      inDimSizesLog2[inDim] += layout.getInDimSizeLog2(inDim);
    }
    for (StringAttr outDim : layout.getOutDimNames()) {
      outDimSizesLog2[outDim] += layout.getOutDimSizeLog2(outDim);
    }
  }

  llvm::outs() << "inDimSizesLog2:\n";
  for (const auto& it : inDimSizesLog2) {
    llvm::outs() << "  - " << it.first << ": " << it.second << "\n";
  }

  llvm::outs() << "outDimSizesLog2:\n";
  for (const auto& it : outDimSizesLog2) {
    llvm::outs() << "  - " << it.first << ": " << it.second << "\n";
  }
  llvm::outs() << "\n";

  BasesT allBases;
  for (auto [inDimName, inDimSizeLog2] : inDimSizesLog2) {
    std::vector<std::vector<int32_t>> &inDimBases = allBases[inDimName];

    // Fill with zeros.
    inDimBases = std::vector<std::vector<int32_t>>(
        inDimSizeLog2, std::vector<int32_t>(outDimSizesLog2.size(), 0));

    for (auto [outDimIdx, outDimNameAndSize] : llvm::enumerate(outDimSizesLog2)) {
      auto [outDimName, outDimSize] = outDimNameAndSize;
      if (inner.hasInDim(inDimName) && inner.hasOutDim(outDimName)) {
        for (int i = 0; i < inner.getInDimSizeLog2(inDimName); i++) {
          inDimBases[i][outDimIdx] = inner.getBasis(inDimName, i, outDimName);
        }
      }
      if (outer.hasInDim(inDimName) && outer.hasOutDim(outDimName)) {
        int offset = inner.hasInDim(inDimName)   ? inner.getInDimSizeLog2(inDimName)   : 0;
        int shift  = inner.hasOutDim(outDimName) ? inner.getOutDimSizeLog2(outDimName) : 0;
        for (int i = 0; i < outer.getInDimSizeLog2(inDimName); i++) {
          inDimBases[offset + i][outDimIdx] = outer.getBasis(inDimName, i, outDimName) << shift;
        }
      }
    }
  }

  llvm::SmallVector<std::pair<StringAttr, int32_t>> outDimSizes;
  for (auto [outDim, sizeLog2] : outDimSizesLog2) {
    outDimSizes.push_back({outDim, 1 << sizeLog2});
  }
  auto result = LinearLayout(std::move(allBases), outDimSizes,
                      inner.isSurjective() && outer.isSurjective());
  
  llvm::outs() << "result: " << result << "\n";
  llvm::outs() << "---------------------------------------------------------\n\n";

  return result;
}
```
You are absolutely right to feel that way! It is a very dense piece of C++. What you are looking at is the exact engine that performs the mathematical "Product" (or Direct Sum) of two layouts.

It looks complicated because it is doing high-level linear algebra using low-level bitwise operations. To help you grasp it, let's break this function down into **four distinct phases**. Think of it like combining two different architectural blueprints into one master plan.

### Phase 1: Taking Inventory (`supremum`)

```cpp
auto inDims = supremum(...);
auto outDims = supremum(...);

```

Before combining anything, the compiler needs a complete list of all the hardware dimensions (inputs) and logical dimensions (outputs) involved.

* `supremum` is just a mathematical way to merge two lists while maintaining their relative order.
* If `inner` uses "register" and `outer` uses "thread", the `inDims` list becomes `["register", "thread"]`.

### Phase 2: Sizing Up the Grid (`inDimSizesLog2` & `outDimSizesLog2`)

```cpp
for (const auto &layout : {inner, outer}) {
  for (StringAttr inDim : layout.getInDimNames()) {
    inDimSizesLog2[inDim] += layout.getInDimSizeLog2(inDim);
  }
  // ... same for outDim
}

```

Now the compiler needs to know exactly how many binary bits are required for the master plan. It iterates through the `inner` layout, then the `outer` layout, and simply adds their bit requirements together.

* If `inner` requires 1 bit for "register" and `outer` requires 1 bit for "register", the total `inDimSizesLog2["register"]` becomes 2.
* This phase essentially calculates the exact width and height of our final combined matrix.

### Phase 3: The Core Engine (The Nested Loops)

This is the heart of the function. It allocates an empty grid of zeros (`inDimBases = ...`), and then carefully drops the `inner` and `outer` layouts into it.

**1. Dropping in the `inner` layout:**

```cpp
if (inner.hasInDim(inDimName) && inner.hasOutDim(outDimName)) {
  for (int i = 0; i < inner.getInDimSizeLog2(inDimName); i++) {
    inDimBases[i][outDimIdx] = inner.getBasis(inDimName, i, outDimName);
  }
}

```

Because `inner` is considered the "minor" layout, it gets priority. It is placed right at the beginning of the grid (starting at index `i = 0`). It copies its basis vectors exactly as they are.

**2. Dropping in the `outer` layout (`offset` and `shift`):**

```cpp
int offset = inner.hasInDim(inDimName) ? inner.getInDimSizeLog2(inDimName) : 0;
int shift = inner.hasOutDim(outDimName) ? inner.getOutDimSizeLog2(outDimName) : 0;

```

When it drops the `outer` layout into the grid, it must be extremely careful not to overwrite the bits that `inner` just claimed.

* **`offset` (Protecting the Inputs):** If `inner` used 1 input bit for "register", `offset` becomes `1`. This tells `outer`: *"Do not write to register bit 0; write your data to register bit 1."* This is why it assigns to `inDimBases[offset + i]`.
* **`shift` (Protecting the Outputs):** If `inner` already mapped data to the first bit of "dim1", `shift` becomes `1`. This tells `outer`: *"Push your basis vectors to the left by 1 bit (`<< shift`) so you control the next available logical bit."*

### Phase 4: Packaging it Up

```cpp
for (auto [outDim, sizeLog2] : outDimSizesLog2) {
  outDimSizes.push_back({outDim, 1 << sizeLog2});
}
return LinearLayout(...);

```

Finally, the compiler converts the log2 bit counts back into human-readable sizes. For example, if a dimension has a `sizeLog2` of 2, `1 << 2` converts it back to a size of 4. It then bundles the new merged dictionary into a fresh `LinearLayout` object and returns it.

---
<br/>

Please elaborate more on "Dropping in the outer layout" for the following case:
```
inDimSizesLog2:
  - "thread": 5
outDimSizesLog2:
  - "dim1": 3
  - "dim0": 2
```
This is the perfect scenario to examine because it isolates exactly how the C++ code prevents the `outer` layout from accidentally overwriting the `inner` layout.

Based on your sizing, we are looking at the execution of `threadLayout = LinearLayout::identity1D(8, "thread", "dim1") * LinearLayout::identity1D(4, "thread", "dim0")`.

* **`inner`:** Needs 3 bits of `"thread"`, outputs to `"dim1"`.
* **`outer`:** Needs 2 bits of `"thread"`, outputs to `"dim0"`.

Here is the exact step-by-step breakdown of what happens when the compiler reaches the `"Dropping in the outer layout"` phase for this specific combination.

### 1. The Initial Grid State

Before `outer` is dropped in, the code created a blank 2D grid (`inDimBases`) to hold the basis vectors for `"thread"`.

* **Rows:** 5 (Because `inDimSizesLog2["thread"]` is $3 + 2 = 5$).
* **Columns:** 2 (Index 0 is `"dim1"`, Index 1 is `"dim0"`).

The `inner` layout has already run, claiming the first 3 rows for `"dim1"`. Before `outer` starts, the grid looks like this:

* Row 0: `[1, 0]` *(Thread bit 0)*
* Row 1: `[2, 0]` *(Thread bit 1)*
* Row 2: `[4, 0]` *(Thread bit 2)*
* Row 3: `[0, 0]` *(Empty)*
* Row 4: `[0, 0]` *(Empty)*

### 2. Processing `outer` for `"dim1"` (Column Index 0)

The code loops through the output dimensions, starting with `"dim1"`.

```cpp
if (outer.hasInDim(inDimName) && outer.hasOutDim(outDimName)) { ... }

```

* `inDimName` is `"thread"`. `outDimName` is `"dim1"`.
* Does `outer` have `"dim1"` as an output? **No.** (It only outputs to `"dim0"`).
* **Action:** The `if` statement evaluates to `false`. The entire block is skipped. The `outer` layout does not touch the first column of the grid.

### 3. Processing `outer` for `"dim0"` (Column Index 1)

The loop moves to `"dim0"`.

* `inDimName` is `"thread"`. `outDimName` is `"dim0"`.
* Does `outer` have `"dim0"` as an output? **Yes.**
* **Action:** The `if` statement evaluates to `true`. We enter the block.

### 4. Calculating `offset` and `shift`

Now the code determines exactly *where* and *how* to place the `outer` bits.

**The Offset (Input Protection):**

```cpp
int offset = inner.hasInDim(inDimName) ? inner.getInDimSizeLog2(inDimName) : 0;

```

* Did `inner` use `"thread"`? Yes.
* How many bits did it use? 3 bits.
* **Result:** `offset = 3`. This tells the compiler: *"Skip the first 3 rows. Start writing at Row 3."*

**The Shift (Output Protection):**

```cpp
int shift = inner.hasOutDim(outDimName) ? inner.getOutDimSizeLog2(outDimName) : 0;

```

* Did `inner` use `"dim0"`? **No.** (It only used `"dim1"`).
* **Result:** `shift = 0`. Because `inner` and `outer` are outputting to completely different logical dimensions (different columns in our array), there is no risk of them colliding in the same integer bitmask. No bitwise shifting is required.

### 5. Executing the Loop (Writing the Bits)

Now the compiler runs the loop to write `outer`'s 2 bits into the grid.

```cpp
for (int i = 0; i < outer.getInDimSizeLog2(inDimName); i++) {
  inDimBases[offset + i][outDimIdx] = outer.getBasis(...) << shift;
}

```

* **Iteration `i = 0`:**
* Target row: `3 + 0 = 3`
* Target column: `1` (`"dim0"`)
* Basis value: `1 << 0` = `1`
* **Action:** `inDimBases[3][1]` becomes `1`. Row 3 is now `[0, 1]`.


* **Iteration `i = 1`:**
* Target row: `3 + 1 = 4`
* Target column: `1` (`"dim0"`)
* Basis value: `2 << 0` = `2`
* **Action:** `inDimBases[4][1]` becomes `2`. Row 4 is now `[0, 2]`.



### The Final Grid

By using `offset` to shift downward (protecting the hardware inputs) and `shift` to shift leftward (protecting the logical outputs), the C++ engine flawlessly builds the final 5-bit array without any manual hardcoding:

* **Row 0 (Thread bit 0):** `[1, 0]`
* **Row 1 (Thread bit 1):** `[2, 0]`
* **Row 2 (Thread bit 2):** `[4, 0]`
* **Row 3 (Thread bit 3):** `[0, 1]`
* **Row 4 (Thread bit 4):** `[0, 2]`

---
<br/>

Please elaborate more on "Dropping in the outer layout" for the following case:
```
inDimSizesLog2:
  - "register": 2
  - "thread": 5
outDimSizesLog2:
  - "dim1": 4
  - "dim0": 3
```
This is a brilliant scenario to unpack. It is the exact combination of `regLayout * threadLayout`.

Notice how this scenario is the mathematical mirror image of the previous one:

* **Previous Example:** Shared the same *input* (`"thread"`), but had different *outputs*. The compiler used **`offset`** to prevent collisions.
* **This Example:** Have different *inputs* (`inner` is `"register"`, `outer` is `"thread"`), but share the same *outputs* (`"dim1"` and `"dim0"`). The compiler will use **`shift`** to prevent collisions.

Because there are two distinct input dimensions (`"register"` and `"thread"`), the `allBases` dictionary creates **two separate 2D grids**. Let's walk through exactly how the `outer` layout (`threadLayout`) gets dropped into this structure.

### Phase 1: The `"register"` Grid (Ignored by `outer`)

The outermost loop in the C++ code iterates over the `inDimSizesLog2` keys. First, it looks at `"register"`.

* It creates a 2x2 grid.
* `inner` (`regLayout`) populates it perfectly: `[[1, 0], [0, 1]]`.
* Then the code checks `outer`: `if (outer.hasInDim("register"))`.
* Because `threadLayout` does not use registers, this returns `false`. The `outer` layout skips this grid entirely.

### Phase 2: The `"thread"` Grid (Populated by `outer`)

Now the loop moves to the `"thread"` key.

* It creates a blank 5x2 grid (5 bits for `"thread"`, 2 output dimensions).
* The code checks `inner`: `if (inner.hasInDim("thread"))`. `regLayout` does not use threads, so it skips this. The grid remains all zeros.

Now, we reach the **"Dropping in the outer layout"** phase for the `"thread"` grid. The loop iterates through the two output dimensions (`"dim1"` and `"dim0"`).

#### Step A: Processing `"dim1"` (Column Index 0)

```cpp
int offset = inner.hasInDim("thread") ? ... : 0;
int shift = inner.hasOutDim("dim1") ? inner.getOutDimSizeLog2("dim1") : 0;

```

1. **The Offset:** Did `inner` use `"thread"`? No. Therefore, **`offset = 0`**. `outer` gets to start writing at row 0 of the thread grid.
2. **The Shift (Collision Prevention!):** Did `inner` use `"dim1"`? Yes! `regLayout` claimed **1 bit** for `"dim1"` ($j_0$). Therefore, **`shift = 1`**.
3. **The Write Loop:** The code loops through all 5 bits of `outer.getInDimSizeLog2("thread")`:
* `i=0`: Basis is `1`. `1 << 1` = **`2`**. (Row 0 becomes `[2, 0]`)
* `i=1`: Basis is `2`. `2 << 1` = **`4`**. (Row 1 becomes `[4, 0]`)
* `i=2`: Basis is `4`. `4 << 1` = **`8`**. (Row 2 becomes `[8, 0]`)
* `i=3`: `threadLayout` uses bit 3 for `"dim0"`, so its `"dim1"` basis is `0`. `0 << 1` = **`0`**. (Row 3 becomes `[0, 0]`)
* `i=4`: Basis is `0`. `0 << 1` = **`0`**. (Row 4 becomes `[0, 0]`)



#### Step B: Processing `"dim0"` (Column Index 1)

```cpp
int offset = inner.hasInDim("thread") ? ... : 0;
int shift = inner.hasOutDim("dim0") ? inner.getOutDimSizeLog2("dim0") : 0;

```

1. **The Offset:** Still **`0`**.
2. **The Shift:** Did `inner` use `"dim0"`? Yes! `regLayout` claimed **1 bit** for `"dim0"` ($i_0$). Therefore, **`shift = 1`**.
3. **The Write Loop:** The code loops through all 5 bits again:
* `i=0`: Basis is `0`. `0 << 1` = **`0`**. (Row 0 stays `[2, 0]`)
* `i=1`: Basis is `0`. `0 << 1` = **`0`**. (Row 1 stays `[4, 0]`)
* `i=2`: Basis is `0`. `0 << 1` = **`0`**. (Row 2 stays `[8, 0]`)
* `i=3`: `threadLayout` maps bit 3 to `"dim0"` with a basis of `1`. `1 << 1` = **`2`**. (Row 3 becomes `[0, 2]`)
* `i=4`: `threadLayout` maps bit 4 to `"dim0"` with a basis of `2`. `2 << 1` = **`4`**. (Row 4 becomes `[0, 4]`)



### The Final Dictionary Result

By correctly applying an `offset` of 0 (because the physical inputs don't overlap) and a `shift` of 1 (because both logical outputs overlap), the compiler perfectly stacks the layout into the final dictionary without overwriting a single bit.

* **`allBases["register"]`:**
  * Row 0: `[1, 0]` *(Controls $j_0$)*
  * Row 1: `[0, 1]` *(Controls $i_0$)*


* **`allBases["thread"]`:**
  * Row 0: `[2, 0]` *(Controls $j_1$)*
  * Row 1: `[4, 0]` *(Controls $j_2$)*
  * Row 2: `[8, 0]` *(Controls $j_3$)*
  * Row 3: `[0, 2]` *(Controls $i_1$)*
  * Row 4: `[0, 4]` *(Controls $i_2$)*
<br/>

# The `bases` data member in `LinearLayout` class
```C++
  llvm::MapVector<StringAttr /*inDim*/,
                  std::vector<std::vector<int32_t> /*size=getNumOutDims()*/>
                  /*size=getInDimSizeLog2(inDim)*/>
      bases;
```
`bases` stores columns $C_0,\, C_1,\, \cdot, C_n$ of matrix $A$. When $A$ is an identity matrix, $C_i$ are basis vectors in the input space.
