# LinearLayout::invertAndCompose()
```C++
LinearLayout LinearLayout::invertAndCompose(const LinearLayout &outer) const {
  // TODO(Lezcano) Make friend and perhaps rename to `convertFrom` or `lstsq`
  // For this, we need to implement our LLVM lowerings by inverting the "outer"
  // layout, and then iterating over the elements from the "this" layout and
  // fetching the corresponding element from the "outer" layout. This exercises
  // the broadcasting that we incentivise via choosing the minimum norm solution
  // in lstsq.

  // The order of dims does not matter. We choose to transpose outer
  auto outDims = llvm::to_vector(getOutDimNames());
  assertDimsEqualIgnoringOrder(outDims, outer.getOutDimNames());
  const auto &B = *this;

  // It creates a transposed version of $A$ so that the order of $A$'s output dimensions perfectly aligns with $B$'s.
  const auto A = outer.transposeOuts(outDims);

  for (auto dim : outDims) {
    assert(A.getOutDimSize(dim) >= B.getOutDimSize(dim) &&
           ("A.invertAndCompose(B) called with incompatible output shapes in " +
            dim.str() + ": " + std::to_string(A.getOutDimSize(dim)) +
            " >= " + std::to_string(B.getOutDimSize(dim)))
               .c_str());
  }

  // Broadcasting heuristic
  // Imagine we have two layouts with `warps = [[0, 0],  [0, 0]]`
  // (broadcasting) on both layouts. We could map any warp to any warp in the
  // conversion. Now, we want to map them as the identity map, to mark that
  // nothing needs to be done there (`lstsq` would map all the warps to the
  // zero warp, minimum norm solution). The heuristic here is as follows:
  // - If a dimension is the same for both layouts, we want to map it as the
  // identity
  //   Equivalently, we don't add it to the conversion
  // - Otherwise, we just call lstsq (i.e. map all the equivalent elements
  //   to the same input element) to take advantage of broadcasting in shared
  //   memory and avoid saving repeated elements in shared memory

  // FIXME: We should check that the other dimensions don't touch the image of
  // this dimension.
  SmallVector<StringAttr> identityDims;
  for (auto dim : A.getInDimNames()) {
    if (B.hasInDim(dim) &&
        A.sublayout(dim, outDims) == B.sublayout(dim, outDims)) {
      identityDims.push_back(dim);
    }
  }
  SmallVector<StringAttr> ANonIdentityInDims;
  SmallVector<StringAttr> BNonIdentityInDims;
  for (auto dim : A.getInDimNames()) {
    if (!llvm::is_contained(identityDims, dim)) {
      ANonIdentityInDims.push_back(dim);
    }
  }
  for (auto dim : B.getInDimNames()) {
    if (!llvm::is_contained(identityDims, dim)) {
      BNonIdentityInDims.push_back(dim);
    }
  }

  auto AReduced = A.sublayout(ANonIdentityInDims, outDims);
  auto BReduced = B.sublayout(BNonIdentityInDims, outDims);

  // If one is empty, the other must be empty as well
  assert((ANonIdentityInDims.empty()) == (BNonIdentityInDims.empty()));
  bool isEmpty = ANonIdentityInDims.empty();

  auto ret = isEmpty ? LinearLayout::empty() : lstsq(AReduced, BReduced);

  // TODO(Lezcano): We should return the reduced layout instead of re-adding the
  // identity maps. With this, we'll be able to kill `minimalCvtLayout`

  // Add the identity maps for the dimensions that are the same for both layouts
  for (auto dim : identityDims) {
    ret *= LinearLayout::identity1D(A.getInDimSize(dim), dim, dim);
  }

  // Reorder the dimensions in the result to match the order expected by the
  // current and outer layouts.
  return ret.transposeIns(llvm::to_vector(B.getInDimNames()))
      .transposeOuts(llvm::to_vector(A.getInDimNames()));
}
```

To understand `LinearLayout::invertAndCompose` deeply, we need to look at it from three angles: **the mathematical goal**, **the compiler optimization goal (broadcasting)**, and **the step-by-step code implementation**.

In the Triton compiler, a `LinearLayout` represents a mathematical mapping (usually a linear transformation over $GF(2)$, the Galois field of two elements) from multidimensional input indices (like thread IDs, block IDs) to output multidimensional indices (like memory addresses or register indices).

Here is the deep dive into what this function does.

### 1. The High-Level Mathematical Goal

The function is called on a layout `this` (let's call it $B$) and takes another layout `outer` (let's call it $A$).

The goal is to find a conversion layout $X$ such that applying $A$ to $X$ yields $B$. Mathematically, you are solving the equation:


$$A \cdot X = B$$

$$X = A^{-1} \cdot B$$

This is exactly why the function is named `invertAndCompose`—it conceptually inverts the `outer` layout ($A$) and composes it with `this` layout ($B$). In the context of Triton, this is used for **layout conversions** (e.g., when a tensor is residing in shared memory with layout $A$ and a thread needs to load it into registers with layout $B$).

### 2. The Step-by-Step Code Walkthrough

Let's break down the code into its logical phases.

#### Phase 1: Alignment and Compatibility

```cpp
auto outDims = llvm::to_vector(getOutDimNames());
assertDimsEqualIgnoringOrder(outDims, outer.getOutDimNames());
const auto &B = *this;
const auto A = outer.transposeOuts(outDims);

```

Before doing any math, the output dimensions of both layouts must match.

* It grabs the output dimensions of $B$ (`this`).
* It ensures $A$ (`outer`) has the exact same output dimensions.
* It creates a transposed version of $A$ so that the *order* of $A$'s output dimensions perfectly aligns with $B$'s.

```cpp
for (auto dim : outDims) {
  assert(A.getOutDimSize(dim) >= B.getOutDimSize(dim) && ...);
}

```

It then verifies that the output size of the target layout $A$ is large enough to accommodate the source layout $B$. You cannot map to a layout that is smaller than what you are coming from.

#### Phase 2: The Broadcasting Heuristic (The "Fast Path")

This is the most heavily commented and crucial optimization in the function.

```cpp
SmallVector<StringAttr> identityDims;
for (auto dim : A.getInDimNames()) {
  if (B.hasInDim(dim) &&
      A.sublayout(dim, outDims) == B.sublayout(dim, outDims)) {
    identityDims.push_back(dim);
  }
}

```

If we strictly used a mathematical solver for everything, it would blindly map unconstrained inputs to 0.

Imagine two layouts that both broadcast a dimension (e.g., they both map `warp_id` to $0$, meaning all warps compute the same thing). The standard Least Squares (`lstsq`) solver would find the minimum norm solution, mapping everything to warp 0.

* **The problem:** Mapping everything to warp 0 means warp 1, 2, and 3 would just read the data that warp 0 computed. This requires syncing through shared memory.
* **The solution (Identity mapping):** If a dimension behaves exactly the same in *both* layout $A$ and layout $B$ (checked by comparing their `sublayout`), the compiler recognizes it can just map that dimension 1-to-1 (Identity). Warp 1 maps to Warp 1, Warp 2 maps to Warp 2. No shared memory transfer is needed because they are already doing the exact same thing!

#### Phase 3: Dimension Separation

```cpp
SmallVector<StringAttr> ANonIdentityInDims;
SmallVector<StringAttr> BNonIdentityInDims;
// ... (loops populating these vectors)
auto AReduced = A.sublayout(ANonIdentityInDims, outDims);
auto BReduced = B.sublayout(BNonIdentityInDims, outDims);

```

The function separates the dimensions that can be trivially mapped (the `identityDims`) from the complex dimensions that require actual mathematical conversion (`NonIdentityInDims`). It creates "reduced" versions of layout $A$ and $B$ that strip away the easy stuff.

#### Phase 4: Solving the Layout Conversion

```cpp
auto ret = isEmpty ? LinearLayout::empty() : lstsq(AReduced, BReduced);

```

This is the heart of the function. For the complex dimensions, it relies on `lstsq` (Least Squares).
Because layout transformations in Triton are linear matrices (often binary matrices), `lstsq` solves $A_{reduced} \cdot X = B_{reduced}$ for $X$.

Why "Least Squares" and not just inverse? Because $A$ might not be a perfectly square, invertible matrix. It might have redundant or broadcasted dimensions. `lstsq` mathematically finds the **minimum norm solution**. In binary layout math, a minimum norm solution maximizes the number of zeros. Maximizing zeros means maximizing the use of the "zero-th" element (e.g., lane 0, warp 0), which naturally facilitates efficient hardware broadcasting.

#### Phase 5: Reassembling the Final Layout

```cpp
for (auto dim : identityDims) {
  ret *= LinearLayout::identity1D(A.getInDimSize(dim), dim, dim);
}

return ret.transposeIns(llvm::to_vector(B.getInDimNames()))
          .transposeOuts(llvm::to_vector(A.getInDimNames()));

```

Finally, the function patches everything back together:

1. It takes the mathematically solved layout (`ret`) and multiplies (`*=`) it by the 1D identity layouts we stripped out in Phase 2. In layout algebra, this acts as a tensor product, stitching the trivial mappings back into the complex mapping.
2. It reorders the input and output dimensions (`transposeIns`, `transposeOuts`) to guarantee the resulting layout's API signature perfectly expects $B$'s inputs and outputs to $A$'s inputs.

### Summary

In short, `invertAndCompose` computes how to rearrange data from layout $A$ to layout $B$. It does this by dividing the problem into two parts:

1. **Identical dimensions**, which are mapped 1-to-1 to bypass shared memory overhead.
2. **Differing dimensions**, which are solved algebraically using a Least Squares solver to naturally incentivize memory-efficient hardware broadcasting.
