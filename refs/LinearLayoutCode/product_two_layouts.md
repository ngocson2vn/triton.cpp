# Product of 2 LinearLayouts
Implementation: lib/Tools/LinearLayout.cpp
```C++
LinearLayout operator*(LinearLayout inner, LinearLayout outer) {
  // Check that dims common to outer and inner have the same relative order.
  auto inDims = supremum(llvm::to_vector(inner.getInDimNames()),
                         llvm::to_vector(outer.getInDimNames()));
  auto outDims = supremum(llvm::to_vector(inner.getOutDimNames()),
                          llvm::to_vector(outer.getOutDimNames()));

  int level = 1;
  auto level_str = std::getenv("SONY_LOG_LEVEL");
  if (level_str != nullptr) {
    level = std::stoi(level_str);
  }

  auto& sonyOs = getSonyOs(level);
  sonyOs << "--- operator*(LinearLayout inner, LinearLayout outer) ---\n";
  
  sonyOs << "inner: " << inner << "\n\n";

  sonyOs << "outer: " << outer << "\n\n";

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

  sonyOs << "inDimSizesLog2:\n";
  for (const auto& it : inDimSizesLog2) {
    sonyOs << "  - " << it.first << ": " << it.second << "\n";
  }

  sonyOs << "outDimSizesLog2:\n";
  for (const auto& it : outDimSizesLog2) {
    sonyOs << "  - " << it.first << ": " << it.second << "\n";
  }
  sonyOs << "\n";

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
  
  sonyOs << "result: " << result << "\n";
  sonyOs << "---------------------------------------------------------\n\n";

  return result;
}
```

## Foundational Algebra
Given the following linear maps:

$L_1: \; F_2^{N_1} \rightarrow F_2^{Q_{dim1}} \times F_2^{Q_{dim0}}$

$L_2: \; F_2^{N_2} \rightarrow F_2^{P_{dim1}} \times F_2^{P_{dim0}}$

Take Cartesian product (direct sum) of $L_1$ and $L_2$:

$L_1 \times L_2: \; F_2^{N_1} \times F_2^{N_2} \rightarrow (F_2^{Q_{dim1}} \times F_2^{Q_{dim0}}) \times (F_2^{P_{dim1}} \times F_2^{P_{dim0}})$

Denote $L = L_1 \times L_2$. 

$L: \; F_2^{N_1} \times F_2^{N_2} \rightarrow (F_2^{Q_{dim1}} \times F_2^{Q_{dim0}}) \times (F_2^{P_{dim1}} \times F_2^{P_{dim0}})$

Let $\mathbf{0}_1$ and $\mathbf{f}_i$ denote the zero vector and the $i$-th standard basis vector in $F_2^{N_1}$, respectively.

Let $\boldsymbol{\theta}_1$ denote the zero vector in $F_2^{Q_{dim1}} \times F_2^{Q_{dim0}}$

Let $\mathbf{0}_2$ and $\mathbf{g}_j$ denote the zero vector and the $j$-th standard basis vector in $F_2^{N_2}$, respectively.

Let $\boldsymbol{\theta}_2$ denote the zero vector in $F_2^{P_{dim1}} \times F_2^{P_{dim0}}$

Let $\mathbf{e}_k$ denote $k$-th standard basis vector in $F_2^{N_1} \times F_2^{N_2}$. 

By definition, $\mathbf{e}_k$ can be represented as follows:

$\mathbf{e}_k = (\mathbf{f}_k, \mathbf{0}_2)$ for $0 \le k < N_1$.

$\mathbf{e}_k = (\mathbf{0}_1, \mathbf{g}_{k-N_1})$ for $N_1 \le k < N_1 + N_2$.
<br/><br/>

Calculate $L(e_k)$:

For $0 \le k < N_1$
- $L(e_k) = L(\mathbf{f}_k, \mathbf{0}_2) = (L_1(\mathbf{f}_k), L_2(\mathbf{0}_2)) = (L_1(\mathbf{f}_k), \boldsymbol{\theta}_2) = \big((L_1(\mathbf{f}_k)_{dim1}, L_1(\mathbf{f}_k)_{dim0}), (\boldsymbol{\theta}_{2,dim1}, \boldsymbol{\theta}_{2,dim0})\big)$

For $N_1 \le k < N_1 + N_2,\; h = k - N_1$
- $L(e_k) = L(\mathbf{0}_1, \mathbf{g}_h) = (L_1(\mathbf{0}_1), L_2(\mathbf{g}_h)) = (\boldsymbol{\theta}_1, L_2(\mathbf{g}_h)) = \big((\boldsymbol{\theta}_{1,dim1}, \boldsymbol{\theta}_{1,dim0}), (L_2(\mathbf{g}_h)_{dim1}, L_2(\mathbf{g}_h)_{dim0})\big)$


<br/><br/>
Let $R$ denote the following **Canonical Isomorphism** (Regrouping):

$R: \; (F_2^{Q_{dim1}} \times F_2^{Q_{dim0}}) \times (F_2^{P_{dim1}} \times F_2^{P_{dim0}}) \rightarrow (F_2^{Q_{dim1}} \times F_2^{P_{dim1}}) \times (F_2^{Q_{dim0}} \times F_2^{P_{dim0}})$

Let $\Omega$ denote the following **Canonical Isomorphism** (Flattening):

$\Omega: \; (F_2^{Q_{dim1}} \times F_2^{P_{dim1}}) \times (F_2^{Q_{dim0}} \times F_2^{P_{dim0}}) \rightarrow F_2^{Q_{dim1} + P_{dim1}} \times F_2^{Q_{dim0} + P_{dim0}}$

Let $T = \Omega \circ R \circ L$ denote a composition of $\Omega$ and $R$ and $L$. Then, $T$ is also a linear map:

$T: \; F_2^{N_1} \times F_2^{N_2} \rightarrow F_2^{Q_{dim1} + P_{dim1}} \times F_2^{Q_{dim0} + P_{dim0}}$

Now let's calculate $T(e_k)$:

$T(e_k) = (\Omega \circ R \circ L)(e_k) = \Omega(R(L(e_k)))$

For $0 \le k < N_1$
- $T(e_k) = \Omega \big( R\big((L_1(\mathbf{f}_k)_{dim1}, L_1(\mathbf{f}_k)_{dim0}), (\boldsymbol{\theta}_{2,dim1}, \boldsymbol{\theta}_{2,dim0})\big)\big) = \Omega\big((L_1(\mathbf{f}_k)_{dim1}, \boldsymbol{\theta}_{2,dim1}), (L_1(\mathbf{f}_k)_{dim0}, \boldsymbol{\theta}_{2,dim0})\big)$

  Since $L_1(\mathbf{f}_k)_{dim1}$ and $L_1(\mathbf{f}_k)_{dim0}$ are binary vectors, we can represent them by the integers $t_{k,dim1}$ and $t_{k,dim0}$, respectively.

  The flattened forms of $(L_1(\mathbf{f}_k)_{dim1}, \boldsymbol{\theta}_{2,dim1})$ and $(L_1(\mathbf{f}_k)_{dim0}, \boldsymbol{\theta}_{2,dim0})$ are binary vectors whose higher-order bits are zero. Obviously, they can be reduced exactly to the integers $t_{k,dim1}$ and $t_{k,dim0}$, respectively.

  $T(e_k) = (t_{k,dim1}$, $t_{k,dim0})$  

For $N_1 \le k < N_1 + N_2,\; h = k - N_1$
- $T(e_k) = \Omega\big(R\big((\boldsymbol{\theta}_{1,dim1}, \boldsymbol{\theta}_{1,dim0}), (L_2(\mathbf{g}_h)_{dim1}, L_2(\mathbf{g}_h)_{dim0})\big)\big) = \Omega\big((\boldsymbol{\theta}_{1,dim1}, L_2(\mathbf{g}_h)_{dim1}), (\boldsymbol{\theta}_{1,dim0}, L_2(\mathbf{g}_h)_{dim0})\big)$

  Similarly, we can represent $L_2(\mathbf{g}_h)_{dim1}$ and $L_2(\mathbf{g}_h)_{dim0}$ by the integers $w_{k,dim1}$ and $w_{k,dim0}$, respectively.

  $T(e_k) = (w_{k,dim1} \ll Q_{dim1}, w_{k,dim0} \ll Q_{dim0})$
<br/><br/>

Obviously, the above **`operator*()`** computes $T(e_k)$.
<br/><br/>

**Side Note:**<br/>
In algebra, taking the Cartesian product of the domain spaces $F_2^{N_1}$ and $F_2^{N_2}$ creates a space of ordered pairs of vectors: $(\mathbf{x}_1, \mathbf{x}_2)$.

Because flattening a pair of vectors (one of length $N_1$ and one of length $N_2$) simply yields a single vector of length $N_1 + N_2$, there is a natural isomorphism:

$F_2^{N_1} \times F_2^{N_2} \cong F_2^{N_1 + N_2}$

Because of this seamless translation, treating the domain as a single, combined space $F_2^{N_1 + N_2}$ is both correct and standard practice.