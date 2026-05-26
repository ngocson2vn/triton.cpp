## Basis Vectors
In linear algebra, any matrix $A$ representing a linear transformation is entirely defined by what it does to the standard basis vectors of the input space.

Here is the formal proof of that claim. It is one of the most elegant and foundational proofs in linear algebra, and it relies entirely on the two defining properties of a linear transformation: **additivity** and **scalar multiplication**.

### The Setup

Let $V$ and $W$ be vector spaces over a field (like our binary field $\mathbb{F}_2$ or the real numbers).

Let $L$ be a linear transformation from $V$ to $W$ ($L: V \to W$).

By definition, a transformation is "linear" if and only if it satisfies two conditions for any vectors $u, v \in V$ and any scalar $c$:

1. **Additivity:** $L(u + v) = L(u) + L(v)$
2. **Scalar Multiplication:** $L(cv) = cL(v)$

Now, let $e_1, e_2, \dots, e_n$ be the standard basis vectors for the input space $V$. By the definition of a basis, *any* arbitrary vector $v \in V$ can be written uniquely as a linear combination of these basis vectors:

$$v = c_1e_1 + c_2e_2 + \dots + c_ne_n$$

where $c_1, c_2, \dots, c_n$ are scalar coefficients (in the case of $\mathbb{F}_2$, these scalars are simply bits, 0 or 1).

### The Proof

We want to find out what the transformation $L$ does to our arbitrary vector $v$. We start by applying $L$ to both sides of our equation:

$$L(v) = L(c_1e_1 + c_2e_2 + \dots + c_ne_n)$$

First, we apply the **additivity** property of linear transformations. We can split the single transformation of a sum into a sum of individual transformations:

$$L(v) = L(c_1e_1) + L(c_2e_2) + \dots + L(c_ne_n)$$

Next, we apply the **scalar multiplication** property. We can pull the scalar coefficients out to the front of each transformation:

$$L(v) = c_1L(e_1) + c_2L(e_2) + \dots + c_nL(e_n)$$

### The Conclusion

Look closely at the final equation. To calculate $L(v)$ for *any* arbitrary vector $v$, the only pieces of information you need are:

1. The scalars $c_1, c_2, \dots, c_n$ (which are just the coordinates of the input vector $v$).
2. The vectors $L(e_1), L(e_2), \dots, L(e_n)$.

The vectors $L(e_1), \dots, L(e_n)$ are exactly the **mapped outputs of the standard basis vectors**.

Because $v$ was chosen to be any arbitrary vector in the entire space, this equation proves that if you know what the transformation does to the basis vectors, you possess all the information needed to calculate what it does to literally every other vector in existence within that space.

When you organize this linear transformation into a matrix $A$, you simply place $L(e_1)$ as the first column, $L(e_2)$ as the second column, and so on. This is why storing the transformed basis vectors—as Triton's C++ code does—is mathematically identical to storing the full transformation matrix $A$.

### The `bases` data member in `LinearLayout` class
```C++
  llvm::MapVector<StringAttr /*inDim*/,
                  std::vector<std::vector<int32_t> /*size=getNumOutDims()*/>
                  /*size=getInDimSizeLog2(inDim)*/>
      bases;
```
`bases` stores images (mapped outputs) of the **standard basis vectors** of the **1D input space** under the linear transformation $L$.