# Column shift function
Given the following function:

$S(\boldsymbol{r}) = 2^v((\boldsymbol{r} \gg p) \mathbin{\&}(2^q - 1))$

where
- $\boldsymbol{r} \in F_2^m$
- $m,\ n,\ p,\ q,\ v$ are positive integers and $m,\ p,\ q,\ v \le n$

Prove that $S(\boldsymbol{r})$ is a linear map: $F_2^m \rightarrow F_2^n$

## Proof
When we move into $F_2^m$ (the vector space of $m$-bit binary strings over the finite field of two elements), the concept of "addition" is no longer standard integer addition; it becomes the **bitwise XOR** operation ($\oplus$).

For a function to be linear in $F_2^m$, it must satisfy the **Boolean superposition** principle:


$S(\boldsymbol{a} \oplus \boldsymbol{b}) = S(\boldsymbol{a}) \oplus S(\boldsymbol{b})$

Here is exactly why your function is a strictly linear transformation in this domain:

### 1. The Right Shift is a Linear Operator

In $F_2^m$, shifting a bit vector to the right by $p$ positions ($\gg p$) simply drops the lowest $p$ bits and pulls in zeros at the top. This operates entirely independently on each bit position. In linear algebra terms, it is a matrix transformation. It distributes perfectly over XOR:


$(\boldsymbol{a} \oplus \boldsymbol{b}) \gg p = (\boldsymbol{a} \gg p) \oplus (\boldsymbol{b} \gg p)$

### 2. The Bitwise AND is a Linear Projection

The term $(2^q - 1)$ represents a constant bitmask of $q$ ones. A bitwise AND with a constant mask merely isolates specific bits (projecting them as they are) and zeroes out the rest. Because there are no "carries" across bits, this is also a linear operation over $F_2^n$:


$(\boldsymbol{x} \oplus \boldsymbol{y}) \mathbin{\&} (2^q - 1) = (\boldsymbol{x} \mathbin{\&} (2^q - 1)) \oplus (\boldsymbol{y} \mathbin{\&} (2^q - 1))$

### 3. The Multiplier Becomes a Left Shift

In standard arithmetic, mixing integer multiplication with bitwise operators usually destroys linearity. However, because your multiplier is $2^v$ (a power of two), multiplying by it is mathematically identical to performing a **logical left shift** by $v$ bits ($\ll v$).

Just like the right shift, a left shift is a linear operator in $F_2^n$:


$((\boldsymbol{x} \oplus \boldsymbol{y}) \ll v) = (\boldsymbol{x} \ll v) \oplus (\boldsymbol{y} \ll v)$

---

### The Final Proof
#### Additivity
Let's define the constant mask $M = 2^q - 1$.

Because multiplication by $2^v$ acts as a left shift, we can rewrite the function purely in bitwise terms:


$S(\boldsymbol{r}) = ((\boldsymbol{r} \gg p) \mathbin{\&} M) \ll v$

Now, evaluate $S(\boldsymbol{a} \oplus \boldsymbol{b})$:


$S(\boldsymbol{a} \oplus \boldsymbol{b}) = (((\boldsymbol{a} \oplus \boldsymbol{b}) \gg p) \mathbin{\&} M) \ll v$

$= (((\boldsymbol{a} \gg p) \oplus (\boldsymbol{b} \gg p)) \mathbin{\&} M) \ll v$

$= (((\boldsymbol{a} \gg p) \mathbin{\&} M) \oplus ((\boldsymbol{b} \gg p) \mathbin{\&} M)) \ll v$

$= (((\boldsymbol{a} \gg p) \mathbin{\&} M) \ll v) \oplus (((\boldsymbol{b} \gg p) \mathbin{\&} M) \ll v)$

$= S(\boldsymbol{a}) \oplus S(\boldsymbol{b})$

#### Scalar multiplication

In linear algebra, for a function to be a true linear transformation over a vector space, it must satisfy both **additivity** (which we proved in the last step) and **homogeneity** (scalar multiplication):


$S(c \cdot \boldsymbol{r}) = c \cdot S(\boldsymbol{r})$


where $c$ is a scalar from the underlying field.

Because the vector space is $F_2^m$, the underlying field is simply $F_2$ (the Galois Field of 2).

In $F_2$, there are only two possible scalars: **$0$ and $1$**.

Let's test both possible values for the scalar $c$:

**Case 1: The scalar is $1$**
If $c = 1$, the property is trivially true for any function:


$S(1 \cdot \boldsymbol{r}) = S(\boldsymbol{r}) = 1 \cdot S(\boldsymbol{r})$

**Case 2: The scalar is $0$**
If $c = 0$, the vector $r$ becomes a zero-vector ($0 \cdot \boldsymbol{r} = \boldsymbol{0}$). We just need to evaluate $S(\boldsymbol{0})$:


$S(\boldsymbol{0}) = 2^v((\boldsymbol{0} \gg p) \mathbin{\&} (2^q - 1))$

* $\boldsymbol{0} \gg p = \boldsymbol{0}$
* $\boldsymbol{0} \mathbin{\&} (2^q - 1) = \boldsymbol{0}$
* $2^v \cdot \boldsymbol{0} = \boldsymbol{0}$

So, $S(\boldsymbol{0}) = \boldsymbol{0}$.
Since $0 \cdot S(\boldsymbol{r})$ is also $\boldsymbol{0}$, the equation holds:


$S(0 \cdot \boldsymbol{r}) = \boldsymbol{0} = 0 \cdot S(\boldsymbol{r})$

### Conclusion

Because the only possible scalars are $0$ and $1$, and the function naturally maps an all-zero input to an all-zero output (meaning the origin is preserved), **scalar multiplication is strictly satisfied.** Combined with the additivity we proved earlier, the function $S(\boldsymbol{r})$ checks every box. It is a completely valid, rigorous linear transformation from the vector space $F_2^m$ to vector space $F_2^n$.

**Side note:** <br/>
3 bitwise operations can be converted to matrix multiplications
- $\boldsymbol{r} \gg p$ is equivalent to multiplication of a right shift matrix $\stackrel{n \times m}{M_1}$ and column vector $\boldsymbol{r}$. 

- Denote $\boldsymbol{a} = \boldsymbol{r} \gg p$ be a binary vector in $F_2^n$.<br/> Then, $\boldsymbol{a} \mathbin{\&} (2^q - 1)$ is equivalent to multiplication of a diagonal matrix $\stackrel{n \times n}{M_2}$ which has only $q$ ones along the main diagonal with the column vector $\boldsymbol{a}$.

- Denote $\boldsymbol{b} = \boldsymbol{a} \mathbin{\&} (2^q - 1)$ be a binary vector in $F_2^n$.<br/> Then, $\boldsymbol{b} \ll 2^v$ is equivalent to multiplication of a left shift matrix $\stackrel{n \times n}{M_3}$ with the column vector $\boldsymbol{b}$

<br/>

$S(\boldsymbol{r})$ can be rewritten as

$`S(\boldsymbol{r}) = \, \stackrel{n \times n}{M_3}(\stackrel{n \times n}{M_2}(\stackrel{n \times m}{M_1}\boldsymbol{r})) = \stackrel{n \times m}{M}\boldsymbol{r}`$, where $M = M_3 M_2 M_1$

<br/>

# Swizzling Linear Map

Start with building a linear map to map memory offsets to tensor indices as follows:

Identity map $L_1$:

$
\begin{aligned}
L_1:\ F_2^m &\rightarrow F_2^m \\
\boldsymbol{r} &\mapsto \boldsymbol{r} \\
\end{aligned}
$

Identity map $L_2$:

$
\begin{aligned}
L_2:\ F_2^n &\rightarrow F_2^n \\
\boldsymbol{c} &\mapsto \boldsymbol{c} \\
\end{aligned}
$

Direct sum $L_1 \times L_2$:

$
\begin{aligned}
L_1 \times L_2:\ F_2^m \times F_2^n &\rightarrow F_2^m \times F_2^n \\
(\boldsymbol{r}, \boldsymbol{c})  &\mapsto (\boldsymbol{r}, \boldsymbol{c})
\end{aligned}
$

$
\begin{aligned}
&\phi:\ F_2^m \times F_2^n \rightarrow F_2^{m + n} \\
&((r_0, \dots, r_{m-1}),(c_0, \dots, c_{n-1})) \mapsto (c_0, \dots, c_{n-1}, r_0, \dots, r_{m-1})
\end{aligned}
$

$
\begin{aligned}
&\phi^{-1}:\ F_2^{m + n} \rightarrow F_2^m \times F_2^n \\
&(c_0, \dots, c_{n-1}, r_0, \dots, r_{m-1}) \mapsto ((r_0, \dots, r_{m-1}),(c_0, \dots, c_{n-1}))
\end{aligned}
$

$
\begin{aligned}
&L = (L_1 \times L_2) \circ \phi^{-1}:\ F_2^{m + n} \rightarrow F_2^m \times F_2^n
\end{aligned}
$

For $0 \le k \lt m$: <br/>
$L(\boldsymbol{e}_k) = (L_1(\boldsymbol{u}_k), \boldsymbol{0}^n) = (\boldsymbol{u}_k, \boldsymbol{0}^n)$, where $\boldsymbol{u}_k$ is $k$-th standard basis vector of $F_2^m$

For $m \le k \lt m + n$: <br/>
$L(\boldsymbol{e}_k) = (\boldsymbol{0}^m, L_2(\boldsymbol{v}_{k-m})) = (\boldsymbol{0}^m, \boldsymbol{v}_{k-m})$, where $\boldsymbol{v}_h$ is $h$-th standard basis vector of $F_2^n$

<br/>

Let $\Omega$ denote the following map:

$
\begin{aligned}
\Omega:\ F_2^m \times F_2^n &\rightarrow F_2^m \times F_2^n \\
(\boldsymbol{r}, \boldsymbol{c})  &\mapsto (\boldsymbol{r}, \boldsymbol{c} \oplus S(\boldsymbol{r}))
\end{aligned}
$

Then, we can easily prove that $\Omega$ is a linear map because $S(\boldsymbol{r})$ is a linear map.

<br/>

Denote $Z = \Omega \circ L:\ F_2^{m + n} \rightarrow F_2^m \times F_2^n$

Then, we can compute $Z(\boldsymbol{e}_k)$ as follows:

For $0 \le k \lt m$: <br/>
$Z(\boldsymbol{e}_k) = (\boldsymbol{u}_k, \boldsymbol{0}^n \oplus S(\boldsymbol{u}_k)) = (\boldsymbol{u}_k, S(\boldsymbol{u}_k))$

For $m \le k \lt m + n$: <br/>
$Z(\boldsymbol{e}_k) = (\boldsymbol{0}^m, \boldsymbol{v}_{k-m} \oplus S(\boldsymbol{0}^m)) = (\boldsymbol{0}^m, \boldsymbol{v}_{k-m})$

$Z$ is called a swizzling linear map.
<br/>

## How does $Z$ help resolve GPU Shared Memory Bank Conflicts?

Firstly, we prove that the swizzling linear map $Z = \Omega \circ L$ is invertible. 

We can evaluate the invertibility of its component maps, $L$ and $\Omega$. <br/>
A composition of two functions is invertible if and only if both individual functions are invertible.

Here is the step-by-step proof.

### 1. Invertibility of $L$

The map $L: F_2^{m+n} \rightarrow F_2^m \times F_2^n$ takes a vector $\boldsymbol{o}$ of length $m+n$ and splits it into two vectors: $\boldsymbol{r}$ of length $m$ and $\boldsymbol{c}$ of length $n$.

This is the canonical isomorphism between a vector space and the direct sum of its subspaces. Because it merely partitions the coordinates of $\boldsymbol{o}$ without losing or altering any information, $L$ is a clear bijection.

* **Inverse of $L$:** The inverse function $L^{-1}$ simply concatenates the two vectors back together: $L^{-1}(\boldsymbol{r}, \boldsymbol{c}) = \boldsymbol{o}$.

### 2. Invertibility of $\Omega$

The map $\Omega: F_2^m \times F_2^n \rightarrow F_2^m \times F_2^n$ is defined by:
$\Omega(\boldsymbol{r}, \boldsymbol{c}) = (\boldsymbol{r}, \boldsymbol{c} \oplus S(\boldsymbol{r}))$

We can prove $\Omega$ is invertible by demonstrating that it is an **involution** (a function that is its own inverse). Let's apply $\Omega$ to its own output:

$\Omega(\Omega(\boldsymbol{r}, \boldsymbol{c})) = \Omega(\boldsymbol{r}, \boldsymbol{c} \oplus S(\boldsymbol{r}))$

By the definition of $\Omega$, we map the first argument identically, and the second argument gets XORed with $S$ applied to the first argument:

$\Omega(\boldsymbol{r}, \boldsymbol{c} \oplus S(\boldsymbol{r})) = (\boldsymbol{r}, (\boldsymbol{c} \oplus S(\boldsymbol{r})) \oplus S(\boldsymbol{r}))$

Because vector addition over the finite field $F_2$ is equivalent to the bitwise XOR operation, any element added to itself results in the zero vector (i.e., $x \oplus x = 0$). Also, vector addition in $F_2$ is associative. Therefore:

$\boldsymbol{c} \oplus (S(\boldsymbol{r}) \oplus S(\boldsymbol{r})) = \boldsymbol{c} \oplus \boldsymbol{0} = \boldsymbol{c}$

Substituting this back gives:

$\Omega(\Omega(\boldsymbol{r}, \boldsymbol{c})) = (\boldsymbol{r}, \boldsymbol{c})$

Since applying $\Omega$ twice returns the original input, $\Omega$ is its own inverse ($\Omega^{-1} = \Omega$). Because it has a well-defined inverse, $\Omega$ is bijective and therefore invertible.

### 3. Conclusion for $Z$

The map $Z$ is defined as the composition of $\Omega$ and $L$:

$Z = \Omega \circ L$

Since both $\Omega$ and $L$ are invertible (bijective) mappings, their composition $Z$ must also be strictly invertible.

We can even construct the explicit inverse map, $Z^{-1} : F_2^m \times F_2^n \rightarrow F_2^{m+n}$, using the property of inverse compositions $(A \circ B)^{-1} = B^{-1} \circ A^{-1}$:

$Z^{-1} = L^{-1} \circ \Omega^{-1}$

Since we established that $\Omega^{-1} = \Omega$, we can simplify this to:

$Z^{-1}(\boldsymbol{r}, \boldsymbol{c}) = L^{-1}(\Omega(\boldsymbol{r}, \boldsymbol{c})) = L^{-1}(\boldsymbol{r}, \boldsymbol{c} \oplus S(\boldsymbol{r}))$

Thus, $Z$ is mathematically invertible, mapping any pair $(\boldsymbol{r}, \boldsymbol{c})$ back to its original combined vector form in $F_2^{m+n}$.

<br/>

When a warp read a column of a tensor with `shape=(m, n)`, the following pipeline happens inside each thread:

**1. Take $(\boldsymbol{r}, \boldsymbol{c})$ coordinate**

Each thread identifies the logical 2D element it is responsible for reading or writing.

**2. Compute 1D offset $o$ from $(\boldsymbol{r}, \boldsymbol{c})$ by applying $Z^{-1}$**

The thread computes a **swizzled 1D physical offset** using $Z^{-1}$.

Let's look at the formula for $Z^{-1}$ that we derived earlier:


$Z^{-1}(\boldsymbol{r}, \boldsymbol{c}) = L^{-1}(\boldsymbol{r}, \boldsymbol{c} \oplus S(\boldsymbol{r}))$

In C++ or CUDA code, this mathematical map translates directly to:

```cpp
// S(r) is often implemented as a bit-shift, e.g., (r >> 1) or just r depending on matrix tile size
int swizzled_c = c ^ S(r); 
int o = (r * row_stride) + swizzled_c; // This is L^-1
```

Because $Z$ is perfectly invertible, no two different $(\boldsymbol{r}, \boldsymbol{c})$ pairs will ever produce the same offset $o$.

To understand why `(r * row_stride) + swizzled_c` perfectly embodies the mathematical inverse map $L^{-1}$, we need to look at how abstract vector concatenation in the finite field $F_2$ translates into standard computer arithmetic.

Here is the breakdown of why this line of C++/CUDA code is the literal implementation of $L^{-1}$.

#### The Mathematical Definition of $L^{-1}$

Recall our original map $L$: it takes a single vector $\boldsymbol{o}$ of length $m+n$ and splits it into a row vector $\boldsymbol{r}$ (length $m$) and a column vector $\boldsymbol{c}$ (length $n$).

Therefore, the inverse map $L^{-1}$ must do the exact opposite. It takes the pair $(\boldsymbol{r}, \boldsymbol{c})$ and **concatenates** them side-by-side to reconstruct the $(m+n)$-bit vector $\boldsymbol{o}$:


$L^{-1}(\boldsymbol{r}, \boldsymbol{c}) = [\boldsymbol{r} \text{ bits}] \text{ concatenated with } [\boldsymbol{c} \text{ bits}]$

#### Concatenation in Hardware

In a computer, variables like `r` and `swizzled_c` are just binary bit-strings. Suppose `swizzled_c` represents an $n$-bit column index.

To concatenate the bits of `r` and `swizzled_c` into a single integer, you cannot just add them together normally. You first have to shift the bits of `r` to the left by $n$ positions to make room for the $n$ bits of `swizzled_c`.
In bitwise logic, concatenation looks like this:
`o = (r << n) | swizzled_c`

#### Bit-Shifting is Multiplication

In binary arithmetic, shifting a number to the left by $n$ bits is mathematically identical to multiplying that number by $2^n$.


$r \ll n \equiv r \times 2^n$

In GPU programming, shared memory blocks are almost always sized as powers of 2 (e.g., 16, 32, or 64 elements wide) to align with warp boundaries. Therefore, the width of a row—the **`row_stride`**—is exactly $2^n$.

By substituting `row_stride` for $2^n$, our bit-shift becomes standard integer multiplication:
`r * row_stride`

At this point, the bits of `r` are sitting in the upper $m$ positions of the integer, and the lower $n$ positions are entirely filled with zeros.

#### Addition is Concatenation (Over $F_2$)

Because the lower $n$ bits of `(r * row_stride)` are all zeros, adding `swizzled_c` to it using the standard `+` operator will not trigger any arithmetic carries.

When there are no carries, arithmetic addition (`+`), bitwise OR (`|`), and bitwise XOR (`^`) all produce the exact same result: they simply drop the bits of `swizzled_c` into those empty zero slots.

#### Conclusion

By piecing this together, we can map the math directly to the code:

1. **$L^{-1}$ requires concatenation:** We must attach the $n$-bit column vector to the $m$-bit row vector.
2. **Make room for the column:** `r * row_stride` shifts the row bits left by $n$ positions.
3. **Attach the column:** `+ swizzled_c` drops the swizzled column bits into the newly opened $n$ positions.

Therefore, the algebraic expression $L^{-1}(\boldsymbol{r}, \boldsymbol{c} \oplus S(\boldsymbol{r}))$ directly compiles down to:
`int o = (r * row_stride) + swizzled_c;`

**3. Compute the physical shared memory location**

The thread takes the shared memory base pointer and adds $o$ (scaled by the byte size of the data type) to get the exact memory address in the GPU's SRAM.

**4. Send the physical shared memory location to GPU**

The thread executes a shared memory load instruction (e.g., `ld.shared`).

**5. GPU computes bank ID and byte offset**

The GPU hardware receives the physical address and routes it to the memory banks.
For 32-bit words, the hardware calculates the bank ID using a simple modulo operation:
`Bank ID = (Physical Address / 4) % 32`

Because your offset $o$ in Step 2 was constructed using the XOR swizzle ($\boldsymbol{c} \oplus S(\boldsymbol{r})$), threads reading down a column (same $\boldsymbol{c}$, different $\boldsymbol{r}$) will inherently calculate different values for $o$. When the hardware applies the modulo 32 operation in Step 5, those different physical addresses gracefully route to completely different Bank IDs, bypassing the conflict entirely!
