# Column shift function
Given the following function:

$S(r) = 2^v((r \gg p) \mathbin{\&}(2^q - 1))$

where, 
- $r \in F_2^m$
- $v$, $p$, and $q$ are positive integers.

Prove that $S(r)$ is a linear map: $F_2^m \rightarrow F_2^m$

## Proof
When we move into $F_2^m$ (the vector space of $m$-bit binary strings over the finite field of two elements), the concept of "addition" is no longer standard integer addition; it becomes the **bitwise XOR** operation ($\oplus$).

For a function to be linear in $F_2^m$, it must satisfy the **Boolean superposition** principle:


$S(a \oplus b) = S(a) \oplus S(b)$

Here is exactly why your function is a strictly linear transformation in this domain:

### 1. The Right Shift is a Linear Operator

In $F_2^m$, shifting a bit vector to the right by $p$ positions ($\gg p$) simply drops the lowest $p$ bits and pulls in zeros at the top. This operates entirely independently on each bit position. In linear algebra terms, it is a matrix transformation. It distributes perfectly over XOR:


$(a \oplus b) \gg p = (a \gg p) \oplus (b \gg p)$

### 2. The Bitwise AND is a Linear Projection

The term $(2^q - 1)$ represents a constant bitmask of $q$ ones. A bitwise AND with a constant mask merely isolates specific bits (projecting them as they are) and zeroes out the rest. Because there are no "carries" across bits, this is also a linear operation over $F_2^m$:


$(x \oplus y) \mathbin{\&} (2^q - 1) = (x \mathbin{\&} (2^q - 1)) \oplus (y \mathbin{\&} (2^q - 1))$

### 3. The Multiplier Becomes a Left Shift

In standard arithmetic, mixing integer multiplication with bitwise operators usually destroys linearity. However, because your multiplier is $2^v$ (a power of two), multiplying by it is mathematically identical to performing a **logical left shift** by $v$ bits ($\ll v$).

Just like the right shift, a left shift is a linear operator in $F_2^m$:


$((x \oplus y) \ll v) = (x \ll v) \oplus (y \ll v)$

---

### The Final Proof
#### Additivity
Let's define the constant mask $M = 2^q - 1$.

Because multiplication by $2^v$ acts as a left shift, we can rewrite the function purely in bitwise terms:


$S(r) = ((r \gg p) \mathbin{\&} M) \ll v$

Now, evaluate $S(a \oplus b)$:


$S(a \oplus b) = (((a \oplus b) \gg p) \mathbin{\&} M) \ll v$

$= (((a \gg p) \oplus (b \gg p)) \mathbin{\&} M) \ll v$

$= (((a \gg p) \mathbin{\&} M) \oplus ((b \gg p) \mathbin{\&} M)) \ll v$

$= (((a \gg p) \mathbin{\&} M) \ll v) \oplus (((b \gg p) \mathbin{\&} M) \ll v)$

$= S(a) \oplus S(b)$

#### Scalar multiplication

In linear algebra, for a function to be a true linear transformation over a vector space, it must satisfy both **additivity** (which we proved in the last step) and **homogeneity** (scalar multiplication):


$S(c \cdot r) = c \cdot S(r)$


where $c$ is a scalar from the underlying field.

Because the vector space is $F_2^m$, the underlying field is simply $F_2$ (the Galois Field of 2).

In $F_2$, there are only two possible scalars: **$0$ and $1$**.

Let's test both possible values for the scalar $c$:

**Case 1: The scalar is $1$**
If $c = 1$, the property is trivially true for any function:


$S(1 \cdot r) = S(r) = 1 \cdot S(r)$

**Case 2: The scalar is $0$**
If $c = 0$, the vector $r$ becomes a zero-vector ($0 \cdot r = 0$). We just need to evaluate $S(0)$:


$S(0) = 2^v((0 \gg p) \mathbin{\&} (2^q - 1))$

* $0 \gg p = 0$
* $0 \mathbin{\&} (2^q - 1) = 0$
* $2^v \cdot 0 = 0$

So, $S(0) = 0$.
Since $0 \cdot S(r)$ is also $0$, the equation holds:


$S(0 \cdot r) = 0 = 0 \cdot S(r)$

### Conclusion

Because the only possible scalars are $0$ and $1$, and the function naturally maps an all-zero input to an all-zero output (meaning the origin is preserved), **scalar multiplication is strictly satisfied.** Combined with the additivity we proved earlier, the function $S(r)$ checks every box. It is a completely valid, rigorous linear transformation in the vector space $F_2^m$.


# Swizzling Linear Map
