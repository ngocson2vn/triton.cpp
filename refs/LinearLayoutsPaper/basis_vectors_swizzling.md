Gemini link: https://gemini.google.com/app/199db705d61e1019

To prove that storing shifts in the row basis vectors correctly applies the shift to the entire row, we must evaluate the full **Linear Layout mapping function $L(x)$**, which transforms a 1D offset $x$ into a 2D coordinate $(Row, Column)$.

Here is the mathematical proof of the final assembly.

### Step 1: Decomposing the 1D Offset

Let’s assume a `tileCols` width of $W$. Because Triton requires $W$ to be a power of 2 (e.g., $W = 64 = 2^6$), any 1D memory offset $x$ can be cleanly split into two non-overlapping binary fields:

1. **Lower bits:** The base column index $c$, where $c < W$.
2. **Upper bits:** The row index $r$.

In standard arithmetic, $x = (r \times W) + c$.
Because $W$ is a power of 2, the bit ranges for $r$ and $c$ never overlap. Therefore, standard addition is perfectly equivalent to bitwise XOR:


$$x = (r \times W) \oplus c$$

### Step 2: Defining the Basis Vectors

Triton’s `getCoreMatrixLinearLayout` assigns a 2D basis vector $\mathbf{b}_i$ to every bit position $i$ of the offset $x$.

* **For the lower bits (columns):** The bit $j$ belongs to the base column $c$. Its basis vector simply increments the column:

$$\mathbf{b}_j = (0, 2^j)$$


* **For the upper bits (rows):** The bit $k$ belongs to the row index $r$. In the offset, this bit is shifted by the column width (e.g., position $k+6$). Its basis vector increments the row by $2^k$, and applies the partial swizzle shift $S(2^k)$:

$$\mathbf{b}_{k+\text{offset}} = (2^k, S(2^k))$$



### Step 3: Evaluating the Layout Function $L(x)$

The 2D coordinate is calculated by taking the bitwise XOR sum of the basis vectors for all active bits in $x$:


$$L(x) = \bigoplus_{i} (x_i \cdot \mathbf{b}_i)$$

We can separate this summation into the lower column bits and the upper row bits:


$$L(x) = \left[ \bigoplus_{j \in \text{lower}} (c_j \cdot \mathbf{b}_j) \right] \oplus \left[ \bigoplus_{k \in \text{upper}} (r_k \cdot \mathbf{b}_{k+\text{offset}}) \right]$$

### Step 4: Solving the Two Halves

**Part A: The Lower Bits (Base Column)**
Substitute the column basis $\mathbf{b}_j = (0, 2^j)$:


$$\bigoplus_{j} \left( c_j \cdot (0, 2^j) \right) = \left( 0, \bigoplus_{j} c_j \cdot 2^j \right)$$


Because reconstructing a binary number from its bits yields the number itself, this perfectly simplifies to:


$$(0, c)$$

**Part B: The Upper Bits (Row + Swizzle)**
Substitute the row basis $\mathbf{b}_{k+\text{offset}} = (2^k, S(2^k))$:


$$\bigoplus_{k} \left( r_k \cdot (2^k, S(2^k)) \right) = \left( \bigoplus_{k} (r_k \cdot 2^k), \bigoplus_{k} (r_k \cdot S(2^k)) \right)$$

1. The first coordinate is simply the reconstruction of the row index: $\bigoplus (r_k \cdot 2^k) = r$.
2. The second coordinate relies on the GF(2) linearity we proved previously. Because $S(r)$ is linear, the XOR sum of the partial shifts equals the total shift for the row: $\bigoplus (r_k \cdot S(2^k)) = S(r)$.

So, the upper bits simplify to:


$$(r, S(r))$$

### Conclusion: The Final Proof

Now, we XOR Part A and Part B together to get the final coordinate mapping:


$$L(x) = (0, c) \oplus (r, S(r))$$

$$L(x) = \mathbf{(r, c \oplus S(r))}$$

**This is the exact mathematical definition of hardware swizzling.** NVIDIA GPUs route shared memory by taking the base memory column and XORing it with a row-dependent shift.

By proving that $L(x) = (r, c \oplus S(r))$, we have mathematically proven that packing the partial shifts into the row basis vectors guarantees that the entire row $r$ receives the correct column shift $S(r)$, successfully modeling the hardware without ever needing runtime arithmetic.
