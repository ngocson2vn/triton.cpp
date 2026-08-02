## 1. Foundational Abstract Algebra
Let's put this on solid footing using linear algebra over $\mathbb{F}_2$ (the field with two elements, where addition is XOR).

### Setup

Let $`A \in \mathbb{F}_2^{\,n_{\text{row}} \times n_{\text{col}}}`$ be the matrix, $`x \in \mathbb{F}_2^{\,n_{\text{col}}}`$ the input vector, and $`y = Ax \in \mathbb{F}_2^{\,n_{\text{row}}}`$ the output.

### Step 1: Construct each $y_{\text{row}}$ independently, purely from the definition

By definition of matrix–vector multiplication over $\mathbb{F}_2$, for each fixed $`\text{row} \in \{0,\dots,n_{\text{row}}-1\}`$:

$`y_{\text{row}} = \bigoplus\limits_{\text{col}=0}^{n_{\text{col}}-1} A(\text{row},\text{col})\cdot x(\text{col})`$ (1)

This is $n_{\text{row}}$ separate, independent scalar equations — one per row, with no reference yet to any other row.

### Step 2: Reindex the sum in equation (1), for this fixed row

Fix a row. As $\text{col}$ ranges over $\{0,\dots,n_{\text{col}}-1\}$, define $i = \text{col} - \text{row}$. The map $\text{col} \mapsto i$ is a **bijection** (it's just a shift by the constant $\text{row}$), $i$ ranges bijectively over $`i \in \{-\text{row},\, -\text{row}+1,\, \dots,\, n_{\text{col}}-1-\text{row}\}`$.

Equation (1) becomes, purely by relabeling the summation variable (valid since a bijective reindexing of a finite sum doesn't change its value):

$`y_{\text{row}} = \bigoplus\limits_{i = -\text{row}}^{\,n_{\text{col}}-1-\text{row}} A(\text{row},\text{row}+i)\cdot x(\text{row}+i)`$ (2)

Notice: the range of $i$ in (2) **depends on $\text{row}$**. This is the obstruction to writing a single vector equation — different rows sum over different index sets.

### Step 3: Extend every row's sum to a common index set $I$
**Definition of $I$**:

Let $I$ denote the set of values $i = \text{col} - \text{row}$ can take, as $\text{row}$ ranges over $\{0,\dots,n_{\text{row}}-1\}$ and $\text{col}$ ranges over $\{0,\dots,n_{\text{col}}-1\}$ **independently** (i.e., the full range of $i = \text{col}-\text{row}$ over the entire index grid of the matrix $A$, not just over pairs that happen to satisfy some other constraint).

- The **maximum** of $\text{col}-\text{row}$ is achieved at the largest possible $\text{col}$ ($=n_{\text{col}}-1$) and smallest possible $\text{row}$ ($=0$), giving $i_{\max} = n_{\text{col}}-1$.
- The **minimum** of $\text{col}-\text{row}$ is achieved at the smallest possible $\text{col}$ ($=0$) and largest possible $\text{row}$ ($=n_{\text{row}}-1$), giving $i_{\min} = 0 - (n_{\text{row}}-1) = -(n_{\text{row}}-1)$.

Every integer between $i_{\min}$ and $i_{\max}$ is attained (e.g. by fixing $\text{col}=0$ and letting $\text{row}$ range down from $n_{\text{row}}-1$ to $0$ for negative $i$, then fixing $\text{row}=0$ and letting $\text{col}$ range up to $n_{\text{col}}-1$ for nonnegative $i$), so $I$ is precisely this contiguous integer interval — there are no gaps.

So $I = \{-(n_{\text{row}}-1),\dots,n_{\text{col}}-1\}$.

For any fixed $\text{row}$, the range of $i$ actually appearing in (2) is a subset of $I$; extend the sum to all of $I$ by inserting the value $0$ for every $i \in I$ not already present in (2) — this is legal because $\oplus$ with $0$ doesn't change a sum ($z \oplus 0 = z$):

$`y_{\text{row}} = \bigoplus\limits_{i \in I} \; \underbrace{\begin{cases} A(\text{row},\text{row}+i)\cdot x(\text{row}+i) & \text{if } 0\le \text{row}+i < n_{\text{col}} \\ 0 & \text{otherwise} \end{cases}}_{\text{call this quantity } s_i(\text{row})}`$ (3)

Equation (3) now sums over the **same** index set $I$ for every row — this is the key structural change from (2).

### Step 4: Induce $\text{term}(i)$ from $s_i(\text{row})$

Fix $`i \in I`$, let row run. The expression $`s_i(\text{row})`$ defined inside (3) is, for that fixed $i$, a rule assigning a scalar in $`\mathbb{F}_2`$ to *each* $`\text{row} \in \{0,\dots,n_{\text{row}}-1\}`$. That is precisely the data of a function

$`\{0,\dots,n_{\text{row}}-1\} \to \mathbb{F}_2,`$

i.e. an element of $`\mathbb{F}_2^{\,n_{\text{row}}}`$ — the same space $y$ lives in. Define

$`\text{term}(i) := \big(s_i(0), s_i(1), \dots, s_i(n_{\text{row}}-1)\big) \in \mathbb{F}_2^{\,n_{\text{row}}}, \qquad \text{term}(i)_{\text{row}} := s_i(\text{row})`$.

So $\text{term}(i)$ is **not introduced independently** — it is exactly the vector obtained by taking the row-indexed family of scalars $\{s_i(\text{row})\}_{\text{row}}$ appearing in equation (3) and packaging them as a single vector, one $\text{term}(i)$ per value of $i \in I$.

### Step 5: Recover the vector equation

With $\text{term}(i)$ so defined, equation (3) reads, for every $\text{row}$:

$y_{\text{row}} = \bigoplus\limits_{i\in I} \text{term}(i)_{\text{row}}$.

We can expand $y_{\text{row}}$ for each row as follows:

| $\text{row}$ | $y_\text{row}$ |
|--------------|----------------|
|$\text{row}_0$|$`\text{term}(i_0)_{\text{row}_0} \oplus \text{term}(i_1)_{\text{row}_0} \oplus \dots \oplus \text{term}(i_K)_{\text{row}_0}`$|
|$\text{row}_1$|$`\text{term}(i_0)_{\text{row}_1} \oplus \text{term}(i_1)_{\text{row}_1} \oplus \dots \oplus \text{term}(i_K)_{\text{row}_1}`$|
| .            | .                                                                                                                          |
|$\text{row}_R$|$`\text{term}(i_0)_{\text{row}_R} \oplus \text{term}(i_1)_{\text{row}_R} \oplus \dots \oplus \text{term}(i_K)_{\text{row}_R}`$|
|              |                                                                                                                            |

where $K$ is the size of the set $I$ minus 1 and $R = n_{\text{row}} - 1$

Since $I$ is now the *same* set for every $\text{row}$, the right-hand side is, for each fixed $\text{row}$, the $\text{row}$-th coordinate of the vector sum $\bigoplus\limits_{i\in I}\text{term}(i)$ (by the coordinatewise definition of vector addition established earlier). As this holds for every $\text{row}$, and two vectors are equal iff equal in every coordinate:

$y = \bigoplus\limits_{i \in I} \text{term}(i)$.

### Summary of the logical order

1. $y_{\text{row}}$ is defined first, directly from matrix multiplication (1).
2. Reindexing by $i = \text{col}-\text{row}$ turns the sum-over-columns into a sum-over-diagonals, but with a row-dependent range (2).
3. Zero-padding unifies the range to a fixed set $I$, independent of row (3) — this is the step that makes a *vector* formulation possible at all.
4. $\text{term}(i)$ is then *induced*, not postulated: it is exactly the vector whose $\text{row}$-th entry is the $i$-th summand appearing in every row's zero-padded equation (3).
5. Summing (3) over all rows simultaneously, using the coordinatewise definition of vector addition, yields $y = \bigoplus\limits_{i\in I}\text{term}(i)$ as a direct consequence — not as an independently asserted fact.

## 2. Compute $\text{term}(i)$
Let’s build $\text{term}(i)$ strictly from the ground up. We will start with your exact definition, apply a simple algebraic substitution, and watch how the math *forces* us to invent both the mask and the shift.

### Step 1: Start with your coordinate definition

From your derivation, we know exactly what scalar value must live at the $\text{row}$-th coordinate of the vector $\text{term}(i)$:

$$\text{term}(i)_{\text{row}} = A(\text{row},\text{row}+i) \cdot x(\text{row}+i)$$

*(Note: We will assume $0 \le \text{row}+i < n_{\text{col}}$ throughout this, treating out-of-bounds as simply multiplying by $0$.)*

Right now, this equation is difficult to turn into a full-vector operation because the input $x$ is being indexed by $(\text{row} + i)$. We want to manipulate the vector $x$ in its native coordinates.

### Step 2: The Algebraic Substitution

Let’s introduce a new variable to represent the column index:


$$c = \text{row} + i$$

If we solve this for $\text{row}$, we get:


$$\text{row} = c - i$$

Now, let's substitute $c$ into your definition. We are changing our perspective from "what goes into this row" to "where does this column's data go."

The value destined for the $(c - i)$-th coordinate of $\text{term}(i)$ is:


$$A(c-i, c) \cdot x(c)$$

This tiny substitution is the breakthrough. It splits the problem into two distinct mathematical operations: **modifying the value** and **moving the index**.

### Step 3: Modifying the Value (Deriving the Mask)

Let's look purely at the value being computed: $A(c-i, c) \cdot x(c)$.

Notice that $c$ is just an index running from $0$ to $n_{\text{col}}-1$. This expression is the point-wise (element-by-element) multiplication of two sets of numbers:

1. The elements of the input vector $x(c)$.
2. The elements of the matrix diagonal $A(c-i, c)$.

In linear algebra, if you want to multiply the corresponding coordinates of two vectors, you use the Hadamard product (denoted $\circ$).

So, the math *demands* that we package those matrix elements $A(c-i, c)$ into a new vector. Let's call it $D^{(i)}$, where its coordinates are defined as:


$$D^{(i)}_c = A(c-i, c)$$

Now we can compute the element-wise product to get an intermediate vector, let's call it $M$:


$$M = D^{(i)} \circ x$$

Because we are working in $\mathbb{F}_2$ (where everything is bits), the element-wise multiplication ($\circ$) of two bit-vectors is exactly the bitwise **AND** operation.

We didn't invent a mask out of nowhere; $D^{(i)}$ *is* the mask. The algebraic need to scale $x(c)$ by the matrix elements forced us to define it.

### Step 4: Moving the Index (Deriving the Shift)

We now have this intermediate vector $M$. Its $c$-th coordinate contains exactly the data we want:


$$M_c = A(c-i, c) \cdot x(c)$$

But look back at the end of Step 2. That data is currently sitting at index $c$ in the vector $M$, but it belongs at index $(c - i)$ in our final vector $\text{term}(i)$.

How do we map a vector $M$ to a new vector $\text{term}(i)$ such that every element at index $c$ moves to index $c - i$?

In abstract algebra, a linear operator that maps coordinate $c \mapsto c - k$ is called a translation or **shift operator**.

* If $i$ is positive (e.g., $i=1$), the data moves from index $c$ to $c - 1$. The index gets smaller, moving elements toward the $0$-th coordinate. In a bit-vector, this is a **right shift** ($\gg$).
* If $i$ is negative (e.g., $i=-1$), the data moves from index $c$ to $c - (-1) = c + 1$. The index gets larger. This is a **left shift** ($\ll$).

### Conclusion of the Induction

By strictly following the algebra, we didn't have to guess the hardware implementation.

1. The substitution $c = \text{row} + i$ isolated the data payload from its destination coordinate.
2. The payload $A(c-i, c) \cdot x(c)$ forced us to define a point-wise multiplication vector (the **mask**).
3. The destination coordinate $(c - i)$ forced us to apply a translation operator to the resulting vector (the **shift**).

## 3. How right shift and left shift won't truncate valid data
### The Safety Condition

Recall that the target row index is defined as $\text{row} = c - i$.
For a coordinate to be valid, the row index must fall strictly within the bounds of the output vector:


$$0 \le c - i < n_{\text{row}}$$

If $c - i$ falls outside this range, that specific element does not exist in the matrix. In our construction, the mask $D^{(i)}_c$ is explicitly set to $0$ for these out-of-bounds coordinates.

Let's test this against both shift directions.

---

### Case 1: Right Shift ($i > 0$)
When $i$ is positive, we shift the vector to the right by $i$ positions.
In a bitwise right shift, the bits at indices $c = 0, 1, \dots, i - 1$ fall off the right edge and are destroyed.

Are we losing valid data? Let's check the lower bound of our safety condition:

$`0 \le c - i`$

$`i \le c`$

The math dictates that for a bit to be valid, its original column index $c$ **must be greater than or equal to $i$**.

What about the bits at $c < i$ (the ones falling off the edge)? Since they violate the lower bound, they correspond to negative row indices. The compiler's masking step ensures that the mask $`D^{(i)}_c = 0`$ for all $`c < i`$.

**Conclusion:** The only bits that fall off the right edge were already forced to $0$ by the mask. The valid data starts safely at index $i$ and shifts perfectly down to index $0$.

### Case 2: Left Shift ($i < 0$)
When $i < 0$, we perform a left shift by $k = \vert{}i\vert{}$. In the hardware, this physically pushes the highest bits off the left edge of the register, destroying them forever.

Here is the exact step-by-step proof of why every single bit that falls off that physical edge was mathematically guaranteed to be a `0`.

#### Step 1: The Physical Truncation Boundary

Let’s assume we are operating inside a standard 32-bit integer register (which is what Triton uses here). The register has indices from `0` to `31`.

A bit starting at column $c$ in vector $M$ will move to the destination $c + k$.
For this bit to be **physically truncated** (pushed off the edge), its destination must exceed the size of the register:


$$c + k \ge 32$$

#### Step 2: The Hardware Constraint on the Matrix

Because this entire algorithm is designed to compute a matrix-vector product inside these 32-bit integers, the dimensions of the matrix $A$ are strictly bounded by the hardware.

The output vector $y$ must fit inside a single integer. Therefore, the maximum number of rows the matrix can possibly have is 32:


$$n_{\text{row}} \le 32$$

#### Step 3: The Transitive Guarantee

Now we bring Step 1 and Step 2 together.

If a bit is physically falling off the edge of the register, we established in Step 1 that its destination is $\ge 32$.
Because $32 \ge n_{\text{row}}$, we can chain these inequalities together:


$$c + k \ge 32 \ge n_{\text{row}}$$

Therefore, for any bit that is physically truncated:


$$c + k \ge n_{\text{row}}$$

#### Step 4: The Mask Neutralizes the Bit

Look at what that final inequality means. The destination index $c + k$ is exactly the row coordinate we asked the matrix $A$ for when building $M$.

Because $c + k \ge n_{\text{row}}$, that bit was trying to pull data from a row that **does not exist** in the matrix.

Because the row does not exist, our piecewise math definition kicked in during the masking phase, explicitly setting the mask $D^{(i)}_c$ to $0$. Therefore, that element in $M$ was exactly $0$.

#### Conclusion

Why is it safe to physically truncate the high bits of $M$?

Because for a bit to shift far enough to fall off the physical register, its destination row had to be larger than the physical size of the register itself. Since the matrix can never be larger than the register, that bit was mathematically destined for an out-of-bounds row. The compiler anticipated this and masked it to `0`.

The hardware aggressively chops off the high bits, but it is only chopping off empty zeros. All valid data from valid matrix rows is safely contained within the bounds of $n_{\text{row}}$, well before the physical cliff of the register.
