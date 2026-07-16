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

Fix $`i \in I$, let row run. The expression $s_i(\text{row})`$ defined inside (3) is, for that fixed $i$, a rule assigning a scalar in $`\mathbb{F}_2$ to *each* $\text{row} \in \{0,\dots,n_{\text{row}}-1\}`$. That is precisely the data of a function

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

## 2. Show the code's `term` equals this abstract $\text{term}(i)$

This is the part that connects the algebra to the implementation. The code computes, for diagonal $i \ge 0$:

$`\text{term} = (x \;\&\; \text{mask}_i) \gg i`$

where $\text{mask}_i$ has bit $c$ set iff $c - i \ge 0$ and $A[c-i,\,c] = 1$ (this is exactly `getMaskAndAllRowsUnique`'s loop, which walks `row = 0, col = i` and sets bit `col` of the mask when `matrix[col]` has bit `row` set).

- **Masking**: bit $c$ of $`x \,\&\, \text{mask}_i`$ equals $`A[c-i,c]\cdot x_c`$ (zero if that matrix entry is 0, and zero for any $c$ not on diagonal $i$).
- **Shifting right by $i$**: this moves bit $c$ to position $c - i = \text{row}$.

So bit $\text{row}$ of the shifted result is $`A[\text{row}, \text{row}+i] \cdot x_{\text{row}+i}`$ — which is *exactly* the definition of $`\text{term}(i)_{\text{row}}`$ from Step 2. (The $i<0$ case with left shift is the same argument with the shift direction reversed, as established earlier.)

## 3. Conclusion

Since the code's shifted-and-masked `term` for diagonal $i$ is bit-for-bit identical to the abstractly-defined $\text{term}(i)$, and Step 2 proved $y = \bigoplus\limits_i \text{term}(i)$ as a pure consequence of reindexing the matrix product, we get:

$`y = \text{term}(i=-n_{\text{row}}+1) \oplus \cdots \oplus \text{term}(i=0) \oplus \cdots \oplus \text{term}(i = n_{\text{col}}-1)`$

exactly matching what the code computes via `ors`/`xors` and the tree reduction. $\blacksquare$

The one subtlety worth flagging: the code doesn't literally XOR *all* diagonal terms — it partitions them into an OR-group and an XOR-group based on `rowsUnique`, then ORs the two partial results together. That's a valid substitution for the *same* formula precisely because, as shown earlier, no output row ever receives contributions from more than one group, so OR and XOR coincide there — but the underlying correctness of decomposing by diagonals in the first place is exactly the proof above, independent of that optimization.