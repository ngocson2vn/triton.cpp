# External Direct Sum
In linear algebra, the **external direct sum** of two linear maps combines them into a single, larger linear map that operates on the direct sum of their respective vector spaces.

Here is how it is formally defined and constructed:

### 1. The Definition

Suppose you have two linear maps over the same field:

* $T: U \rightarrow V$
* $S: W \rightarrow X$

The external direct sum of these two maps, denoted as $T \oplus S$, is a new linear map that goes from the direct sum of the domain spaces ($U \oplus W$) to the direct sum of the codomain spaces ($V \oplus X$).

Formally, it is written as:


$$T \oplus S: \; U \oplus W \rightarrow V \oplus X$$

### 2. How It Acts on Vectors

Elements in the external direct sum space $U \oplus W$ are typically written as ordered pairs $(u, w)$, where $u \in U$ and $w \in W$.

The map $T \oplus S$ operates on these pairs component-wise. It applies the first map to the first component, and the second map to the second component:


$$(T \oplus S)(u, w) = (T(u), S(w))$$

Because $T$ and $S$ are both linear, their external direct sum $T \oplus S$ is also guaranteed to be a linear map.

### 3. Matrix Representation

If the vector spaces are finite-dimensional, you can represent the linear maps as matrices. Suppose:

* Matrix $A$ represents $T$ (with dimensions $\dim(V) \times \dim(U)$)
* Matrix $B$ represents $S$ (with dimensions $\dim(X) \times \dim(W)$)

The matrix representation of the external direct sum $T \oplus S$ is simply the **block diagonal matrix** constructed from $A$ and $B$. If we call this new matrix $C$, it looks like this:


$$C = \begin{bmatrix} A & 0 \\ 0 & B \end{bmatrix}$$

When you multiply this block matrix by a column vector built by stacking $u$ and $w$, you get exactly the component-wise result defined above:


$$\begin{bmatrix} A & 0 \\ 0 & B \end{bmatrix} \begin{bmatrix} u \\ w \end{bmatrix} = \begin{bmatrix} Au \\ Bw \end{bmatrix}$$
