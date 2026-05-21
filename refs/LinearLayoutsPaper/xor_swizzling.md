# XOR Swizzling
Given a matrix $A$ with shape = (ROWS, COLS), and a shift function $S(r) = \text{VEC} * (r \mod K)$, <br/>
where,
- $r$ is row index
- $\text{VEC}$ is the number of consecutive elements for a read request. The common value is $8$.
- $K$ is selected so that $\text{VEC} * (K - 1) < \text{COLS}$. $K$ will determine the number of rows after which the swizzling pattern repeats.

$\text{swizzle}(A)$ operation is defined as follows:<br/>
For every $A(i, j)$, shift it to the position of $A(i, j \oplus S(i))$.

### Example
[xor_swizzling.py](./xor_swizzling.py)

<img src="./swizzledA.png">
