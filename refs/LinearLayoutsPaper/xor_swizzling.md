# XOR Swizzling
Given a matrix $A$ with shape = (ROWS, COLS), and a shift function $S(r) = r * k$, <br/>
where $k$ is selected so that $(\text{ROWS} - 1) * k < \text{COLS}$, we define $\text{swizzle}(A)$ operation as follows:<br/>
For every $A(i, j)$, we shift it to $A(i, j \oplus S(i))$.