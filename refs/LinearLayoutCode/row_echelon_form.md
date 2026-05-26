If **Reduced Row Echelon Form (RREF)** is the "fully solved" state of a matrix, REF is the "halfway solved" state. 
You reach REF by using standard **Gaussian elimination** (stopping before the "Jordan" part).

### The Three Rules of Row Echelon Form

For a matrix to be in Row Echelon Form, it only needs to satisfy the first three rules of RREF. The fourth, strictest rule is dropped.

1. **Zero Rows at the Bottom:** Any rows made completely of zeros must be at the very bottom of the matrix.
2. **The Leading Entry (The Pivot):** The first non-zero number from the left in any row must be strictly to the right of the leading number in the row above it. (This creates that descending staircase pattern).
3. **Zeros Below the Pivot:** All numbers strictly *below* a pivot must be **$0$**.

**The Crucial Difference:** Notice what is missing? In standard REF, the numbers *above* the pivot do not have to be zero. Furthermore, in many math textbooks, the pivot itself doesn't even strictly have to be a $1$ (though making it a $1$ is standard practice and makes the math easier).

---

### A Visual Comparison

Let's look at a system in Row Echelon Form versus Reduced Row Echelon Form so the difference is glaringly obvious.

**1. Row Echelon Form (REF)**

$$\begin{bmatrix}
1 & 4 & -3 & 7 \\
0 & 1 & 6 & 2 \\
0 & 0 & 1 & 5
\end{bmatrix}$$

* Notice the staircase pattern of the $1$s.
* Notice that everything *below* the $1$s is a $0$.
* However, the numbers *above* the $1$s (like the $4$, $-3$, and $6$) are still there. This matrix is in REF, but it is **not** reduced.

**2. Reduced Row Echelon Form (RREF)**
If we continued applying row operations to clear out the numbers above the pivots, we would reach RREF:

$$\begin{bmatrix}
1 & 0 & 0 & -2 \\
0 & 1 & 0 & -28 \\
0 & 0 & 1 & 5
\end{bmatrix}$$

* Now, every pivot is the *only* non-zero number in its entire column.
