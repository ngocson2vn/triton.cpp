# Identity Maps

### 1. Are identity maps matrices?
**Yes, exactly.** Because this paper models everything as linear transformations over the finite field $\mathbb{F}_2$ (where the only numbers are 0 and 1, and addition is XOR), every single map discussed in that proof can be written as a matrix.

Specifically, these identity maps are just matrices made of 1s and 0s. 
* If a map transfers 3 bits straight across (like our thread-to-column map $\text{id}_{T_1}^{\text{Thr}, 1}$), its matrix representation is simply a $3 \times 3$ identity matrix:
$$
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$
* If a map transfers 1 bit (like $\text{id}_{R_1}^{\text{Reg}, 1}$), it is just a $1 \times 1$ matrix: $[1]$.

When the paper says "map," you can always mentally substitute the word "matrix."

---

### 2. Does the symbol $\times$ mean the direct sum operation?
**Functionally, yes! You hit the nail on the head.**

Formally, in abstract algebra, the $\times$ symbol here denotes the **Cartesian product** of functions acting on a Cartesian product of vector spaces (e.g., $\mathbb{F}_2^{R_1} \times \mathbb{F}_2^{R_2}$). 

However, when you translate that Cartesian product of functions into *matrices*, it becomes exactly the **matrix direct sum (often denoted as $\oplus$)**. 

When you take the direct sum of matrices, you are taking those smaller matrices and stacking them along the diagonal to build a larger block matrix, filling the rest of the space with zeros.

The entire $8 \times 8$ matrix $A$ from your very first screenshot is built by taking the direct sum of the Register matrix, the Thread matrix, and the Warp matrix, placing them along the diagonal!


## Registers
Let's look at your formula: $\text{id}_R^o = \text{id}_{R_1}^{\text{Reg}, 1} \times \text{id}_{R_2}^{\text{Reg}, 2}$

From our earlier Layout A example:
* $\text{id}_{R_1}^{\text{Reg}, 1}$ takes 1 register bit and routes it to the column dimension (logical index $j$). It is a $1 \times 1$ matrix: $[1]$.
* $\text{id}_{R_2}^{\text{Reg}, 2}$ takes 1 register bit and routes it to the row dimension (logical index $i$). It is also a $1 \times 1$ matrix: $[1]$.

When you combine them with the $\times$ operator, you construct a $2 \times 2$ block diagonal matrix (the direct sum):
$$
\text{id}_R^o = 
\begin{bmatrix}
[1] & 0 \\
0 & [1] 
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 \\
0 & 1 
\end{bmatrix}
$$

This perfectly describes the logic: "The first input bit affects only the first output slot, and the second input bit affects only the second output slot. They do not mix." 

## Threads
Let's build the Thread matrix block ($\text{id}_T^o$) step-by-step using the direct sum operation we just talked about.

From the paper's proof, the Thread map is defined as:
$$\text{id}_T^o = \text{id}_{T_1}^{\text{Thr}, 1} \times \text{id}_{T_2}^{\text{Thr}, 2}$$

### 1. The Building Blocks
From our Layout A ($16 \times 16$) example, we know the 5 Thread bits are split into 3 bits for the columns ($T_1$) and 2 bits for the rows ($T_2$).

* **The Column Map ($\text{id}_{T_1}^{\text{Thr}, 1}$):** Since $T_1 = 3$, this is simply a $3 \times 3$ identity matrix. It takes 3 thread bits and outputs 3 column bits.
$$
\text{id}_{T_1} = 
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

* **The Row Map ($\text{id}_{T_2}^{\text{Thr}, 2}$):** Since $T_2 = 2$, this is a $2 \times 2$ identity matrix. It takes the remaining 2 thread bits and outputs 2 row bits.
$$
\text{id}_{T_2} = 
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

### 2. The Direct Sum ($\times$ / $\oplus$)
When we apply the Cartesian product ($\times$) to these linear maps, we are taking their matrix direct sum ($\oplus$). This means we place our $3 \times 3$ matrix and our $2 \times 2$ matrix along the diagonal of a new, larger matrix, and fill the empty spaces with zeros.

$$
\text{id}_T^o = \text{id}_{T_1} \oplus \text{id}_{T_2} = 
\begin{bmatrix}
\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} & \begin{matrix} 0 & 0 \\ 0 & 0 \\ 0 & 0 \end{matrix} \\
\begin{matrix} 0 & 0 & 0 \\ 0 & 0 & 0 \end{matrix} & \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
\end{bmatrix}
$$

When you erase the inner brackets, you are left with a perfect $5 \times 5$ block diagonal matrix:

$$
\text{id}_T^o = 
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$