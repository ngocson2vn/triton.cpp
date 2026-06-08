# Gauss-Jordan Elimination
Gauss-Jordan elimination is the step-by-step algorithm used to transform a standard matrix into the Reduced Row Echelon Form (RREF) we just discussed. It is a highly systematic way of solving a system of linear equations by applying a specific set of rules to an "augmented matrix."

Here is exactly how the algorithm works from start to finish, using a straightforward example.

### 1. The Three Rules (Elementary Row Operations)

When performing Gauss-Jordan elimination, you are only allowed to make three types of moves (operations) on the rows of the matrix. These moves mathematically preserve the relationships of the equations:

1. **Swap:** You can swap the positions of any two entire rows.
2. **Scale:** You can multiply or divide an entire row by any non-zero number.
3. **Pivot:** You can add or subtract a scaled version of one row to another row.

### 2. The Concrete Example

Let's take a simple system of two equations with two variables ($x$ and $y$):

* $2x + y = 5$
* $x - y = 1$

**Step 1: Create the Augmented Matrix**
First, we strip away the variables and write the coefficients and the results as an augmented matrix. The vertical line separates the left side of the equations from the right side.

$$\begin{bmatrix}
2 & 1 & | & 5 \\
1 & -1 & | & 1
\end{bmatrix}$$

**Step 2: Get a Leading 1 in the Top-Left**
Our first goal is to make the top-left number a $1$. The top-left number is currently a $2$. However, the row below it starts with a $1$. The easiest way to get a $1$ on top is to use our **Swap** rule.

* *Action:* Swap Row 1 ($R_1$) and Row 2 ($R_2$).

$$\begin{bmatrix}
1 & -1 & | & 1 \\
2 & 1 & | & 5
\end{bmatrix}$$

**Step 3: Eliminate the Numbers Below the Leading 1**
Now that we have our first pivot (the leading $1$), we must make every other number in that column a $0$. We need to turn the $2$ in the bottom row into a $0$. We use our **Pivot** rule.

* *Action:* Replace $R_2$ with ($R_2 - 2 \cdot R_1$).
* *Math:* * Column 1: $2 - (2 \cdot 1) = 0$
* Column 2: $1 - (2 \cdot -1) = 3$
* Column 3: $5 - (2 \cdot 1) = 3$



$$\begin{bmatrix}
1 & -1 & | & 1 \\
0 & 3 & | & 3
\end{bmatrix}$$

**Step 4: Get the Next Leading 1**
We move diagonally down to the right. The next pivot needs to be in row 2, column 2. Currently, that number is a $3$. To turn it into a $1$, we use our **Scale** rule.

* *Action:* Divide $R_2$ by $3$ ($R_2 \to \frac{1}{3}R_2$).

$$\begin{bmatrix}
1 & -1 & | & 1 \\
0 & 1 & | & 1
\end{bmatrix}$$

**Step 5: Eliminate the Numbers Above the New Pivot (The "Jordan" part)**
Standard *Gaussian elimination* stops here, but *Gauss-Jordan* goes all the way to RREF. This means we must clear out the column above our new leading $1$. We need to turn the $-1$ in the top row into a $0$. We use the **Pivot** rule again.

* *Action:* Replace $R_1$ with ($R_1 + R_2$).
* *Math:*
* Column 1: $1 + 0 = 1$
* Column 2: $-1 + 1 = 0$
* Column 3: $1 + 1 = 2$



$$\begin{bmatrix}
1 & 0 & | & 2 \\
0 & 1 & | & 1
\end{bmatrix}$$

---

### The Result

The matrix is now in perfect Reduced Row Echelon Form. The left side is the Identity matrix, and the right side gives us our final, perfectly isolated answers:

* $1x + 0y = 2 \implies x = 2$
* $0x + 1y = 1 \implies y = 1$

By feeding an augmented matrix $[A \mid B]$ into a computer and programmatically applying these three simple rules until the left side becomes an Identity matrix, systems like Triton can solve massive layout conversions entirely through linear algebra.


# Gauss-Jordan Elimination for solving the matrix equation AX = B
To solve the matrix equation $AX = B$ where $B$ is also a matrix using Gauss-Jordan elimination, you perform row operations on a combined augmented matrix. Instead of augmenting $A$ with a single column vector, you augment $A$ with the entire matrix $B$. 
The final goal is to transform the augmented matrix $[A \mid B]$ into $[I \mid X]$, where $I$ is the identity matrix and $X$ is your solution matrix.

---

## 1. Form the Augmented Matrix

* Setup: Place matrix $A$ on the left and matrix $B$ on the right of the dividing bar.
* Structure: The augmented matrix has the form $[A \mid B]$.
* Dimensions: If $A$ is $n \times n$ and $B$ is $n \times m$, the augmented matrix will be $n \times (n + m)$.

---

## 2. Perform Forward Elimination

* Objective: Convert the left side ($A$) into an upper triangular matrix.
* Process: Work column by column from left to right.
* Step: Eliminate all entries below the main diagonal pivoting elements using elementary row operations.

---

## 3. Scale the Pivot Rows

* Objective: Create leading ones along the main diagonal of the left matrix.
* Process: Divide each row by its diagonal (pivot) element.
* Result: The left matrix now has $1\text{s}$ on the diagonal and $0\text{s}$ below it.

---

## 4. Perform Backward Elimination

* Objective: Convert the left side into the identity matrix $I$.
* Process: Work from the bottom row up to the top row.
* Step: Eliminate all entries above the main diagonal pivoting elements.

---

## Concrete Example
Let $A = \begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$ and $B = \begin{pmatrix} 4 & 5 \\ 3 & 2 \end{pmatrix}$. We want to find the $2 \times 2$ matrix $X$.

### Step 1: Set up the augmented matrix
$$
[A \mid B] = \left( \begin{array}{cc|cc} 2 & 1 & 4 & 5 \\ 1 & 1 & 3 & 2 \end{array} \right)
$$ 

### Step 2: Eliminate below the first pivot
Swap Row 1 ($R_1$) and Row 2 ($R_2$) to get a $1$ in the top-left corner easily:
$$
\left( \begin{array}{cc|cc} 1 & 1 & 3 & 2 \\ 2 & 1 & 4 & 5 \end{array} \right)
$$ 
Subtract $2 \times R_1$ from $R_2$ ($R_2 \leftarrow R_2 - 2R_1$):
$$
\left( \begin{array}{cc|cc} 1 & 1 & 3 & 2 \\ 0 & -1 & -2 & 1 \end{array} \right)
$$ 

### Step 3: Scale the second pivot row
Multiply $R_2$ by $-1$ ($R_2 \leftarrow -1 \times R_2$):
$$
\left( \begin{array}{cc|cc} 1 & 1 & 3 & 2 \\ 0 & 1 & 2 & -1 \end{array} \right)
$$ 

### Step 4: Eliminate above the second pivot
Subtract $R_2$ from $R_1$ ($R_1 \leftarrow R_1 - R_2$):
$$
\left( \begin{array}{cc|cc} 1 & 0 & 1 & 3 \\ 0 & 1 & 2 & -1 \end{array} \right)
$$ 
The left side is now the identity matrix $I$. The right side is our solution matrix $X$.

---

### ✅ Final Answer
The solution matrix $X$ is explicitly isolated on the right side of the fully reduced augmented matrix.
$$
X = \begin{pmatrix} 1 & 3 \\ 2 & -1 \end{pmatrix}
$$ 

---
