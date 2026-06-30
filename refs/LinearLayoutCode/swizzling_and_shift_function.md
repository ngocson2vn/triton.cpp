# Column shift function
Given the following function:

$S(\boldsymbol{r}) = 2^v((\boldsymbol{r} \gg p) \mathbin{\&}(2^q - 1))$

where
- $\boldsymbol{r} \in F_2^m$
- $m,\ n,\ p,\ q,\ v$ are positive integers and $m,\ p,\ q,\ v \le n$

Prove that $S(\boldsymbol{r})$ is a linear map: $F_2^m \rightarrow F_2^n$

## Proof
When we move into $F_2^m$ (the vector space of $m$-bit binary strings over the finite field of two elements), the concept of "addition" is no longer standard integer addition; it becomes the **bitwise XOR** operation ($\oplus$).

For a function to be linear in $F_2^m$, it must satisfy the **Boolean superposition** principle:


$S(\boldsymbol{a} \oplus \boldsymbol{b}) = S(\boldsymbol{a}) \oplus S(\boldsymbol{b})$

Here is exactly why your function is a strictly linear transformation in this domain:

### 1. The Right Shift is a Linear Operator

In $F_2^m$, shifting a bit vector to the right by $p$ positions ($\gg p$) simply drops the lowest $p$ bits and pulls in zeros at the top. This operates entirely independently on each bit position. In linear algebra terms, it is a matrix transformation. It distributes perfectly over XOR:


$(\boldsymbol{a} \oplus \boldsymbol{b}) \gg p = (\boldsymbol{a} \gg p) \oplus (\boldsymbol{b} \gg p)$

### 2. The Bitwise AND is a Linear Projection

The term $(2^q - 1)$ represents a constant bitmask of $q$ ones. A bitwise AND with a constant mask merely isolates specific bits (projecting them as they are) and zeroes out the rest. Because there are no "carries" across bits, this is also a linear operation over $F_2^n$:


$(\boldsymbol{x} \oplus \boldsymbol{y}) \mathbin{\&} (2^q - 1) = (\boldsymbol{x} \mathbin{\&} (2^q - 1)) \oplus (\boldsymbol{y} \mathbin{\&} (2^q - 1))$

### 3. The Multiplier Becomes a Left Shift

In standard arithmetic, mixing integer multiplication with bitwise operators usually destroys linearity. However, because your multiplier is $2^v$ (a power of two), multiplying by it is mathematically identical to performing a **logical left shift** by $v$ bits ($\ll v$).

Just like the right shift, a left shift is a linear operator in $F_2^n$:


$((\boldsymbol{x} \oplus \boldsymbol{y}) \ll v) = (\boldsymbol{x} \ll v) \oplus (\boldsymbol{y} \ll v)$

---

### The Final Proof
#### Additivity
Let's define the constant mask $M = 2^q - 1$.

Because multiplication by $2^v$ acts as a left shift, we can rewrite the function purely in bitwise terms:


$S(\boldsymbol{r}) = ((\boldsymbol{r} \gg p) \mathbin{\&} M) \ll v$

Now, evaluate $S(\boldsymbol{a} \oplus \boldsymbol{b})$:


$S(\boldsymbol{a} \oplus \boldsymbol{b}) = (((\boldsymbol{a} \oplus \boldsymbol{b}) \gg p) \mathbin{\&} M) \ll v$

$= (((\boldsymbol{a} \gg p) \oplus (\boldsymbol{b} \gg p)) \mathbin{\&} M) \ll v$

$= (((\boldsymbol{a} \gg p) \mathbin{\&} M) \oplus ((\boldsymbol{b} \gg p) \mathbin{\&} M)) \ll v$

$= (((\boldsymbol{a} \gg p) \mathbin{\&} M) \ll v) \oplus (((\boldsymbol{b} \gg p) \mathbin{\&} M) \ll v)$

$= S(\boldsymbol{a}) \oplus S(\boldsymbol{b})$

#### Scalar multiplication

In linear algebra, for a function to be a true linear transformation over a vector space, it must satisfy both **additivity** (which we proved in the last step) and **homogeneity** (scalar multiplication):


$S(c \cdot \boldsymbol{r}) = c \cdot S(\boldsymbol{r})$


where $c$ is a scalar from the underlying field.

Because the vector space is $F_2^m$, the underlying field is simply $F_2$ (the Galois Field of 2).

In $F_2$, there are only two possible scalars: **$0$ and $1$**.

Let's test both possible values for the scalar $c$:

**Case 1: The scalar is $1$**
If $c = 1$, the property is trivially true for any function:


$S(1 \cdot \boldsymbol{r}) = S(\boldsymbol{r}) = 1 \cdot S(\boldsymbol{r})$

**Case 2: The scalar is $0$**
If $c = 0$, the vector $r$ becomes a zero-vector ($0 \cdot \boldsymbol{r} = \boldsymbol{0}$). We just need to evaluate $S(\boldsymbol{0})$:


$S(\boldsymbol{0}) = 2^v((\boldsymbol{0} \gg p) \mathbin{\&} (2^q - 1))$

* $\boldsymbol{0} \gg p = \boldsymbol{0}$
* $\boldsymbol{0} \mathbin{\&} (2^q - 1) = \boldsymbol{0}$
* $2^v \cdot \boldsymbol{0} = \boldsymbol{0}$

So, $S(\boldsymbol{0}) = \boldsymbol{0}$.
Since $0 \cdot S(\boldsymbol{r})$ is also $\boldsymbol{0}$, the equation holds:


$S(0 \cdot \boldsymbol{r}) = \boldsymbol{0} = 0 \cdot S(\boldsymbol{r})$

### Conclusion

Because the only possible scalars are $0$ and $1$, and the function naturally maps an all-zero input to an all-zero output (meaning the origin is preserved), **scalar multiplication is strictly satisfied.** Combined with the additivity we proved earlier, the function $S(\boldsymbol{r})$ checks every box. It is a completely valid, rigorous linear transformation from the vector space $F_2^m$ to vector space $F_2^n$.

**Side note:** <br/>
3 bitwise operations can be converted to matrix multiplications
- $\boldsymbol{r} \gg p$ is equivalent to multiplication of a right shift matrix $\stackrel{n \times m}{M_1}$ and column vector $\boldsymbol{r}$. 

- Denote $\boldsymbol{a} = \boldsymbol{r} \gg p$ be a binary vector in $F_2^n$.<br/> Then, $\boldsymbol{a} \mathbin{\&} (2^q - 1)$ is equivalent to multiplication of a diagonal matrix $\stackrel{n \times n}{M_2}$ which has only $q$ ones along the main diagonal with the column vector $\boldsymbol{a}$.

- Denote $\boldsymbol{b} = \boldsymbol{a} \mathbin{\&} (2^q - 1)$ be a binary vector in $F_2^n$.<br/> Then, $\boldsymbol{b} \ll 2^v$ is equivalent to multiplication of a left shift matrix $\stackrel{n \times n}{M_3}$ with the column vector $\boldsymbol{b}$

<br/>

$S(\boldsymbol{r})$ can be rewritten as

$`S(\boldsymbol{r}) = \, \stackrel{n \times n}{M_3}(\stackrel{n \times n}{M_2}(\stackrel{n \times m}{M_1}\boldsymbol{r})) = \stackrel{n \times m}{M}\boldsymbol{r}`$, where $M = M_3 M_2 M_1$

<br/>

# Swizzling Linear Map

Start with building a linear map to map memory offsets to tensor indices as follows:

Identity map $L_1$:

$
\begin{aligned}
L_1:\ F_2^m &\rightarrow F_2^m \\
\boldsymbol{r} &\mapsto \boldsymbol{r} \\
\end{aligned}
$

Identity map $L_2$:

$
\begin{aligned}
L_2:\ F_2^n &\rightarrow F_2^n \\
\boldsymbol{c} &\mapsto \boldsymbol{c} \\
\end{aligned}
$

Direct sum $L_1 \times L_2$:

$
\begin{aligned}
L_1 \times L_2:\ F_2^m \times F_2^n &\rightarrow F_2^m \times F_2^n \\
(\boldsymbol{r}, \boldsymbol{c})  &\mapsto (\boldsymbol{r}, \boldsymbol{c})
\end{aligned}
$

$
\begin{aligned}
&\phi:\ F_2^m \times F_2^n \rightarrow F_2^{m + n} \\
&((r_0, \dots, r_{m-1}),(c_0, \dots, c_{n-1})) \mapsto (c_0, \dots, c_{n-1}, r_0, \dots, r_{m-1})
\end{aligned}
$

$
\begin{aligned}
&\phi^{-1}:\ F_2^{m + n} \rightarrow F_2^m \times F_2^n \\
&(c_0, \dots, c_{n-1}, r_0, \dots, r_{m-1}) \mapsto ((r_0, \dots, r_{m-1}),(c_0, \dots, c_{n-1}))
\end{aligned}
$

$
\begin{aligned}
&L = (L_1 \times L_2) \circ \phi^{-1}:\ F_2^{m + n} \rightarrow F_2^m \times F_2^n
\end{aligned}
$

For $0 \le k \lt m$: <br/>
$L(\boldsymbol{e}_k) = (L_1(\boldsymbol{u}_k), \boldsymbol{0}^n)$, where $\boldsymbol{u}_k$ is $k$-th standard basis vector of $F_2^m$

For $m \le k \lt m + n$: <br/>
$L(\boldsymbol{e}_k) = (\boldsymbol{0}^m, L_2(\boldsymbol{v}_{k-m}))$, where $\boldsymbol{v}_h$ is $h$-th standard basis vector of $F_2^n$

<br/>

Let $\Omega$ denote the following map:

$
\begin{aligned}
\Omega:\ F_2^m \times F_2^n &\rightarrow F_2^m \times F_2^n \\
(\boldsymbol{r}, \boldsymbol{c})  &\mapsto (\boldsymbol{r}, \boldsymbol{c} \oplus S(\boldsymbol{r}))
\end{aligned}
$

Then, we can easily prove that $\Omega$ is a linear map because $S(\boldsymbol{r})$ is a linear map.

<br/>

Denote $Z = \Omega \circ L:\ F_2^{m + n} \rightarrow F_2^m \times F_2^n$

Then, we can compute $Z(\boldsymbol{e}_k)$ as follows:

For $0 \le k \lt m$: <br/>
$Z(\boldsymbol{e}_k) = (L_1(\boldsymbol{u}_k), \boldsymbol{0}^n \oplus S(L_1(\boldsymbol{u}_k))) = (L_1(\boldsymbol{u}_k), S(L_1(\boldsymbol{u}_k)))$

For $m \le k \lt m + n$: <br/>
$Z(\boldsymbol{e}_k) = (\boldsymbol{0}^m, L_2(\boldsymbol{v}_{k-m}) \oplus S(\boldsymbol{0}^m)) = (\boldsymbol{0}^m, L_2(\boldsymbol{v}_{k-m}))$

$Z$ is called a swizzling linear map.
