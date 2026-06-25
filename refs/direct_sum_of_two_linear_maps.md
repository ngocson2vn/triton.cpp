# Direct Sum Of Two Linear Maps
Given vector spaces $V,\; W,\; U,\; X$ and linear maps $F:\; V \rightarrow U$ and $G:\; W \rightarrow X$,

The direct sum $F \times G:\; V \times U \rightarrow W \times X$ (often denoted as $F \oplus G$) is defined by its action on an ordered pair $(v, w) \in V \times W$:

$(F \times G)(v, w) = (F(v), G(w))$

Prove that the direct sum $F \times G$ is also a linear map.

## Proof

To prove that the direct sum (or Cartesian product) of two linear maps is also a linear map, we need to show that it satisfies the two foundational properties of linearity: 
- **additivity** and 
- **scalar multiplication** (homogeneity).

Let $(v_1, w_1)$ and $(v_2, w_2)$ be elements in $V \times W$, and let $c \in \mathbb{F}$ be a scalar.

---

### Part 1: Proof of Additivity

We must show that $(F \times G)((v_1, w_1) + (v_2, w_2)) = (F \times G)(v_1, w_1) + (F \times G)(v_2, w_2)$.

1\. **Add the vectors in the domain:** By the definition of vector addition in the product space $V \times W$, we add component-wise:

$(F \times G)((v_1, w_1) + (v_2, w_2)) = (F \times G)(v_1 + v_2, w_1 + w_2)$


2\. **Apply the map definition:** By the definition of the map $F \times G$, we apply $F$ to the first component and $G$ to the second:

$= (F(v_1 + v_2), G(w_1 + w_2))$


3\. **Use the linearity of $F$ and $G$:** Since $F$ and $G$ are individually linear maps, 

$F(v_1 + v_2) = F(v_1) + F(v_2)$ and $G(w_1 + w_2) = G(w_1) + G(w_2)$. 

Substituting these in yields:

$= (F(v_1) + F(v_2), G(w_1) + G(w_2))$


4\. **Separate the vectors in the codomain:** By the definition of vector addition in the product space $U \times X$, we can split the ordered pair into a sum of two ordered pairs:

$= (F(v_1), G(w_1)) + (F(v_2), G(w_2))$


5\. **Substitute the map definition back:** 

$= (F \times G)(v_1, w_1) + (F \times G)(v_2, w_2)$

This confirms that additivity holds.

---

### Part 2: Proof of Scalar Multiplication

We must show that $(F \times G)(c(v_1, w_1)) = c(F \times G)(v_1, w_1)$.

1\. **Multiply by the scalar in the domain:** By the definition of scalar multiplication in $V \times W$:

$(F \times G)(c(v_1, w_1)) = (F \times G)(cv_1, cw_1)$


2\. **Apply the map definition:** 

$= (F(cv_1), G(cw_1))$


3\. **Use the linearity of $F$ and $G$:** Since $F$ and $G$ are individually linear, we can pull the scalar $c$ out of both functions:

$= (cF(v_1), cG(w_1))$


4\. **Factor the scalar out in the codomain:** By the definition of scalar multiplication in $U \times X$, we pull $c$ out of the ordered pair:

$= c(F(v_1), G(w_1))$


5\. **Substitute the map definition back:**

$= c(F \times G)(v_1, w_1)$



This confirms that scalar multiplication holds.

---

### Conclusion

Because the map $F \times G$ preserves both vector addition and scalar multiplication, **$F \times G$ is a linear map**.
<br/><br/>

In algebra, given the following two linear maps:

$L_1:\; F_2^{N_1} \rightarrow F_2^{D_1}$

$L_2:\; F_2^{N_2} \rightarrow F_2^{D_2}$

Direct sum of $L_1$ and $L_2$:

$L_1 \times L_2:\; F_2^{N_1} \times F_2^{N_2} \rightarrow F_2^{D_1} \times F_2^{D_2}$

Let $\phi$ denote linear map $\phi:\; F_2^{N_1} \times F_2^{N_2} \rightarrow F_2^{N_1+N_2}$

Let $T$ denote linear map $T:\; F_2^{N_1+N_2} \rightarrow F_2^{D_1} \times F_2^{D_2}$

Let $\mathbf{e}_k,\; 0 \ll k < N_1 + N_2$ denote standard basis in $F_2^{N_1+N_2}$

How to strictly compute $T(\mathbf{e}_k)$ via $L_1$, $L_2$, and $\phi$?