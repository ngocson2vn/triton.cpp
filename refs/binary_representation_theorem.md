### The Base-2 Representation Theorem

The theorem states that every non-negative integer $N$ can be expressed in exactly **one** way as a sum of distinct powers of 2.

Mathematically, any integer $N \ge 0$ can be written uniquely as:


$N = \sum\limits_{i=0}^{m} c_i 2^i$


where the coefficients $c_i \in \{0, 1\}$.

## Proof
Here is the constructive proof using the Division Algorithm.

### The "No-Choice" Constructive Proof

Let $N$ be any non-negative integer. We want to write it in base-2:


$N = c_0 2^0 + c_1 2^1 + c_2 2^2 + \dots + c_m 2^m$


where every coefficient $c_i \in \{0, 1\}$.

**Step 1: Isolate the first bit ($c_0$)**

Let's factor out a 2 from every term *except* the first one ($c_0 2^0$, which is just $c_0$):


$N = c_0 + 2(c_1 + c_2 2^1 + c_3 2^2 + \dots + c_m 2^{m-1})$

Notice the structure of this equation. It is exactly the Euclidean Division Algorithm:


$N = \text{Remainder} + 2 \times \text{Quotient}$

Let $Q_1 = (c_1 + c_2 2^1 + \dots + c_m 2^{m-1})$.
So, our equation is simply:


$N = 2Q_1 + c_0$

**Step 2: The Parity Lock**

Here is where the uniqueness becomes obvious. If you divide any integer $N$ by 2, there is absolutely no ambiguity about what the remainder is.

* If $N$ is even, the remainder $c_0$ **must** be 0.
* If $N$ is odd, the remainder $c_0$ **must** be 1.

You have no choice in the matter. The value of $c_0$ is entirely locked in by whether $N$ is even or odd. Furthermore, the quotient $Q_1$ is also entirely locked in (it is exactly $\lfloor N / 2 \rfloor$).

Because $c_0$ can only be exactly one value, the first bit is uniquely determined.

**Step 3: Repeat the process**

Now we look at our locked-in quotient, $Q_1$. We know its formula from Step 1:


$Q_1 = c_1 + c_2 2^1 + c_3 2^2 + \dots + c_m 2^{m-1}$

We do the exact same thing. We factor out a 2 from everything except $c_1$:


$Q_1 = c_1 + 2(c_2 + c_3 2^1 + \dots + c_m 2^{m-2})$

$Q_1 = c_1 + 2Q_2$

Once again, if we divide $Q_1$ by 2, the remainder $c_1$ is locked in by the parity of $Q_1$. You have no choice. The value of $c_1$ is strictly uniquely determined.

**Conclusion**

As you iterate this process, every single coefficient ($c_0, c_1, c_2, \dots$) is sequentially locked into place as the deterministic remainder of dividing by 2.

Because standard division only ever yields one specific quotient and one specific remainder, the algorithm generates exactly one specific sequence of 1s and 0s. There is zero mathematical wiggle room to generate an alternative set of coefficients, making the representation absolutely and intuitively **unique**.
