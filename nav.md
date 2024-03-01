Musings...Just a start....


Let's quantize into buckets of unit length:

p(x) = $\sum_i \alpha_i I \\{ i<=x<=i+1\\}$

We move forward by $\Delta x$:

$$ p'(x) = p(x-\Delta x) $$

So:

```math
\alpha'_i = \int_{-\infty}^\infty p'(x) dx$$
```

$$ \alpha'_i = \int p(x-\Delta x) dx $$

```math
\alpha'_i = \int_{i}^{i+1} \sum_j \alpha_j I \{ j\leq x-\Delta x\leq j+1 \} dx
```
