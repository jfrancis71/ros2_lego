Musings...Just a start....


Let's quantize into buckets of unit length:

p(x) = $\sum_i \alpha_i I \\{ i<=x<=i+1\\}$

$\alpha x$

We move forward by $\Delta x$:



$$ p'(x) = \int p(x-\Delta x) dx $$

So:

$$ \alpha'_i = \int p'(x) dx $$

$$ \alpha'_i = \int \int p(x'-\Delta x) dx' dx $$

$$ \alpha'_i = \int \int \sum_j \alpha_j I \\{ j<=x'<=j+1\\} dx' dx $$
