# Sampling Importance Resampling

We can estimate $E_{x \sim p(x)}[f(x)]$ by sampling taking N samples $x^i$ from p(x)

and computing:

$$
\frac{1}{N} \sum_{i=1}^N x^i
$$

What if we do not have samples from p(x) but from q(x) instead? We can use importance sampling to weight the samples appropriately.

## Importance Sampling

$$
E_{x \sim p(x)}[f(x)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = E_{x \sim q(x)}[f(x) \frac{p(x)}{q(x)}]
$$
