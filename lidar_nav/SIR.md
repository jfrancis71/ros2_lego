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

We can estimate this by computing:

$$
\frac{1}{N} \sum_{i=1}^N x^i \frac{p(x^i)}{q(x^i)}
$$

## SIR

SIR (Sampling Importance Resampling) is the technique of resampling a set of samples from q.

We construct a discrete probability distribution over our samples from q and for each of these elements we give it probability $\frac{p(x)}{q(x)}$ and normalise this probability distribution. We then generate N samples from this distribution. This is like a set of samples drawn directly from the distribution p, even though it has been drawn from q.

It is not exact as, for example if p is a normal distribution our samples may repeat themselves which has zero probability for a normal distribution (no two seperate samples are likely to be exactly equal). Nevertheless for the purposes of computing expectations this is approximately correct and improves with larger N.

## Reference

Bishop, C.M. (2006), Pattern Recognition and Machine Learning, p.534
