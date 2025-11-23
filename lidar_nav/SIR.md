# Sampling Importance Resampling

We can estimate $E_{z \sim p(x)}[f(x)]$ by sampling taking N samples $x^i$ from p(x)

and computing:

$$
\frac{1}{N} \sum_{i=1}^N x^i
$$
