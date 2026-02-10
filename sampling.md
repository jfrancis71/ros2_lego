# Sampling

Topic here is sampling from distribution p(x) where p(x) is difficult to sample directly from. However it is fairly straightforward to evaluate p(x).

``` mermaid
mindmap
  root((Sampling))
    Importance Sampling
      Uses a proposal distribution q
      Limitations
        May be inefficient if q is very different from p
    Gibbs Sampling
      Aimed at sampling from a multivariate distribution
      Limitations
        id2["Assumes that the conditional distribution p xt given x excluding t is easy to sample from."]
        Can get stuck in islands if variables strongly correlated
    Metropolis Hastings
      Uses a proposal distribution q
      Limitations
        If acceptance ratio not close to 0.5, can mix slowly.
