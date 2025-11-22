# Particle Filter

Bishop p645

We have a robot that provides a stream of observations from a sensor (eg a Lidar) and want to estimate its position.
We will call the sequence of observations x and positions z.

We are ignoring control signals for the initial analysis:

The setup is we would like to compute an expectation $E_{z\sim p(z_n|x_{1..n})}[f(z)]$

## Section 1

We demonstrate how to compute $E_{z\sim p(z_n|x_{1..n})}[f(z)]$ using samples from $p(z_n|x_{1..n-1})$

$$
E_{z\sim p(z_n|x_{1..n})}[f(z)] = \int p(z_n|x_{1..n}) f(z_n) dz_n
$$

Let's break out the $x_n$ from $x_{1..n-1}$

$$
= \int p(z_n|x_n, x_{1..n-1}) f(z_n) dz_n
$$

Using Bayes Theorem:

$$
= \int \frac{p(x_n|z_n, x_{1..n-1}) p(z_n | x_{1..n-1})}{p(x_n|x_{1..n-1})} f(z_n) dz_n
$$

Using conditional independence:

$$
= \int \frac{p(x_n|z_n) p(z_n | x_{1..n-1})}{p(x_n|x_{1..n-1})} f(z_n) dz_n
$$

The denominator is just a number:

$$
= \frac{\int p(x_n|z_n) p(z_n | x_{1..n-1}) f(z_n) dz_n}{p(x_n|x_{1..n-1})}
$$

Reverse marginalising the denominator

$$
= \frac{\int p(x_n|z_n) p(z_n | x_{1..n-1}) f(z_n) dz_n}{ \int p(x_n,z_n|x_{1..n-1}) dz_n}
$$

Refactoring the denominator by chain rule:

$$
= \frac{\int p(x_n|z_n) p(z_n | x_{1..n-1}) f(z_n) dz_n}{ \int p(z_n|x_{1..n-1}) p(x_n|z_n, x_{1..n-1}) dz_n}
$$

By conditional independence:

$$
= \frac{\int p(x_n|z_n) p(z_n | x_{1..n-1}) f(z_n) dz_n}{ \int p(z_n|x_{1..n-1}) p(x_n|z_n) dz_n}
$$

Turning into expectations:

$$
= \frac{E_{z_n \sim p(z_n|x_{1..n-1})}[f(z) p(x_n|z_n)]}{E_{z_n \sim p(z_n|x_{1..n-1})}[p(x_n|z_n)]}
$$

We can estimate this by taking N samples from $p(z_n|x_{1..n-1})$ and computing:

$$
So E_{z\sim p(z_n|x_{1..n})}[f(z)] \\approx \sum_i f(z_i) w_i
$$

where:

$$
w_i = \frac{p(x_n|z_i)}{\sum_i' p(x_n|z_i')}
$$

## Section 2

Reverse marginalising:

$$
p(z_n|x_{1..n-1}) = \int p(z_n,z_{n-1}|x_{1..n-1}) dz_{n-1}
$$

Chain rule:

$$
= \int p(z_n|z_{n-1},x_{1..n-1}) p(z_{n-1}|x_{1..n-1}) dz_{n-1}
$$

Conditional independence:

$$
= \int p(z_n|z_{n-1}) p(z_{n-1}|x_{1..n-1}) dz_{n-1}
$$

Seperate out $x_{n-1}$

$$
= \int p(z_n|z_{n-1}) p(z_{n-1}|x_{n-1}, x_{1..n-2}) dz_{n-1}
$$

Apply Bayes rule:

$$
= \int p(z_n|z_{n-1}) \frac{p(z_{n-1}|x_{1..n-2}) p(x_{n-1}|z_{n-1}, x_{1..n-2})}{p(x_{n-1}|x_{1..n-2})} dz_{n-1}
$$

Apply conditional independence rule:

$$
= \int p(z_n|z_{n-1}) \frac{p(z_{n-1}|x_{1..n-2}) p(x_{n-1}|z_{n-1})}{p(x_{n-1}|x_{1..n-2})} dz_{n-1}
$$


Denominator is constant wrt $z_{n-1}$:

$$
= \frac{\int p(z_n|z_{n-1}) p(z_{n-1}|x_{1..n-2}) p(x_{n-1}|z_{n-1}) dz_{n-1}}{p(x_{n-1}|x_{1..n-2})}
$$

Reverse marginalising the denominator wrt z_n:

$$
= \frac{\int p(z_n|z_{n-1}) p(z_{n-1}|x_{1..n-2}) p(x_{n-1}|z_{n-1}) dz_{n-1}}{\int p(z_{n-1}, x_{n-1}|x_{1..n-2}) dz_{n-1}}
$$

Applying chain rule to denominator:

$$
= \frac{\int p(z_n|z_{n-1}) p(z_{n-1}|x_{1..n-2}) p(x_{n-1}|z_{n-1}) dz_{n-1}}{\int p(x_{n-1}|z_{n-1}, x_{1..n-2}) p(z_{n-1}|x_{1..n-2}) dz_{n-1}}
$$

Applying conditional independence to denominator

$$
= \frac{\int p(z_n|z_{n-1}) p(z_{n-1}|x_{1..n-2}) p(x_{n-1}|z_{n-1}) dz_{n-1}}{\int p(x_{n-1}|z_{n-1}) p(z_{n-1}|x_{1..n-2}) dz_{n-1}}
$$
