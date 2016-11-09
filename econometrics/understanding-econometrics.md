# Econometrics Understanding

## Introduction

`Expectation` is the mean value for the population. This means that for a discrete possibility the $\mathbb{E}(x) = x \cdot P(x)$. For a continuous variable this is $\mathbb{E}(x) = \int x f(x) dx$ where $f(x)$ is the [density function](#pdf).

`Variance` is the square of the deviation from the expected value. For a discrete possibility this means that $V(x) = \sigma ^2 = \sum (x - \mathbb{E}(x))^2$, or the sum of the squared differences between the outcome and the expected outcome. In a continuous situation this is $V(x) = \int (x - \mathbb{E}(x))^2 f(x) dx$.

`Density Function` <a name="pdf"></a> (`f(x)`) is the function which describes the chance of a given variable occuring. In discrete variables this is the weight variable and equivlent to $P(x = x_0)$, while for continuous variables this is just defined as $f(x_0) = frac{dP(x<=x_0)}{dx_0}$ where $P(x < x_0)$ is the [cumulative distribution function](#cdf)

`Cumulative Distribution Function` <a name="cdf"></a> is the chance of a given threshold value being greater than or equal to the case. This can be defined in terms of the density function as $$P(x <= x_0) = \int_{-\infty}^{x_0} f(x)$$ for continuous distributions.

`Joint Distributions` in real-life most probabilities won't be a function of just a single variable. This then gives rise to functions taking multiple variables for the density & distribution functions. The distribution function would then be denoted as $P(y <= y_0, x <= x0)$ and the density function as the derivative ($frac{dP(y <= y_0, x <= x0)}{dx_0 dy_0}$).

`Conditional Distribution/Expectation` When dealing with joint distributions mostly we will be considering one variable at a time, and as such want to look at the probability of one outcome given another (denoted as $P(y | x = x_0)$). This has the density function defined as $f(y|x) = frac{f(y,x)}{f(x)}$. 

This then has the expectation $$\mathbb{E}(y | x = x_0) = \int y f(y|x_0) dy$$ and variance $$V(y|x) = \int (y - \mathbb{E}(y|x))^2 f(y|x) dy$$

In the case that $\mathbb{E}(Y | X = x)$ isn't for a fixed value of X, then the expectation is then a function with respect to x. (So $\mathbb{E}(Y | X = x) = g_x(Y)$)

`Independence`

## Sampling

`Random Variables`

`iid`

`Estimators/Models`

`Law of Large Numbers`

`Central Limit Theorem`

`Continuous Mapping`

`Slutzky Lemma`

## Linear Regression Model

`Model` $y_i = \beta_0 + x_i \beta_1 + u_i$ //TODO Break down

`covariates`

`mean independence`

## Estimation

`OLS`

`SSR`

`β` vs $\hat \beta$

`R^2 regression`

`Standard Error regression`

## OLS Properties

`Sampling properties`

``