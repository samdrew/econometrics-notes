# Econometrics Understanding

## Introduction

`Expectation` is the mean value for the population. This means that for a discrete possibility the $\mathbb{E}(x) = x \cdot P(x)$. For a continuous variable this is $\mathbb{E}(x) = \int x f(x) dx$ where $f(x)$ is the [density function](#pdf).

Other things to note about expectations: 

$$\mathbb{E}(x_1 + x_2) = \mathbb{E}(x_1) + \mathbb{E}(x_2)$$

This can also be shown in matricies with the following

$$\mathbb{E}(X_1 + X_2) = \mathbb{E}\left( \begin{array} 
X_{11} + X_{12} \\ X_{21} + X_{22} \\ ... \\ X_{1N} + X_{2N} \end{array} \right)\\
= \left( \begin{array} 
\mathbb{E}(X_{11} + X_{12}) \\ \mathbb{E}(X_{21} + X_{22}) \\ ... \\ \mathbb{E}(X_{N1} + X_{N2}) \end{array} \right) \\
= \left( begin{array} \mathbb{E}(X_{11}) \\  \mathbb{E}(X_{21}) \\ ... \\  \mathbb{E}(X_{N1}) \end{array} \right) + \left( begin{array} \mathbb{E}(X_{12}) \\  \mathbb{E}(X_{22}) \\ ... \\  \mathbb{E}(X_{N2}) \end{array} \right) \\
= \mathbb{E}(X_{1}) + \mathbb{E}(X_{2})$$

This also works for linear scaling with a matrix 

$$\mathbb{E}(AX) = A \mathbb{E}(X)$$

`Variance` is the square of the deviation from the expected value. For a discrete possibility this means that $V(x) = \sigma ^2 = \sum (x - \mathbb{E}(x))^2$, or the sum of the squared differences between the outcome and the expected outcome. In a continuous situation this is $V(x) = \int (x - \mathbb{E}(x))^2 f(x) dx$.

To do this with a vector X

$$ Var(X) = \mathbb{E}\left( (X - \mu) (X - \mu)' \right)$$

The variance of a random vector is a square variance-covariance matrix, which has the variance of each element of a random vector in the main diagonal and the interaction terms in the quadrants. This turns out to be a symmetric matrix, meaning that $A = A'$. 

$$Var(X) = \left( \begin{array} 
Var(X_1) & Cov(X_1, X_2) & ... & Covar(X_1, X_N) \\
Cov(X_2, X_1) & Var(X_2) & ... & ... \\
... & ... & ... & ... \\
Cov(X_N, X_1) & ... & ... & Var(X_N) \end{array} \right)$$

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

`Model` A simple single-variate linear regression model might be described as the following $y_i = \beta _0 + x_i \beta _1 + u_i$ for $i = 1, \ldots , N$

* $y_i$ is the outcome variable
* $\beta _0$ is the constant term
* $\beta _1$ is the slope
* $x_i$ is the covariate
* $u_i$ is the disturbance or error term. 

`covariates` are your random variables. These are also known as 

`mean independence`

## Estimation

`Ordinary Least Square` is what happens when you try to find an estimator which minimizes the square of the error term in the linear regression model. 

To find it, 

* rearrange the linear regression model in terms of u
* square it (or multiply it by it's transpose in the case of matrix formulation)
* differentiate that and solve for the minimum
* rearrange in terms of $\hat \beta$

This will give $$\hat \beta = (X'X)^{-1}X'y$$

`SSR` Rearranging our linear model to find $\hat u$ 

$$SSR(\beta) = \sum_{i=1}^N \hat u_i^2 = \sum_{i=1}^N (y_i - \hat \beta_0 -\hat \beta x_i)\\
S = (\hat u' \hat u) = (Y - \hat \beta X)'(Y - X \hat \beta)\\
 = y' y  - y' X \hat \beta - \hat \beta' X' y - \hat \beta ' X' X \hat \beta \\
 frac{dSSR}{d\hat \beta} = - X' y - X' y + 2 X'X \hat \beta = 0 $$

`β` vs $\hat \beta$

`R^2 regression`

`Standard Error regression`

## OLS Properties

`Assumptions`

`Sample Properties`

`Asymptotic Variance`

## Hypothesis Testing

`Hypothesis formulation`

`Test statistics`

`Types of errors`

`Single co-effient testing` Two-sided

One-sided

`t-statistics`

`Asymptotic t-test`

`Joint Hypothesis testing`

`Asymptotic f-test`

`Homoskedacity`

## Confidence Intervals



## Binary Response Models

`Notation`

`Single-index model`

### Parametric

`Parametric functions` Logit/Probit

`Log-likelihood function`

`Likelihood Estimators` ℒ

`Asymptotic Variance`

`Newton-Raphson`

### Semi-Parametric