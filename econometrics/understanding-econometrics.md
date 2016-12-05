# Econometrics Understanding

## Introduction

`Expectation` is the mean value for the population. This means that for a discrete possibility the $\mathbb{E}(x) = x \cdot P(x)$. For a continuous variable this is $\mathbb{E}(x) = \int x f(x) dx$ where $f(x)$ is the [density function](#pdf).

Other things to note about expectations: 

$$\mathbb{E}(x_1 + x_2) = \mathbb{E}(x_1) + \mathbb{E}(x_2)$$

This can also be shown in matricies with the following

$$\begin{aligned}
\mathbb{E}(X_1 + X_2) &= \mathbb{E}\left( \begin{array} 
X_{11} + X_{12} \\ X_{21} + X_{22} \\ ... \\ X_{1N} + X_{2N} \end{array} \right)\\
&= \left( \begin{array} 
\mathbb{E}(X_{11} + X_{12}) \\ \mathbb{E}(X_{21} + X_{22}) \\ ... \\ \mathbb{E}(X_{N1} + X_{N2}) \end{array} \right) \\
&= \left( \begin{array} \mathbb{E}(X_{11}) \\  \mathbb{E}(X_{21}) \\ ... \\  \mathbb{E}(X_{N1}) \end{array} \right) + \left( \begin{array} \mathbb{E}(X_{12}) \\  \mathbb{E}(X_{22}) \\ ... \\  \mathbb{E}(X_{N2}) \end{array} \right) \\
&= \mathbb{E}(X_{1}) + \mathbb{E}(X_{2})
\end{aligned}$$

This also works for linear scaling with a matrix 

$$\mathbb{E}(AX) = A \mathbb{E}(X)$$

`Variance` <a name="var"></a> is the square of the deviation from the expected value. For a discrete possibility this means that $V(x) = \sigma ^2 = \sum (x - \mathbb{E}(x))^2$, or the sum of the squared differences between the outcome and the expected outcome. In a continuous situation this is $V(x) = \int (x - \mathbb{E}(x))^2 f(x) dx$.

The variance of a random vector is a square variance-covariance matrix, which has the variance of each element of a random vector in the main diagonal and the interaction terms in the quadrants. This turns out to be a symmetric matrix, meaning that $A = A'$. 

$$Var(X) = \left( \begin{array} 
Var(X_1) & Cov(X_1, X_2) & ... & Covar(X_1, X_N) \\
Cov(X_2, X_1) & Var(X_2) & ... & ... \\
... & ... & ... & ... \\
Cov(X_N, X_1) & ... & ... & Var(X_N) \end{array} \right)$$

This can then be written in terms of the expectations where

$$ Var(X) = \mathbb{E}\left( (X - \mathbb{E}(X)) (X - \mathbb{E}(X))' \right) \\
\left( \begin{array} 
(X_1 - \mathbb{E}(X_1))^2 & (X_1 - \mathbb{E}(X_1))(X_2 - \mathbb{E}(X_2)) & ... & ... \\
... & (X_2 - \mathbb{E}(X_2))^2  & ... & ... \\
... & ... & ... & ... \\
Cov(X_N, X_1) & ... & ... & Var(X_N) \end{array} \right)$$

When we then want to find Var(AX)

$$Var(AX) = \mathbb{E}\left( (AX - A\mathbb{E}(X))(AX - A\mathbb{E}(X)') \right) \\
= \mathbb{E}\left( A(X - \mathbb{E}(X))(X - \mathbb{E}(X)')A' \right)  \\
= A \mathbb{E}\left( (X - \mathbb{E}(X))(X - \mathbb{E}(X)') \right) A' \\
= A Var(X) A' $$

`Density Function` <a name="pdf"></a> (`f(x)`) is the function which describes the chance of a given variable occuring. In discrete variables this is the weight variable and equivlent to $P(x = x_0)$, while for continuous variables this is just defined as $f(x_0) = frac{dP(x<=x_0)}{dx_0}$ where $P(x < x_0)$ is the [cumulative distribution function](#cdf)

`Cumulative Distribution Function` <a name="cdf"></a> is the chance of a given threshold value being greater than or equal to the case. This can be defined in terms of the density function as $$P(x <= x_0) = \int_{-\infty}^{x_0} f(x)$$ for continuous distributions.

`Joint Distributions` in real-life most probabilities won't be a function of just a single variable. This then gives rise to functions taking multiple variables for the density & distribution functions. The distribution function would then be denoted as $P(y <= y_0, x <= x0)$ and the density function as the derivative ($frac{dP(y <= y_0, x <= x0)}{dx_0 dy_0}$).

`Conditional Distribution/Expectation` When dealing with joint distributions mostly we will be considering one variable at a time, and as such want to look at the probability of one outcome given another (denoted as $P(y | x = x_0)$). This has the density function defined as $f(y|x) = \frac{f(y,x)}{f(x)}$. 

This then has the expectation $$\mathbb{E}(y | x = x_0) = \int y f(y|x_0) dy$$ and variance $$V(y|x) = \int (y - \mathbb{E}(y|x))^2 f(y|x) dy$$

In the case that $\mathbb{E}(Y | X = x)$ isn't for a fixed value of X, then the expectation is then a function with respect to x. (So $\mathbb{E}(Y | X = x) = g_x(Y)$)

`Independence` is an important assumption of variables in many of the models, and has the effect of massively simplifying the maths so that $\mathbb{E}(Y | X) = \mathbb{E}(Y)$. Mostly this is relevant in the case that 

## Sampling

`Random Variables` 

`iid`

`Estimators/Models`

`Law of Large Numbers`

`Central Limit Theorem`

`Continuous Mapping` states that for a given (continuous) function $f(x)$, as the value of $\hat x$ approaches a given value $x$, the function $f(\hat x)$ also approaches a value $f(x)$. This only works if the 

`Slutzky Lemma` 

## Linear Regression Model

`Model` A simple single-variate linear regression model might be described as the following $y_i = \beta _0 + x_i \beta _1 + u_i$ for $i = 1, \ldots , N$

* $y_i$ is the outcome variable
* $\beta _0$ is the constant term
* $\beta _1$ is the slope
* $x_i$ is the covariate (dependent variable)
* $u_i$ is the disturbance or error term. 

Given that this is extremely basic, we'd normally want to extend it to have lots of dependent variables, which in turn means lots of co-efficient terms ($\beta _k$)

`covariates` are your random variables. These are also known as 

`bias` Given our OLS estimator $\hat \beta = (X' X)^{-1} X' y$ we can plug in the linear regression model $y = X\beta + u$ to give us 
$$\begin{align}
\hat \beta &= (X' X)^{-1} X' (X\beta + u)\\
 &= (X' X)^{-1} X' X \beta + (X' X)^{-1} X' u \\
 &= \beta + (X' X)^{-1} X' u
 \end{align}$$

 We can then take expectations of this, and given the mean independence of the error term ($\mathbb{E}(u | X) = 0$) we get 

 $$\mathbb{E}(\hat \beta) = \beta + (X' X) X' \mathbb{E}(u) \\
  = \beta $$

`variance` we can start with the identity we found in the previous [variance](#var) section, $Var(AX) = A Var(X) A'$ and apply that to our $\hat \beta = (X' X)^{-1} X y$ where $A = (X' X)^{-1} X'$. This gives us 

$$Var(\hat \beta) = (X' X)^{-1} X' Var(y) X (X' X)^{-1}\\
 = (X' X)^{-1} X' \sigma I X (X' X)^{-1}\\
 = \sigma^2 (X' X)^{-1} X' X (X' X)^{-1}\\
 = \sigma^2 (X' X)^{-1} $$

`Gauss-Markov` provides proof that $\hat \beta _{OLS}$ is the Best Unbiased Linear Estimator for a linear regression, under the G-M Conditions (e.g. zero-mean conditional variance)

We can start by looking for a better linear estimator. Any linear estimator will be in the form $$\tilde \beta = \alpha y + u$$ where $\alpha = \left( \begin{array} \alpha _0 \\ \alpha _1 \\ ... \\ \alpha _N \end{array} \right)$. This can then be described in terms of the $\hat \beta_{OLS}$ as $\tilde \beta = a$

## Estimation

`Ordinary Least Square` is what happens when you try to find an estimator which minimizes the square of the error term in the linear regression model. 

To find it, 

* rearrange the linear regression model in terms of u :$$u = y - X'\beta$$
* square it (or multiply it by it's transpose in the case of matrix formulation) : 
$$\begin{aligned}uu' &= (y- X'\beta)(y-X'\beta)' \\
&= yy' - 2 yX\beta + X'X
\end{aligned}$$
* differentiate that and solve for the minimum
* rearrange in terms of $\hat \beta$

This will give $$\hat \beta = (X'X)^{-1}X'y$$

`SSR` Rearranging our linear model to find $\hat u$ 

$$\begin{aligned}
SSR(\beta) &= \sum_{i=1}^N \hat u_i^2 = \sum_{i=1}^N (y_i - \hat \beta_0 -\hat \beta x_i)\\
S &= (\hat u' \hat u) = (Y - \hat \beta X)'(Y - X \hat \beta)\\
 &= y' y  - y' X \hat \beta - \hat \beta' X' y - \hat \beta ' X' X \hat \beta \\
 \frac{\delta SSR}{\delta \hat \beta} &= - X' y - X' y + 2 X'X \hat \beta = 0 
 \end{aligned}$$

`$\beta$ vs $\hat \beta$`

`R^2 regression`

`Standard Error regression`

## OLS Properties

`Assumptions` about the data are required for the OLS estimator to be valid. There are a set of assumptions named Gauss-Markov which include

* Homoskadicity
* iid
* 

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

These models have the properties that for -∞ the function should return 0 and and the function for +∞ tends towards 1, and the function of 0 returning 1/2. The models generally used are the `logit` (L or $\Lamda$) and `probit` ($\Phi$).

`Notation` Binary response functions tend to be denoted as $1( \cdot )$. In a generic form these would then contain a condition, e.g. $1(\gamma_0(x_i, u_i) \geq 0 )$.

`Single-index model` is a set of conditions which are applied to the binary response model. These are 

* `additive separability` meaning that the disturbance term is not really a function of $\gamma_0$, and can be stated as $\gamma_0(x_i, u_i) = \gamma_0(x_i) - u_i$
* linear function $\gamma_0$ is assumed to be a linear transformation which can be described in the form of a matrix $\beta_0$, meaning that $\gamma_0(x_i, u_i) = x_i'\beta_0 - u_i$
* which allows single outcome binary outcome model to be described as a function where the outcome from $x_i'\beta_0$ is larger than $u_i$. This can be written as $y_i = 1(x_i'\beta_0 \geq u_i)$

### Parametric

`Parametric functions` Logit/Probit

`Log-likelihood function`

`Likelihood Estimators` ℒ

`Asymptotic Variance`

`Newton-Raphson` is an iterative heuristic algorithm for finding solutions to equations. The reason it is relevant is that it is able to provide results for problems which cannot otherwise be analytically solved.

### Semi-Parametric

## Maximum Likelihood Estimator

We are trying to find the smallest possible value for the error term ($\theta$) when using a binary outcome model to estimate an outcome. This term shall be known as $\theta_0$

`Log-likelihood estimator` Starting with the log-likelihood function $$ln \mathcal{L}(\theta) = \sum\limits_{i=1}^n ln(f(z_i,\theta))$$ we need to find the likelihood estimator where $$\theta_n := arg \max_{\theta} ln \mathcal{L}(\theta)$$

