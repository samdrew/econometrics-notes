---
title: "Econometrics Understanding"
author: "Sam Drew"
header-includes:
   - \usepackage{xfrac}
output:
    html_document
---
\newcommand{\L}{\mathcal{L}}
\newcommand{\E}{\mathbb{E}}


# Introduction

`Expectation` is the mean value for the population. This means that for a discrete possibility the $\E(x) = x \cdot P(x)$. For a continuous variable this is $\E(x) = \int x f(x) dx$ where $f(x)$ is the [density function](#pdf).

Other things to note about expectations: 

$$\E(x_1 + x_2) = \E(x_1) + \E(x_2)$$

This can also be shown in matricies with the following

$$\begin{aligned}
\mathbb{E}(X_1 + X_2) &= \mathbb{E}\left( \begin{array} 
X_{11} + X_{12} \\ X_{21} + X_{22} \\ \ldots \\ X_{1N} + X_{2N} \end{array} \right)\\
&= \left( \begin{array} 
\mathbb{E}(X_{11} + X_{12}) \\ \mathbb{E}(X_{21} + X_{22}) \\ \ldots \\ \mathbb{E}(X_{N1} + X_{N2}) \end{array} \right) \\
&= \left( \begin{array} \mathbb{E}(X_{11}) \\  \mathbb{E}(X_{21}) \\ \ldots \\  \mathbb{E}(X_{N1}) \end{array} \right) + \left( \begin{array} \mathbb{E}(X_{12}) \\  \mathbb{E}(X_{22}) \\ \ldots \\  \mathbb{E}(X_{N2}) \end{array} \right) \\
&= \mathbb{E}(X_{1}) + \mathbb{E}(X_{2})
\end{aligned}$$

This also works for linear scaling with a matrix 

$$\mathbb{E}(AX) = A \mathbb{E}(X)$$

`Variance` <a name="var"></a> is the square of the deviation from the expected value. For a discrete possibility this means that $V(x) = \sigma ^2 = \sum (x - \mathbb{E}(x))^2$, or the sum of the squared differences between the outcome and the expected outcome. In a continuous situation this is $V(x) = \int (x - \mathbb{E}(x))^2 f(x) dx$.

The variance of a random vector is a square variance-covariance matrix, which has the variance of each element of a random vector in the main diagonal and the interaction terms in the quadrants. This turns out to be a symmetric matrix, meaning that $A = A'$. 

$$Var(X) = \left( \begin{array} 
Var(X_1) & Cov(X_1, X_2) & \ldots & Covar(X_1, X_N) \\
Cov(X_2, X_1) & Var(X_2) & \ddots & \vdots \\
\ldots & \ldots & \ddots & \vdots \\
Cov(X_N, X_1) & \ldots & \ldots & Var(X_N) \end{array} \right)$$

This can then be written in terms of the expectations where

$$ Var(X) = \mathbb{E}\left( (X - \mathbb{E}(X)) (X - \mathbb{E}(X))' \right) \\
\left( \begin{array} 
(X_1 - \mathbb{E}(X_1))^2 & (X_1 - \mathbb{E}(X_1))(X_2 - \mathbb{E}(X_2)) & \ldots & \vdots \\
\ldots & (X_2 - \mathbb{E}(X_2))^2  & \ddots & \vdots \\
\ldots & \ldots & \ddots & \vdots \\
Cov(X_N, X_1) & \ldots & \ldots & Var(X_N) \end{array} \right)$$

When we then want to find Var(AX)

$$Var(AX) = \mathbb{E}\left( (AX - A\mathbb{E}(X))(AX - A\mathbb{E}(X)') \right) \\
= \mathbb{E}\left( A(X - \mathbb{E}(X))(X - \mathbb{E}(X)')A' \right)  \\
= A \mathbb{E}\left( (X - \mathbb{E}(X))(X - \mathbb{E}(X)') \right) A' \\
= A Var(X) A' $$

Note that for scalar A we get the result

$$ Var(AX) = A^2 Var(X)$$

`Density Function` <a name="pdf"></a> (`f(x)`) is the function which describes the chance of a given variable occuring. In discrete variables this is the weight variable and equivlent to $P(x = x_0)$, while for continuous variables this is just defined as $f(x_0) = \frac{dP(x \leq x_0)}{dx_0}$ where $P(x < x_0)$ is the [cumulative distribution function](#cdf)

`Cumulative Distribution Function` <a name="cdf"></a> is the chance of a given threshold value being greater than or equal to the case. This can be defined in terms of the density function as $$P(x \leq x_0) = \int_{-\infty}^{x_0} f(x)$$ for continuous distributions.

`Joint Distributions` in real-life most probabilities won't be a function of just a single variable. This then gives rise to functions taking multiple variables for the density & distribution functions. The distribution function would then be denoted as $P(y \leq y_0, x \leq x0)$ and the density function as the derivative ($\frac{dP(y \leq y_0, x \leq x_0)}{dx_0 dy_0}$).

`Conditional Distribution/Expectation` When dealing with joint distributions mostly we will be considering one variable at a time, and as such want to look at the probability of one outcome given another (denoted as $P(y | x = x_0)$). This has the density function defined as $f(y|x) = \frac{f(y,x)}{f(x)}$. 

This then has the expectation $$\mathbb{E}(y | x = x_0) = \int y f(y|x_0) dy$$ and variance $$V(y|x) = \int (y - \mathbb{E}(y|x))^2 f(y|x) dy$$

In the case that $\mathbb{E}(Y | X = x)$ isn't for a fixed value of X, then the expectation is then a function with respect to x. (So $\mathbb{E}(Y | X = x) = g_x(Y)$)

`Independence` is an important assumption of variables in many of the models, and has the effect of massively simplifying the maths so that $\mathbb{E}(Y | X) = \mathbb{E}(Y)$. Mostly this is relevant in the case that 

# Sampling

`Random Variables` These are the covariates which effect

`iid`, or identically independent distributed, is a core assumption of many variables and samples (including [Gauss-Markov](#g-m)). 

`Estimators/Models`

`Law of Iterated Expectations` states that for $\mathbb{E}(\mathbb{E}(Y|X)) = \mathbb{E}(Y)$. This is because the notation $\mathbb{E}(Y|X)$ is saying that the expectation of Y given a value of X equals a given value, therefore the expected outcome across all of these possible values of X is going to be Y. 

This can be better understood with an example. Given Y as wage, and X as gender we can calculate the following can be found

$$\begin{aligned}
\mathbb{E}(\text{Wage}) &= \mathbb{E}(\mathbb{E}(\text{Wage}|\text{Sex})) \\
&=\sum P(\text{Sex} = sex) \cdot \mathbb{E}(\text{Wage}|\text{Sex} = sex) \\
&= P(\text{Sex} = f) \cdot \mathbb{E}(\text{Wage}|\text{Sex} = f) + P(\text{Sex} = m) \cdot \mathbb{E}(\text{Wage}|\text{Sex} = m)
\end{aligned}$$

`Total law of Variance` This law, otherwise known as Eve's law, allows the variance of a given value to be found, knowing only its conditional variance and expectations. 

$$Var(Y)=\mathbb{E}[Var(Y|X)]+Var(\mathbb{E}[Y|X])$$

`Law of Large Numbers` means that if you take enough samples, the sample mean will approach the population mean. The below equation states that the sample mean ($\hat \mu)

$$\begin{aligned}
\hat \mu = n^{-1} \sum_{i=1}^n y_i\\
\lim_{n \rightarrow \infty} P (|\hat \mu - \mu| > \varepsilon) = 0 && \forall \varepsilon > 0
\end{aligned}$$

`Central Limit Theorem`

`Continuous Mapping` states that for a given (continuous) function $f(x)$, as the value of $\hat x$ approaches a given value $x$, the function $f(\hat x)$ also approaches a value $f(x)$. This only works if the function $f(x)$ is differentiable. 

`Slutzky Lemma` similar to the continuous mapping function, however it is applied to probability convergence. Given $\hat \mu$, $\hat \tau$ as random variable where 

# Linear Regression Model

`Model` A simple single-variate linear regression model might be described as the following $y_i = \beta _0 + x_i \beta _1 + u_i$ for $i = 1, \ldots , N$

* $y_i$ is the outcome variable
* $\beta _0$ is the constant term
* $\beta _1$ is the slope
* $x_i$ is the covariate (dependent variable)
* $u_i$ is the disturbance or error term. 

Given that this is extremely basic, we'd normally want to extend it to have lots of dependent variables, which in turn means lots of co-efficient terms ($\beta _k$)

`covariates` are your random variables. These are also known as 

`model estimation` the model is built up by using a sample to find the [BLUE](#g-m) by minimizing the sum of squared residuals of the sample. So in the general (matrix form) model $Y = \beta X + u$ we are trying to minimize the value of $u'u$, as $u$ is a matrix. Rearranging the model for $u$, we have $u = Y - \beta X$.

$$\begin{aligned}
\text{SSR} &= u'u = (Y - \hat \beta X)'(Y - \hat \beta X) \\
&= (Y' - \hat \beta ' X')(Y - X \hat \beta) \\
&= Y'Y - Y'X \hat \beta - \hat \beta ' X'Y - \hat \beta ' X' Y + \hat \beta ' X' X \hat \beta \\
\frac{d\text{SSR}}{d\hat \beta} &= - X'Y - X'Y + 2X'X \hat \beta \\
0 &= 2X'X \hat \beta - 2X'Y \\
\hat \beta &= (2X'X)^{-1} 2X'Y \\
&= (X'X)^{-1} X'Y
\end{aligned}$$

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

$$\begin{aligned}
Var(\hat \beta) &= (X' X)^{-1} X' Var(y) X (X' X)^{-1}\\
 &= (X' X)^{-1} X' \sigma^2 \mathbf{I} X (X' X)^{-1}\\
 &= \sigma^2 (X' X)^{-1} X' X (X' X)^{-1}\\
 &= \sigma^2 (X' X)^{-1} 
 \end{aligned}$$

`Gauss-Markov` <a name="g-m"></a>  provides proof that $\hat \beta _{OLS}$ is the Best Unbiased Linear Estimator for a linear regression, under the G-M Conditions (e.g. zero-mean conditional variance)

We can start by looking for a better linear estimator. Any linear estimator will be in the form $$\tilde \beta = \alpha y + u$$ where $\alpha = \left( \begin{array} \alpha _0 \\ \alpha _1 \\ \ldots \\ \alpha _N \end{array} \right)$. This can then be described in terms of the $\hat \beta_{OLS}$ as $\tilde \beta = a$

# Estimation

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

# OLS Properties

`Assumptions` about the data are required for the OLS estimator to be valid. There are a set of assumptions named Gauss-Markov which include

* Homoskedicity
	- Mean-zero error $\mathbb{E}{u} = 0$
	- $u_i$ and $x_i$ are independent
	- $Var(u_i) = \sigma^2 \mathbf{I}$
* iid
	- iid error term $Cov(u_i, u_j) = 0$
* $u$ is independent of $x$
* $X'X$ is a non-singular matrix $det(X'X) \neq 0$

`Sample Properties`

`Asymptotic Variance`

# Hypothesis Testing

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

# Confidence Intervals



# Binary Response Models

These models have the properties that for -∞ the function should return 0 and and the function for +∞ tends towards 1, and the function of 0 returning 1/2. The models generally used are the `logit` (L or $\Lamda$) and `probit` ($\Phi$).

`Notation` Binary response functions tend to be denoted as $1( \cdot )$. In a generic form these would then contain a condition, e.g. $1(\gamma_0(x_i, u_i) \geq 0 )$.

`Single-index model` is a set of conditions which are applied to the binary response model. These are 

* `additive separability` meaning that the disturbance term is not really a function of $\gamma_0$, and can be stated as $\gamma_0(x_i, u_i) = \gamma_0(x_i) - u_i$
* linear function $\gamma_0$ is assumed to be a linear transformation which can be described in the form of a matrix $\beta_0$, meaning that $\gamma_0(x_i, u_i) = x_i'\beta_0 - u_i$
* which allows single outcome binary outcome model to be described as a function where the outcome from $x_i'\beta_0$ is larger than $u_i$. This can be written as $y_i = 1(x_i'\beta_0 \geq u_i)$

## Parametric

`Parametric functions` Logit/Probit

`Logit` The logistic function or logit is defined as $\Lambda(x) = \frac{e^x}{1+e^x}$. This function, while not having quite such good properties as the probit, ie less steep transition and longer tails, is very easy to differentiate. 

$$\begin{aligned}
\Lambda(x) &= \frac{e^x}{1+e^x} \\
&= \frac{1}{1+e^{-1}} \\
&= (1+e^{-1})^{-1} \\
\frac{\delta \Lambda(x)}{\delta x} &= (-1)\cdot(-e^{-x})\cdot(1+e^{-x})^{-2} \\
&= \frac{e^{-x}}{(1+e^{-x})^{2}} \\
&= \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} \\
&= \frac{1}{1+e^{-x}} \cdot \frac{1 + e^{-x} - 1}{1+e^{-x}} \\
&= \frac{1}{1+e^{-x}} \cdot \left( \frac{1 + e^{-x}}{1+e^{-x}} - \frac{1}{1+e^{-x}} \right)\\
&= \Lambda(x) \cdot \left( 1 - \Lambda(x) \right)\\
\end{aligned}$$

Similarly if we differentiate $\Lambda(x \beta)$ we find [^Logit]

$$\begin{aligned}
\frac{\delta \Lambda(x \beta_o)}{\delta x} &= \frac{\beta_o e^{-x \beta_o}}{(1+e^{-x \beta_o})^{2}} \\
&= \Lambda(x \beta_o) \cdot \left[ 1 - \Lambda(x \beta_o) \right] \beta_o\\
\end{aligned}$$

[^Logit]: This is $\Lambda(x'_i \beta_o)[1-\Lambda(x'_i \beta_o)]\beta_{ko}$ in the notes

`Log-likelihood function` $$ln\L(\beta) = \sum_{i=1}^{n} \left( (1-y_i) ln(1-\Lambda(x'_i \beta)) + y_i ln (\Lambda(x'_i \beta)) \right)$$

`Likelihood Estimators` 

`Asymptotic Variance`

`Newton-Raphson` is an iterative, heuristic algorithm for finding solutions to equations. The reason it is relevant is that it is able to provide results for problems which cannot otherwise be analytically solved.

## Semi-Parametric

# Maximum Likelihood Estimator

We are trying to find the smallest possible value for the error term ($\theta$) when using a binary outcome model to estimate an outcome. This term shall be known as $\theta_0$

`Log-likelihood estimator` Starting with the log-likelihood function $$ln \mathcal{L}(\theta) = \sum\limits_{i=1}^n ln(f(z_i,\theta))$$ we need to find the likelihood estimator where $$\theta_n := arg \max_{\theta} ln \mathcal{L}(\theta)$$

# Hetroskedastic Estimators

Sometimes we can't assume that the errors are going to be homoskedastic. In these cases we'll take a look at the properties of the OLS estimator, and look at some of the other available estimators.

$$\begin{aligned}
Y &= X \beta + u && Var(Y|X) = \sigma^2 \Omega \\
\hat \beta_{OLS} &= (X'X)^{-1} X'Y \\
Var(\hat \beta_{OLS}|X) &= (X'X)^{-1} X' Var(Y|X) X' (X'X)^{-1} \\
&= (X'X)^{-1} X' \sigma^2 \Omega X' (X'X)^{-1} \\
&= \sigma^2 (X'X)^{-1} X' \Omega X' (X'X)^{-1}
\end{aligned}$$

It is then possible to prove that the variance of $\hat \beta_{OLS}$ is greater in the case of hetroskedacity than it is in the case of homoskedacity, ie $\sigma^2 (X'X)^{-1} X' \Omega X' (X'X)^{-1} > \sigma^2 (X'X)^{-1}$. Given that G-M is no longer upheld, OLS is no longer BLUE.

## GLS

Generalized least squares methods are ways of finding the linear regression terms in the case of hetroskedacity. They achieve it by transforming the system using a transformation matrix $P$ to remove the hetroskedacity and then using OLS on the new model.

$$\begin{aligned}
PY &= PX\beta + Pu && \text{s.t. } Var(Pu|X) = \sigma^2 \mathbf{I} \\
Var(Pu|X) &= P Var(u|X) P' = \sigma^2 P \Omega P' \\
&= \sigma^2 \mathbf{I} \\
\Rightarrow \mathbf{I} &= P \Omega P' \\
\end{aligned}$$

This gives us a relationship between $P$, $\mathbf{I}$ and $\Omega$. What we really want to find is the value of $P$ in terms of $\Omega$.

$$\begin{aligned}
P^{-1} &= P^{-1} P \Omega P' = \Omega P' \\
P^{-1} P'^{-1} &= \Omega \\
\Rightarrow \Omega &= (P'P)^{-1} \\
&&&\text{Assume matrix P is symmetric} \Rightarrow P = P' \\
\Omega &= (P^2)^{-1} = P^{-2} \\
P &= \Omega^{-1/2} \\
Var(\Omega^{-1/2} u | X) &= \sigma^2 \Omega^{-1/2} \Omega \Omega^{-1/2} \\
&= \sigma^2 \mathbf{I}
\end{aligned}$$

Using this transformation matrix $\Omega^{-1/2}$ we can then look at our original definition of the GLS model and substitute in the new value for $P$. This will give us the GLS model.

$$\begin{aligned}
PY &= PX\beta + Pu \\
\Omega^{-1/2} Y &= \Omega^{-1/2} X \beta + \Omega^{-1/2} u \\
&& z = \Omega^{-1/2} Y \\
&& \omega = \Omega^{-1/2} X \\
&& \varepsilon = \Omega^{-1/2} u\\
z &= \omega \beta + \varepsilon \\
\hat \beta_{OLS} &= (\omega' \omega)^{-1} \omega' z \\
\hat \beta_{GLS} &= (X'P'PX)^{-1}X'P'PY \\
&= (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} Y
\end{aligned}$$

Now we want to know the properties of this model. Let's look at the Variance of this model.

$$\begin{aligned}
Y &= X\beta + u && Var(u | X) = \sigma^2 \Omega \\
\hat \beta_{GLS} &= (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} Y \\
&= (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} (X \beta + u) \\
&= \beta + (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} u \\
\mathbb{E} (\hat \beta_{GLS} | X) &= \beta + (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} \mathbb{E} (u|X) \\
&= \beta \\
Var(\hat \beta_{GLS} | X) &= (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} Var(Y|X) \Omega^{-1} X (X' \Omega^{-1} X)^{-1} \\
&= (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} \sigma^2 \Omega \Omega^{-1} X (X' \Omega^{-1} X)^{-1} \\
&= \sigma^2 (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} X (X' \Omega^{-1} X)^{-1} \\
&= \sigma^2 (X' \Omega^{-1} X)^{-1} 
\end{aligned}$$

Given the law of total variance this we can say $Var(\hat \beta_{GLS}) = \sigma^2 (X' \Omega^{-1} X)^{-1}$. Alternatively the $Var(\hat \beta_{GLS})$ can be calculated by looking at the square error, to find the same result. 

$$\begin{aligned}
\hat \beta_{GLS} &= \beta + (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} u \\
\hat \beta_{GLS} - \beta &= (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} u \\
(\hat \beta_{GLS} - \beta)(\hat \beta_{GLS} - \beta)' &= (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} u ((X' \Omega^{-1} X)^{-1} X' \Omega^{-1} u )' \\
&= (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} u u' \Omega^{-1} X (X' \Omega^{-1} X)^{-1} \\
\mathbb{E}((\hat \beta_{GLS} - \beta)(\hat \beta_{GLS} - \beta)'|X) &= (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} \mathbb{E}(u u'|X) \Omega^{-1} X (X' \Omega^{-1} X)^{-1} \\
&= \sigma^2 (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} \Omega \Omega^{-1} X (X' \Omega^{-1} X)^{-1} \\
&= \sigma^2 (X' \Omega^{-1} X)^{-1} \\
Var(\hat \beta_{GLS}) &= \sigma^2 \mathbb{E}((X' \Omega^{-1} X)^{-1})
\end{aligned}$$

Note: Sometimes GLS, when applied to hetroskedasity is referred to as Weighted Least Squares (WLS).