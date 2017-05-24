---
title: "Behaviour Economics Notes"
author: "Sam Drew"
output: html_document
---

# Cognative Biases

## Availability

Availability of information about a subject acts as a proxy for prevalence. i.e. if you hear a lot about a subject then you assume it's common-place. This is why media has a lot of influence.

## Representativeness (/Stereotyping)



## Overconfidence

## Anchoring

Given a hint of an answer, e.g. as part of a question, without strong priors people will base their judgement on this information. This is why carefully worded survey questions are vitally important.

## Adjustment

# Efficient Market Hypothesis

Evidence for EMH from Fama, random walk behaviour (1965), event study: adjust rapidly to new information (1969).

## Weak-form Efficiency

Random-walk -> past performance/events will not effect future prices. This rejects the possibility of momentum pricing.

Prices not always accurate, meaning that fundamental analysis provides an opportunity for profit.

## Semi-strong form Efficiency

Stocks are fundamentally valued correctly according to all past and present historical data, however they do not include private data in the estimations, so that it is theoretically possible to make a profit.

## Strong-form Efficiency

All data, past and present, public and private is taken into account in the pricing and it is always correct. There is never any profit to be made.

## Inefficiencies

- Weak-form: momentum portfolios, longer term reversal
- Semi-strong: 
	+ Small outperform large, particularly in January
	+ High Book-to-Market outperform low
	+ Inclusion in major indicies bumps value

# Ambiguity Aversion

Mean-variance portfolio selection

$$ \max_w \sum_{i=1}^N w_i \mathbb{E}(r_i) - \gamma \sum_{i=1}^N \sum_{j=1}^N Cov(w_i r_i, w_j r_j) $$

where $\gamma > 0$ and represents risk aversion, and s.t. $\sum_{i=1}^N w_i = 1$

## Maximin framework

Adding ambiguity to the mean-variance framework can be achieved by defining an ambiguous mean return of  $$ \mu \in \{\underline{\mu}, \overline{\mu} \} $$ where $\underline{\mu} = \mu - \delta$ and $\overline{\mu} = \mu + \delta$, and $\sigma_\mu \equiv \sigma_\underline{\mu} \equiv \sigma_\overline{\mu}$.

The maximin framework aims to maximize the worst-case outcomes, so assumes $$\mu = \begin{cases} \mu - \delta & w > 0 \\ \mu + \delta & w < 0 \end{cases}$$

Substituting this into the original mean-variance framwork gives us 

$$ \max_w \sum_{i=1}^N w_i \mu_i - \sum_{i=1}^N \lvert w_i \delta_i \rvert - \gamma \sum_{i=1}^N \sum_{j=1}^N Cov(w_i r_i, w_j r_j) $$

Solving this for a two-item portfolio, with one risk-free asset with known return r, and one risky asset of uncertain return and variance $\sigma^2$, we get 

$$ \max_w (1-w)r +  w\mu - \lvert w \delta \rvert - \gamma (w \sigma)^2 $$

$$\begin{aligned}
\frac{dU}{dw} &= \mu - r - \lvert \delta \rvert - 2 \gamma w \sigma^2 \\
0 &= \mu - r - \lvert \delta \rvert - 2 \gamma w \sigma^2 \\
2 \gamma w \sigma^2 &= \mu - r - \lvert \delta \rvert \\
w &=
\begin{cases}
\frac{\mu - r - \delta }{2 \gamma \sigma^2} && w > 0\\
\frac{\mu - r + \delta }{2 \gamma \sigma^2} && w < 0\\
0 && \text{Otherwise}
\end{cases}
\end{aligned}$$

From this we can get $w > 0$, where

$$0 < \frac{\mu - r - \delta }{2 \gamma \sigma^2}\\
0 < \mu - r - \delta \\
r < \mu - \delta$$

And similarly $w < 0$, where

$$0 > \frac{\mu - r + \delta }{2 \gamma \sigma^2}\\
0 > \mu - r + \delta \\
r > \mu + \delta$$

leaving us with $w = 0 \quad \forall \  \mu - \delta < r < \mu + \delta$ 

## Subadditive probabilities

Another way of modelling ambiguity aversion is through the use of subadditive probabilities, such that $Pr(A) + Pr(B) < Pr(A \text{ or } B)$, or more generally $\int_{-\infty}^{\infty} f(x) \ dx < 1$ where $f(x)$ is the probability density function.

For a binary outcome $Y \in {Y_H, Y_L}$ with probabilities $\pi_H$ and $\pi_L$ respectively, we therefore have $\pi_H + \pi_L < 1$. Given a basic investor's problem we start with the framework, purchase if $$p < \pi_H Y_H + \pi_L Y_L$$

Given that $\pi_H + \pi_L < 1$ we have $\pi_H + \pi_L + \delta = 1$, with $\delta > 0$

$$\begin{aligned} 
p_b &< \pi_H Y_H + \pi_L Y_L \\
p_b &< (1 - \pi_L - \delta) Y_H + \pi_L Y_L \\
p_b &< Y_H (1 - \delta) + \pi_L(Y_L - Y_H) \\
p_b + \delta Y_H &< Y_H + \pi_L(Y_L - Y_H) \\
p_b < p + \delta_H Y_H &< Y_H + \pi_L(Y_L - Y_H) 
\end{aligned}$$

or to match the notes

$$\begin{aligned} 
p_b &< \pi_H Y_H + \pi_L Y_L \\
p_b &< \pi_H Y_H + (1 - \pi_H - \delta) Y_L \\
p_b &< \pi_H (Y_H - Y_L) + Y_L(1 - \delta) \\
p_b < p_b + \delta Y_L &< Y_L + \pi_H (Y_H - Y_L)
\end{aligned}$$

The second form is better because $Y_L + \pi_H (Y_H - Y_L) < Y_H + \pi_L(Y_L - Y_H)$, so less information will be lost in the final step.

Similarly for selling we have 

$$\begin{aligned} 
p_s &> \pi_H Y_H + \pi_L Y_L \\
p_s &> (1 - \pi_L - \delta) Y_H + \pi_L Y_L \\
p_s &> Y_H (1 - \delta) + \pi_L (Y_L - Y_H) \\
p_s \ngtr p_s + \delta Y_H &> Y_H - \pi_L (Y_H - Y_L) \\
\end{aligned}$$

# Prospect Theory

## Equity Premium Puzzle

According to most models equity returns are above rates for others.

Benartzi Thaler (1995) justify it with

- Loss Aversion - the loss regions are greater in magnitude than the profit
- Narrow Framing ("Mental Accounting") - looking at only the individual risks not the composites


- Skewness Premium
	+ Right-skewed stocks are lower returns than more symmetrical
- Disposition Effect
- Insurance Premia and Deductibles

# Corporate Finance

M&M