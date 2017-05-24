---
title: "Behaviour Economics Question"
author: "Sam Drew"
output: pdf_document
---
## Non-additive probabilities

Another way of modelling ambiguity aversion is through the use of subadditive probabilities, such that $Pr(A) + Pr(B) < Pr(A \text{ or } B)$, or more generally $\int_{-\infty}^{\infty} f(x) \ dx < 1$ where $f(x)$ is the probability density function.

For a binary outcome $Y \in {Y_H, Y_L}$ with probabilities $\pi_H$ and $\pi_L$ respectively, we therefore have $\pi_H + \pi_L < 1$. Given a basic investor's problem we start with the framework, purchase if $$p < \pi_H Y_H + \pi_L Y_L$$

Given that $\pi_H + \pi_L < 1$ we have $\pi_H + \pi_L + \delta = 1$, with $\delta > 0$, looking at the worst-case for buy (when $\delta$ is on the Low) we have 

$$\begin{aligned} 
p_b &< \pi_H Y_H + (\pi_L + \delta) Y_L \\
p_b &< \pi_H Y_H + (1 - \pi_H) Y_L \\
p_b &< Y_L + \pi_H (Y_H - Y_L)
\end{aligned}$$

Similarly for selling we have the worst-case when $\delta$ is on high, so

$$\begin{aligned} 
p_s &> (\pi_H + \delta) Y_H + \pi_L Y_L \\
p_s &> (1 - \pi_L) Y_H + \pi_L Y_L \\
p_s &> Y_H + \pi_L (Y_L - Y_H) \\
p_s &> Y_H - \pi_L (Y_H - Y_L) \\
\end{aligned}$$