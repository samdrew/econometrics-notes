---
title: "Microeconomics Prep"
author: "Sam Drew"
output:
    html_document
---
$$
\newcommand{\B}{\mathscr{B}}
\newcommand{\R}{\mathbb{R}}
$$

# Preference Relations

People want things, and rational people are assumed to want the same things each time they are given a choice. This is called a preference relation. We look at this from two directions, preferences and choices. 

## Preferences

We start with a description of some outcome being preferred ($\succsim$) to another outcome. This relationship can also be written the other way around, so $x \succsim y$ and $y \precsim x$ are identical by definition.

This relationship is said to be *rational*, meaning that it is

- Complete: $\forall x,y \in X$, we have that $x \succsim y$ or $y \succsim x$, or both.
- Transitive: $\forall x,y \in X$, if $x \succsim y$ and $y \succsim z$, we have $x \succsim z$.

From this relationship we then define more preference relations. These are 

- $\succ$ defined as $\succsim \text{ and not } \precsim$, and 
- $\sim$ defined as $\succsim \text{ and } \precsim$

## Questions

### Preferences

- (2015 & 2016) Show that if $f: \R \rightarrow \R$ is a strictly increasing function and $u: X \rightarrow \R$ is a utility function representing the preference relation $\succsim$, then the function$v: X \rightarrow \R$ defined by $v(x) = f(u(x))$ is also a utility function representing the preference relation $\succsim$.

> Let $x, y \in X$
> 
> Since $u(\cdot)$ represents $x$, $u(x) \geq u(y) \iff x \succsim y$
> 
> Given $f(\cdot)$ is strictly increasing, $u(x) \geq u(y) \iff v(x) \geq v(y)$
> 
> So $x \succsim y \iff v(x) \geq v(y)$
> 
> Therefore v(\cdot) is a preference relationship $\succsim$.

- (Tutorial 1) Show that if $\succsim$ is rational then, if $x \succ y \succsim z$, then $x \succ z$

> $x \succ y$ implies $x \succsim y$
> 
> Therefore $x \succsim y \succsim z$ and by transitivity $x \succsim z$
> 
> Now, assume $z \succsim x$, then we have $y \succsim z \succsim x$, and by transitivity $y \succsim x$.
> 
> This violates the continuity condition on $x \succ y$, and is therefore false by contradiction.
> 
> As such $x \succsim z$ and not $z \succsim x$, which is $x \succ z$.

- Show that $\succ$ is both irreflexive ($x \succ x$ never holds) and transitive (if $x \succ y$ and $y \succ z$, then $x \succ z$)

> *Irreflexitivity:* Let $x \in X$, by completeness $x \succsim x$
> 
> Therefore there is no $x$ for which $x \succ x$
> 
> *Transitivity:* Suppose that $x \succ y$ and $y \succ z$. This implies $y \succsim z$
> 
> This gives $x \succ y \succsim z$, which we've already shows implies $x \succ z$.

- Show that $\sim$ is reflexive ($x \sim x$ for all $x$), transitive (if $x\sim y$ and $y \sim z$, then $x \sim z$), and symmetric (if $x \sim y$, then $y\sim x$)[^](say something)

> *Reflexivity:* Let $x \in X$, by completeness $x \succsim x$
> 
> Given $x \succsim x$ and $x \precsim x$ always hold, we can say $x \sim x$
> 
> *Transitivity:* Suppose $x \sim y$ and $y \sim z$, we can imply
> 
> 1. $x \succsim y$ and $y \succsim z$, and by transitivity $x \succsim z$
> 2. $y \succsim x$ and $z \succsim y$, and by transitivity $z \succsim x$
> 
> From which we can conclude $x \sim z$.
> 
> *Symmetry:* Suppose $x \sim y$ we can say $x \succsim y$ and $y \succsim x$.
> 
> Therefore $y \succsim x$ and $x \succsim y$, giving us $y \sim x$.

### Choice

- (2015 & Tutorial 1) Consider the choice structure $(\B, C(\cdot))$ with $\B = (\{x, y\}, \{x, y, z\})$ and $C(\{x, y\}) = \{x\}$. Show that if $(\B, C(\cdot))$ satisfies the weak axiom, then we must have $C(\{x, y, z\}) = \{x\},= \{z\}, \text{ or }= \{x, z\}$.

# Convexity and Monotonicity

## Questions

- (Mock & Tutorial) Show the following: If $\succsim$ is strongly monotone, then it is monotone.

> Strongly monotone definition: If $x \geq y$ and $x \neq y$, then $x \succ y$
> 
> Assume $\succsim$ is strongly monotone and $x \gg y$
> 
> In which case, $x \geq y$ and $x \neq y$, so $x \succ y$
> 
> Therefore $\succsim$ is monotone.

- (Tutorial) Show the following: If $\succsim$ is monotone, then it is locally nonsatiated.

> Let $\varepsilon > 0$ and $x \in X$
> 
> $i \rightarrow \R_+^L$
> 
> $y = x + \frac{\varepsilon}{\parallel i \parallel}\cdot i$
> 
> Therefore $\parallel y - x \parallel \leq \varepsilon$ and $y \succsim x$, and preference relationship $\succsim$ is locally nonsatiated.

- (Mas-Colell) Verify that lexicographic ordering is complete, transitive, strongly monotone and strictly convex.

> **Definition**: Preference relationship $x \succsim y$ if $x_1 \geq y_1$ and either $x_1 \neq y_1 \text{ or } x_2 \geq y_2$.
> 
> *Completeness*: Assume $\succsim$ does not exist, and $x,y \in X$ we get
> 
> 1. $y_1 \succ x_1$, or 
> 2. $x_1 = y_1 \text{ and } y_2 > x_2$
> 
> Therefore if $x \succsim y$ does not exist, $y \succ x$ must, which in turn implies $y \succsim x$, prooving completeness by contradiction.
> 
> *Transitivity*: If preference relations $x \succsim y$ and $y \succsim z$ represent lexicograpic ordering, then either
> 
> 1. ($x_1 \geq y_1$ and $x_1 \neq y_1$) and ($y_1 \geq z_1$ and $y_1 \neq z_1$) $\Rightarrow x_1 \geq z_2 \text{ and } x_1 \neq z_1$, or
> 1. ($x_1 \geq y_1$ and $x_1 \neq y_1$) and ($y_1 \geq z_1$ and $y_2 \geq z_2$) $\Rightarrow x_1 \geq z_2 \text{ and } x_1 \neq z_1$, or
> 1. ($x_1 \geq y_1$ and $x_2 \geq y_2$) and ($y_1 \geq z_1$ and $y_1 \neq z_1$) $\Rightarrow x_1 \geq z_2 \text{ and } x_1 \neq z_1$, or
> 1. ($x_1 \geq y_1$ and $x_2 \geq y_2$) and ($y_1 \geq z_1$ and $y_2 \geq z_2$) $\Rightarrow x_1 \geq z_2 \text{ and } x_2 \geq z_2$.
> 
> Therefore $x_1 \geq z_1$ and either $x_1 \neq z_1 \text{ or } x_2 \geq z_2$, and $x \succsim z$.

- Show that if $f(\cdot)$ is a continuous utility function representing $\succsim$, then $\succsim$ is continuous

# Budget Sets

Budget sets are a formal way of describing a set of options.

## Questions 

- (Tutorial) A consumer consumes one consumption good x and hours of leisure h. The price of the consumption good is p, and the consumer can work at a wage rate of s = 1. What is the consumer's Walrasian budget set?

> $s+h \leq 24$ and $px \leq s$, therefore the budget set can be described as
> 
> $$\{(x,h) \in \R_+^2 : h \leq 24 \text{ and } px \leq 24-h \}$$

- (2016) Show formally that the Walrasian budget set $B_{p,w} = \{x \in \R^L_+ : px \leq w \}$ is convex.

# Market Clearing and Competitive Equilibria

## Questions

- (2013 ) Show that, if an allocation $\left(x_1^*, \ldots ,x_I^*, y_1^*, \ldots , y_I^* \right)$ and price vector $p^* \gg 0$ constitute a competitive equilibrium, then for any strictly positive scalar $\alpha > 0$, allocation $\left(x_1^*, \ldots ,x_I^*, y_1^*, \ldots , y_J^* \right)$ and price vector $\alpha p^* \gg 0$ also constitue a competitive equilibrium.

- (2014 & Tutorial) Show that, if an allocation $\left(x_1, \ldots ,x_I, y_1, \ldots , y_J \right)$ and price vector $p \gg 0$ satisfy the parket clearing condition $\displaystyle \sum_{i=1}^I x_{li} = \omega_l + \sum_{j=1}^J y_{lj}$ for all goods $l \neq k$ and if every consumer's budget constraint is satisfied with equality, so that $\displaystyle px_i = p \omega_i + \sum_{j=1}^J \theta_{ij} p y_j$ for all $i$, then the market for good $k$ also clears.

> The budget constraint for all goods, $l$ is 
> $$\begin{aligned}\sum_{l=1}^L p_l x_{il} &= \sum_{l=1}^L p_l \omega_{il} + \sum_{l=1}^L \sum_{j=1}^J \theta_{ij} p_l y_jl \\
> 0 &= \sum_{l=1}^L p_l \left( x_{il} - \omega_{il} - \sum_{j=1}^J \theta_{ij} y_{jl} \right) \\
> \text{Summed across all individuals} \\
> &= \sum_{l=1}^L p_l \left(\sum_{i=1}^I x_{il} - \omega_{l} - \sum_{j=1}^J y_{jl} \right) && \text{since } \sum_{i=1}^I \theta_{ij} = 1_j\\
> \sum_{l \neq k} p_l \left(\sum_{i=1}^I x_{il} - \omega_{l} - \sum_{j=1}^J y_{jl} \right) &= - p_k \left(\sum_{i=1}^I x_{ik} - \omega_{k} - \sum_{j=1}^J y_{jk} \right) \\ 
> \text{Given market clearing for } l \neq k \\
> 0 &= - p_k \left(\sum_{i=1}^I x_{ik} - \omega_{k} - \sum_{j=1}^J y_{jk} \right) \\ 
> \text{Implies market clears, since } p > 0 \end{aligned}$$

# Game Theory - Signalling

## Questions 

Consider the signalling model of Spence analysed in the lecture with the following
specifications: The productivity of a worker with education level e is 2+(e/4) if he is of
a high type while it is 1 regardless of e if he is of a low type. The worker’s cost of
obtaining e is e/2 if he is of a high type while it is e if he is of a low type. The worker's outside option value is 1 for both types.

1. Describe the equilibrium when the worker's type is publicly known.

    > In equilibrium when the worker's type is known, each worker's wage will be equal to their productivity, so $w_H = 2 \text{ and } w_L = 1$. This is because in 
    > 
    > - If they were paid less than that, someone else would hire they away at a greater wage, and take the additional profit.
    > - This is $\geq$ than the reserver prices for their outside options.
    >
    > Additionally neither will undertake education, as the marginal increase in productivity ($\frac{e}{4}$ and $0$) is less than the cost ($\frac{e}{2}$ and $e$)

	Answer the following questions assuming that the worker's type, which is high with probability $\lambda < 1/2$, is private information.

1. Describe the equilibrium when the education opportunity is absent.
	
	> In the case of no education opportunity, there will be no separation. Wages will be a pooled at the mean productivity for a worker. This is given by $$w_p = \lambda p_H + (1 - \lambda) p_L = 1 + \lambda$$. Again
    > 
    > - If they were paid less than that, someone else would hire they away at a greater wage, and take the additional profit.
    > - This is $\geq$ than the reserver prices for their outside options.

1. Describe separating equilibria when the education opportunity is present.
	
	> Separating equilibria occurs when the cost of education for a low-type worker is higher than the increase in wages that they achieve. This can be described by the equation $$w_H \leq w_L + e_L \cdot h$$ where h is the amount of education required to be paid at $w_H$. 
	> 
	> The separating wages will again be equal to the productivities of the individuals taking part. As such $w_L = 1$ and in this case $w_H = 2 + \frac{h}{4}$. This means that the amount of education required can be defined as $$2 + \frac{h}{4} = 1 + h \\ h = \frac{4}{3} (2 - 1) \\ h = \frac{4}{3}$$

1. Describe pooling equilibria when the education opportunity is present.
1. Find the separating equilibrium that satisfies the Intuitive Criterion.
1. Which type(s) of worker benefits from the education opportunity? Does the benefit stem from the signalling role or the productivity-enhancing role of education? Justify your answers