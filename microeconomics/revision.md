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
- (Tutorial 1) Show that if $\succsim$ is rational then, if $x \succ y \succsim z$, then $x \succ z$
- Show that $\succ$ is both irreflexive ($x \succ x$ never holds) and transitive (if $x \succ y$ and $y \succ z$, then $x \succ z$)
- Show that $\sim$ is reflexive ($x \sim x$ for all $x$), transitive (if $x\sim y$ and $y \sim z$, then $x \sim z$), and symmetric (if $x \sim y$, then $y\sim x$)

### Choice

- (2015 & Tutorial 1) Consider the choice structure $(\B, C(\cdot))$ with $\B = (\{x, y\}, \{x, y, z\})$ and $C(\{x, y\}) = \{x\}$. Show that if $(\B, C(\cdot))$ satisfies the weak axiom, then we must have $C(\{x, y, z\}) = \{x\},= \{z\}, \text{ or }= \{x, z\}$.

# Convexity and Monotonicity

## Questions

- (Mock) Show the following: If $\succsim$ is strongly monotone, then it is monotone.
- (Tutorial) Show the following: If $\succsim$ is monotone, then it is locally nonsatiated.
- (Mas-Colell) Verify that lexicographic ordering is complete, transitive, strongly monotone and strictly convex.
- Show that if $f(\cdot)$ is a continuous utility function representing $\succsim$, then $\succsim$ is continuous

# Budget Sets

Budget sets are a formal way of describing a set of options.

## Questions 

- (Tutorial) A consumer consumes one consumption good x and hours of leisure h. The price of the consumption good is p, and the consumer can work at a wage rate of s = 1. What is the consumer's Walrasian budget set?
- (2016) Show formally that the Walrasian budget set $B_{p,w} = \{x \in \R^L_+ : px \leq w \} is convex.
