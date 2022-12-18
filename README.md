# MVA - Optimal Transport project

This repository contains the code for the MVA Optimal Transport course project.
It implements the Osborne algorithm for matrix balancing, an algorithm whose aim is to balance the row and column sums
of a matrix : given a matrix $K$, we want to find a diagonal matrix $D$ such that

$$A = DKD^{-1}$$

has balanced row and column sums :

$$r(A) = \left(\sum_{j=1}^n A_{ij}\right)_{i \in [1, ..., n]}$$

and

$$c(A) = \left(\sum_{i=1}^n A_{ij}\right)_{j \in [1, ..., n]}$$

are equal.

This is an ubiquitous pre-conditioning task for many linear algebra operations such as eigenvalue decomposition.
The project aims to study the behavior of the algorithm variants, its integration into linear algebra routines, and
its connection to Optimal Transport theory. Specifically, the algorithm is very much related to Sinkhorn's algorithm,
used to solve Optimal Transport problems. Therefore, one may extend theoretical results from the Osborne algorithm to
Sinkhorn's algorithm.

The project is based on the following paper : (Near-linear convergence of the Random Osborne algorithm for
Matrix Balancing, Altschuler, Parrilo, 2021)[https://arxiv.org/abs/2004.02837v2]
