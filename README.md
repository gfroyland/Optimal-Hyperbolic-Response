# Optimal-Hyperbolic-Response

<b>Code to optimise linear response functionals for hyperbolic dynamical systems</b> 

First, run `optimal_response_efficient` for suitably chosen value of $n$, for example 

`a1, a2, Ṫ, Ṫcoarse, ffine = optimal_response_efficient(16)`

This will store the arrays of Fourier coefficients for $\dot{T}$ in the arrays `a1`, `a2`, the vector field `Ṫ` (a function which takes a 2-vector as input and outputs a 2-vector), `Ṫcoarse` ($\dot{T}$ evaluated on an $n\times n$ grid), and `ffine` (the SRB measure estimate evaluated on a $4n\times 4n$ grid).
Plots of the estimates of the optimal vector field $\dot{T}$ and the SRB measure of $T$ are also displayed and saved.

One may then run 

`oldexpectation, newexpectation, ffine0, ffine1 = response_compare(16, Ṫ, ffine)`

which will add a small increment of $\dot{T}$ to the map $T$ and compute and plot the resulting perturbed SRB measure. The value of the integral of the observation $c$ with respect to the unperturbed and perturbed SRB measures are also reported and stored as the numbers `oldexpectation` and `newexpectation`, respectively. The unperturbed and perturbed SRB measures are evaluated on $4n\times 4n$ grids and stored in the arrays `ffine0` and `ffine1`, respectively.
