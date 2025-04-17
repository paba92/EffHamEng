# Efficient Hamiltonian engineering

>Create plots for the related article: [arXiv:2410.19903](https://arxiv.org/abs/2410.19903)

Efficiently generate robust single-qubit pulses to engineer an arbitrary system Hamiltonian.

## Overview
* *utilities/hardware.py*: Generate coupling coefficients and interactions, modeling certain system Hamiltonians.
* *utilities/reshaping.py*: Efficient and robust Hamiltonian engineering methods
  * Method **genV** generates a matrix V leading to a feasible linear program.
  * Method **LP** solves the linear program and returns the single-qubit pulses and relative evolution times λ to implement the target Hamiltonian.
* *utilities/simulation.py*: Simulates the time evolution of the efficient Hamiltonian engineering methods.
