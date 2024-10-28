# Efficient Hamiltonian engineering

>Create plots for the related article: [arXiv:9999.99999](https://arxiv.org/abs/9999.99999)

Efficiently generate single-qubit pulses to engineer an arbitrary system Hamiltonian.

## Overview
* *utilities/hardware.py*: Generate coupling coefficients and interactions, modeling certain system Hamiltonians.
* *utilities/reshaping.py*: Efficient Hamiltonian engineering methods
  * Method **genV** generates a matrix $V$ leading to a feasible linear program.
  * Method **LP** solves the linear program and returns the single-qubit pulses and relative evolution times $\lambda$ to implement the target Hamiltonian.
* *utilities/simulation.py*: Simulates the time evolution of the efficient Hamiltonian engineering methods.
