#!/usr/bin/env python
""" Efficient Hamiltonian engineering

@Author: Pascal Baßler
"""
import numpy as np
import itertools
import random
import pickle
from typing import Union
from joblib import Parallel, delayed

import cvxpy as cp
import mosek

#################
### Gate sets ###
#################
# The effect (sign-flip and permutation) of conjugation with elements from the gate-sets are encoded in look-up-tables
# in __init__, see i.e. self._LUTsqrt. The python objects representing gates are defined in simulation.py.

# List of labels for single-qubit operations
## Pauli matrices
# 0: Id
# 1: X
# 2: Y
# 3: Z

## single-qubit Clifford gate sets
# sqrt:
# 0: Id
# 1: X
# 2: Y
# 3: Z
# 4: sqrt(X)
# 5: sqrt(Y)
# 6: sqrt(Z)
# 7: sqrt(X)^(-1)
# 8: sqrt(Y)^(-1)
# 9: sqrt(Z)^(-1)
#
# sqrtsqrt:
# 0: Id
# 1: X
# 2: Y
# 3: Z
# 4: sqrt(X)*sqrt(Y)
# 5: sqrt(X)*sqrt(Y)^(-1)
# 6: sqrt(X)^(-1)*sqrt(Y)
# 7: sqrt(X)^(-1)*sqrt(Y)^(-1)
# 8: sqrt(Y)*sqrt(X)
# 9: sqrt(Y)*sqrt(X)^(-1)
# 10: sqrt(Y)^(-1)*sqrt(X)
# 11: sqrt(Y)^(-1)*sqrt(X)^(-1)

class Reshaping:
    def __init__(self, n: int = None, J: np.ndarray = None, pauli_string_list: np.ndarray = None,
                 dimension_factor: float = 2.5, nr_samples: int = 1, gate_set: str = 'sqrt', nr_jobs: int = 0,
                 generate_V: bool = True, tol: float = 1e-9, filename: str = None, solve_optimal: bool = False):
        """
        Constructor for a Reshaping object.

        :param n: nr. of qubits
        :param J: list/array of Pauli basis coefficients
        :param pauli_string_list: 2D list/array with shape (d,n) for d Pauli strings on n qubits of the quantum object.
        :param dimension_factor: for the relaxation: r = dimension_factor * d
        :param nr_samples: for the heuristic: nr. of samples of possibly feasible V
        :param gate_set: What gate-set to use: 'pauli': Paulis
                                               'sqrt': Paulis and sqrt(Paulis),
                                               'sqrtsqrt': Paulis and sqrt(Paulis)*sqrt(Paulis)
        :param nr_jobs: nr. of parallel threads
        :param generate_V: automatically generate V at the end of __init___
        :param tol: tolerance for testing a float value against zero
        :param filename: load a Reshaping object from a file
        :param solve_optimal: If True: Uses the inefficient optimal linear program.
        """
        if not filename is None:
            with open(filename, 'rb') as inp:  # Load data from a file
                obj = pickle.load(inp)
                n = obj.n
                J = obj.J
                pauli_string_list = obj.pauli_string_list
                dimension_factor = obj.dimension_factor
                nr_samples = obj.nr_samples
                gate_set = obj.gate_set
                nr_jobs = obj.nr_jobs
                tol = obj.tol
                solve_optimal = obj.solve_optimal
                self._J_permutation = obj.J_permutation
                self._J_sign = obj.J_sign
                self._V = obj.V
                self._operator_strings = obj.operator_strings
        else:
            self._V = None
            self._operator_strings = None
            # Required for the fast update of J
            self._J_permutation = None
            self._J_sign = None

        # Required for the fast quad2int method
        self._quad_digits = ['0', '1', '2', '3']
        # These are the look-up tables for the conjugation of sqrt(pauli)
        self._LUTsqrt = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Permutation look-up table
                                  [1, 1, 1, 1, 1, 3, 2, 1, 3, 2],
                                  [2, 2, 2, 2, 3, 2, 1, 3, 2, 1],
                                  [3, 3, 3, 3, 2, 1, 3, 2, 1, 3]])
        self._sgnLUTsqrt = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sign-flip look-up table
                                     [0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
                                     [0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                                     [0, 1, 1, 0, 0, 1, 0, 1, 0, 0]])  # (-1)**(sgnLUT)
        # These are look-up tables for the conjugation of sqrt(pauli)*sqrt(pauli)
        self._LUTsqrtsqrt = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Permutation look-up table
                                      [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2],
                                      [2, 2, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3],
                                      [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]])
        self._sgnLUTsqrtsqrt = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sign-flip look-up table
                                         [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                                         [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                                         [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]])
        self._n = int(n)
        self._J = J
        self._pauli_string_list = np.array(pauli_string_list)
        if self._pauli_string_list is None:
            self._J_idx = None
        else:
            # Convert list/array of Pauli strings to array of integer representing each Pauli string
            self._J_idx = self.quad2int_fast(self._pauli_string_list)
        self._nr_samples = nr_samples
        self._nr_jobs = nr_jobs
        self._tol = tol
        # Copy the look-up tables corresponding to the desired gate-set
        if gate_set=='sqrt':
            self._is_sqrt = 1
            self._LUT = np.copy(self._LUTsqrt)
            self._sgnLUT = np.copy(self._sgnLUTsqrt)
        elif gate_set=='sqrtsqrt':
            self._is_sqrt = 0
            self._LUT = np.copy(self._LUTsqrtsqrt)
            self._sgnLUT = np.copy(self._sgnLUTsqrtsqrt)
        elif gate_set=='pauli':
            self._is_sqrt = -1
            self._LUT = np.copy(self._LUTsqrtsqrt)
            self._sgnLUT = np.copy(self._sgnLUTsqrtsqrt)
        else:
            assert False, "reshaping.py, __init__(): ERROR. Invalid gate-set."

        self._p_list = None  # internal usage
        self._solve_optimal = solve_optimal
        if dimension_factor>=1:
            self.dimension_factor = dimension_factor
        else:
            print("reshaping.py, __init__(): WARNING. Parameter dimension_factor set too small. Setting it to 2.")
            self.dimension_factor = 2
        if not (self._J is None or self._pauli_string_list is None):
            if self._V is None:
                if generate_V:
                    self.genV()

    ##############################
    ## Getter and setter functions
    ##############################
    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, value: int):
        self._n = int(value)
        self._J_idx = None
        self._J = None
        self._J_permutation = None
        self._J_sign = None
        self._V = None
        self._operator_strings = None

    @property
    def J(self) -> np.ndarray:
        return self._J

    @J.setter
    def J(self, value: np.ndarray):
        if len(value)!=len(self._J_idx):
            self._J_idx = None
            self._J_permutation = None
            self._J_sign = None
            self._V = None
            self._operator_strings = None
            self._J = value
            return
        if np.linalg.norm(np.nonzero(value)[0] - np.nonzero(self._J)[0])<self._tol:
            self.update_J_values(value)
        else:
            self._J_permutation = None
            self._J_sign = None
            self._V = None
            self._operator_strings = None
            self._J = value
        return

    @property
    def pauli_string_list(self) -> np.ndarray:
        return self._pauli_string_list

    @pauli_string_list.setter
    def pauli_string_list(self, value: np.ndarray):
        self._pauli_string_list = np.array(value)
        if self._pauli_string_list is None:
            self._J_idx = None
        else:
            self._J_idx = self.quad2int_fast(self._pauli_string_list)
        self._J = None
        self._J_permutation = None
        self._J_sign = None
        self._V = None
        self._operator_strings = None

    @property
    def V(self) -> np.ndarray:
        if self._V is None:
            assert not self._J_idx is None, "reshaping.py, V(): ERROR. Jidx is not defined."
            assert not self._J is None, "reshaping.py, V(): ERROR. J is not defined."
            self.genV()
        return self._V

    @property
    def gate_set(self) -> str:
        if self._is_sqrt==1:
            return "sqrt"
        elif self._is_sqrt==0:
            return "sqrtsqrt"
        elif self._is_sqrt==-1:
            return "pauli"

    @gate_set.setter
    def gate_set(self, value: str):
        if value=='sqrt':
            self._is_sqrt = 1
            self._LUT = np.copy(self._LUTsqrt)
            self._sgnLUT = np.copy(self._sgnLUTsqrt)
        elif value=='sqrtsqrt':
            self._is_sqrt = 0
            self._LUT = np.copy(self._LUTsqrtsqrt)
            self._sgnLUT = np.copy(self._sgnLUTsqrtsqrt)
        elif value=='pauli':
            self._is_sqrt = -1
            self._LUT = np.copy(self._LUTsqrtsqrt)
            self._sgnLUT = np.copy(self._sgnLUTsqrtsqrt)
        else:
            assert False, "reshaping.py, gate_set(): ERROR. Invalid gate-set."
        self._J_permutation = None
        self._J_sign = None
        self._V = None
        self._operator_strings = None

    @property
    def operator_strings(self) -> np.ndarray:
        if self._operator_strings is None:
            assert not self._J_idx is None, "reshaping.py, operator_strings(): ERROR. Jidx is not defined."
            assert not self._J is None, "reshaping.py, operator_strings(): ERROR. J is not defined."
            self.genV()
        return self._operator_strings

    @property
    def nr_samples(self) -> int:
        return self._nr_samples

    @nr_samples.setter
    def nr_samples(self, value: int):
        self._nr_samples = value

    @property
    def nr_jobs(self) -> int:
        return self._nr_jobs

    @nr_jobs.setter
    def nr_jobs(self, value: int):
        self._nr_jobs = value

    @property
    def tol(self) -> float:
        return self._tol

    @tol.setter
    def tol(self, value: float):
        self._tol = value

    @property
    def dimension_factor(self) -> float:
        return self._dimension_factor

    @dimension_factor.setter
    def dimension_factor(self, value: float):
        self._dimension_factor = value

    @property
    def J_permutation(self) -> np.ndarray:
        return self._J_permutation

    @property
    def J_sign(self) -> np.ndarray:
        return self._J_sign

    @property
    def operator_strings(self) -> np.ndarray:
        return self._operator_strings

    ##################
    ## Utility methods
    ##################
    def save2file(self, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def update_J_values(self, J: np.ndarray, pauli_string_list: np.ndarray = None):
        """
        Fast method to update the J values and the corresponding V matrix.

        :param J: list/array of Pauli basis coefficients
        :param pauli_string_list: 2D list/array with shape (d,n) for d Pauli strings on n qubits of the quantum object.
        """
        self._J = J
        if self._V is None:
            if pauli_string_list is None:
                assert not self._pauli_string_list is None, "reshaping.py, update_J_values(): ERROR. Pauli strings are required."
            else:
                self._pauli_string_list = np.array(pauli_string_list)
            # Convert list/array of Pauli strings to array of integer representing each Pauli string
            self._J_idx = self.quad2int_fast(self._pauli_string_list)

            self.genV()
        else:
            if not pauli_string_list is None:
                print("reshaping.py, update_J_values(): WARNING. pauli_string_list will be ignored.")
        self._V = np.zeros(self._J_sign.shape)
        for idx in range(self._J_sign.shape[1]):
            self._V[:, idx] = self._J_sign[:, idx] * self._J[self._J_permutation[:, idx]]

    def int2base(self, k_list: np.ndarray, b: int, n: int = None) -> np.ndarray:
        """
        Converts a list/array of integers into an array of digits representing the integers in a given base.

        :param k_list: list/array of integers, each element represents a number in base b with n digits
        :param b: the base of the number system
        :param n: nr. of digits
        :return: array of the digits of the integers in base b
        """
        if n is None:
            n = self._n
        ret = []
        for k in k_list:
            if k == 0:
                ret.append([0] * n)
            else:
                digits = []
                while k:
                    digits.append(int(k % b))
                    k //= b
                res = digits[::-1]
                ret.append([0] * (n - len(res)) + res)
        return np.array(ret)

    def base2int(self, k_list: np.ndarray, b: int, n: int=None) -> np.ndarray:
        """
        Converts a list/array of digits representing numbers in base b into an array of integers.

        :param k_list: list/array of digits representing numbers in base b with n digits
        :param b: the base of the number system
        :param n: nr. of digits
        :return: array of integers
        """
        if n is None:
            n = self._n
        ret = []
        for k in k_list:
            ret.append(sum([k[i] * b ** (n - 1 - i) for i in range(n)]))
        return np.array(ret)

    def quad2int_fast(self, k_list: np.ndarray) -> np.ndarray:
        """
        Fast method to convert a list/array of digits in base 4 to an array of integers

        :param k_list: list/array of digits representing numbers in base 4
        :return: array of integers
        """
        ret = [0] * len(k_list) #np.zeros(len(k_list))
        for idx, x in enumerate(k_list):
            ret[idx] = int("".join([self._quad_digits[y] for y in x]), 4)
        return np.array(ret)

    def is_V_feasible(self, V: np.ndarray) -> bool:
        """
        Checks if V leads to a feasible linear program for all M
            min ||lam||_1
            s.t. V@lam = M

        :param V: 2D array representing the conjugation of the single-qubit gates with the quantum object
        :return: True: V is feasible, False: V is not feasible
        """
        if np.linalg.matrix_rank(V) < V.shape[0]:  # Full row rank?
            return False
        # Construct the linear program to check if the origin is contained in conv(col(V))
        x = cp.Variable(V.shape[1])
        objective = cp.Minimize(cp.norm(x, 1))  # L1 objective function
        constraints = [V @ x == np.zeros(V.shape[0]),
                       x >= np.ones(V.shape[1])]  # linear equation system
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except:
            return False
        if x.value is None:
            return False
        else:
            return True

    ###############
    ## Main methods
    ###############

    ## Pauli conjugation
    ####################
    def parallelCalcVnonZeros(self, s_list: np.ndarray) -> np.ndarray:
        """
        Calculates the conjugation of Pauli Strings with one Pauli string (for parallel execution)

        :param s_list: Pauli string
        :return: Conjugation of Pauli Strings with one Pauli string
        """
        return (-1) ** (self._p_list @ s_list.T)  # From the Pauli commutation relation

    def genV_pauli_exact(self):
        """
        Calculates V, representing the conjugation of d Pauli strings from the quantum object with all combinations of
        Pauli gates.
        V has 4^n columns for the gate set 'pauli'.
        """
        if self._p_list is None:
            self._p_list = self.int2base(self._J_idx, 2, n=2 * self._n)  # Convert the integer labels of J to binary labels
        s_list = np.array(list(itertools.product([0, 1], repeat=2 * self._n)))  # Generate all 4^n possible Pauli string labels
        if not self._nr_jobs:
            V = (-1) ** (self._p_list @ s_list.T)  # Calculate the (partial) Walsh-Hadamard matrix
        else:
            split_s_list = np.array_split(s_list, self._nr_jobs, axis=0)
            V = np.hstack(Parallel(n_jobs=self._nr_jobs)(
                delayed(self.parallelCalcVnonZeros)(s_list) for s_list in split_s_list))

        s_list = self.int2base(self.base2int(s_list, 2, n=2 * self._n), 4)
        s_list[s_list == 1] = -1
        s_list[s_list == 2] = 1
        s_list[s_list == -1] = 2

        self._operator_strings = s_list  # Save the Pauli pulses
        # Save data for fast J update
        self._J_sign = V
        self._V = np.multiply(V.T, self._J).T
        self._J_permutation = np.array([np.arange(len(self._J_idx))] * V.shape[1]).T.astype(int)
        return

    def sampleV_pauli(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs one V matrix, representing the conjugation of d Pauli strings from the quantum object with
        k*d randomly sampled Pauli strings.
        Without feasibility check!

        :param k: factor for the nr. of Pauli string samples (k*d sampled Pauli strings)
        :return: V: d x k*d matrix,
                 s_list: contains the sampled Pauli strings
        """
        if self._p_list is None:
            self._p_list = self.int2base(self._J_idx, 2, n=2 * self._n)  # Convert the integer labels of J to binary labels
        dim = 4**self._n
        # Choose nr_of_s many random samples from all Pauli strings
        nr_of_s = int(len(self._J_idx) * k)
        if nr_of_s > dim:
            s_list = np.arange(dim)
        elif dim < 2 ** 63:
            s_list = random.sample(range(dim), nr_of_s)
        else:
            s_list = [random.randint(1, dim) for ii in range(nr_of_s)]
        s_list = self.int2base(s_list, 2, n=2 * self._n)  # Convert the integer labels of J to binary labels
        if not self._nr_jobs:
            V = (-1) ** (self._p_list @ s_list.T)  # Calculate the partial Walsh-Hadamard matrix
        else:
            split_s_list = np.array_split(s_list, self._nr_jobs, axis=0)
            V = np.hstack(Parallel(n_jobs=self._nr_jobs)(
                delayed(self.parallelCalcVnonZeros)(s_list) for s_list in split_s_list))
        return V, s_list

    def genV_pauli(self):
        """
        Samples a matrix V, representing the conjugation of d Pauli strings from the quantum object with
        k*d randomly sampled Pauli strings.
        Then checks if V is feasible. Increase k if V is infeasible.

        """
        for k in np.linspace(self.dimension_factor, self.dimension_factor+5, 30):
            for smpl in range(self._nr_samples):
                V, s_list = self.sampleV_pauli(k)  # Generate the constraint matrix V and the Pauli strings
                if self.is_V_feasible(V):
                    V, unique_idx = np.unique(V, axis=1, return_index=True)  # Remove redundant columns
                    s_list = self.int2base(self.base2int(s_list[unique_idx, :], 2, n=2*self._n), 4)
                    s_list[s_list == 1] = -1
                    s_list[s_list == 2] = 1
                    s_list[s_list == -1] = 2

                    self._operator_strings = s_list  # Save the Pauli pulses
                    # Save data for fast J update
                    self._J_sign = V
                    self._V = np.multiply(V.T, self._J).T
                    self._J_permutation = np.array([np.arange(len(self._J_idx))] * V.shape[1]).T.astype(int)
                    return
        assert False, "reshaping.py, genV_pauli(): ERROR. No feasible V was found. Increase dimension_factor."


    ## Clifford conjugation
    #######################
    def genV_clifford_exact(self):
        """
        Calculates V, representing the conjugation of d Pauli strings from the quantum object with all combinations of
        single-qubit Clifford gates (either 'sqrt' or 'sqrtsqrt').
        V has 10^n or 12^n columns for the gate sets 'sqrt' and 'sqrtsqrt', respectively.
        """
        # Generate all self._LUT.shape[1]^n (12^n or 10^n) possible Clifford string labels
        all_s_list = itertools.product(range(self._LUT.shape[1]), repeat=self._n)
        if not self._nr_jobs:
            len_s_list = self._LUT.shape[1] ** self._n
            # Initialze required matrices
            V = np.zeros((len(self._J_idx), len_s_list))  # Matrix for LP constraints
            J_permutation = np.zeros((len(self._J), len_s_list)).astype(int)  # For fast J update
            J_sign = np.zeros((len(self._J), len_s_list))  # For fast J update
            for idx, oper in enumerate(all_s_list):
                V_idx, sign_idx, permutation_idx = self.parallelCalcV(oper)
                J_permutation[:, idx]  = permutation_idx
                J_sign[:, idx] = sign_idx
                V[:, idx] = V_idx
        else:
            data = np.array(
                Parallel(n_jobs=self._nr_jobs)(delayed(self.parallelCalcV)(oper) for oper in all_s_list)).T
            V = data[:, 0, :]
            J_sign = data[:, 1, :]
            J_permutation = data[:, 2, :].astype(int)
        V, unique_idx = np.unique(V, axis=1, return_index=True)  # Remove redundant columns
        self._V = V
        # Save the single-qubit Clifford pulses
        self._operator_strings = np.array(list(itertools.product(range(self._LUT.shape[1]), repeat=self._n)))[unique_idx, :]
        # Save data for fast J update
        self._J_permutation = J_permutation
        self._J_sign = J_sign
        return

    def parallelCalcV(self, oper, save_sign_permutation: bool = True):
        """
        Calculates the conjugation of Pauli Strings with one single-qubit Clifford string (for parallel execution)

        :param s_list: list of single-qubit Clifford strings depending on the gate set 'sqrt' or 'sqrtsqrt'
        :param save_sign_permutation: True: Save data for fast update of J values, see self.update_J_values().
        :return: one column of V,
                 sign flips of one column of V,
                 permutation of rows
        """
        # Calculate the permutation of the interactions
        new_idx = self.quad2int_fast(self._LUT[self._pauli_string_list, oper])
        permutation = (self._J_idx == new_idx[:, None]).argmax(axis=0)
        # Calculate the sign-flips of the interactions
        sign = (-1) ** np.sum(self._sgnLUT[self._pauli_string_list, oper], axis=1)[permutation]
        # Generate one column corresponding to the Clifford string oper
        V = sign * self._J[permutation]
        if save_sign_permutation:
            return V, sign, permutation
        else:
            return V

    def sampleV_clifford(self, k: int, save_sign_permutation: bool = True):
        """
        Constructs one V matrix, representing the conjugation of d Pauli strings from the quantum object with
        k*d randomly sampled single-qubit Clifford strings.
        Without feasibility check!

        :param k: factor for the nr. of single-qubit Clifford string samples (k*d sampled single-qubit Clifford strings)
        :param save_sign_permutation: True: Save data for fast update of J values, see self.update_J_values().
        :return: V: d x k*d matrix,
                 J_sign: sign flips of V,
                 J_permutation: permutation of rows for each column of V,
                 s_list: contains the sampled single-qubit Clifford strings
        """
        nz_J = np.nonzero(self._J)[0]
        ## Sample single-qubit Clifford strings
        if self._is_sqrt==1:  # the 'sqrt' gate-set
            len_nz_J = len(nz_J)
            s_list = np.random.choice(self._LUT.shape[1], size=(int(k * (len(self._J) - len_nz_J)), self._n))
            s_list_pauli = np.random.choice(4, size=(int(k * (len_nz_J)), self._n))
            s_list = np.vstack((s_list, s_list_pauli))
        elif self._is_sqrt==0:  # the 'sqrtsqrt' gate-set
            s_list = np.random.choice(self._LUT.shape[1], size=(int(k * (len(self._J))), self._n))
        else:
            assert False, "reshaping.py, sampleV_clifford(): ERROR. Wrong gate-set. Expected 'sqrt' or 'sqrtsqrt'."

        # Initialze required matrices
        V = np.zeros((len(self._J_idx), s_list.shape[0]))  # Matrix for LP constraints
        if save_sign_permutation:
            J_permutation = np.zeros((len(self._J), s_list.shape[0])).astype(int)  # For fast J update
            J_sign = np.zeros((len(self._J), s_list.shape[0]))  # For fast J update
        if not self._nr_jobs:
            for idx, oper in enumerate(s_list):
                if save_sign_permutation:
                    V_idx, sign_idx, permutation_idx = self.parallelCalcV(oper)
                    J_permutation[:, idx] = permutation_idx
                    J_sign[:, idx] = sign_idx
                    V[:, idx] = V_idx
                else:
                    V[:, idx] = self.parallelCalcV(oper, save_sign_permutation=False)
        else:
            data = np.array(Parallel(n_jobs=self._nr_jobs)(
                delayed(self.parallelCalcV)(oper, save_sign_permutation=save_sign_permutation) for oper in s_list)).T
            if save_sign_permutation:
                V = data[:, 0, :]
                J_sign = data[:, 1, :]
                J_permutation = data[:, 2, :].astype(int)
            else:
                V = data
        if save_sign_permutation:
            return V, J_sign, J_permutation, s_list
        else:
            return V, s_list

    def genV_clifford(self):
        """
        Samples a matrix V, representing the conjugation of d Pauli strings from the quantum object with
        k*d randomly sampled single-qubit Clifford strings.
        Then checks if V is feasible. Increase k if V is infeasible.
        """
        for k in np.linspace(self.dimension_factor, self.dimension_factor + 5, 30):
            for smpl in range(self._nr_samples):
                V, J_sign, J_permutation, s_list = self.sampleV_clifford(k)
                if self.is_V_feasible(V):  # Check if V is feasible
                    V, unique_idx = np.unique(V, axis=1, return_index=True)

                    self._V = V
                    self._operator_strings = s_list[unique_idx, :]
                    # Save data for fast J update
                    self._J_permutation = J_permutation[:, unique_idx]
                    self._J_sign = J_sign[:, unique_idx]
                    return
        assert False, "reshaping.py, genV_clifford(): ERROR. No feasible V was found. Increase dimension_factor."

    ## Generate feasible V
    ######################
    def genV(self):
        """
        Generates a feasible V depending on the gate set, the nr. of non-zeros in J and the dimension factor
        """
        if self._is_sqrt==-1:
            assert len(np.nonzero(self._J)[0])==len(self._J_idx), \
                "reshaping.py, genV(): ERROR. Pauli gate-set expects only non-zero values of J."
            if 4 ** self._n < len(self._J) * self.dimension_factor or self._solve_optimal:
                # Consider all commbinations -> for optimal solution in LP
                self.genV_pauli_exact()
            else:
                # Calculate V and check feasibility
                self.genV_pauli()
            return
        elif self._is_sqrt==1 or self._is_sqrt==0:
            if self._LUT.shape[1]**self._n<len(self._J)*self.dimension_factor or self._solve_optimal:
                # Consider all commbinations -> for optimal solution in LP
                self.genV_clifford_exact()
            else:
                # Calculate V and check feasibility
                self.genV_clifford()
            return
        else:
            assert False, "reshaping.py, genV(): ERROR. Wrong gate-set."

    def LP(self, target: np.ndarray, obj_fct_norm: Union[int, str] = 1, lam_bounds: list = None, solver: str = 'MOSEK'):
        """
        Solves the linear program
            min ||lam||_p
            s.t. V@lam = target
                 lam>=0
        where the p-norm in the objective function can be chosen.

        :param target: the reshaping target
        :param obj_fct_norm: the norm in the objective function ('inf' for infinity norm)
        :param lam_bounds: lower and upper bounds on lam (lam is not sparse anymore!)
        :param solver: either use 'MOSEK' or 'GLPK' solver.
        :return: lam: non-zero coefficients for conic combination,
                 used_operator_strings: the corresponding operator strings to conjugate the quantum object
        """
        if self._V is None:
            assert not self._J_idx is None, "reshaping.py, LP(): ERROR. Jidx is not defined."
            assert not self._J is None, "reshaping.py, LP(): ERROR. J is not defined."
            self.genV()
        assert len(target)==self._V.shape[0], "reshaping.py, LP(): ERROR. Wrong target dimension."
        if lam_bounds is None:
            x = cp.Variable(self._V.shape[1], nonneg=True)  # non-negativity constraint
            objective = cp.Minimize(cp.norm(x, obj_fct_norm))  # \ell_1 or \infty objective function: obj_fct_norm = 1 or 'inf'
            constraints = [self._V @ x == target]  # linear equation system
            prob = cp.Problem(objective, constraints)
            if solver=='MOSEK':
                prob.solve()
            else:
                prob.solve(solver=cp.GLPK, glpk={'msg_lev': 'GLP_MSG_OFF'})
            solution = x.value
            solution[solution < self._tol] = 0.0
            nz = np.nonzero(solution)[0]

            lam = solution[nz]
            used_operator_strings = self._operator_strings[nz, :]
        else:
            # Solve the LP with lower/upper bounds on lambda (there is no sparse solution anymore)
            x = cp.Variable(self._V.shape[1])
            objective = cp.Minimize(cp.norm(self._V @ x - target, obj_fct_norm))
            constraints = [x >= lam_bounds[0]*np.ones(self._V.shape[1]),
                           x <= lam_bounds[1]*np.ones(self._V.shape[1])]
            prob = cp.Problem(objective, constraints)
            if solver == 'MOSEK':
                prob.solve()
            else:
                prob.solve(solver=cp.GLPK, glpk={'msg_lev': 'GLP_MSG_OFF'})
            err = cp.norm(self._V @ x.value - target, obj_fct_norm).value
            if err>self._tol:
                print("reshaping.py, LP(): WARNING. The solution has an error of "+str(err)+".")
            lam = x.value
            used_operator_strings = self._operator_strings
        return lam, used_operator_strings
