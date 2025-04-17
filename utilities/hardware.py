#!/usr/bin/env python
""" Quantum Hardware
Generates the Pauli string and coefficients for different system Hamiltonians representing the interactions in different
quantum platforms.

@Author: Pascal Baßler
"""
import numpy as np
import itertools
import random

from utilities import couplings as cou

class QuantumHardware:
    def __init__(self, n: int = None, N: int = None, J_value: str = 'random', interaction_type: int = 0):
        """
        Constructor for a QuantumHardware object.

        :param n: nr. of qubits (will be ignored for 2D lattice interactions)
        :param N: Locality of the interactions
        :param J_value: 'ones': J is the all-ones vector,
                        'random': Is element is sampled uniformly from [-1,1],
                        'increasing': J = [1, 2, 3, ...],
                        'random_normed': Same as 'random' but the ell_1 norm is normalized.
        :param interaction_type: 0, all N local interactions.
                                 k (int), randomly choose k interaction supports for each 1,...,N and generate all possible interactions on these supports.
        """
        self._n = n
        self._N = N
        self._J_value = J_value
        self._interaction_type = interaction_type
        self._pauli_string_list = None
        self._nz_idx = None
        self._J = None
        return

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = value
        self._nz_idx = None
        self._J = None

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        self._N = value
        self._nz_idx = None
        self._J = None

    @property
    def J_value(self):
        return self._J_value

    @J_value.setter
    def J_value(self, value):
        self._J_value = value
        self._nz_idx = None
        self._J = None

    @property
    def pauli_string_list(self):
        return self._pauli_string_list

    @pauli_string_list.setter
    def pauli_string_list(self, value):
        self._pauli_string_list = value

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, value):
        self._J = value

    ## Utility methods
    ##################
    def int2base(self, k_list, b, n=None):
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

    def base2int(self, k_list, b, n=None):
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
            ret.append(sum([k[i] * pow(b, n - 1 - i) for i in range(n)]))
        return np.array(ret)

    def Ham_k_int(self, k):
        """
        Generate an array of all Pauli X strings on n qubits with support k.

        :param k: Support of the Pauli X strings.
        :return: All Pauli X strings on n qubits with support k.
        """
        max_int = 2 ** int(self._n) - 1
        res = [int(sum([2 ** i for i in range(k)]))]
        if res[-1] < max_int:
            while res[-1] < max_int:
                c = res[-1] & (-res[-1])
                r = res[-1] + c
                res.append(int(((r ^ res[-1]) >> 2) / c) | r)
            return res[:-1]
        else:
            return res

    def Ham_k_int_quad(self, k):
        """
        Generate an array of all Pauli strings on n qubits with support k.

        :param k: Support of the Pauli strings.
        :return: All Pauli strings on n qubits with support k.
        """
        all_comb = np.array(list(itertools.product(range(1, 4), repeat=k)))
        temp = self.int2base(self.Ham_k_int(k), 2)
        res = []
        for i in temp:
            nz = np.nonzero(i)[0]
            for j in all_comb:
                res.append(sum([4 ** int(nz[t]) * int(j[t]) for t in range(k)]))
        return res

    def random_Ham_k_int_quad(self, k, nr_samples):
        """
        Randomly choose Pauli X strings with support k.
        Then generate all possible Pauli strings on these supports.

        :param k: Support of the Pauli strings.
        :param nr_samples: Nr. of samples
        :return: All possible Pauli string on random support k.
        """
        all_comb = np.array(list(itertools.product(range(1, 4), repeat=k)))
        first = [1] * k + [0] * (self._n - k)
        temp = []
        for i in range(nr_samples):
            random.shuffle(first)
            temp.append(first.copy())
        res = []
        for i in temp:
            nz = np.nonzero(i)[0]
            for j in all_comb:
                res.append(sum([4 ** int(nz[t]) * int(j[t]) for t in range(k)]))
        return res

    ## Generate many-body interactions
    ##################################
    def N_body_interaction(self):
        """
        Generate all Pauli strings on n qubits with support<=N.
        """
        all_N_orders = []
        for i in range(1, self._N + 1):
            all_N_orders = all_N_orders + self.Ham_k_int_quad(i)
        all_N_orders = sorted(all_N_orders)
        self._pauli_string_list = self.int2base(all_N_orders, 4)

    def N_body_commuting_interaction(self):
        """
        Generate all commuting Pauli X strings on n qubits with support<=N.
        """
        all_N_orders = []
        for i in range(1, self._N + 1):
            all_N_orders = all_N_orders + self.Ham_k_int(i)
        all_N_orders = sorted(all_N_orders)
        self._pauli_string_list = self.int2base(all_N_orders, 2)

    def random_interaction(self, nr_of_interactions):
        """
        Generate random Pauli strings

        :param nr_of_interactions: Nr. of random Pauli strings
        """
        dim = 4 ** self._n

        if nr_of_interactions > dim-1:
            Jidx = np.arange(1, dim)
        elif dim < 2 ** 63:
            Jidx = random.sample(range(1, dim), nr_of_interactions)
        else:
            Jidx = [random.randint(1, dim) for i in range(nr_of_interactions)]
        self._pauli_string_list = self.int2base(Jidx, 4)

    def random_sparse_N_body_interaction(self):
        """
        Randomly choose Pauli X strings with support<=N.
        Then generate all possible Pauli strings on these supports.
        """
        order_one = self.Ham_k_int(1)
        if self._interaction_type<len(order_one):
            order_one = random.sample(order_one, k=self._interaction_type)

        temp = self.int2base(order_one, 2)
        order_one = []
        for i in temp:
            nz = np.nonzero(i)[0]
            for j in [1, 2, 3]:
                order_one.append(4 ** int(nz[0]) * int(j))

        all_N_orders = []
        for i in range(2, self._N + 1):
            all_N_orders = all_N_orders + self.random_Ham_k_int_quad(i, self._interaction_type)
        all_N_orders = all_N_orders + order_one
        self._pauli_string_list = self.int2base(np.unique(all_N_orders).tolist(), 4)

    def lattice2D_NextNeig_N_body_interaction(self, shape, commuting=True):
        """
        Generate all possible (commuting) N-body Pauli strings on a 2D square lattice. (with N=2 or N=3)

        :param shape: The "length" and "width" of the square lattice
        :param commuting: True: Only generate commuting interactions
        """
        self._n = shape[0] * shape[1]
        a = np.reshape(np.arange(self._n),shape)
        if commuting:
            all_comb_N2 = np.array([[1,1]])
            all_comb_N3 = np.array([[1,1,1]])
        else:
            all_comb_N2 = np.array(list(itertools.product(range(1, 4), repeat=2)))
            all_comb_N3 = np.array(list(itertools.product(range(1, 4), repeat=3)))

        def neighbors(position):
            row_number = position[0]+1
            column_number = position[1]+1
            return [[a[i][j] if i >= 0 and i < len(a) and j >= 0 and j < len(a[0]) else -1
                     for j in range(column_number - 2, column_number + 1)]
                    for i in range(row_number - 2, row_number + 1)]

        quadJidx = []
        for row in range(shape[0]):
            for col in range(shape[1]):
                mat = neighbors((row, col))
                if self._N>=2:
                    temps = [[a[row, col], mat[0][1]], # (1,2)
                             [a[row, col], mat[1][2]], # (1,3)
                             [a[row, col], mat[2][1]], # (1,4)
                             [a[row, col], mat[1][0]]] # (1,5)
                    for i in temps:
                        if not -1 in i:
                            for j in all_comb_N2:
                                quadJidx_temp = np.zeros(self._n)
                                quadJidx_temp[i] = j
                                quadJidx.append(quadJidx_temp)
                    if self._N==3:
                        temps = [[a[row, col], mat[0][1], mat[1][2]],  # (1,2,3)
                                 [a[row, col], mat[0][1], mat[2][1]],  # (1,2,4)
                                 [a[row, col], mat[1][2], mat[2][1]],  # (1,3,4)
                                 [a[row, col], mat[1][2], mat[1][0]],  # (1,3,5)
                                 [a[row, col], mat[2][1], mat[1][0]],  # (1,4,5)
                                 [a[row, col], mat[1][0], mat[0][1]]]  # (1,5,2)
                        for i in temps:
                            if not -1 in i:
                                for j in all_comb_N3:
                                    quadJidx_temp = np.zeros(self._n)
                                    quadJidx_temp[i] = j
                                    quadJidx.append(quadJidx_temp)
        self._pauli_string_list = np.unique(np.array(quadJidx), axis=0).astype(int)

    ## Generate coupling coefficients
    #################################
    def minimal_non_zero_J(self, random_choice=False, nr_of_interactions=None):
        """
        Generate interactions with the minimal number of non-zero interactions, such that Clifford conjugation is
        still able to generate all interactions.

        :param random_choice: True: The non-zero entries in J are chosen randomly.
        :param nr_of_interactions: Nr. of random interactions.
        """
        if nr_of_interactions is None:
            if self._interaction_type:
                self.random_sparse_N_body_interaction()
            else:
                self.N_body_interaction()
        else:
            self.random_interaction(nr_of_interactions)

        if nr_of_interactions is None:
            all_N_orders_com = np.unique(self._pauli_string_list.astype(bool).astype(int), axis=0)
            if random_choice:
                N_orders = np.random.choice([1, 2, 3], size=(len(all_N_orders_com), self._n)) * all_N_orders_com
            else:
                N_orders = all_N_orders_com
            Jidx = self.base2int(self._pauli_string_list, 4).tolist()
            self._nz_idx = [Jidx.index(i) for i in self.base2int(N_orders, 4)]
        else:
            a = self.base2int(self._pauli_string_list.astype(bool).astype(int), 2)
            self._nz_idx = sorted([random.sample(np.nonzero(a == j)[0].tolist(), 1)[0] for j in np.unique(a)])

        self._J = np.zeros(len(self._pauli_string_list))
        if self._J_value=='ones':
            self._J[self._nz_idx] = np.ones(len(self._nz_idx))
        elif self._J_value=='random':
            self._J[self._nz_idx] = np.random.uniform(low=-1, high=1, size=len(self._nz_idx))
        elif self._J_value == 'increasing':
            self._J[self._nz_idx] = np.arange(len(self._nz_idx))
        elif self._J_value=='random_normed':
            temp = np.random.uniform(low=-1, high=1, size=len(self._nz_idx))
            self._J[self._nz_idx] = temp/sum(np.abs(temp))

    def non_zero_J(self, nr_of_interactions=None):
        """
        Generate interactions with only non-zero interactions.

        :param nr_of_interactions: Nr. of random interactions.
        """
        if nr_of_interactions is None:
            if self._interaction_type:
                self.random_sparse_N_body_interaction()
            else:
                self.N_body_interaction()
        else:
            self.random_interaction(nr_of_interactions)

        if self._J_value=='ones':
            self._J = np.ones(len(self._pauli_string_list))
        elif self._J_value=='random':
            self._J = np.random.uniform(low=-1, high=1, size=len(self._pauli_string_list))
        elif self._J_value=='increasing':
            self._J = np.arange(len(self._pauli_string_list))
        elif self._J_value=='random_normed':
            temp = np.random.uniform(low=-1, high=1, size=len(self._pauli_string_list))
            self._J = temp/sum(np.abs(temp))

    def commuting_hamiltonian(self):
        """
        Generate all commuting interactions with support<=N.
        """
        self.N_body_commuting_interaction()

        if self._J_value=='ones':
            self._J = np.ones(len(self._pauli_string_list))
        elif self._J_value=='random':
            self._J = np.random.uniform(low=-1, high=1, size=len(self._pauli_string_list))
        elif self._J_value=='increasing':
            self._J = np.arange(len(self._pauli_string_list))
        elif self._J_value=='random_normed':
            temp = np.random.uniform(low=-1, high=1, size=len(self._pauli_string_list))
            self._J = temp/sum(np.abs(temp))

    def random_non_zero_J(self, sparsity=0.8, random_sparsity=False, nr_of_interactions=None):
        """
        Generate interactions with a random number of non-zero interactions, such that Clifford conjugation is
        still able to generate all interactions.

        :param sparsity: How many zero J elements? 0: All elements are non-zero, 1: minimal number of non-zero J
        :param random_sparsity: True: Choose the sparsity random.
        :param nr_of_interactions: Nr. of random interactions.
        """
        self.minimal_non_zero_J(random_choice=True, nr_of_interactions=nr_of_interactions)

        mask = np.ones(len(self._pauli_string_list), bool)
        mask[self._nz_idx] = False
        if random_sparsity:
            p = np.random.uniform(low=0, high=1)
        else:
            p = sparsity
        if self._J_value=='ones':
            fact = np.ones(len(self._pauli_string_list) - len(self._nz_idx))
        elif self._J_value=='random':
            fact = np.random.uniform(low=-1, high=1, size=len(self._pauli_string_list) - len(self._nz_idx))
        elif self._J_value=='increasing':
            fact = np.arange(len(self._pauli_string_list) - len(self._nz_idx))
        elif self._J_value=='random_normed':
            fact = np.random.uniform(low=-1, high=1, size=len(self._nz_idx))
        self._J[mask] = np.random.choice([1.0, 0.0], size=len(fact), p=[1-p, p]) * fact
        if self._J_value=='random_normed':
            self._J = self._J/sum(np.abs(self._J))

    def ion_trap_harm_pot(self, include_non_commuting_interactions=False):
        """
        Generate Ising ZZ interactions with coupling strength given by an ion trap with harmonic trapping potential.

        :param include_non_commuting_interactions: True: Also include non-commuting interactions and set corresponding J to zero.
        """
        self._pauli_string_list = self.int2base(self.Ham_k_int(2), 2)*3
        nz_idx = np.nonzero(self._pauli_string_list)
        external = cou.Quadratic1D(mw2=1.)  # Assume quadratic trap potential
        Hinv = cou.coupling_matrix(self._n, external)  # Calculate the inverse Hessian
        self._J = [Hinv[i[0],i[1]]for i in np.reshape(nz_idx[1],(int(len(nz_idx[1])/2),2))]
        if include_non_commuting_interactions:
            all_comb = np.array(list(itertools.product(range(1, 4), repeat=2)))[:-1]
            res = np.empty((0,self._n)).astype(int)
            for i in self._pauli_string_list:
                nz = np.nonzero(i)[0]
                temp = np.zeros((len(all_comb),len(i))).astype(int)
                temp[:, nz] = all_comb
                res = np.vstack([res,temp])
            self._pauli_string_list = np.vstack([self._pauli_string_list, res])
            self._J = self._J + [0.]*res.shape[0]
        self._J = np.array(self._J)

    def square_lattice_2D(self, shape, commuting=True, only_non_zero_J=True, random_choice=False):
        """
        Generate all possible (commuting) N-body interactions on a 2D square lattice. (with N=2 or N=3)

        :param shape: The "length" and "width" of the square lattice
        :param commuting: True: Only generate commuting interactions
        :param only_non_zero_J: True: All J are non-zero.
        :param random_choice: True: The non-zero entries in J are chosen randomly.
        """
        self.lattice2D_NextNeig_N_body_interaction(shape, commuting=commuting)

        if not only_non_zero_J:
            all_N_orders_com = np.unique(self._pauli_string_list.astype(bool).astype(int), axis=0)
            if random_choice:
                N_orders = np.random.choice([1, 2, 3], size=(len(all_N_orders_com), self._n)) * all_N_orders_com
            else:
                N_orders = all_N_orders_com
            self._nz_idx = [np.where((self._pauli_string_list == i).all(axis=1))[0][0] for i in N_orders]
        else:
            self._nz_idx = list(range(self._pauli_string_list.shape[0]))

        self._J = np.zeros(self._pauli_string_list.shape[0])
        if self._J_value=='ones':
            self._J[self._nz_idx] = np.ones(len(self._nz_idx))
        elif self._J_value=='random':
            self._J[self._nz_idx] = np.random.uniform(low=-1, high=1, size=len(self._nz_idx))
        elif self._J_value == 'increasing':
            self._J[self._nz_idx] = np.arange(len(self._nz_idx))
        elif self._J_value=='random_normed':
            temp = np.random.uniform(low=-1, high=1, size=len(self._nz_idx))
            self._J[self._nz_idx] = temp/sum(np.abs(temp))

