#!/usr/bin/env python
""" Simulates the time evolution of the Hamiltonian engineering methods

@Author: Pascal Baßler
"""
import numpy as np
from qutip import *

from utilities import hardware as hw

## Pauli matrices
pauli = {0: identity(2),  # Id
         1: sigmax(),  # X
         2: sigmay(),  # Y
         3: sigmaz()}  # Z

## Pauli matrices
pauli_list = {0: [pauli[0], pauli[0]],  # Id
              1: [pauli[1], pauli[1]],  # X
              2: [pauli[2], pauli[2]],  # Y
              3: [pauli[3], pauli[3]]}  # Z

## single-qubit Clifford gate sets
sqrtsqrt_list = {0: pauli_list[0],  # Id
                 1: pauli_list[1],  # X
                 2: pauli_list[2],  # Y
                 3: pauli_list[3],  # Z
                 4: [pauli[1], pauli[2]],  # sqrt(X)*sqrt(Y)
                 5: [pauli[1], -pauli[2]],  # sqrt(X)*sqrt(Y)^(-1)
                 6: [-pauli[1], pauli[2]],  # sqrt(X)^(-1)*sqrt(Y)
                 7: [-pauli[1], -pauli[2]],  # sqrt(X)^(-1)*sqrt(Y)^(-1)
                 8: [pauli[2], pauli[1]],  # sqrt(Y)*sqrt(X)
                 9: [pauli[2], -pauli[1]],  # sqrt(Y)*sqrt(X)^(-1)
                 10: [-pauli[2], pauli[1]],  # sqrt(Y)^(-1)*sqrt(X)
                 11: [-pauli[2], -pauli[1]]}  # sqrt(Y)^(-1)*sqrt(X)^(-1)

class HamiltonianSimulation:
    def __init__(self, hardware, target, lam, used_pauli_strings, gate_set, calc_diamond_norm: bool = False,
                 calc_evolution_time: bool = False, J_unit: float = 1.):
        """
        Constructor for a HamiltonianSimulation object.

        :param hardware: Instance of a QuantumHardware object
        :param target: list/array of the target Pauli basis coefficients
        :param lam: list/array of the free evolution times.
        :param used_pauli_strings: 2D list/array with shape (s,n) for s single-qubit gate strings on n qubits.
        :param gate_set: What gate-set to use: 'pauli': Paulis
                                               'sqrtsqrt': Paulis and sqrt(Paulis)*sqrt(Paulis)
        :param calc_diamond_norm: True: Calculate the diamond distance between target and simulation unitary.
        :param calc_evolution_time: True: Calculate the total evolution time (sum(lam)+single-qubit gate time)
        :param J_unit: Factor of J
        """
        assert isinstance(hardware, hw.QuantumHardware), "simulation.py, __init__(): ERROR. Invalid input."
        self._n = hardware.n
        if gate_set=='sqrtsqrt':
            self._is_pauli_gate = 0
        elif gate_set=='pauli':
            self._is_pauli_gate = 1
        else:
            assert False, "simulation.py, __init__(): ERROR. Invalid gate-set."
        self._pauli_string_list = np.array(hardware.pauli_string_list)
        self._lam = lam/J_unit
        self._used_pauli_strings = used_pauli_strings
        self._calc_diamond_norm = calc_diamond_norm
        self._calc_evolution_time = calc_evolution_time
        self._J = hardware.J * J_unit
        self._H_S = self.genHamiltonian(self._J)  # Generate system Hamiltonian
        self._target = target
        self._H_T = self.genHamiltonian(self._target)  # Generate Target Hamiltonian
        self._U_T = None

    def genHamiltonian(self, coef: np.ndarray):
        """
        Generates a Hamiltonian in the Pauli basis with specified coefficients.

        :param coef: list/array of coefficients for each Pauli string
        :return: H: the full Hamiltonian
        """
        H = tensor([identity(2)] * self._n)
        H = H - tensor([identity(2)] * self._n)
        for i in range(len(coef)):
            H = H + coef[i] * tensor([pauli[j] for j in self._pauli_string_list[i]])
        return H

    def calc_product(self, i_list, time_list, t_pulse, combine_pulses=False):
        """
        Calculates the unitary corresponding to the matrix exponential product of conjugating a system Hamiltonian with
        single-qubit gates.

        :param i_list: Contains indices of _used_pauli_strings corresponding to the times in time_list
        :param time_list: Contains the free evolution times
        :param t_pulse: Pulse duration of a pi pulse
        :param combine_pulses: (only for Pauli gate-set) combine two adjacent Pauli gate layers
        :return:
        """
        if self._is_pauli_gate:
            pulse_list = pauli_list
        else:
            pulse_list = sqrtsqrt_list

        def get_S1S2(pauli_string):  # Calculate the single-qubit generators
            pulse1 = tensor([identity(2)] * self._n)
            pulse2 = tensor([identity(2)] * self._n)
            for u in range(self._n):
                a = [identity(2)] * (self._n - 1)
                a.insert(u, pulse_list[pauli_string[u]][0])
                pulse1 += tensor(a)
                a = [identity(2)] * (self._n - 1)
                a.insert(u, pulse_list[pauli_string[u]][1])
                pulse2 += tensor(a)
            S1 = np.pi / 4 * pulse1
            S2 = np.pi / 4 * pulse2
            return S1, S2

        U_res = tensor([identity(2)] * self._n)  # Resulting evolution operator
        if combine_pulses:  # Combine adjacent Pauli strings to one Pauli string (saving half of t_pulse)
            assert self._is_pauli_gate, \
                "simulation.py, calc_product(): ERROR. The combine_pulses parameter can only be used with the Pauli gate-set."
            mat = np.array([[0, 1, 2, 3],
                            [1, 0, 3, 2],
                            [2, 3, 0, 1],
                            [3, 2, 1, 0]])  # Look-up table for the product of two Pauli matrices
            S1, S2 = get_S1S2(self._used_pauli_strings[i_list[0], :])
            U = ((-1j * (S2 + t_pulse / 2 * self._H_S)).expm() *  # Single-qubit pulses with finite pulse time
                 (-1j * (S1 + t_pulse / 2 * self._H_S)).expm() *
                 (-1j * time_list[0] * self._H_S).expm())  # Free evolution under system Hamiltonian
            U_res = U_res * U
            for idx in range(1,len(i_list)):
                #print_out = 'Simulation progress: '+str(np.round(100*idx/len(i_list), decimals=1))+'%'
                #print("\r", print_out, end="")
                S1, S2 = get_S1S2(mat[(self._used_pauli_strings[i_list[idx-1], :],self._used_pauli_strings[i_list[idx], :])])
                U = ((-1j * (S2 + t_pulse / 2 * self._H_S)).expm() *  # Single-qubit pulses with finite pulse time
                     (-1j * (S1 + t_pulse / 2 * self._H_S)).expm() *
                     (-1j * time_list[idx] * self._H_S).expm())  # Free evolution under system Hamiltonian
                U_res = U_res * U
            S1, S2 = get_S1S2(self._used_pauli_strings[i_list[-1], :])
            U = ((-1j * (S2 + t_pulse/2 * self._H_S)).expm() *
                 (-1j * (S1 + t_pulse/2 * self._H_S)).expm())
            U_res = U_res * U
        else:
            for idx, i in enumerate(i_list):
                S1, S2 = get_S1S2(self._used_pauli_strings[i, :])
                U = ((-1j * (-S2 + t_pulse/2 * self._H_S)).expm() *  # Single-qubit pulses with finite pulse time
                     (-1j * (-S1 + t_pulse/2 * self._H_S)).expm() *
                     (-1j * time_list[idx] * self._H_S).expm() *  # Free evolution under system Hamiltonian
                     (-1j * (S1 + t_pulse/2 * self._H_S)).expm() *  # Single-qubit pulses with finite pulse time
                     (-1j * (S2 + t_pulse/2 * self._H_S)).expm())
                U_res = U_res * U
        return U_res, 'unitary'

    def calc_metrics(self, res, unitary_or_state):
        """
        Calculates the difference of the target state/unitary and the simulated state/unitary.

        :param res: The simulated state/unitary.
        :param unitary_or_state:
        :return: For unitaries: (diamond distance), process fidelity and average gate fidelity
                 For states: Fidelity, trace distance and the Hilber-Schmidt distance
        """
        if unitary_or_state=="unitary":
            process_fidelity = metrics.process_fidelity(res, self._U_T)
            avrg_gate_fidelity = metrics.average_gate_fidelity(res, self._U_T)
            if self._calc_diamond_norm:
                diamond_norm = (res - self._U_T).dnorm()  # Diamond distance
                return [diamond_norm, process_fidelity, avrg_gate_fidelity]
            return [process_fidelity, avrg_gate_fidelity]
        elif unitary_or_state=="density_mat":
            target_state = self._U_T * self._initial_state * self._U_T.dag()
            fidelity = metrics.fidelity(res, target_state)
            trace_dist = metrics.tracedist(res, target_state)
            hilbert_schmidt_dist = metrics.hilbert_dist(res, target_state)
            return [fidelity, trace_dist, hilbert_schmidt_dist]
        else:
            assert False, "simulation.py, calc_metrics(): ERROR. Unknown keyword for parameter unitary_or_state."

    def trotter(self, effective_evolution_time: float = 1, t_pulse: float = 0.0, r: int = 6):
        """
        Simulation of the second order Trotter method implementing the reshaping method.

        :param effective_evolution_time: t_eff, the effective evolution time of the target Hamiltonian: exp(-i t_eff H_T)
        :param t_pulse: pi pulse time
        :param r: Nr. of cycles of the Trotter method
        :return: The distance of the simulated vs. exact evolution
        """
        if self._U_T is None:
            self._U_T = (-1j * self._H_T * effective_evolution_time).expm()  # Generate target unitary

        s = effective_evolution_time / r
        lam_s = self._lam * s

        # 2nd order Trotter
        times_idx_list = list(range(len(lam_s))) + list(reversed(range(len(lam_s))))
        lam_s = lam_s/2
        lam_s = np.append(lam_s, np.flip(lam_s))

        times_idx_list = times_idx_list * r
        times_list = lam_s.tolist() * r

        # Calculate the product of matrix exponentials to get the resulting unitary
        res, unitary_or_state = self.calc_product(times_idx_list, times_list, t_pulse, combine_pulses=self._is_pauli_gate)
        if self._calc_evolution_time:  # Calculate total evolution time inclugin single-qubit pulses
            evolution_time = len(times_list) + sum(np.array(times_list))
            print("Total evolution time [sec]: ", evolution_time)
        return self.calc_metrics(res, unitary_or_state)

    def commutingHamiltonian(self, effective_evolution_time: float = 1, t_pulse: float = 0.0):
        """
        Simulation of our reshaping method for commuting Hamiltonians.

        :param effective_evolution_time: t_eff, the effective evolution time of the target Hamiltonian: exp(-i t_eff H_T)
        :param t_pulse: pi pulse time
        :return: The distance of the simulated vs. exact evolution
        """
        if self._U_T is None:
            self._U_T = (-1j * self._H_T * effective_evolution_time).expm()  # Generate target unitary
        lam = self._lam * effective_evolution_time
        indizes = np.arange(len(self._lam))

        # Calculate the product of matrix exponentials to get the resulting unitary
        res, unitary_or_state = self.calc_product(indizes, lam, t_pulse, combine_pulses=self._is_pauli_gate)
        if self._calc_evolution_time:  # Calculate total evolution time inclugin single-qubit pulses
            evolution_time = len(lam)*t_pulse + sum(lam)
            print("Total evolution time [sec]: ", evolution_time)
        return self.calc_metrics(res, unitary_or_state)
