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
    def __init__(self, hardware, target, gate_set, calc_diamond_norm: bool = False,
                 calc_evolution_time: bool = False):
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
        self._calc_diamond_norm = calc_diamond_norm
        self._calc_evolution_time = calc_evolution_time
        self._J = hardware.J
        self._H_S = self.genHamiltonian(self._J)  # Generate system Hamiltonian
        self._target = target
        self._H_T = self.genHamiltonian(self._target)  # Generate Target Hamiltonian
        self._U_T = None

    def norm_of_H_S(self):
        return self._H_S.norm()

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

    def calc_product(self, i_list, time_list, sign_list, t_pulse, robust_CP=False, combine_pulses=False, rotation_error=0.0, off_resonance_error=0.0):
        """
        Calculates the unitary corresponding to the matrix exponential product of conjugating a system Hamiltonian with
        single-qubit gates.

        :param i_list: Contains indices of _used_pauli_strings corresponding to the times in time_list
        :param time_list: Contains the free evolution times
        :param t_pulse: Pulse duration of a pi pulse
        :param robust_CP: 0: Clifford conjugation with pi/2 pulses
                          1: Clifford conjugation with SCROFULOUS pulses
                          2: Clifford conjugation with SCROBUTUS pulses
        :param combine_pulses: (only for Pauli gate-set) combine two adjacent Pauli gate layers
        :param rotation_error: relative rotation angle error
        :param off_resonance_error: off-resonance error
        :return:
        """
        if self._is_pauli_gate:
            pulse_list = pauli_list
        else:
            pulse_list = sqrtsqrt_list
        angle_errors = np.random.uniform(low=0, high=rotation_error, size=self._n)
        off_res_error = np.random.uniform(low=0, high=off_resonance_error, size=self._n)

        def get_S1S2(pauli_string, signs, angle_errors):  # Calculate the single-qubit generators
            pulse1 = tensor([identity(2)] * self._n)
            pulse2 = tensor([identity(2)] * self._n)
            for u in range(self._n):
                a = [identity(2)] * (self._n - 1)
                a.insert(u, (np.pi / 4 * signs[u] * pulse_list[pauli_string[u]][0] + sigmaz() * off_res_error[u]))
                pulse1 +=  (1 + angle_errors[u]) * tensor(a)
                a = [identity(2)] * (self._n - 1)
                a.insert(u, (np.pi / 4 * signs[u] * pulse_list[pauli_string[u]][1] + sigmaz() * off_res_error[u]))
                pulse2 += (1 + angle_errors[u]) * tensor(a)
            S1 = pulse1
            S2 = pulse2
            return S1, S2

        # Hamiltonian for one single-qubit gate layer
        def H_p(theta_list, phi_list, angle_errors):
            pulse = tensor([identity(2)] * self._n)
            theta = max(np.abs(theta_list))
            for u in range(self._n):
                a = [identity(2)] * (self._n - 1)
                if np.abs(theta_list[u]) > 1e-14:
                    a.insert(u, (sigmax() * np.cos(phi_list[u]) + sigmay() * np.sin(phi_list[u]) + sigmaz() * off_res_error[u]))
                else:
                    a.insert(u, identity(2))
                pulse += theta * np.sign(theta_list[u]) * (1 + angle_errors[u]) * tensor(a)
            return pulse

        if robust_CP:
            #                 [phi_1, phi_2]
            operator_2_phi = np.array([[0., 0.],  # Id
                                       [0., 0.],  # X
                                       [1., 1.],  # Y
                                       [0., 0.],  # Z=Id (ToDo: commutes with Ising interactions)
                                       [0., 1.],  # sqrt(X)*sqrt(Y)
                                       [0., 1.],  # sqrt(X)*sqrt(Y)^(-1)
                                       [0., 1.],  # sqrt(X)^(-1)*sqrt(Y)
                                       [0., 1.],  # sqrt(X)^(-1)*sqrt(Y)^(-1)
                                       [1., 0.],  # sqrt(Y)*sqrt(X)
                                       [1., 0.],  # sqrt(Y)*sqrt(X)^(-1)
                                       [1., 0.],  # sqrt(Y)^(-1)*sqrt(X)
                                       [1., 0.]])  # sqrt(Y)^(-1)*sqrt(X)^(-1)

            operator_2_thetasign = np.array([[0., 0.],  # Id
                                             [1., 1.],  # X
                                             [1., 1.],  # Y
                                             [0., 0.],  # Z=Id (ToDo: commutes with Ising interactions)
                                             [1., 1.],  # sqrt(X)*sqrt(Y)
                                             [1., -1.],  # sqrt(X)*sqrt(Y)^(-1)
                                             [-1., 1.],  # sqrt(X)^(-1)*sqrt(Y)
                                             [-1., -1],  # sqrt(X)^(-1)*sqrt(Y)^(-1)
                                             [1., 1.],  # sqrt(Y)*sqrt(X)
                                             [1., -1.],  # sqrt(Y)*sqrt(X)^(-1)
                                             [-1., 1.],  # sqrt(Y)^(-1)*sqrt(X)
                                             [-1., -1.]])  # sqrt(Y)^(-1)*sqrt(X)^(-1)

            U_res = tensor([identity(2)] * self._n)  # Resulting evolution operator
            for idx in range(len(i_list)):
                operator = i_list[idx,:]

                thetas = operator_2_thetasign[operator, :]
                phis = operator_2_phi[operator, :] * np.pi / 2

                theta_1 = 2.0103114334664382626
                if robust_CP==1:
                    ## Parameters for the SCROFOLOUS pulse sequence
                    arg = -np.pi * np.cos(theta_1) / (2 * theta_1 * np.sin(np.pi / 4))
                    phi_1 = phis + np.arccos(arg)
                    phi_2 = phi_1 - np.arccos(-np.pi / (2 * theta_1))
                    ##
                elif robust_CP==2:
                    ## Parameters for the SCROBUTUS pulse sequence
                    theta_r = np.arccos(0.5 * (1 - np.pi / theta_1 * np.sin(theta_1 / 2) ** 2))
                    theta_2 = np.pi + 2 * theta_r
                    arg = -np.pi * np.cos(theta_1) / (2 * theta_1 * np.sin(np.pi / 4))
                    phi_1 = phis + np.arccos(arg)
                    phi_2 = phi_1 - np.arccos(-np.pi / (2 * theta_1))
                    phi_r = phi_2 + np.pi
                    ##

                theta_list = []
                phi_list = []
                t_p_list = []
                for j in range(2):
                    if robust_CP == 1:
                        theta_list.append(thetas[:, j] * theta_1 / 2)
                        phi_list.append((thetas[:, j] != 0) * phi_1[:, j])
                        t_p_list.append(t_pulse * theta_1 / np.pi)

                        theta_list.append(thetas[:, j] * np.array([np.pi / 2] * self._n))
                        phi_list.append((thetas[:, j] != 0) * phi_2[:, j])
                        t_p_list.append(t_pulse)

                        theta_list.append(thetas[:, j] * theta_1 / 2)
                        phi_list.append((thetas[:, j] != 0) * phi_1[:, j])
                        t_p_list.append(t_pulse * theta_1 / np.pi)
                    elif robust_CP == 2:
                        theta_list.append(thetas[:, j] * theta_1 / 2)
                        phi_list.append((thetas[:, j] != 0) * phi_1[:, j])
                        t_p_list.append(t_pulse * theta_1 / np.pi)

                        theta_list.append(thetas[:, j] * theta_r / 2)  # np.array([theta_r/2]*self._n))
                        phi_list.append((thetas[:, j] != 0) * phi_r[:, j])
                        t_p_list.append(t_pulse * theta_r / np.pi)

                        theta_list.append(thetas[:, j] * theta_2 / 2)  # np.array([theta_2/2]*self._n))
                        phi_list.append((thetas[:, j] != 0) * phi_2[:, j])
                        t_p_list.append(t_pulse * theta_2 / np.pi)

                        theta_list.append(thetas[:, j] * theta_r / 2)  # np.array([theta_r/2]*self._n))
                        phi_list.append((thetas[:, j] != 0) * phi_r[:, j])
                        t_p_list.append(t_pulse * theta_r / np.pi)

                        theta_list.append(thetas[:, j] * theta_1 / 2)
                        phi_list.append((thetas[:, j] != 0) * phi_1[:, j])
                        t_p_list.append(t_pulse * theta_1 / np.pi)
                H_p_list = []
                for j in range(len(theta_list)):
                    H_p_list.append(H_p(theta_list[j], phi_list[j], angle_errors))

                pulses = tensor([identity(2)] * self._n)
                for j in range(len(theta_list)):
                    pulses = pulses * (-1j * (H_p_list[j] + t_p_list[j] * self._H_S)).expm()
                pulses_dag = tensor([identity(2)] * self._n)
                for j in range(len(theta_list)):
                    pulses_dag = (-1j * (-H_p_list[j] + t_p_list[j] * self._H_S)).expm() * pulses_dag
                U = pulses_dag
                if time_list[idx] > 1e-13:
                   U = U * (-1j * time_list[idx] * self._H_S).expm()  # Free evolution under system Hamiltonian
                U = U * pulses
                U_res = U_res * U
        else:
            U_res = tensor([identity(2)] * self._n)  # Resulting evolution operator
            if combine_pulses:  # Combine adjacent Pauli strings to one Pauli string (saving half of t_pulse)
                assert self._is_pauli_gate, \
                    "simulation.py, calc_product(): ERROR. The combine_pulses parameter can only be used with the Pauli gate-set."
                mat = np.array([[0, 1, 2, 3],
                                [1, 0, 3, 2],
                                [2, 3, 0, 1],
                                [3, 2, 1, 0]])  # Look-up table for the product of two Pauli matrices
                S1, S2 = get_S1S2(i_list[0,:], sign_list[0], angle_errors)
                U = ((-1j * (S2 + t_pulse / 2 * self._H_S)).expm() *  # Single-qubit pulses with finite pulse time
                     (-1j * (S1 + t_pulse / 2 * self._H_S)).expm())
                if time_list[0] > 1e-13:
                    U = U * (-1j * time_list[0] * self._H_S).expm()  # Free evolution under system Hamiltonian
                U_res = U_res * U
                for idx in range(1,len(i_list)):
                    sign_temp = np.array(sign_list[idx])
                    S1, S2 = get_S1S2(mat[(i_list[idx-1,:],i_list[idx,:])], sign_temp, angle_errors)
                    U = ((-1j * (S2 + t_pulse / 2 * self._H_S)).expm() *  # Single-qubit pulses with finite pulse time
                         (-1j * (S1 + t_pulse / 2 * self._H_S)).expm())
                    if time_list[idx] > 1e-13:
                        U = U * (-1j * time_list[idx] * self._H_S).expm()  # Free evolution under system Hamiltonian
                    U_res = U_res * U

                S1, S2 = get_S1S2(i_list[-1,:], sign_list[-1], angle_errors)
                U = ((-1j * (S2 + t_pulse/2 * self._H_S)).expm() *
                     (-1j * (S1 + t_pulse/2 * self._H_S)).expm())
                U_res = U_res * U
            else:
                for idx in range(len(i_list)):
                    S1, S2 = get_S1S2(i_list[idx,:], sign_list[idx], angle_errors)
                    U = ((-1j * (-S2 + t_pulse/2 * self._H_S)).expm() *  # Single-qubit pulses with finite pulse time
                         (-1j * (-S1 + t_pulse/2 * self._H_S)).expm())
                    if time_list[idx] > 1e-13:
                        U = U * (-1j * time_list[idx] * self._H_S).expm()  # Free evolution under system Hamiltonian
                    U = U * ((-1j * (S1 + t_pulse/2 * self._H_S)).expm() *  # Single-qubit pulses with finite pulse time
                             (-1j * (S2 + t_pulse/2 * self._H_S)).expm())
                    U_res = U_res * U
        return U_res

    def calc_metrics(self, res, unitary_or_state="unitary"):
        """
        Calculates the difference of the target state/unitary and the simulated state/unitary.

        :param res: The simulated state/unitary.
        :param unitary_or_state:
        :return: For unitaries: (diamond distance), process fidelity and average gate fidelity
                 For states: Fidelity, trace distance and the Hilber-Schmidt distance
        """
        if self._U_T is None:
            self._U_T = (-1j * self._H_T).expm()  # Generate target unitary

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
