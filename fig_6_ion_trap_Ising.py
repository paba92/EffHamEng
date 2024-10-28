#!/usr/bin/env python
""" Generate plots for the simulation with an ion trap model in Fig. 6

@Author: Pascal Baßler
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed

from utilities import reshaping as r
from utilities import hardware as hw
from utilities import simulation as s
########################
# ToDo: Uncomment the relevant parts:

t_eff = 1.  # effective evolution time

## Figure 6 left
hamiltonian_always_on = True  # If True: System Hamiltonian is on during single-qubit pulses
J_error_amplitude = 0  # Calibration error of J
t_p_factor = 1
parameter_sweep_all = range(0,52,5)
parameter_is_n = False
n = 10

# ## Figure 6 middle
# hamiltonian_always_on = False  # If True: System Hamiltonian is on during single-qubit pulses
# J_error_amplitude = 2e-4  # Calibration error of J
# t_p_factor = 1
# parameter_sweep_all = range(0,52,5)
# parameter_is_n = False
# n = 10

# ## Figure 6 right
# hamiltonian_always_on = True  # If True: System Hamiltonian is on during single-qubit pulses
# J_error_amplitude = 1e-2  # Calibration error of J
# t_p_factor = 25
# parameter_sweep_all = range(3,11)
# parameter_is_n = True
# n = None
########################

reshape_ising_to_heisenberg = False  # If True: Reshape Ion trap Ising interactions to Heisenberg model
all_samples = range(50)  # Nr. of samples/threads
nr_jobs = len(all_samples)
filename = "filename"
load_data = False

# Ion trap parameters
B1 = 40 # [T/m] Magnetic field gradient
f_trap = 400 # [kHz] Trap frequency
f_rabi = f_trap/2 # [kHz] Rabi frequency
J_unit = (B1/f_trap)**2/109.919368e-6 # [Hz] units for J (including physical constants)
t_pulse = t_p_factor*np.pi/(2*np.pi*f_rabi*1e3) # [sec] pi-pulse time
if not hamiltonian_always_on:
    t_pulse = 0.

def calc_sample(qhw, parameter):
    ## Target couplings
    ###################
    if reshape_ising_to_heisenberg:
        gate_set = "sqrtsqrt"
        # Heisenberg model
        XX_terms = qhw.pauli_string_list[:int(qhw.n / 2 * (qhw.n - 1)), :]
        XX_idx = list(range(int(qhw.n / 2 * (qhw.n - 1))))
        YY_idx = []
        ZZ_idx = []
        for i in range(XX_terms.shape[0]):
            YY_idx.append(np.where(np.all(qhw.pauli_string_list == 2 * XX_terms[i, :], axis=1))[0][0])
        for i in range(XX_terms.shape[0]):
            ZZ_idx.append(np.where(np.all(qhw.pauli_string_list == 3 * XX_terms[i, :], axis=1))[0][0])
        A = np.zeros(qhw.pauli_string_list.shape[0])
        # Random target couplings
        A[XX_idx] = np.random.uniform(low=0, high=t_eff, size=len(XX_idx))
        A[YY_idx] = np.random.uniform(low=0, high=t_eff, size=len(YY_idx))
        A[ZZ_idx] = np.random.uniform(low=0, high=t_eff, size=len(ZZ_idx))
    else:
        gate_set = "pauli"
        A = np.random.uniform(low=0, high=t_eff, size=len(qhw.J))  # Random target couplings

    ## Calculate reshaping times and operations
    ###########################################
    a = r.Reshaping(qhw.n, qhw.J, qhw.pauli_string_list, gate_set=gate_set, dimension_factor=6)
    lam, used_pauli_strings = a.LP(A, solver='GLPK')

    ## Simulate reshaping
    #####################
    ## Add calibration noise
    noise = np.random.uniform(low=-1, high=1, size=qhw.J.shape) * J_error_amplitude * parameter
    qhw.J = qhw.J + qhw.J * noise
    ## finite pulse time
    t_pulse_factor = parameter

    sim = s.HamiltonianSimulation(qhw, A, lam, used_pauli_strings, gate_set, J_unit=J_unit)
    if reshape_ising_to_heisenberg:
        trotter_err = sim.trotter(t_pulse=t_pulse * t_pulse_factor, r=parameter)
        return trotter_err[1]
    else:
        commuting_err = sim.commutingHamiltonian(t_pulse=t_pulse * t_pulse_factor)
        return commuting_err[1]

if not load_data:
    avrg_err = []
    qhw = hw.QuantumHardware()
    for parameter in parameter_sweep_all:
        print("Parameter: ", parameter)
        if parameter_is_n:
            qhw.n = parameter
            error_factor = 1
        else:
            qhw.n = n
            error_factor = parameter
        qhw.ion_trap_harm_pot(include_non_commuting_interactions=reshape_ising_to_heisenberg)  # Consider an ion trap with harmonic trap potential
        smpl_err = Parallel(n_jobs=nr_jobs)(delayed(calc_sample)(qhw, error_factor) for sample in all_samples)
        print("Average of average gate infidelity [10e-3]: ", (1-np.mean(smpl_err, axis=0))*1e3)
        avrg_err.append(smpl_err)

        with open(filename+'.npy', 'wb') as f:
            np.save(f, np.array(avrg_err))
else:
    with open(filename+'.npy', 'rb') as f:
        avrg_err = np.load(f, allow_pickle=True)

err_mean = []
err_var = []
for i in range(len(parameter_sweep_all)):
    err_mean.append(np.mean(avrg_err[i]))
    err_var.append(np.sqrt(np.var(avrg_err[i])))

fig = plt.figure(figsize=(1.6, 1.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_ylabel(r'$1-F_{\mathrm{avg}}$', labelpad=2)
ax.ticklabel_format(style='sci',scilimits=(-3,-3),axis='y')
cmap_g = mpl.cm.get_cmap('Blues')

if hamiltonian_always_on:
    if parameter_is_n:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        ax.errorbar(np.array(parameter_sweep_all), (1 - np.array(err_mean)), np.array(err_var), color=cmap_g(1.2))
        ax.set_xlim(min(parameter_sweep_all), max(parameter_sweep_all) + +0.2)
        ax.set_xlabel(r'nr. of qubits $n$', labelpad=2)
    else:
        fact = t_pulse * 1e6
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        ax.errorbar(np.array(parameter_sweep_all) * fact, (1 - np.array(err_mean)), np.array(err_var), color=cmap_g(1.2))
        ax.set_xlim(min(parameter_sweep_all)*fact, max(parameter_sweep_all)*fact + 2)
        ax.set_xlabel(r'$t_p$ [$\mu sec$]', labelpad=2)
else:
    ax.ticklabel_format(style='sci',scilimits=(-2,-2),axis='x')
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2 / 100))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1 / 100))
    ax.errorbar(np.array(parameter_sweep_all) * J_error_amplitude, (1 - np.array(err_mean)), np.array(err_var), color=cmap_g(1.2))
    ax.set_xlim(min(parameter_sweep_all)*J_error_amplitude, (max(parameter_sweep_all)+1)*J_error_amplitude+0.03/100)
    ax.set_xlabel(r'cal. error $\varepsilon$', labelpad=2)

plt.savefig(filename+'.pdf', transparent=False, bbox_inches='tight')
plt.show()
