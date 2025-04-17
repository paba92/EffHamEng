#!/usr/bin/env python
""" Generate plots for the simulation with an ion trap model in Fig. 8

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
n = 8  # Nr. of qubits
t_eff = 1.  # effective evolution time
c_all = range(2,21)  # Nr. of Trotter cycles c
rot_error = 0.01  # Rotation angle error
off_res_error = 0.01  # Off-resonance error
use_robust_composite_pulse = 0 # 0: Clifford conjugation with pi/2 pulses
                               # 1: Clifford conjugation with SCROFULOUS pulses
                               # 2: Clifford conjugation with SCROBUTUS pulses
# ToDo: The calculation of the finite pulse time effect for the robust SCROFULOUS/SCROBUTUS pulse sequences
#  is not yet optimized for speed. It is in the file utilities/reshaping.py in the method clifford_CP_error_matrix().
########################

all_samples = range(50)  # Nr. of samples
nr_jobs = 50  # Nr. of threads
filename = "filename"
load_data = False

# Ion trap parameters
B1 = 40 # [T/m] Magnetic field gradient
f_trap = 500 # [kHz] Trap frequency
f_rabi = f_trap/2 # [kHz] Rabi frequency
J_unit = (B1/f_trap)**2/109.919368e-6 # [Hz] units for J (including physical constants)
t_pulse = np.pi/(2*np.pi*f_rabi*1e3) # [sec] pi-pulse time

def calc_sample(J, n, pauli_string_list, parameter):
    ## Target couplings
    ###################
    gate_set = "sqrtsqrt"

    qhw = hw.QuantumHardware()
    qhw.n = n
    qhw.pauli_string_list = np.copy(pauli_string_list)
    qhw.J = np.copy(J)

    # Heisenberg model
    ZZ_terms = qhw.pauli_string_list[:int(qhw.n / 2 * (qhw.n - 1)), :]
    XX_idx = []
    YY_idx = []
    ZZ_idx = list(range(int(qhw.n / 2 * (qhw.n - 1))))
    for i in range(ZZ_terms.shape[0]):
        YY_idx.append(np.where(np.all(qhw.pauli_string_list == 2 * np.sign(ZZ_terms[i, :]), axis=1))[0][0])
    for i in range(ZZ_terms.shape[0]):
        XX_idx.append(np.where(np.all(qhw.pauli_string_list == 1 * np.sign(ZZ_terms[i, :]), axis=1))[0][0])
    A = np.zeros(qhw.pauli_string_list.shape[0])
    # Random target couplings
    A[XX_idx] = np.random.uniform(low=0.1, high=1, size=len(XX_idx))
    A[YY_idx] = np.random.uniform(low=0.1, high=1, size=len(YY_idx))
    A[ZZ_idx] = np.random.uniform(low=0.1, high=1, size=len(ZZ_idx))


    ## Calculate reshaping times and operations
    ###########################################
    a = r.Reshaping(qhw.n, qhw.J, qhw.pauli_string_list, gate_set=gate_set, dimension_factor=6)

    ## Robust Clifford conjugation
    times_list_robust, operator_list_robust, sign_list_robust = a.trotter(A,
                                                                          t_pulse=t_pulse * J_unit/t_eff,
                                                                          order=2,
                                                                          cycles=parameter,
                                                                          robust_CP=use_robust_composite_pulse,
                                                                          use_MILP=True)
    times_list_robust = times_list_robust * t_eff / J_unit

    ## "naive" Clifford conjugation
    times_list, operator_list, sign_list = a.trotter(A, solver='GLPK', t_pulse=0.0, order=2, cycles=parameter)
    times_list = times_list * t_eff / J_unit

    ## Simulate reshaping
    #####################
    qhw.J = qhw.J * J_unit
    sim = s.HamiltonianSimulation(qhw, A * t_eff, gate_set)
    # Robust method
    err_robustCP = sim.calc_metrics(sim.calc_product(operator_list_robust,
                                                     times_list_robust,
                                                     sign_list_robust,
                                                     t_pulse,
                                                     robust_CP=use_robust_composite_pulse,
                                                     rotation_error=rot_error,
                                                     off_resonance_error=off_res_error))
    # Naive method
    err_naive = sim.calc_metrics(sim.calc_product(operator_list,
                                                  times_list,
                                                  sign_list,
                                                  t_pulse,
                                                  rotation_error=rot_error,
                                                  off_resonance_error=off_res_error))
    # Only rotation angle error
    err_rot_angle = sim.calc_metrics(sim.calc_product(operator_list,
                                                      times_list,
                                                      sign_list,
                                                      0.,
                                                      rotation_error=rot_error))
    # Only Off-resonance error
    err_off_res = sim.calc_metrics(sim.calc_product(operator_list,
                                                    times_list,
                                                    sign_list,
                                                    0.,
                                                    rotation_error=0,
                                                    off_resonance_error=off_res_error))
    # Exact (Trotter)
    no_err = sim.calc_metrics(sim.calc_product(operator_list,
                                               times_list,
                                               sign_list,
                                               0.0))
    return (err_robustCP[1], err_naive[1], err_rot_angle[1], err_off_res[1], no_err[1],
            sum(times_list_robust), 2*len(times_list_robust), sum(times_list), 2*len(times_list))

if not load_data:
    avrg_err = []
    qhw = hw.QuantumHardware()
    for c in c_all:
        print("Nr. of Trotter cycles: ", c)
        parameter = c
        qhw.n = n
        qhw.ion_trap_harm_pot(include_non_commuting_interactions=True)  # Consider an ion trap with harmonic trap potential
        smpl_err = Parallel(n_jobs=nr_jobs)(delayed(calc_sample)(qhw.J, qhw.n, qhw.pauli_string_list, parameter) for sample in all_samples)
        print("Average of average gate infidelity [10e-3]: ")
        print("Robust:                       ", (1 - np.mean(np.array(smpl_err)[:, 0], axis=0)) * 1e3)
        print("Naive :                       ", (1 - np.mean(np.array(smpl_err)[:, 1], axis=0)) * 1e3)
        print("Rotation angle error:         ", (1 - np.mean(np.array(smpl_err)[:, 2], axis=0)) * 1e3)
        print("Off-resonance error:          ", (1 - np.mean(np.array(smpl_err)[:, 3], axis=0)) * 1e3)
        print("exact (Trotter):              ", (1 - np.mean(np.array(smpl_err)[:, 4], axis=0)) * 1e3)
        avrg_err.append(smpl_err)
        with open(filename+'.npy', 'wb') as f:
            np.save(f, np.array(avrg_err))
else:
    with open(filename+'.npy', 'rb') as f:
        avrg_err = np.load(f, allow_pickle=True)
avrg_err = np.array(avrg_err)

labels = ["robust", "naive", "angle error", "off-resonance error", "exact (Trotter)"]
cmap_r = mpl.cm.get_cmap('Reds')
colors = ["#0072B2", cmap_r(0.75), "#E69F00", "#F0E442", "black"]
all_err_mean = []
all_err_var = []
for j in range(len(labels)):
    err_mean = []
    err_var = []
    for i in range(len(c_all)):
        err_mean.append(np.mean(avrg_err[:,:,j][i]))
        err_var.append(np.sqrt(np.var(avrg_err[:,:,j][i])))
    all_err_mean.append(err_mean)
    all_err_var.append(err_var)

fig = plt.figure(figsize=(2, 2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_ylabel(r'$1-F_{\mathrm{avg}}$', labelpad=2)
ax.ticklabel_format(style='sci',scilimits=(-3,-3),axis='y')


ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
for j in range(len(labels)):
    if j==1:
        ax.errorbar(np.array(c_all), (1-np.array(all_err_mean[j])), np.array(all_err_var[j]), label=labels[j],
                    color=colors[j], zorder=10)
    else:
        if j==0:
            ax.errorbar(np.array(c_all), (1-np.array(all_err_mean[j])), np.array(all_err_var[j]), label=labels[j],
                        color=colors[j], zorder=9)
        else:
            ax.errorbar(np.array(c_all), (1-np.array(all_err_mean[j])), np.array(all_err_var[j]), label=labels[j],
                        color=colors[j])
ax.set_xlim(min(c_all)-0.2, max(c_all)+0.2)
ax.set_ylim(1e-4, 1.)
ax.set_xlabel(r'nr. of Trotter cycles $n_{\mathrm{Tro}}$', labelpad=2)
ax.set_yscale('log')
plt.tight_layout()
plt.legend(loc='lower left', bbox_to_anchor=(-0.04,-0.05), prop={'size': 9}, frameon=False)#, bbox_to_anchor=(1,0.62))
plt.savefig(filename+'.pdf', transparent=False, bbox_inches='tight')
plt.show()
