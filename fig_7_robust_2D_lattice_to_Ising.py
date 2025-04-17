#!/usr/bin/env python
""" Generate plots for the simulation with a 2D lattice model in Fig. 7

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
lattice_dim = (2,3)
t_eff = 1.  # effective evolution time
c_all = range(1,5)  # Nr. of Trotter cycles c
rot_error = 0.01  # Rotation angle error
########################

all_samples = range(50)  # Nr. of samples
nr_jobs = 50  # Nr. of threads
filename = "filename"
load_data = False

# Device parameters
J_unit = 1e3 # [Hz] units for J
t_pulse = 1e-7 # [sec] pi-pulse time

def calc_sample(J, n, pauli_string_list, parameter):
    ## Target couplings
    ###################
    gate_set = "pauli"

    qhw = hw.QuantumHardware()
    qhw.n = n
    qhw.pauli_string_list = np.copy(pauli_string_list)
    qhw.J = np.copy(J)

    # 2D lattice model with two- and three-body interactions
    idx_N3 = np.nonzero(np.sum(qhw.pauli_string_list, axis=1) == 3)[0]
    idx_N2 = np.nonzero(np.sum(qhw.pauli_string_list, axis=1) == 2)[0]
    qhw.pauli_string_list[idx_N2] = 3*qhw.pauli_string_list[idx_N2]

    qhw.J[idx_N3] = 0.1  # Coupling coefficient for the three-body interactions
    qhw.J[idx_N2] = 1.  # Coupling coefficient for the two-body interactions
    # Random target two-body couplings
    A = np.zeros(qhw.pauli_string_list.shape[0])
    A[idx_N2] = np.random.uniform(low=0.1, high=1, size=len(idx_N2))

    ## Calculate reshaping times and operations
    ###########################################
    a = r.Reshaping(qhw.n, qhw.J, qhw.pauli_string_list, tol=1e-13, gate_set=gate_set, dimension_factor=4)

    ## Pauli conjugation robust against finite pulse time error and rotation angle error
    times_list_robust, operator_list_robust, sign_list_robust = a.trotter(A,
                                                                          t_pulse=t_pulse * J_unit/t_eff,
                                                                          order=1,
                                                                          cycles=parameter,
                                                                          use_MILP=True)
    times_list_robust = times_list_robust * t_eff / J_unit

    # "naive" Pauli conjugation
    times_list, operator_list, sign_list = a.trotter(A, solver='GLPK', t_pulse=0.0, order=1, cycles=parameter)
    times_list = times_list * t_eff / J_unit

    ## Simulate reshaping
    #####################
    ## randomly sample three-body couplings
    qhw.J[idx_N3] = np.random.uniform(low=-0.1, high=0.1, size=len(idx_N3))
    qhw.J = qhw.J * J_unit
    sim = s.HamiltonianSimulation(qhw, A * t_eff, gate_set)

    # Robust method
    err_robust = sim.calc_metrics(sim.calc_product(operator_list_robust,
                                                   times_list_robust,
                                                   sign_list_robust,
                                                   t_pulse,
                                                   rotation_error=rot_error))
    # Naive method
    err_naive = sim.calc_metrics(sim.calc_product(operator_list,
                                                  times_list,
                                                  sign_list,
                                                  t_pulse,
                                                  rotation_error=rot_error))
    # Only rotation angle error
    err_rot = sim.calc_metrics(sim.calc_product(operator_list,
                                                times_list,
                                                sign_list,
                                                0.0,
                                                rotation_error=rot_error))
    # Exact (Trotter)
    no_err = sim.calc_metrics(sim.calc_product(operator_list,
                                               times_list,
                                               sign_list,
                                               0.0))
    return (err_robust[1], err_naive[1], err_rot[1], no_err[1],
            sum(times_list_robust), 2*len(times_list_robust), sum(times_list), 2*len(times_list))


if not load_data:
    avrg_err = []
    qhw = hw.QuantumHardware(N=3)
    for c in c_all:
        print("Nr. of Trotter cycles: ", c)
        parameter = c
        qhw.square_lattice_2D(lattice_dim)  # Consider an ion trap with harmonic trap potential
        smpl_err = Parallel(n_jobs=nr_jobs)(delayed(calc_sample)(qhw.J, qhw.n, qhw.pauli_string_list, parameter) for sample in all_samples)
        print("Average of average gate infidelity [10e-3]: ")
        print("Robust:                 ", (1 - np.mean(np.array(smpl_err)[:, 0], axis=0)) * 1e3)
        print("Naive :                 ", (1 - np.mean(np.array(smpl_err)[:, 1], axis=0)) * 1e3)
        print("Rotation angle error:   ", (1 - np.mean(np.array(smpl_err)[:, 2], axis=0)) * 1e3)
        print("exact (Trotter):        ", (1 - np.mean(np.array(smpl_err)[:, 3], axis=0)) * 1e3)
        avrg_err.append(smpl_err)
        with open(filename+'.npy', 'wb') as f:
            np.save(f, np.array(avrg_err))
else:
    with open(filename+'.npy', 'rb') as f:
        avrg_err = np.load(f, allow_pickle=True)

avrg_err = np.array(avrg_err)


labels = ["robust Pauli", "naive", "angle error", "exact (Trotter)"]
cmap_b = mpl.cm.get_cmap('Blues')
cmap_g = mpl.cm.get_cmap('Greens')
cmap_r = mpl.cm.get_cmap('Reds')
cmap_p = mpl.cm.get_cmap('Purples')
colors = ["#009E73", cmap_r(0.75), "#E69F00", "black"]
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

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
for j in range(len(labels)):
    if j==0:
        ax.errorbar(np.array(c_all), (1-np.array(all_err_mean[j])), np.array(all_err_var[j]), label=labels[j],
                    color=colors[j], zorder=5)
    else:
        ax.errorbar(np.array(c_all), (1 - np.array(all_err_mean[j])), np.array(all_err_var[j]), label=labels[j],
                    color=colors[j])
ax.set_xlim(min(c_all)-0.1, max(c_all)+0.1)
ax.set_ylim(5*1e-5, 3*1e-2)
ax.set_xlabel(r'nr. of Trotter cycles $n_{\mathrm{Tro}}$', labelpad=2)
ax.set_yscale('log')
plt.tight_layout()
plt.legend(loc='upper right', bbox_to_anchor=(1.7,1.), prop={'size': 9}, frameon=False)
plt.savefig(filename+'.pdf', transparent=False, bbox_inches='tight')
plt.show()
