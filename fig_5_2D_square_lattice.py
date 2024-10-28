#!/usr/bin/env python
""" Generate heuristic optimality plots for 2D square lattices in Fig. 5

@Author: Pascal Baßler
"""
import numpy as np
import time
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt

from utilities import reshaping as r
from utilities import hardware as hw

N = 2
sqrt_n_all = range(2,16)
gate_set = 'pauli'

all_dimension_factors = range(3,7)
all_samples = range(50)
filename = "filename"
load_data = False

if not load_data:
    qhw = hw.QuantumHardware(N=N, J_value="ones")
    all_lam_a = []
    for sqrt_n in sqrt_n_all:
        avrg_lam_a = []
        for sample in all_samples:
            qhw.square_lattice_2D((sqrt_n, sqrt_n), commuting=False)
            ## Target couplings
            A = np.random.uniform(low=-1, high=1, size=len(qhw.J))
            dim_lam_a = []
            for dimension_factor in all_dimension_factors:
                print("n: ", qhw.n, "   Sample: ", sample, "   Dim. Fact: ", dimension_factor)
                a = r.Reshaping(qhw.n, qhw.J, qhw.pauli_string_list, gate_set=gate_set, dimension_factor=dimension_factor)
                start = time.time()
                lam, used_pauli_strings = a.LP(A)
                runtime = time.time() - start
                print("runtime: ", runtime)
                dim_lam_a.append([sum(lam), runtime])
            avrg_lam_a.append(dim_lam_a)
        all_lam_a.append(avrg_lam_a)
        with open(filename+'.npy', 'wb') as f:
            np.save(f, np.array(all_lam_a))
else:
    with open(filename + '.npy', 'rb') as f:
        all_lam_a = np.load(f, allow_pickle=True)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)

n_all = [i**2 for i in sqrt_n_all]

mean_a = []
var_a = []
for idx, j in enumerate(all_lam_a):
    mean_a.append(np.mean(j, axis=0))
    var_a.append(np.sqrt(np.var(j, axis=0)))

cmap_r = mpl.cm.get_cmap('Reds')

mean_a = np.array(mean_a)
var_a = np.array(var_a)
for i in range(mean_a.shape[1]):  # for over dim factor
    ax1.errorbar(n_all, mean_a[:, i, 0], var_a[:, i, 0], label=f'$r=$' + str(all_dimension_factors[i]) + f'$d$',
                 color=cmap_r(0.25 + 0.25 * i))
    ax2.errorbar(n_all, mean_a[:, i, 1], var_a[:, i, 1], label=f'$r=$' + str(all_dimension_factors[i]) + f'$d$',
                 color=cmap_r(0.25 + 0.25 * i))

ax1.set_ylabel(f'$\sum \lambda$')
ax1.set_xlabel("n")
ax2.set_ylabel('runtime[sec]')
ax2.set_xlabel("n")

plt.tight_layout()
plt.legend()
plt.savefig(filename + '.pdf', transparent=False, bbox_inches='tight')
plt.show()
