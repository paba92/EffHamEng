#!/usr/bin/env python
""" Generate the feasibility plots in Fig. 3

@Author: Pascal Baßler
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from utilities import reshaping as r
from utilities import hardware as hw
########################
# ToDo: Uncomment the relevant parts:

## Figure 3 (a)
N = 2
n_all = range(3,31)
gate_set = 'sqrtsqrt'
interaction_type = 0

# ## Figure 3 (b)
# N = 3
# n_all = range(3,14)
# gate_set = 'sqrtsqrt'
# interaction_type = 0

# ## Figure 3 (c)
# N = 5
# n_all = range(5,31)
# gate_set = 'sqrtsqrt'
# interaction_type = 10
########################

len_fact = 5
all_fact = np.linspace(1.5, 2.5, len_fact)
all_samples = range(50)
filename = "filename"
load_data = False

if not load_data:
    result = np.zeros((0, len_fact))
    runtime = np.zeros((0,len_fact))
    qhw = hw.QuantumHardware(N=N, J_value='random', interaction_type=interaction_type)
    for n in n_all:
        avrg_result = []
        avrg_runtime = []
        qhw.n = n
        for k in all_fact:
            print("n: ", n, "   k: ", k)
            res = []
            rt = []
            for sample in all_samples:
                print("n: ", n, "   k: ", k, "    sample: ", sample)
                qhw.random_non_zero_J(random_sparsity=True)
                ## If certain qubits do not interact at all, reduce nr. of qubits
                supp = np.sum(qhw.pauli_string_list, axis=0).astype(bool)
                n_red = sum(supp)

                a = r.Reshaping(n_red, qhw.J, qhw.pauli_string_list[:,supp], gate_set=gate_set, generate_V=False)
                start = time.time()
                V, s_list = a.sampleV_clifford(k, save_sign_permutation=False)
                res.append(int(a.is_V_feasible(V)))
                rt.append(time.time() - start)
            avrg_result.append(np.mean(res))
            avrg_runtime.append(np.mean(rt))
            print(avrg_result)
        result = np.vstack((result,np.array(avrg_result)))
        runtime = np.vstack((runtime,np.array(avrg_runtime)))
        with open('data.npy', 'wb') as f:
            np.save(f, np.array([result, runtime]))
else:
    with open('data.npy', 'rb') as f:
        data = np.load(f, allow_pickle=True)
    result = data[0,:]
    runtime = data[1, :]

plt.imshow(np.flip(result.T, axis=0))
plt.yticks(ticks=range(len(all_fact)), labels=np.flip(all_fact))#, rotation=90)
plt.xticks(ticks=range(len(n_all)), labels=list(n_all))
plt.xlabel("n")
plt.ylabel("r/d")
plt.colorbar()
plt.savefig(filename+'.pdf', transparent=False, bbox_inches='tight')
plt.show()
