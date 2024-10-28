#!/usr/bin/env python
""" Generate the feasibility plots in Fig. 1

@Author: Pascal Baßler
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from utilities import reshaping as r
from utilities import hardware as hw
########################
# ToDo: Uncomment the relevant parts:

## Figure 1 (a)
N = 2
n_all = range(3,31)
gate_set = 'pauli'
interaction_type = 0
nn_random_interactions = False

# ## Figure 1 (b)
# N = 3
# n_all = range(3,14)
# gate_set = 'pauli'
# interaction_type = 0
# nn_random_interactions = False

# ## Figure 1 (c)
# N = 99  # N is ignored in this case
# n_all = range(3,31)
# gate_set = 'pauli'
# interaction_type = 0
# nn_random_interactions = True
########################

len_fact = 5
all_fact = np.linspace(1.5, 2.5, len_fact)
all_samples = range(50)
filename = "filename"
load_data = False

if not load_data:
    result = np.zeros((0, len_fact))
    runtime = np.zeros((0,len_fact))
    qhw = hw.QuantumHardware(N=N, J_value='ones', interaction_type=interaction_type)
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
                if nn_random_interactions:
                    qhw.non_zero_J(nr_of_interactions=n * n)
                else:
                    qhw.non_zero_J()

                a = r.Reshaping(qhw.n, qhw.J, qhw.pauli_string_list, gate_set=gate_set, generate_V=False)
                start = time.time()
                V, s_list = a.sampleV_pauli(k)
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
