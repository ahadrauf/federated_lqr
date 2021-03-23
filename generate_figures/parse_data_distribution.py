import numpy as np
import matplotlib.pyplot as plt
from utils import *

file_name10Q10R = "../data/20210318_03_02_42_fedadmm_v2_with_centralized.npy"
file_name1Q10R = "../data/20210318_03_02_22_fedadmm_v2_with_centralized.npy"
file_name10Q1R = "../data/20210318_03_02_53_fedadmm_v2_with_centralized.npy"
# file_name10Q10R = "data/20210317_18_56_09_fedadmm_v2_with_centralized.npy"
# file_name1Q10R = "data/20210317_20_19_21_fedadmm_v2_with_centralized.npy"
# file_name10Q1R = "data/20210317_20_20_24_fedadmm_v2_with_centralized.npy"
# file_name10Q10R = "data/20210317_18_56_09_fedadmm_v2_with_centralized.npy"
# file_name1Q10R = "data/20210317_18_55_03_fedadmm_v2_with_centralized.npy"
# file_name10Q1R = "data/20210317_01_10_58_fedadmm_v2_with_centralized.npy"
# file_name1Q1R = "data/20210315_20_59_15_fedadmm_v2_with_centralized.npy"
file_name1Q1R = "../data/20210315_20_59_15_fedadmm_v2_with_centralized.npy"
file_names = ['data/20210318_03_02_42_fedadmm_v2_with_centralized.npy',
              # 'data/20210318_03_02_22_fedadmm_v2_with_centralized.npy',
              'data/20210318_00_58_46_fedadmm_v2_with_centralized.npy',
              # 'data/20210316_23_38_24_fedadmm_v2_with_centralized.npy',
              'data/20210318_03_02_53_fedadmm_v2_with_centralized.npy',
              'data/20210317_20_20_24_fedadmm_v2_with_centralized.npy',
              'data/20210317_20_19_21_fedadmm_v2_with_centralized.npy']

n, m = 4, 2
M = 5
N = 5
traj_range = np.arange(1, 16)
Wctrl = 3
Wdyn = 1
W = Wdyn * np.eye(n)
VQ = np.eye(n) / n
VR = np.eye(m) / m

Q_trues = []
R_trues = []
Q_learned = []
R_learned = []
xs = []
ys = []

for i, file_name in enumerate(file_names):
    data = np.load(file_name, allow_pickle=True)
    A, B, K_trues, P_trues, _Q_trues, _R_trues, costs_lr, costs_admm, costs_centralized, costs_fedadmm, costs_pfedadmm, \
    out_lr, out_admm, out_centralized, out_fedadmm, out_pfedadmm, costs_lr_K, costs_admm_KQR, costs_centralized_KQR, \
    costs_fedadmm_KQR, costs_pfedadmm_KQR = data
    Q_trues.append(_Q_trues)
    R_trues.append(_R_trues)
    traj_length = len(costs_lr)
    Q_learned.append(np.mean([out_pfedadmm[(traj_length - 1, 1, robot)][2] for robot in range(M)], axis=0))
    R_learned.append(np.mean([out_pfedadmm[(traj_length - 1, 1, robot)][3] for robot in range(M)], axis=0))
    print()

# fig = plt.figure()

color = ['blue', 'green', 'purple', 'red', 'orange']
markers = ['o', '+', 's', '^', '<']
for i in range(len(Q_trues)):
    learned = plt.scatter([np.linalg.norm(Q_learned[i], 2) / np.linalg.norm(R_learned[i], 2)],
                [np.linalg.norm(R_learned[i], 2)],
                s=200, c=color[i], marker=markers[i], edgecolors='k', alpha=0.75)
    real = plt.scatter([np.linalg.norm(Q, 2) / np.linalg.norm(R, 2) for Q, R, in zip(Q_trues[i], R_trues[i])],
                [np.linalg.norm(R, 2) for R in R_trues[i]],
                c=color[i], marker=markers[i], edgecolors='k')
    vline = plt.axvline(np.mean([np.linalg.norm(Q, 2) / np.linalg.norm(R, 2) for Q, R, in zip(Q_trues[i], R_trues[i])]),
                color='k', linestyle='--')
    if i == 0:
        plt.legend([real, vline, learned], ['Robot Parameters', 'Average Robot Parameters', 'Learned Distribution Mean'])

plt.semilogx(True)
plt.title("Learning Distributions, 20 Demonstrations, M = 5 Robots")
plt.xlabel(r"$||Q||_2 / ||R||_2$")
plt.ylabel(r"$||R||_2$")
plt.show()
