import numpy as np
import matplotlib.pyplot as plt
from utils import *

file_name10Q10R = "data/20210317_18_56_09_fedadmm_v2_with_centralized.npy"
file_name1Q10R = "data/20210317_20_19_21_fedadmm_v2_with_centralized.npy"
file_name10Q1R = "data/20210317_20_20_24_fedadmm_v2_with_centralized.npy"
# file_name10Q10R = "data/20210317_18_56_09_fedadmm_v2_with_centralized.npy"
# file_name1Q10R = "data/20210317_18_55_03_fedadmm_v2_with_centralized.npy"
# file_name10Q1R = "data/20210317_01_10_58_fedadmm_v2_with_centralized.npy"
file_name1Q1R = "data/20210315_20_59_15_fedadmm_v2_with_centralized.npy"

n, m = 4, 2
M = 5
N = 5
traj_range = np.arange(1, 16)
Wctrl = 3
Wdyn = 1
W = Wdyn*np.eye(n)
VQ = np.eye(n)/n
VR = np.eye(m)/m

data = np.load(file_name1Q1R, allow_pickle=True)
A, B, K_trues, P_trues, Q_trues, R_trues, costs_lr, costs_admm, costs_centralized, costs_fedadmm, costs_pfedadmm,\
out_lr, out_admm, out_centralized, out_fedadmm, out_pfedadmm, costs_lr_K, costs_admm_KQR, costs_centralized_KQR,\
costs_fedadmm_KQR, costs_pfedadmm_KQR = data
Q_trues1Q1R = Q_trues
R_trues1Q1R = R_trues
Q_pfedadmm1Q1R = np.mean([out_pfedadmm[(traj_range[-1], 1, robot)][2] for robot in range(M)], axis=0)
R_pfedadmm1Q1R = np.mean([out_pfedadmm[(traj_range[-1], 1, robot)][3] for robot in range(M)], axis=0)

data = np.load(file_name10Q1R, allow_pickle=True)
A, B, K_trues, P_trues, Q_trues, R_trues, costs_lr, costs_admm, costs_centralized, costs_fedadmm, costs_pfedadmm,\
out_lr, out_admm, out_centralized, out_fedadmm, out_pfedadmm, costs_lr_K, costs_admm_KQR, costs_centralized_KQR,\
costs_fedadmm_KQR, costs_pfedadmm_KQR = data
Q_trues10Q1R = Q_trues
R_trues10Q1R = R_trues
Q_pfedadmm10Q1R = np.mean([out_pfedadmm[(traj_range[-1], 1, robot)][2] for robot in range(M)], axis=0)
R_pfedadmm10Q1R = np.mean([out_pfedadmm[(traj_range[-1], 1, robot)][3] for robot in range(M)], axis=0)

data = np.load(file_name1Q10R, allow_pickle=True)
A, B, K_trues, P_trues, Q_trues, R_trues, costs_lr, costs_admm, costs_centralized, costs_fedadmm, costs_pfedadmm,\
out_lr, out_admm, out_centralized, out_fedadmm, out_pfedadmm, costs_lr_K, costs_admm_KQR, costs_centralized_KQR,\
costs_fedadmm_KQR, costs_pfedadmm_KQR = data
Q_trues1Q10R = Q_trues
R_trues1Q10R = R_trues
Q_pfedadmm1Q10R = np.mean([out_pfedadmm[(traj_range[-1], 1, robot)][2] for robot in range(M)], axis=0)
R_pfedadmm1Q10R = np.mean([out_pfedadmm[(traj_range[-1], 1, robot)][3] for robot in range(M)], axis=0)

data = np.load(file_name10Q10R, allow_pickle=True)
A, B, K_trues, P_trues, Q_trues, R_trues, costs_lr, costs_admm, costs_centralized, costs_fedadmm, costs_pfedadmm,\
out_lr, out_admm, out_centralized, out_fedadmm, out_pfedadmm, costs_lr_K, costs_admm_KQR, costs_centralized_KQR,\
costs_fedadmm_KQR, costs_pfedadmm_KQR = data
Q_trues10Q10R = Q_trues
R_trues10Q10R = R_trues
Q_pfedadmm10Q10R = np.mean([out_pfedadmm[(traj_range[-1], 1, robot)][2] for robot in range(M)], axis=0)
R_pfedadmm10Q10R = np.mean([out_pfedadmm[(traj_range[-1], 1, robot)][3] for robot in range(M)], axis=0)

fig = plt.figure()
plt.scatter([np.linalg.norm(x) for x in Q_trues1Q1R], [np.linalg.norm(x) for x in R_trues1Q1R], c="blue", marker='o', alpha=0.5)
plt.scatter([np.linalg.norm(Q_pfedadmm1Q1R)], [np.linalg.norm(R_pfedadmm1Q1R)], s=100, c="blue", marker="o")

plt.scatter([np.linalg.norm(x) for x in Q_trues10Q1R], [np.linalg.norm(x) for x in R_trues10Q1R], c="green", marker='+', alpha=0.5)
plt.scatter([np.linalg.norm(Q_pfedadmm10Q1R)], [np.linalg.norm(R_pfedadmm10Q1R)], s=100, c="green", marker="+")

plt.scatter([np.linalg.norm(x) for x in Q_trues1Q10R], [np.linalg.norm(x) for x in R_trues1Q10R], c="purple", marker='s', alpha=0.5)
plt.scatter([np.linalg.norm(Q_pfedadmm1Q10R)], [np.linalg.norm(R_pfedadmm1Q10R)], s=100, c="purple", marker="s")

plt.scatter([np.linalg.norm(x) for x in Q_trues10Q10R], [np.linalg.norm(x) for x in R_trues10Q10R], c="red", marker='^', alpha=0.5)
# plt.scatter([np.linalg.norm(Q_pfedadmm10Q10R)], [np.linalg.norm(R_pfedadmm10Q10R)], s=100, c="red", marker="^")

plt.show()
