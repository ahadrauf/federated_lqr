import numpy as np
import matplotlib.pyplot as plt
from utils import *

M5 = "data/20210316_20_03_16_fedadmm_v2_with_centralized.npy"
M10 = "data/20210317_02_43_07_fedadmm_v2_with_centralized.npy"
M15 = "data/20210317_03_47_48_fedadmm_v2_with_centralized.npy"
M20 = "data/20210316_23_37_57_fedadmm_v2_with_centralized.npy"
M25 = "data/20210317_02_43_24_fedadmm_v2_with_centralized.npy"

n, m = 4, 2
M = 5
N = 5
traj_range = np.arange(1, 21)
Wctrl = 3
Wdyn = 1
W = Wdyn*np.eye(n)
VQ = np.eye(n)/n
VR = np.eye(m)/m

x = [1, 2, 3, 4, 5]

ys = []
ks = []
qs = []
rs = []

data = np.load(M5, allow_pickle=True)
A, B, K_trues, P_trues, Q_trues, R_trues, costs_lr, costs_admm, costs_centralized, costs_fedadmm, costs_pfedadmm,\
out_lr, out_admm, out_centralized, out_fedadmm, out_pfedadmm, costs_lr_K, costs_admm_KQR, costs_centralized_KQR,\
costs_fedadmm_KQR, costs_pfedadmm_KQR = data
ys.append(costs_pfedadmm[-1])
ks.append(costs_pfedadmm_KQR[-1]['K'])
qs.append(costs_pfedadmm_KQR[-1]['Q'])
rs.append(costs_pfedadmm_KQR[-1]['R'])
cost_true = np.nanmean([np.trace(P_trues[i]@W) for i in range(M)], axis=0)

data = np.load(M10, allow_pickle=True)
A, B, K_trues, P_trues, Q_trues, R_trues, costs_lr, costs_admm, costs_centralized, costs_fedadmm, costs_pfedadmm,\
out_lr, out_admm, out_centralized, out_fedadmm, out_pfedadmm, costs_lr_K, costs_admm_KQR, costs_centralized_KQR,\
costs_fedadmm_KQR, costs_pfedadmm_KQR = data
ys.append(costs_pfedadmm[-1])
ks.append(costs_pfedadmm_KQR[-1]['K'])
qs.append(costs_pfedadmm_KQR[-1]['Q'])
rs.append(costs_pfedadmm_KQR[-1]['R'])



data = np.load(M15, allow_pickle=True)
A, B, K_trues, P_trues, Q_trues, R_trues, costs_lr, costs_admm, costs_centralized, costs_fedadmm, costs_pfedadmm,\
out_lr, out_admm, out_centralized, out_fedadmm, out_pfedadmm, costs_lr_K, costs_admm_KQR, costs_centralized_KQR,\
costs_fedadmm_KQR, costs_pfedadmm_KQR = data
ys.append(costs_pfedadmm[-1] + 0.)
ks.append(costs_pfedadmm_KQR[-1]['K'])
qs.append(costs_pfedadmm_KQR[-1]['Q'])
rs.append(costs_pfedadmm_KQR[-1]['R'])


data = np.load(M20, allow_pickle=True)
A, B, K_trues, P_trues, Q_trues, R_trues, costs_lr, costs_admm, costs_centralized, costs_fedadmm, costs_pfedadmm,\
out_lr, out_admm, out_centralized, out_fedadmm, out_pfedadmm, costs_lr_K, costs_admm_KQR, costs_centralized_KQR,\
costs_fedadmm_KQR, costs_pfedadmm_KQR = data
ys.append(costs_pfedadmm[-1])
ks.append(costs_pfedadmm_KQR[-1]['K'])
qs.append(costs_pfedadmm_KQR[-1]['Q'])
rs.append(costs_pfedadmm_KQR[-1]['R'])


data = np.load(M25, allow_pickle=True)
A, B, K_trues, P_trues, Q_trues, R_trues, costs_lr, costs_admm, costs_centralized, costs_fedadmm, costs_pfedadmm,\
out_lr, out_admm, out_centralized, out_fedadmm, out_pfedadmm, costs_lr_K, costs_admm_KQR, costs_centralized_KQR,\
costs_fedadmm_KQR, costs_pfedadmm_KQR = data
ys.append(costs_pfedadmm[-1])
ks.append(costs_pfedadmm_KQR[-1]['K'])
qs.append(costs_pfedadmm_KQR[-1]['Q'])
rs.append(costs_pfedadmm_KQR[-1]['R'])


fig = plt.figure()
# plt.scatter([np.linalg.norm(x) for x in Q_trues1Q1R], [np.linalg.norm(x) for x in R_trues1Q1R], c="blue", marker='o', alpha=0.5)
# plt.scatter([np.linalg.norm(Q_pfedadmm1Q1R)], [np.linalg.norm(R_pfedadmm1Q1R)], s=100, c="blue", marker="o")
#
# plt.scatter([np.linalg.norm(x) for x in Q_trues10Q1R], [np.linalg.norm(x) for x in R_trues10Q1R], c="green", marker='+', alpha=0.5)
# plt.scatter([np.linalg.norm(Q_pfedadmm10Q1R)], [np.linalg.norm(R_pfedadmm10Q1R)], s=100, c="green", marker="+")
#
# plt.scatter([np.linalg.norm(x) for x in Q_trues1Q10R], [np.linalg.norm(x) for x in R_trues1Q10R], c="purple", marker='s', alpha=0.5)
# plt.scatter([np.linalg.norm(Q_pfedadmm1Q10R)], [np.linalg.norm(R_pfedadmm1Q10R)], s=100, c="purple", marker="s")
#
# plt.scatter([np.linalg.norm(x) for x in Q_trues10Q10R], [np.linalg.norm(x) for x in R_trues10Q10R], c="red", marker='^', alpha=0.5)
# # plt.scatter([np.linalg.norm(Q_pfedadmm10Q10R)], [np.linalg.norm(R_pfedadmm10Q10R)], s=100, c="red", marker="^")
# plt.boxplot(ks, labels=x)
# plt.boxplot(qs, labels=x)
# plt.boxplot(rs, labels=x)

plt.boxplot(ys, labels=x)
plt.axhline(cost_true, ls='--', c='k', label='Optimal')
plt.legend()
plt.title("LQR Cost After 20 Demonstrations, 5 Robots")
plt.xlabel(r"Communication Interval $H$")
plt.ylabel("LQR Cost")

fig2 = plt.figure()
bytes = np.divide(len(traj_range), x)*(n**2 + m**2)*M
bytes_saved = np.divide(bytes, len(traj_range)*(n+m)*M*N)
plt.plot(x, bytes_saved)
plt.xlabel(r"Communication Interval $H$")
plt.ylabel("# Bytes Sent / # Bytes Sent via Centralized Learning")
plt.title("Bandwidth Savings of Federated Learning")
plt.grid(True)
plt.show()
