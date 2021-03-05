import numpy as np
from scipy.linalg import solve_discrete_are
import cvxpy as cp
from ADMM_v2 import *
from utils import *
import matplotlib.pyplot as plt
import warnings

# warnings.filterwarnings("ignore")
from datetime import datetime
import time

M = 10
n, m = 4, 2
nrandom = 1
covarQR = 0.1
niter = 3
Wdyn = 3
Wctrl = 5
# A = np.array([
#     [.99997, .039, 0, -.322],
#     [-.065, .99681, 7.74, 0],
#     [.02, -.101, .99571, 0],
#     [0, 0, 1, 1]
# ])
# B = np.array([
#     [.0001, 0],
#     [-.0018, -.0004],
#     [-.0116, .00598],
#     [0, 0]
# ])
# W = .1*np.array([
#     [1.00092109, -0.02610491, 0.016055, 0.],
#     [-0.02610491, 0.99785518, -0.10197781, 0.],
#     [0.016055, -0.10197781, 0.010601, 0.],
#     [0., 0., 0., 0.]
# ])
# A = np.random.randn(n, n)
# A = A / np.abs(np.linalg.eig(A)[0]).max()
# B = np.random.randn(n, m)
A = np.array([[0.13741826, 0.02876351, 0.50096039, 0.24711964],
              [-0.08638155, -0.29891056, 0.16163153, 0.24719121],
              [0.06460169, 0.24025215, -0.29127423, -0.28410344],
              [0.18903618, 0.43957661, -0.43899608, -0.38508461]])
B = np.array([[1.12563572, -0.59218407],
              [-0.01612495, -0.18979434],
              [-1.38525459, -0.51277622],
              [-1.43032195, 2.17007322]])
W = Wdyn*np.eye(n)

Q_trues = []
R_trues = []
P_trues = []
K_trues = []

ATA = lambda A: A.T@A
for i in range(M):
    Q_trues.append(np.eye(n) + covarQR*ATA(np.random.rand(n, n)))
    R_trues.append(np.eye(m) + covarQR*ATA(np.random.rand(m, m)))
    P_trues.append(solve_discrete_are(A, B, Q_trues[i], R_trues[i]))
    K_trues.append(-np.linalg.solve(R_trues[i] + B.T@P_trues[i]@B, B.T@P_trues[i]@A))
print("Q_trues", Q_trues)
print("R_trues", R_trues)
print("K_trues", K_trues)


def simulate(K, i, N=10, seed=None, add_noise=False, train=False):
    if seed is not None:
        np.random.seed(seed)
    if train:
        x = np.random.randint(-100, 100, size=(n,))
    else:
        x = np.random.multivariate_normal(np.zeros(n), W)
    xs = []
    us = []
    cost = 0.0
    for _ in range(N):
        u = K@x
        if add_noise:
            u += Wctrl*np.random.randn(m)
        xs.append(x)
        us.append(u)
        cost += (x@Q_trues[i]@x + u@R_trues[i]@u)/N
        x = A@x + B@u + np.random.multivariate_normal(np.zeros(n), W)
    xs = np.array(xs)
    us = np.array(us)

    return cost, xs, us


def simulate_cost(K, i, N=10, seed=None, add_noise=False, N_test=1000, train=False):
    tests = [simulate(K, i, N=N, seed=seed, add_noise=add_noise, train=train) for _ in range(N_test)]
    return np.nanmean([cost for cost, _, _ in tests])


N_test = 10000
# cost_true = np.nanmean([np.trace(P_trues[i]@W) for i in range(M)], axis=0)
# cost_noise = np.nanmean([np.trace(P_trues[0]@(W + Wctrl*Wctrl*B@B.T)) for i in range(M)], axis=0)
cost_true = np.nanmean([simulate_cost(K_trues[i], i, N=N_test, add_noise=False, N_test=1) for i in range(M)], axis=0)
cost_noise = np.nanmean([simulate_cost(K_trues[i], i, N=N_test, add_noise=True, N_test=1) for i in range(M)], axis=0)
print("Cost true: {}, cost noise: {}".format(cost_true, cost_noise))

Q_avg = np.mean(Q_trues, axis=0)
R_avg = np.mean(R_trues, axis=0)
cost_LQ_true = np.nanmean([np.linalg.norm([ATA(1./n/n*np.random.randn(n, n)) - Q_avg]) for _ in range(
    N_test)], axis=0)
cost_LR_true = np.nanmean([np.linalg.norm([ATA(1./n/n*np.random.randn(m, m)) - R_avg]) for _ in range(
    N_test)], axis=0)
cost_fLQ_true = np.nanmean([np.linalg.norm(Q_trues[i] - Q_avg) for i in range(M)])
cost_fLR_true = np.nanmean([np.linalg.norm(R_trues[i] - R_avg) for i in range(M)])

print("Baseline error on LQ: {}, on LR: {}".format(cost_LQ_true, cost_LR_true))
print("Error for true federated learning on LQ: {}, on LR: {}".format(cost_fLQ_true, cost_fLR_true))

# For saving files
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H_%M_%S")
print(timestamp)

costs_lr = []
costs_admm = []
costs_fedadmm = []
out_lr = {}
out_admm = {}
out_fedadmm = {}
costs_lr_K = []
costs_admm_KQR = []
costs_fedadmm_KQR = []
Ns = np.arange(1, 22, 2)
seed_range = np.arange(1, 2)


def plot_losses(costs_lr, costs_admm, costs_fedadmm, verbose=False):
    costs_lr = np.array(costs_lr)
    costs_admm = np.array(costs_admm)
    costs_fedadmm = np.array(costs_fedadmm)

    np.save('data/' + timestamp + "_fedadmm_v2.npy", [costs_lr, costs_admm, costs_fedadmm,
                                                      out_lr, out_admm, out_fedadmm,
                                                      costs_lr_K, costs_admm_KQR, costs_fedadmm_KQR])

    mean_lr = np.nanmean(costs_lr, axis=1)
    std_lr = np.nanstd(costs_lr, axis=1)
    mean_admm = np.nanmean(costs_admm, axis=1)
    std_admm = np.nanstd(costs_admm, axis=1)
    mean_fedadmm = np.nanmean(costs_fedadmm, axis=1)
    std_fedadmm = np.nanstd(costs_fedadmm, axis=1)

    idx = np.arange(len(costs_lr_K))
    mean_lr_K = {'K': np.array([np.nanmean(costs_lr_K[i]['K']) for i in idx])}
    std_lr_K = {'K': np.array([np.nanstd(costs_lr_K[i]['K']) for i in idx])}
    mean_admm_KQR = {k: np.array([np.nanmean(costs_admm_KQR[i][k]) for i in idx]) for k in 'KQR'}
    std_admm_KQR = {k: np.array([np.nanstd(costs_admm_KQR[i][k]) for i in idx]) for k in 'KQR'}
    mean_fedadmm_KQR = {k: np.array([np.nanmean(costs_fedadmm_KQR[i][k]) for i in idx]) for k in 'KQR'}
    std_fedadmm_KQR = {k: np.array([np.nanstd(costs_fedadmm_KQR[i][k]) for i in idx]) for k in 'KQR'}

    if verbose:
        print("Mean LR", mean_lr)
        print("Mean ADMM", mean_admm)
        print("Mean FedADMM", mean_fedadmm)

    latexify(fig_width=6*2.5, fig_height=2.7*2.5)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].axhline(cost_noise, ls='--', c='k', label='expert (with noise)')
    axs[0, 0].axhline(cost_true, ls='-', c='k', label='optimal (without noise)')
    axs[0, 0].scatter(idx, mean_lr, s=8, marker='o', c='blue', label='PF')
    axs[0, 0].fill_between(idx, mean_lr - std_lr/3, mean_lr + std_lr/3, alpha=.5, color='blue')
    axs[0, 0].scatter(idx, mean_admm, s=8, marker='*', c='green', label='ADMM')
    axs[0, 0].fill_between(idx, mean_admm - std_admm/3, mean_admm + std_admm/3, alpha=.5, color='green')
    axs[0, 0].scatter(idx, mean_fedadmm, s=8, marker='*', c='purple', label='FedADMM')
    axs[0, 0].fill_between(idx, mean_fedadmm - std_fedadmm/3, mean_fedadmm + std_fedadmm/3, alpha=.3, color='purple')
    axs[0, 0].semilogy()
    axs[0, 0].axhline(cost_true, ls='-', c='k', label='optimal')
    axs[0, 0].set_ylabel('cost')
    axs[0, 0].set_xlabel('demonstrations')
    axs[0, 0].set_title("Cost vs. Trajectory Length")
    axs[0, 0].legend()

    # Plot K
    axs[1, 0].scatter(idx, mean_lr_K['K'], s=8, marker='o', c='cyan', label='policy fitting')
    axs[1, 0].fill_between(idx, mean_lr_K['K'] - std_lr_K['K']/3, mean_lr_K['K'] + std_lr_K['K']/3, alpha=.5,
                           color='cyan')
    axs[1, 0].scatter(idx, mean_admm_KQR['K'], s=8, marker='o', c='green', label='ADMM')
    axs[1, 0].fill_between(idx, mean_admm_KQR['K'] - std_admm_KQR['K']/3, mean_admm_KQR['K'] + std_admm_KQR['K']/3/3,
                           alpha=.5, color='green')
    axs[1, 0].scatter(idx, mean_fedadmm_KQR['K'], s=8, marker='o', c='purple', label='FedADMM')
    axs[1, 0].fill_between(idx, mean_fedadmm_KQR['K'] - std_fedadmm_KQR['K']/3,
                           mean_fedadmm_KQR['K'] + std_fedadmm_KQR['K']/3,
                           alpha=.5, color='purple')
    axs[1, 0].grid(True)
    axs[1, 0].set_xlabel(r"Trajectory Number $\tau$")
    axs[1, 0].set_ylabel(r'$||K - K_{true}||$')
    axs[1, 0].set_title('K Loss, N=' + str(N) + ', M=' + str(M))
    axs[1, 0].legend()

    # Plot Q Loss
    axs[0, 1].axhline(cost_LQ_true, ls='--', c='k', label='Random Guessing')
    axs[0, 1].scatter(idx, mean_admm_KQR['Q'], s=8, marker='o', c='green', label='ADMM')
    axs[0, 1].fill_between(idx, mean_admm_KQR['Q'] - std_admm_KQR['Q']/3, mean_admm_KQR['Q'] + std_admm_KQR['Q']/3/3,
                           alpha=.5, color='green')
    axs[0, 1].scatter(idx, mean_fedadmm_KQR['Q'], s=8, marker='o', c='purple', label='FedADMM')
    axs[0, 1].fill_between(idx, mean_fedadmm_KQR['Q'] - std_fedadmm_KQR['Q']/3,
                           mean_fedadmm_KQR['Q'] + std_fedadmm_KQR['Q']/3,
                           alpha=.5, color='purple')
    axs[0, 1].grid(True)
    axs[0, 1].set_xlabel(r'Trajectory Number $\tau$')
    axs[0, 1].set_ylabel(r'$||Q - Q_{true}||$')
    axs[0, 1].set_title('Q Loss, N=' + str(N) + ', M=' + str(M))
    axs[0, 1].legend()

    # Plot R Loss
    axs[1, 1].axhline(cost_LR_true, ls='--', c='k', label='Random Guessing')
    axs[1, 1].scatter(idx, mean_admm_KQR['R'], s=8, marker='o', c='green', label='ADMM')
    axs[1, 1].fill_between(idx, mean_admm_KQR['R'] - std_admm_KQR['R']/3, mean_admm_KQR['R'] + std_admm_KQR['R']/3/3,
                           alpha=.5, color='green')
    axs[1, 1].scatter(idx, mean_fedadmm_KQR['R'], s=8, marker='o', c='purple', label='FedADMM')
    axs[1, 1].fill_between(idx, mean_fedadmm_KQR['R'] - std_fedadmm_KQR['R']/3,
                           mean_fedadmm_KQR['R'] + std_fedadmm_KQR['R']/3,
                           alpha=.5, color='purple')
    axs[1, 1].grid(True)
    axs[1, 1].set_xlabel(r'Trajectory Number $\tau$')
    axs[1, 1].set_ylabel(r'$||R - R_{true}||$')
    axs[1, 1].set_title('R Loss, N=' + str(N) + ', M=' + str(M))
    axs[1, 1].legend()

    plt.tight_layout()
    fig_name = "figures/" + timestamp + "_fedadmm_v2_random_M={}_Wctrl={}_Wdyn={}_nrandom={}_covarQR={}_niter=" \
                                        "{}_nseed={}_Nmax={}".format(
        M, Wctrl, Wdyn, nrandom, covarQR, niter, len(seed_range), Ns[-1]
    )
    plt.savefig(fig_name + ".png")


for i in range(M):
    for k in seed_range:
        out_fedadmm[(Ns[0] - 1, k, i)] = (np.zeros((m, n)),
                                          np.zeros((n, n)),
                                          np.zeros((n, n)),
                                          np.zeros((m, m)))  # m, k, N --> (K, P, Q)

Nprev = Ns[0] - 1
for N in Ns:
    print("Traj # =", N, end=" - ", flush=True)
    # start = datetime.now()
    start = time.clock()

    costs_lr += [[]]
    costs_admm += [[]]
    costs_fedadmm += [[]]
    costs_lr_K += [{'K': []}]
    costs_admm_KQR += [{'K': [], 'Q': [], 'R': []}]
    costs_fedadmm_KQR += [{'K': [], 'Q': [], 'R': []}]
    for k in seed_range:
        print(k, end=", (", flush=True)
        for i in range(M):
            print(i, end=", ", flush=True)
            _, xs, us = simulate(K_trues[i], i, N=N, seed=k, add_noise=True, train=True)
            # print(np.shape(xs))


            def L(K):
                loss = cp.sum_squares(xs@K.T - us)
                return loss


            def r(K):
                return cp.sum_squares(K), []


            rPQR = lambda Q, R: (cp.sum_squares(Q) + cp.sum_squares(R))


            LK = lambda K: np.linalg.norm(K - K_trues[i])
            LQ = lambda Q: np.linalg.norm(Q - Q_trues[i])
            LR = lambda R: np.linalg.norm(R - R_trues[i])

            Klr = policy_fitting(L, r, xs, us)
            Kadmm, Padmm, Qadmm, Radmm = policy_fitting_with_a_kalman_constraint(L, r, xs, us, A, B, n_random=nrandom,
                                                                                 niter=niter)
            # print(k, end=", ", flush=True)
            _, prevfedP, prevfedQ, prevfedR = out_fedadmm[(Nprev, k, i)]
            # prevfedP, prevfedQ, prevfedR = P_trues[i], Q_trues[i], R_trues[i]
            LPQR = lambda Q, R: 0.1*(cp.sum_squares(Q - prevfedQ) + cp.sum_squares(R - prevfedR))
            Kfedadmm, Pfedadmm, Qfedadmm, Rfedadmm = policy_fitting_with_a_kalman_constraint(L, r, xs, us, A, B,
                                                                                             niter=niter,
                                                                                             n_random=nrandom,
                                                                                             P0=prevfedP,
                                                                                             Q0=prevfedQ, R0=prevfedR,
                                                                                             LPQR=LPQR, rPQR=rPQR)

            cost_lr = simulate_cost(Klr, i, N=N_test, seed=0, N_test=1)
            out_lr[(N, k, i)] = Klr
            cost_admm = simulate_cost(Kadmm, i, N=N_test, seed=0, N_test=1)
            out_admm[(N, k, i)] = (Kadmm, Padmm, Qadmm, Radmm)
            cost_fedadmm = simulate_cost(Kfedadmm, i, N=N_test, seed=0, N_test=1)
            out_fedadmm[(N, k, i)] = (Kfedadmm, Pfedadmm, Qfedadmm, Rfedadmm)

            if np.isnan(cost_lr) or cost_lr > 1e5 or cost_lr == np.inf:
                cost_lr = np.nan
            costs_lr[-1].append(cost_lr)
            costs_admm[-1].append(cost_admm)
            costs_fedadmm[-1].append(cost_fedadmm)

            costs_lr_K[-1]['K'].append(LK(Klr))
            costs_admm_KQR[-1]['K'].append(LK(Kadmm))
            costs_admm_KQR[-1]['Q'].append(LQ(Qadmm))
            costs_admm_KQR[-1]['R'].append(LR(Radmm))
            costs_fedadmm_KQR[-1]['K'].append(LK(Kfedadmm))
            costs_fedadmm_KQR[-1]['Q'].append(LQ(Qfedadmm))
            costs_fedadmm_KQR[-1]['R'].append(LR(Rfedadmm))

    avg_admmQ = np.zeros((n, n))
    avg_admmR = np.zeros((m, m))
    avg_fedadmmQ = np.zeros((n, n))
    avg_fedadmmR = np.zeros((m, m))
    for k in seed_range:
        for i in range(M):
            avg_admmQ += out_fedadmm[(N, k, i)][2]/len(seed_range)/M
            avg_admmR += out_fedadmm[(N, k, i)][3]/len(seed_range)/M
            avg_fedadmmQ += out_fedadmm[(N, k, i)][2]/len(seed_range)/M
            avg_fedadmmR += out_fedadmm[(N, k, i)][3]/len(seed_range)/M
    true_admmQ_loss = np.linalg.norm(avg_admmQ - Q_avg)
    true_admmR_loss = np.linalg.norm(avg_admmR - R_avg)
    true_fedadmmQ_loss = np.linalg.norm(avg_fedadmmQ - Q_avg)
    true_fedadmmR_loss = np.linalg.norm(avg_fedadmmR - R_avg)

    end = time.clock()
    Nprev = N
    print(
        " %03d | %3.3f | %3.3f | %3.3f (%3.3f - %3.3f - %3.3f), LK = %3.3f | %3.3f (%3.3f - %3.3f - %3.3f), "
        "LK = %3.3f, LQ = %3.3f (%3.3f), LR = %3.3f (%3.3f) | %3.3f (%3.3f - %3.3f - %3.3f), LK = %3.3f, LQ = %3.3f (%3.3f), "
        "LR = %3.3f (%3.3f) - time elapsed = %3.3f "%
        (N, cost_true, cost_noise,
         np.nanmean(costs_lr[-1]), np.nanmin(costs_lr[-1]), np.nanstd(costs_lr[-1]), np.nanmax(costs_lr[-1]),
         np.nanmean(costs_lr_K[-1]['K']),
         np.nanmean(costs_admm[-1]), np.nanmin(costs_admm[-1]), np.nanstd(costs_admm[-1]), np.nanmax(costs_admm[-1]),
         np.nanmean(costs_admm_KQR[-1]['K']),
         np.nanmean(costs_admm_KQR[-1]['Q']), true_admmQ_loss,
         np.nanmean(costs_admm_KQR[-1]['R']), true_admmR_loss,
         np.nanmean(costs_fedadmm[-1]), np.nanmin(costs_fedadmm[-1]), np.nanstd(costs_fedadmm[-1]),
         np.nanmax(costs_fedadmm[-1]),
         np.nanmean(costs_fedadmm_KQR[-1]['K']),
         np.nanmean(costs_fedadmm_KQR[-1]['Q']), true_fedadmmQ_loss,
         np.nanmean(costs_fedadmm_KQR[-1]['R']), true_fedadmmR_loss,
         end - start
         ))

    plot_losses(costs_lr, costs_admm, costs_fedadmm, verbose=True)

plot_losses(costs_lr, costs_admm, costs_fedadmm, verbose=True)
plt.show()
