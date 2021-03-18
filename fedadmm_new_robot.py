import numpy as np
from scipy.linalg import solve_discrete_are
import cvxpy as cp
from ADMM_slack import *
from utils import *
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings("ignore")
from datetime import datetime
import time
from scipy.stats import wishart

M = 1  # number of robots
n, m = 4, 2  # dimension of state (n) and input (m)
nrandom = 4  # number of random starts on ADMM (optimize over which random start gives the best L(K))
covarQR = 0.1  # covariance of generated Q and R (only matters if use_wishart = False)
niter = 10  # number of iterations within ADMM
Wdyn = 1  # magnitude of dynamics noise
Wctrl = 3  # magnitude of control noise
N = 5  # trajectory length
traj_range = np.arange(1, 16)  # # of demonstrations (each of length N)
seed_range = np.arange(1, 11)  # # of seeds to average over (the value actually doesn't matter, just the range length)
admm_QRreg = True  # whether to add regularization on Q and R to Boyd's ADMM implementation
use_wishart = False  # whether to initialize the Q's and R's using a wishart distribution
noise_on_output_loss = False
noise_on_input_data = True
average_over = 10
alpha = 0  # multiplier on r(K, Q, R)
alphaK = 0.05
beta = 0.01*100  # multiplier on l_D
name_clarifier = "ADMMslack_fixedrandom_newrobot"
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
# A = A/np.abs(np.linalg.eig(A)[0]).max()
# B = np.random.randn(n, m)
A = np.array([[0.13741826, 0.02876351, 0.50096039, 0.24711964],
              [-0.08638155, -0.29891056, 0.16163153, 0.24719121],
              [0.06460169, 0.24025215, -0.29127423, -0.28410344],
              [0.18903618, 0.43957661, -0.43899608, -0.38508461]])
B = np.array([[1.12563572, -0.59218407],
              [-0.01612495, -0.18979434],
              [-1.38525459, -0.51277622],
              [-1.43032195, 2.17007322]])
W = Wdyn * np.eye(n)
VQ = np.eye(n) / n  # /n for normalizing to eye(n)
VR = np.eye(m) / m  # /m

Q_trues = []
R_trues = []
P_trues = []
K_trues = []

ATA = lambda A: A.T @ A

Q_starter = np.array([[1.65775798, -0.12087396, 0.42784446, 0.7738255],
                      [-0.12087396, 0.15703623, 0.05796011, -0.09531927],
                      [0.42784446, 0.05796011, 0.1870196, 0.14953155],
                      [0.7738255, -0.09531927, 0.14953155, 0.40412161]])

R_starter = np.array([[1.25851049, 0.83728649],
                      [0.83728649, 3.77638641]])

for i in range(M):
    if use_wishart:
        Q = np.reshape(wishart.rvs(n * n, VQ), (n, n))
        R = np.reshape(wishart.rvs(m * m, VR), (m, m))
    else:
        Q = Q_starter + covarQR * ATA(np.random.rand(n, n))
        R = R_starter + covarQR * ATA(np.random.rand(m, m))
    Q_trues.append(Q)
    R_trues.append(R)
    P_trues.append(solve_discrete_are(A, B, Q_trues[i], R_trues[i]))
    K_trues.append(-np.linalg.solve(R_trues[i] + B.T @ P_trues[i] @ B, B.T @ P_trues[i] @ A))
print("Q_trues", Q_trues)
print("R_trues", R_trues)
print("K_trues", K_trues)


def simulate(K, robot, N=10, seed=None, add_noise=False, train=False):
    if seed is None:
        np.random.seed(np.random.randint(0, 1000))
    x = np.random.multivariate_normal(np.zeros(n), W)
    # if train:
    # x = np.random.randint(-Wdyn*3, Wdyn*3, size=(n,))
    # else:
    #     x = np.random.multivariate_normal(np.zeros(n), np.eye(n))
    xs = []
    us = []
    cost = 0.0
    for _ in range(N):
        u = K @ x
        # print('u', u)
        if add_noise:
            u += Wctrl * np.random.randn(m)
        xs.append(x)
        us.append(u)
        cost += (x @ Q_trues[robot] @ x + u @ R_trues[robot] @ u) / N
        # print('x', A@x + B@u)
        x = A @ x + B @ u + np.random.multivariate_normal(np.zeros(n), W)
    xs = np.array(xs)
    us = np.array(us)

    return cost, xs, us


def simulate_cost(K, robot, N=10, seed=None, add_noise=None, average_over=10, train=False):
    if add_noise is None:
        add_noise = noise_on_output_loss
    tests = [simulate(K, robot, N=N, seed=seed, add_noise=add_noise, train=train) for _ in range(average_over)]
    return np.nanmean([cost for cost, _, _ in tests])


N_test = 1000
# cost_true = np.nanmean([np.trace(P_trues[i]@W) for i in range(M)], axis=0)
# cost_noise = np.nanmean([np.trace(P_trues[0]@(W + Wctrl*Wctrl*B@B.T)) for i in range(M)], axis=0)
cost_true = np.nanmean(
    [simulate_cost(K_trues[i], i, N=N_test, add_noise=False, average_over=average_over) for i in range(M)], axis=0)
cost_noise = np.nanmean(
    [simulate_cost(K_trues[i], i, N=N_test, add_noise=True, average_over=average_over) for i in range(M)], axis=0)
print("Cost true: {}, cost noise: {}".format(cost_true, cost_noise))

Q_avg = np.mean(Q_trues, axis=0)
R_avg = np.mean(R_trues, axis=0)
cost_LQ_true = np.nanmean([np.linalg.norm([ATA(1. / n * np.random.randn(n, n)) - Q_avg]) for _ in range(
    N_test)], axis=0)
cost_LR_true = np.nanmean([np.linalg.norm([ATA(1. / n * np.random.randn(m, m)) - R_avg]) for _ in range(
    N_test)], axis=0)
cost_fLQ_true = np.nanmean([np.linalg.norm(Q_trues[i] - Q_avg) for i in range(M)])
cost_fLR_true = np.nanmean([np.linalg.norm(R_trues[i] - R_avg) for i in range(M)])

print("Baseline error on LQ: {}, on LR: {}".format(cost_LQ_true, cost_LR_true))
print("Error for true federated learning on LQ: {}, on LR: {}".format(cost_fLQ_true, cost_fLR_true))

# For saving files
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H_%M_%S")
print(timestamp)

costs_admm = []
costs_pfedadmm = []
out_admm = {}
out_pfedadmm = {}
costs_lr_K = []
costs_admm_KQR = []
costs_pfedadmm_KQR = []

latexify(fig_width=6 * 2.5, fig_height=2.7 * 2.5)


def plot_losses(costs_admm, costs_pfedadmm, verbose=False, plot=False):
    # global plotted_before, fig, axs
    # fig.clf()
    fig, axs = plt.subplots(2, 2)
    costs_admm = np.array(costs_admm)
    costs_pfedadmm = np.array(costs_pfedadmm)

    np.save('data/' + timestamp + "_fedadmmv3_new_robot.npy", [A, B, K_trues, P_trues, Q_trues, R_trues,
                                                               costs_admm, costs_pfedadmm,
                                                               out_admm, out_pfedadmm,
                                                               costs_admm_KQR, costs_pfedadmm_KQR])

    mean_admm = np.nanmean(costs_admm, axis=1)
    std_admm = np.nanstd(costs_admm, axis=1)
    mean_pfedadmm = np.nanmean(costs_pfedadmm, axis=1)
    std_pfedadmm = np.nanstd(costs_pfedadmm, axis=1)

    idx = np.arange(0, len(costs_lr_K))
    idx_plot = np.arange(1, len(costs_lr_K) + 1)
    mean_admm_KQR = {k: np.array([np.nanmean(costs_admm_KQR[i][k]) for i in idx]) for k in 'KQR'}
    std_admm_KQR = {k: np.array([np.nanstd(costs_admm_KQR[i][k]) for i in idx]) for k in 'KQR'}
    mean_pfedadmm_KQR = {k: np.array([np.nanmean(costs_pfedadmm_KQR[i][k]) for i in idx]) for k in 'KQR'}
    std_pfedadmm_KQR = {k: np.array([np.nanstd(costs_pfedadmm_KQR[i][k]) for i in idx]) for k in 'KQR'}

    if verbose:
        print("Mean ADMM", mean_admm)
        print("Mean pFedADMM", mean_pfedadmm)

    axs[0, 0].axhline(cost_true, ls='-', c='k', label='optimal (without noise)')
    # axs[0, 0].axhline(cost_noise, ls='--', c='k', label='expert (with noise)')
    axs[0, 0].scatter(idx_plot, mean_admm, s=8, marker='*', c='green', label='ADMM')
    axs[0, 0].fill_between(idx_plot, mean_admm - std_admm / 3, mean_admm + std_admm / 3, alpha=.3, color='green')
    axs[0, 0].scatter(idx_plot, mean_pfedadmm, s=8, marker='*', c='purple', label='pFedADMM')
    axs[0, 0].fill_between(idx_plot, mean_pfedadmm - std_pfedadmm / 3, mean_pfedadmm + std_pfedadmm / 3, alpha=.3,
                           color='purple')
    axs[0, 0].semilogy()
    axs[0, 0].set_ylabel('cost')
    axs[0, 0].set_xlabel(r"# demonstrations $\tau_n$")
    axs[0, 0].set_title("Cost vs. Trajectory Length")
    axs[0, 0].legend()

    # Plot K
    axs[1, 0].scatter(idx_plot, mean_admm_KQR['K'], s=8, marker='o', c='green', label='ADMM')
    axs[1, 0].fill_between(idx_plot, mean_admm_KQR['K'] - std_admm_KQR['K'] / 3,
                           mean_admm_KQR['K'] + std_admm_KQR['K'] / 3 / 3,
                           alpha=.3, color='green')
    axs[1, 0].scatter(idx_plot, mean_pfedadmm_KQR['K'], s=8, marker='o', c='purple', label='pFedADMM')
    axs[1, 0].fill_between(idx_plot, mean_pfedadmm_KQR['K'] - std_pfedadmm_KQR['K'] / 3,
                           mean_pfedadmm_KQR['K'] + std_pfedadmm_KQR['K'] / 3,
                           alpha=.3, color='purple')
    axs[1, 0].grid(True)
    axs[1, 0].set_xlabel(r"# demonstrations $\tau_n$")
    axs[1, 0].set_ylabel(r'$||K - K_{true}||$')
    axs[1, 0].set_title('K Loss, N=' + str(N) + ', M=' + str(M))
    axs[1, 0].legend()

    # Plot Q Loss
    # axs[0, 1].axhline(cost_LQ_true, ls='-', c='k', label='Random Guessing')
    axs[0, 1].axhline(cost_fLQ_true, ls='--', c='k', label='FedADMM with True Qavg')
    axs[0, 1].scatter(idx_plot, mean_admm_KQR['Q'], s=8, marker='o', c='green', label='ADMM')
    axs[0, 1].fill_between(idx_plot, mean_admm_KQR['Q'] - std_admm_KQR['Q'] / 3,
                           mean_admm_KQR['Q'] + std_admm_KQR['Q'] / 3 / 3,
                           alpha=.3, color='green')
    axs[0, 1].scatter(idx_plot, mean_pfedadmm_KQR['Q'], s=8, marker='o', c='purple', label='pFedADMM')
    axs[0, 1].fill_between(idx_plot, mean_pfedadmm_KQR['Q'] - std_pfedadmm_KQR['Q'] / 3,
                           mean_pfedadmm_KQR['Q'] + std_pfedadmm_KQR['Q'] / 3,
                           alpha=.3, color='purple')
    axs[0, 1].grid(True)
    axs[0, 1].set_xlabel(r"# demonstrations $\tau_n$")
    axs[0, 1].set_ylabel(r'$||Q - Q_{true}||$')
    axs[0, 1].set_title('Q Loss, N=' + str(N) + ', M=' + str(M))
    axs[0, 1].legend()

    # Plot R Loss
    # axs[1, 1].axhline(cost_LR_true, ls='-', c='k', label='Random Guessing')
    axs[1, 1].axhline(cost_fLR_true, ls='--', c='k', label='FedADMM with True Ravg')
    axs[1, 1].scatter(idx_plot, mean_admm_KQR['R'], s=8, marker='o', c='green', label='ADMM')
    axs[1, 1].fill_between(idx_plot, mean_admm_KQR['R'] - std_admm_KQR['R'] / 3,
                           mean_admm_KQR['R'] + std_admm_KQR['R'] / 3 / 3,
                           alpha=.3, color='green')
    axs[1, 1].scatter(idx_plot, mean_pfedadmm_KQR['R'], s=8, marker='o', c='purple', label='pFedADMM')
    axs[1, 1].fill_between(idx_plot, mean_pfedadmm_KQR['R'] - std_pfedadmm_KQR['R'] / 3,
                           mean_pfedadmm_KQR['R'] + std_pfedadmm_KQR['R'] / 3,
                           alpha=.3, color='purple')
    axs[1, 1].grid(True)
    axs[1, 1].set_xlabel(r"# demonstrations $\tau_n$")
    axs[1, 1].set_ylabel(r'$||R - R_{true}||$')
    axs[1, 1].set_title('R Loss, N=' + str(N) + ', M=' + str(M))
    axs[1, 1].legend()

    fig_name = "figures/" + timestamp + "_fedadmm_v2_random_M={}_Wctrl={}_Wdyn={}_nrandom={}_covarQR={}_niter=" \
                                        "{}_nseed={}\n_N={}_Ntraj={}_admmQRreg={}_usewishartQR={}_" \
                                        "noisyinput={}_noisyoutput={}" \
                                        "_alpha={}_beta={}\n{}".format(
        M, Wctrl, Wdyn, nrandom, covarQR, niter, len(seed_range), N, traj_range[-1], admm_QRreg, use_wishart,
        noise_on_input_data, noise_on_output_loss, alpha, beta, name_clarifier
    )
    fig.suptitle(fig_name)
    plt.tight_layout()
    plt.savefig(fig_name.replace('\n', '') + ".png")
    if not plot:
        plt.close(fig)


for i in range(M):
    for k in seed_range:
        out_admm[(traj_range[0] - 1, k, i)] = (np.zeros((m, n)),
                                               np.zeros((n, n)),
                                               np.zeros((n, n)),
                                               np.zeros((m, m)))  # m, k, N --> (K, P, Q)
        out_pfedadmm[(traj_range[0] - 1, k, i)] = (np.zeros((m, n)),
                                                   np.zeros((n, n)),
                                                   np.zeros((n, n)),
                                                   np.zeros((m, m)))  # m, k, N --> (K, P, Q)

traj_prev = traj_range[0] - 1
xs_agg = {i: np.zeros((1, n)) for i in range(M)}
us_agg = {i: np.zeros((1, m)) for i in range(M)}
for traj in traj_range:
    print("Traj # =", traj, end=" - ", flush=True)
    # start = datetime.now()
    start = time.time()

    costs_admm += [[]]
    costs_pfedadmm += [[]]
    costs_lr_K += [{'K': []}]
    costs_admm_KQR += [{'K': [], 'Q': [], 'R': []}]
    costs_pfedadmm_KQR += [{'K': [], 'Q': [], 'R': []}]
    for k in seed_range:
        print(k, end=", (", flush=True)
        prevfedQ = np.nanmean([out_admm[(traj_prev, k, i)][2] for i in range(M)], axis=0)
        prevfedR = np.nanmean([out_admm[(traj_prev, k, i)][3] for i in range(M)], axis=0)
        prevpfedQ = np.nanmean([out_pfedadmm[(traj_prev, k, i)][2] for i in range(M)], axis=0)
        prevpfedR = np.nanmean([out_pfedadmm[(traj_prev, k, i)][3] for i in range(M)], axis=0)
        if traj == 1:
            prevfedP = np.zeros((n, n))
            prevfedK = np.zeros((m, n))

            prevpfedQ = Q_starter
            prevpfedR = R_starter
        else:
            prevfedP = solve_discrete_are(A, B, prevfedQ, prevfedR)
            prevfedK = -np.linalg.solve(prevfedR + B.T @ prevfedP @ B, B.T @ prevfedP @ A)

            prevpfedP = solve_discrete_are(A, B, prevpfedQ, prevpfedR)
            prevpfedK = -np.linalg.solve(prevpfedR + B.T @ prevpfedP @ B, B.T @ prevpfedP @ A)


        # _, prevfedP, prevfedQ, prevfedR = out_pfedadmm[(traj_prev, k, i)]
        for robot in range(M):
            print(robot, end=", ", flush=True)
            _, xs, us = simulate(K_trues[robot], robot, N=N, seed=k, add_noise=noise_on_input_data, train=True)
            xs_agg[robot] = np.append(xs_agg[robot], xs, axis=0)  # shape = N x n
            us_agg[robot] = np.append(us_agg[robot], us, axis=0)

            if traj == 1:
                prevfedP = np.zeros((n, n))
                prevfedK = np.zeros((m, n))
                prevpfedP = np.zeros((n, n))
                prevpfedK = np.zeros((m, n))
            else:
                prevpfedP = solve_discrete_are(A, B, prevpfedQ, prevpfedR)
                prevpfedK = -np.linalg.solve(prevpfedR + B.T @ prevpfedP @ B, B.T @ prevpfedP @ A)

            L = lambda K: cp.sum_squares(xs_agg[robot] @ K.T - us_agg[robot])
            LPQR = lambda Q, R: beta * (cp.sum_squares(Q - prevfedQ) + cp.sum_squares(R - prevfedR))

            r = lambda K: alphaK * cp.sum_squares(K)
            rPQR = lambda Q, R: alpha * (cp.sum_squares(Q) + cp.sum_squares(R))
            # 200*(cp.sum_squares(Q-Q_trues[robot]) + cp.sum_squares(R-R_trues[robot]))#

            LK = lambda K: np.linalg.norm(K - K_trues[robot])
            LQ = lambda Q: np.linalg.norm(Q - Q_trues[robot])
            LR = lambda R: np.linalg.norm(R - R_trues[robot])

            if admm_QRreg:
                Kadmm, Padmm, Qadmm, Radmm = policy_fitting_with_a_kalman_constraint(L, r, A, B,
                                                                                     n_random=nrandom,
                                                                                     niter=niter,
                                                                                     P0=prevfedP,
                                                                                     Q0=prevfedQ,
                                                                                     R0=prevfedR,
                                                                                     rPQR=rPQR,
                                                                                     LPQR=LPQR)
            else:
                Kadmm, Padmm, Qadmm, Radmm = policy_fitting_with_a_kalman_constraint(L, r, A, B,
                                                                                     n_random=nrandom,
                                                                                     P0=prevfedP,
                                                                                     Q0=prevfedQ,
                                                                                     R0=prevfedR,
                                                                                     niter=niter)

            # pFedADMM
            LpPQR = lambda Q, R: beta * (cp.sum_squares(Q - prevpfedQ) + cp.sum_squares(R - prevpfedR))
            Kpfedadmm, Ppfedadmm, Qpfedadmm, Rpfedadmm = policy_fitting_with_a_kalman_constraint(L, r,
                                                                                                 A, B,
                                                                                                 niter=niter,
                                                                                                 n_random=nrandom,
                                                                                                 P0=prevpfedP,
                                                                                                 Q0=prevpfedQ,
                                                                                                 R0=prevpfedR,
                                                                                                 LPQR=LpPQR, rPQR=rPQR)

            cost_admm = simulate_cost(Kadmm, robot, N=N_test, seed=0, average_over=average_over)
            out_admm[(traj, k, robot)] = (Kadmm, Padmm, Qadmm, Radmm)
            cost_pfedadmm = simulate_cost(Kpfedadmm, robot, N=N_test, seed=0, average_over=average_over)
            out_pfedadmm[(traj, k, robot)] = (Kpfedadmm, Ppfedadmm, Qpfedadmm, Rpfedadmm)

            if np.isnan(cost_admm) or cost_admm > 1e5 or cost_admm == np.inf:
                cost_admm = np.nan
            if np.isnan(cost_pfedadmm) or cost_pfedadmm > 1e5 or cost_pfedadmm == np.inf:
                cost_pfedadmm = np.nan
            costs_admm[-1].append(cost_admm)
            costs_pfedadmm[-1].append(cost_pfedadmm)

            costs_admm_KQR[-1]['K'].append(LK(Kadmm))
            costs_admm_KQR[-1]['Q'].append(LQ(Qadmm))
            costs_admm_KQR[-1]['R'].append(LR(Radmm))
            costs_pfedadmm_KQR[-1]['K'].append(LK(Kpfedadmm))
            costs_pfedadmm_KQR[-1]['Q'].append(LQ(Qpfedadmm))
            costs_pfedadmm_KQR[-1]['R'].append(LR(Rpfedadmm))

            loss_admm = np.linalg.norm(xs_agg[robot] @ Kadmm.T - us_agg[robot]) ** 2
            loss_pfedadmm = np.linalg.norm(xs_agg[robot] @ Kpfedadmm.T - us_agg[robot]) ** 2
            print('Ladmm = {}, Lpfedadmm = {}'.format(loss_admm,
                                                      loss_pfedadmm))

            print(LQ(Q_starter), LR(R_starter))
            print(LK(Kadmm), LQ(Qadmm), LR(Radmm))
            print(LK(Kpfedadmm), LQ(Qpfedadmm), LR(Rpfedadmm))

    avg_admmQ = np.zeros((n, n))
    avg_admmR = np.zeros((m, m))
    avg_pfedadmmQ = np.zeros((n, n))
    avg_pfedadmmR = np.zeros((m, m))
    for k in seed_range:
        for robot in range(M):
            avg_admmQ += out_admm[(traj, k, robot)][2] / len(seed_range) / M
            avg_admmR += out_admm[(traj, k, robot)][3] / len(seed_range) / M
            avg_pfedadmmQ += out_pfedadmm[(traj, k, robot)][2] / len(seed_range) / M
            avg_pfedadmmR += out_pfedadmm[(traj, k, robot)][3] / len(seed_range) / M
    true_admmQ_loss = np.linalg.norm(avg_admmQ - Q_avg)
    true_admmR_loss = np.linalg.norm(avg_admmR - R_avg)
    true_pfedadmmQ_loss = np.linalg.norm(avg_pfedadmmQ - Q_avg)
    true_pfedadmmR_loss = np.linalg.norm(avg_pfedadmmR - R_avg)

    end = time.time()
    traj_prev = traj
    print(
        " %03d | %3.3f | %3.3f | "
        "ADMM: %3.3f (%3.3f - %3.3f - %3.3f), LK = %3.3f, LQ = %3.3f (%3.3f), LR = %3.3f (%3.3f) | "
        "pFedADMM: %3.3f (%3.3f - %3.3f - %3.3f), LK = %3.3f, LQ = %3.3f (%3.3f), LR = %3.3f (%3.3f) - "
        "time elapsed = %3.3f " %
        (N, cost_true, cost_noise,
         np.nanmean(costs_admm[-1]), np.nanmin(costs_admm[-1]), np.nanstd(costs_admm[-1]), np.nanmax(costs_admm[-1]),
         np.nanmean(costs_admm_KQR[-1]['K']),
         np.nanmean(costs_admm_KQR[-1]['Q']), true_admmQ_loss,
         np.nanmean(costs_admm_KQR[-1]['R']), true_admmR_loss,
         np.nanmean(costs_pfedadmm[-1]), np.nanmin(costs_pfedadmm[-1]), np.nanstd(costs_pfedadmm[-1]),
         np.nanmax(costs_pfedadmm[-1]),
         np.nanmean(costs_pfedadmm_KQR[-1]['K']),
         np.nanmean(costs_pfedadmm_KQR[-1]['Q']), true_pfedadmmQ_loss,
         np.nanmean(costs_pfedadmm_KQR[-1]['R']), true_pfedadmmR_loss,
         end - start
         ))

    plot_losses(costs_admm, costs_pfedadmm, verbose=True)

plot_losses(costs_admm, costs_pfedadmm, verbose=True, plot=True)
plt.show()
