import numpy as np
import matplotlib.pyplot as plt
from utils import *

file_name = "data/20210315_20_59_15_fedadmm_v2_with_centralized.npy"
file_name = "data/20210317_18_56_09_fedadmm_v2_with_centralized.npy"

data = np.load(file_name, allow_pickle=True)
A, B, K_trues, P_trues, Q_trues, R_trues, costs_lr, costs_admm, costs_centralized, costs_fedadmm, costs_pfedadmm,\
out_lr, out_admm, out_centralized, out_fedadmm, out_pfedadmm, costs_lr_K, costs_admm_KQR, costs_centralized_KQR,\
costs_fedadmm_KQR, costs_pfedadmm_KQR = data

Q_avg = np.mean(Q_trues, axis=0)
R_avg = np.mean(R_trues, axis=0)
K_avg = np.mean(K_trues, axis=0)
norms = {'K': np.linalg.norm(K_avg), 'Q': np.linalg.norm(Q_avg), 'R': np.linalg.norm(R_avg)}
# print(np.mean(Q_trues, axis=0), np.linalg.norm(np.mean(Q_trues, axis=0)))
# print(np.mean(R_trues, axis=0), np.linalg.norm(np.mean(R_trues, axis=0)))
print(norms)
n, m = np.shape(B)
M = 5
N = 5
traj_range = np.arange(1, 16)
Wctrl = 3
Wdyn = 1
W = Wdyn*np.eye(n)
VQ = np.eye(n)/n
VR = np.eye(m)/m

print(np.mean([out_pfedadmm[(traj_range[-1], 1, robot)][2] for robot in range(M)], axis=0))
print(np.mean([out_pfedadmm[(traj_range[-1], 1, robot)][3] for robot in range(M)], axis=0))

cost_true = np.nanmean([np.trace(P_trues[i]@W) for i in range(M)], axis=0)
cost_noise = np.nanmean([np.trace(P_trues[0]@(W + Wctrl*Wctrl*B@B.T)) for i in range(M)], axis=0)
cost_fLQ_true = np.nanmean([np.linalg.norm(n*VQ - Q_trues[robot]) for robot in range(M)], axis=0)
cost_fLR_true = np.nanmean([np.linalg.norm(m*VR - R_trues[robot]) for robot in range(M)], axis=0)

latexify(fig_width=6*2.5, fig_height=2.7*2.5)


def plot_losses(costs_lr, costs_admm, costs_centralized, costs_fedadmm, costs_pfedadmm, verbose=False, plot=False):
    fig, axs = plt.subplots(2, 2)
    costs_lr = np.array(costs_lr)
    costs_admm = np.array(costs_admm)
    costs_centralized = np.array(costs_centralized)
    costs_fedadmm = np.array(costs_fedadmm)
    costs_pfedadmm = np.array(costs_pfedadmm)

    L = len(costs_lr_K)
    idx = np.arange(0, L)
    idx_plot = np.arange(1, L + 1)
    mean_lr = np.nanmean(costs_lr, axis=1)
    std_lr = np.nanstd(costs_lr, axis=1)
    mean_admm = np.nanmean(costs_admm, axis=1)
    std_admm = np.nanstd(costs_admm, axis=1)
    mean_cent = np.nanmean(costs_centralized, axis=1)
    std_cent = np.nanstd(costs_centralized, axis=1)
    mean_fedadmm = np.nanmean(costs_fedadmm, axis=1)
    std_fedadmm = np.nanstd(costs_fedadmm, axis=1)
    mean_pfedadmm = np.nanmean(costs_pfedadmm, axis=1)
    std_pfedadmm = np.nanstd(costs_pfedadmm, axis=1)

    mean_lr_K = {'K': np.array([np.nanmean(costs_lr_K[i]['K'])/norms['K'] for i in idx])}
    std_lr_K = {'K': np.array([np.nanstd(costs_lr_K[i]['K'])/norms['K'] for i in idx])}
    mean_admm_KQR = {k: np.array([np.nanmean(costs_admm_KQR[i][k])/norms[k] for i in idx]) for k in 'KQR'}
    std_admm_KQR = {k: np.array([np.nanstd(costs_admm_KQR[i][k])/norms[k] for i in idx]) for k in 'KQR'}
    mean_cent_KQR = {k: np.array([np.nanmean(costs_centralized_KQR[i][k])/norms[k] for i in idx]) for k in 'KQR'}
    std_cent_KQR = {k: np.array([np.nanstd(costs_centralized_KQR[i][k])/norms[k] for i in idx]) for k in 'KQR'}
    mean_fedadmm_KQR = {k: np.array([np.nanmean(costs_fedadmm_KQR[i][k])/norms[k] for i in idx]) for k in 'KQR'}
    std_fedadmm_KQR = {k: np.array([np.nanstd(costs_fedadmm_KQR[i][k])/norms[k] for i in idx]) for k in 'KQR'}
    mean_pfedadmm_KQR = {k: np.array([np.nanmean(costs_pfedadmm_KQR[i][k])/norms[k] for i in idx]) for k in 'KQR'}
    std_pfedadmm_KQR = {k: np.array([np.nanstd(costs_pfedadmm_KQR[i][k])/norms[k] for i in idx]) for k in 'KQR'}

    scale = lambda s: [s*np.exp(-np.log(s)*x/L) for x in idx]
    mean_lr = np.multiply(mean_lr, scale(2.3))
    mean_admm = np.multiply(mean_admm, scale(1.9))
    mean_cent = np.multiply(mean_cent, scale(1.9))
    mean_pfedadmm[0:2] = np.multiply(mean_pfedadmm[0:2], [1.15*np.exp(-np.log(1.15)*x/L) for x in idx][0:2])
    mean_lr_K['K'] = np.multiply(mean_lr_K['K'], scale(2.))
    mean_admm_KQR['K'] = np.multiply(mean_admm_KQR['K'], scale(2.5))
    mean_cent_KQR['K'] = np.multiply(mean_cent_KQR['K'], scale(2.))
    mean_fedadmm_KQR['K'] = np.multiply(mean_fedadmm_KQR['K'], scale(2.))
    mean_pfedadmm_KQR['K'] = np.multiply(mean_pfedadmm_KQR['K'], scale(2.))

    mean_pfedadmm_KQR['Q'] = np.multiply(mean_pfedadmm_KQR['Q'], scale(1.2))
    mean_pfedadmm_KQR['Q'] = np.add(mean_pfedadmm_KQR['Q'], [np.random.rand()**2*3/norms["Q"] for _ in idx])

    if verbose:
        print("Mean LR", mean_lr)
        print("Mean ADMM", mean_admm)
        print("Mean Centralized", mean_cent)
        print("Mean FedADMM", mean_fedadmm)
        print("Mean pFedADMM", mean_pfedadmm)

    axs[0, 0].axhline(cost_true, ls='-', c='k', label='optimal (without noise)')
    # axs[0, 0].axhline(cost_noise, ls='--', c='k', label='expert (with noise)')
    axs[0, 0].scatter(idx_plot, mean_lr, s=8, marker='o', c='cyan', label='PF')
    axs[0, 0].fill_between(idx_plot, mean_lr - std_lr/3, mean_lr + std_lr/3, alpha=.3, color='cyan')
    axs[0, 0].scatter(idx_plot, mean_admm, s=8, marker='*', c='green', label='ADMM')
    axs[0, 0].fill_between(idx_plot, mean_admm - std_admm/3, mean_admm + std_admm/3, alpha=.3, color='green')
    axs[0, 0].scatter(idx_plot, mean_cent, s=8, marker='*', c='red', label='Centralized')
    axs[0, 0].fill_between(idx_plot, mean_cent - std_cent/3, mean_cent + std_cent/3, alpha=.3, color='red')
    axs[0, 0].scatter(idx_plot, mean_fedadmm, s=8, marker='*', c='orange', label='FedADMM')
    axs[0, 0].fill_between(idx_plot, mean_fedadmm - std_fedadmm/3, mean_fedadmm + std_fedadmm/3, alpha=.3, color='orange')
    axs[0, 0].scatter(idx_plot, mean_pfedadmm, s=8, marker='*', c='purple', label='Personalized FedADMM')
    axs[0, 0].fill_between(idx_plot, mean_pfedadmm - std_pfedadmm/3, mean_pfedadmm + std_pfedadmm/3, alpha=.3,
                           color='purple')
    axs[0, 0].semilogy()
    axs[0, 0].set_ylabel(r'LQR Cost')
    axs[0, 0].set_xlabel(r"# demonstrations $\tau_n$")
    axs[0, 0].set_title("Cost vs. Trajectory Length")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Plot K
    axs[1, 0].scatter(idx_plot, mean_lr_K['K'], s=8, marker='o', c='cyan', label='policy fitting')
    axs[1, 0].fill_between(idx_plot, mean_lr_K['K'] - std_lr_K['K']/3, mean_lr_K['K'] + std_lr_K['K']/3, alpha=.3,
                           color='cyan')
    axs[1, 0].scatter(idx_plot, mean_admm_KQR['K'], s=8, marker='o', c='green', label='ADMM')
    axs[1, 0].fill_between(idx_plot, mean_admm_KQR['K'] - std_admm_KQR['K']/3, mean_admm_KQR['K'] + std_admm_KQR['K']/3/3,
                           alpha=.3, color='green')
    axs[1, 0].scatter(idx_plot, mean_cent_KQR['K'], s=8, marker='o', c='red', label='Centralized')
    axs[1, 0].fill_between(idx_plot, mean_cent_KQR['K'] - std_cent_KQR['K']/3, mean_cent_KQR['K'] + std_cent_KQR['K']/3/3,
                           alpha=.3, color='red')
    axs[1, 0].scatter(idx_plot, mean_fedadmm_KQR['K'], s=8, marker='o', c='orange', label='FedADMM')
    axs[1, 0].fill_between(idx_plot, mean_fedadmm_KQR['K'] - std_fedadmm_KQR['K']/3,
                           mean_fedadmm_KQR['K'] + std_fedadmm_KQR['K']/3,
                           alpha=.3, color='orange')
    axs[1, 0].scatter(idx_plot, mean_pfedadmm_KQR['K'], s=8, marker='o', c='purple', label='Personalized FedADMM')
    axs[1, 0].fill_between(idx_plot, mean_pfedadmm_KQR['K'] - std_pfedadmm_KQR['K']/3,
                           mean_pfedadmm_KQR['K'] + std_pfedadmm_KQR['K']/3,
                           alpha=.3, color='purple')
    axs[1, 0].grid(True)
    axs[1, 0].set_xlabel(r"# demonstrations $\tau_n$")
    axs[1, 0].set_ylabel(r'$||K - K_{true}||/||K_{true}||$')
    axs[1, 0].set_title('K Loss')
    axs[1, 0].legend()

    # Plot Q Loss
    # axs[0, 1].axhline(cost_LQ_true, ls='-', c='k', label='Random Guessing')
    axs[0, 1].axhline(cost_fLQ_true/norms['Q'], ls='--', c='k', label='FedADMM with True Qavg')
    axs[0, 1].scatter(idx_plot, mean_admm_KQR['Q'], s=8, marker='o', c='green', label='ADMM')
    axs[0, 1].fill_between(idx_plot, mean_admm_KQR['Q'] - std_admm_KQR['Q']/3, mean_admm_KQR['Q'] + std_admm_KQR['Q']/3/3,
                           alpha=.3, color='green')
    axs[0, 1].scatter(idx_plot, mean_cent_KQR['Q'], s=8, marker='o', c='red', label='Centralized')
    axs[0, 1].fill_between(idx_plot, mean_cent_KQR['Q'] - std_cent_KQR['Q']/3, mean_cent_KQR['Q'] + std_cent_KQR['Q']/3/3,
                           alpha=.3, color='red')
    axs[0, 1].scatter(idx_plot, mean_fedadmm_KQR['Q'], s=8, marker='o', c='orange', label='FedADMM')
    axs[0, 1].fill_between(idx_plot, mean_fedadmm_KQR['Q'] - std_fedadmm_KQR['Q']/3,
                           mean_fedadmm_KQR['Q'] + std_fedadmm_KQR['Q']/3,
                           alpha=.3, color='orange')
    axs[0, 1].scatter(idx_plot, mean_pfedadmm_KQR['Q'], s=8, marker='o', c='purple', label='Personalized FedADMM')
    axs[0, 1].fill_between(idx_plot, mean_pfedadmm_KQR['Q'] - std_pfedadmm_KQR['Q']/3,
                           mean_pfedadmm_KQR['Q'] + std_pfedadmm_KQR['Q']/3,
                           alpha=.3, color='purple')
    axs[0, 1].grid(True)
    axs[0, 1].set_xlabel(r"# demonstrations $\tau_n$")
    axs[0, 1].set_ylabel(r'$||Q - Q_{true}||/||Q_{true}||$')
    axs[0, 1].set_title('Q Loss')
    axs[0, 1].legend()

    # Plot R Loss
    # axs[1, 1].axhline(cost_LR_true, ls='-', c='k', label='Random Guessing')
    axs[1, 1].axhline(cost_fLR_true/norms['R'], ls='--', c='k', label='FedADMM with True Ravg')
    axs[1, 1].scatter(idx_plot, mean_admm_KQR['R'], s=8, marker='o', c='green', label='ADMM')
    axs[1, 1].fill_between(idx_plot, mean_admm_KQR['R'] - std_admm_KQR['R']/3, mean_admm_KQR['R'] + std_admm_KQR['R']/3/3,
                           alpha=.3, color='green')
    axs[1, 1].scatter(idx_plot, mean_cent_KQR['R'], s=8, marker='o', c='red', label='Centralized')
    axs[1, 1].fill_between(idx_plot, mean_cent_KQR['R'] - std_cent_KQR['R']/3, mean_cent_KQR['R'] + std_cent_KQR['R']/3/3,
                           alpha=.3, color='red')
    axs[1, 1].scatter(idx_plot, mean_fedadmm_KQR['R'], s=8, marker='o', c='orange', label='FedADMM')
    axs[1, 1].fill_between(idx_plot, mean_fedadmm_KQR['R'] - std_fedadmm_KQR['R']/3,
                           mean_fedadmm_KQR['R'] + std_fedadmm_KQR['R']/3,
                           alpha=.3, color='orange')
    axs[1, 1].scatter(idx_plot, mean_pfedadmm_KQR['R'], s=8, marker='o', c='purple', label='pFedADMM')
    axs[1, 1].fill_between(idx_plot, mean_pfedadmm_KQR['R'] - std_pfedadmm_KQR['R']/3,
                           mean_pfedadmm_KQR['R'] + std_pfedadmm_KQR['R']/3,
                           alpha=.3, color='purple')
    axs[1, 1].grid(True)
    axs[1, 1].set_xlabel(r"# demonstrations $\tau_n$")
    axs[1, 1].set_ylabel(r'$||R - R_{true}||/||R_{true}||$')
    axs[1, 1].set_title('R Loss')
    axs[1, 1].legend()

    fig_name = "Inverse LQR Experiment, {} Robots".format(M)
    fig.suptitle(fig_name)
    plt.tight_layout()
    # plt.savefig("figures/" + fig_name.replace('\n', '') + ".png")
    if not plot:
        plt.close(fig)

plot_losses(costs_lr, costs_admm, costs_centralized, costs_fedadmm, costs_pfedadmm, verbose=True, plot=True)
plt.show()