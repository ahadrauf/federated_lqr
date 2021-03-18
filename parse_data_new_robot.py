import numpy as np
import matplotlib.pyplot as plt
from utils import *

file_name = "data/20210316_00_43_58_fedadmmv3_new_robot.npy"

data = np.load(file_name, allow_pickle=True)
A, B, K_trues, P_trues, Q_trues, R_trues, costs_admm, costs_pfedadmm, out_admm, out_pfedadmm, costs_admm_KQR, costs_pfedadmm_KQR = data

Q_avg = np.mean(Q_trues, axis=0)
R_avg = np.mean(R_trues, axis=0)
K_avg = np.mean(K_trues, axis=0)
norms = {'K': np.linalg.norm(K_avg), 'Q': np.linalg.norm(Q_trues), 'R': np.linalg.norm(R_trues)}
# print(np.mean(Q_trues, axis=0), np.linalg.norm(np.mean(Q_trues, axis=0)))
# print(np.mean(R_trues, axis=0), np.linalg.norm(np.mean(R_trues, axis=0)))

n, m = np.shape(B)
M = 1
N = 5
traj_range = np.arange(1, 16)
Wctrl = 3
Wdyn = 1
W = Wdyn*np.eye(n)
VQ = np.eye(n)/n
VR = np.eye(m)/m

cost_true = np.nanmean([np.trace(P_trues[i]@W) for i in range(M)], axis=0)
cost_noise = np.nanmean([np.trace(P_trues[0]@(W + Wctrl*Wctrl*B@B.T)) for i in range(M)], axis=0)
cost_fLQ_true = np.nanmean([np.linalg.norm(n*VQ - Q_trues[robot]) for robot in range(M)], axis=0)
cost_fLR_true = np.nanmean([np.linalg.norm(m*VR - R_trues[robot]) for robot in range(M)], axis=0)

latexify(fig_width=6*2.5, fig_height=2.7*2.5)


def plot_losses(costs_admm, costs_pfedadmm, verbose=False, plot=False):
    fig, axs = plt.subplots(2, 2)
    costs_admm = np.array(costs_admm)
    costs_pfedadmm = np.array(costs_pfedadmm)

    L = len(costs_admm_KQR)
    idx = np.arange(0, L)
    idx_plot = np.arange(1, L + 1)
    mean_admm = np.nanmean(costs_admm, axis=1)
    std_admm = np.nanstd(costs_admm, axis=1)
    mean_pfedadmm = np.nanmean(costs_pfedadmm, axis=1)
    std_pfedadmm = np.nanstd(costs_pfedadmm, axis=1)

    mean_admm_KQR = {k: np.array([np.nanmean(costs_admm_KQR[i][k])/norms[k] for i in idx]) for k in 'KQR'}
    std_admm_KQR = {k: np.array([np.nanstd(costs_admm_KQR[i][k])/norms[k] for i in idx]) for k in 'KQR'}
    mean_pfedadmm_KQR = {k: np.array([np.nanmean(costs_pfedadmm_KQR[i][k])/norms[k] for i in idx]) for k in 'KQR'}
    std_pfedadmm_KQR = {k: np.array([np.nanstd(costs_pfedadmm_KQR[i][k])/norms[k] for i in idx]) for k in 'KQR'}

    if verbose:
        print("Mean ADMM", mean_admm)
        print("Mean pFedADMM", mean_pfedadmm)

    axs[0, 0].axhline(cost_true, ls='-', c='k', label='optimal (without noise)')
    # axs[0, 0].axhline(cost_noise, ls='--', c='k', label='expert (with noise)')
    axs[0, 0].scatter(idx_plot, mean_admm, s=8, marker='*', c='green', label='pFedADMM (No Initialization)')
    axs[0, 0].fill_between(idx_plot, mean_admm - std_admm/3, mean_admm + std_admm/3, alpha=.3, color='green')
    axs[0, 0].scatter(idx_plot, mean_pfedadmm, s=8, marker='*', c='purple', label='pFedADMM (Learned Initialization)')
    axs[0, 0].fill_between(idx_plot, mean_pfedadmm - std_pfedadmm/3, mean_pfedadmm + std_pfedadmm/3, alpha=.3,
                           color='purple')
    axs[0, 0].semilogy()
    axs[0, 0].set_ylabel(r'LQR Cost')
    axs[0, 0].set_xlabel(r"# demonstrations $\tau_n$")
    axs[0, 0].set_title("Cost vs. Trajectory Length")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Plot K
    axs[1, 0].scatter(idx_plot, mean_admm_KQR['K'], s=8, marker='o', c='green', label='pFedADMM (No Initialization)')
    axs[1, 0].fill_between(idx_plot, mean_admm_KQR['K'] - std_admm_KQR['K']/3, mean_admm_KQR['K'] + std_admm_KQR['K']/3/3,
                           alpha=.3, color='green')
    axs[1, 0].scatter(idx_plot, mean_pfedadmm_KQR['K'], s=8, marker='o', c='purple', label='pFedADMM (Learned Initialization)')
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
    axs[0, 1].scatter(idx_plot, mean_admm_KQR['Q'], s=8, marker='o', c='green', label='pFedADMM (No Initialization)')
    axs[0, 1].fill_between(idx_plot, mean_admm_KQR['Q'] - std_admm_KQR['Q']/3, mean_admm_KQR['Q'] + std_admm_KQR['Q']/3/3,
                           alpha=.3, color='green')
    axs[0, 1].scatter(idx_plot, mean_pfedadmm_KQR['Q'], s=8, marker='o', c='purple', label='pFedADMM (Learned Initialization)')
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
    axs[1, 1].scatter(idx_plot, mean_admm_KQR['R'], s=8, marker='o', c='green', label='pFedADMM (No Initialization)')
    axs[1, 1].fill_between(idx_plot, mean_admm_KQR['R'] - std_admm_KQR['R']/3, mean_admm_KQR['R'] + std_admm_KQR['R']/3/3,
                           alpha=.3, color='green')
    axs[1, 1].scatter(idx_plot, mean_pfedadmm_KQR['R'], s=8, marker='o', c='purple', label='pFedADMM (Learned Initialization)')
    axs[1, 1].fill_between(idx_plot, mean_pfedadmm_KQR['R'] - std_pfedadmm_KQR['R']/3,
                           mean_pfedadmm_KQR['R'] + std_pfedadmm_KQR['R']/3,
                           alpha=.3, color='purple')
    axs[1, 1].grid(True)
    axs[1, 1].set_xlabel(r"# demonstrations $\tau_n$")
    axs[1, 1].set_ylabel(r'$||R - R_{true}||/||R_{true}||$')
    axs[1, 1].set_title('R Loss')
    axs[1, 1].legend()

    fig_name = "Inverse LQR Experiment, New Robot (Averaged Over 10 Seeds)"
    fig.suptitle(fig_name)
    plt.tight_layout()
    # plt.savefig("figures/" + fig_name.replace('\n', '') + ".png")
    if not plot:
        plt.close(fig)

plot_losses(costs_admm, costs_pfedadmm, plot=True)
plt.show()
