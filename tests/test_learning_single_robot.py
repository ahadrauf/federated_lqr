import numpy as np
from lqr import LQR
import matplotlib.pyplot as plt


def initialize_computer_LQR():
    A = np.array([[1.]])
    B = np.array([[1.]])
    Q = np.array([[1.]])
    R = np.array([[1.]])
    F = np.array([[1.]])
    var_dyn = np.array([[0]])  # np.array([[1e-1**2]])
    var_ctrl = np.array([[0]])  # np.array([[1e-1**2]])
    return LQR(A, B, Q, R, F, var_dyn, var_ctrl)


def initialize_human_LQR(qh, rh):
    comp = initialize_computer_LQR()
    comp.Q *= qh
    comp.R *= rh
    return comp


if __name__ == "__main__":
    qh, rh = 1, 3

    comp = initialize_computer_LQR()
    human = initialize_human_LQR(qh, rh)

    x0 = np.array([[10]])

    # num_to_sim = 100
    num_iters = 30
    N = 20
    resolution_dyn = 1
    resolution_ctrl = 0.1
    eps = 1e-70

    traj_probs_human = []
    state_transition_probs_human = []

    avg_traj_human = []
    avg_traj_comp = []

    fig, axs = plt.subplots(1, 2)
    zQs = []
    zRs = []
    Qs = [comp.Q]
    Rs = [comp.R]
    eta = 0.01
    for iteration in range(num_iters):
        x_human, u_human, metadata_human = human.generate_trajectory(x0, N)
        x_comp, u_comp, metadata_comp = comp.generate_trajectory(x0, N)

        # Local update
        zQ = comp.dJ_dQ(x_comp, x_human)
        zR = comp.dJ_dR(u_comp, u_human)
        print(zR)
        # comp.Q -= eta*zQ
        comp.R -= eta*zR

        Qs.append(comp.Q)
        Rs.append(comp.R)
        zQs.append(zQ)
        zRs.append(zR)

        if iteration == 0:
            axs[0].plot([x[0][0] for x in x_human], 'xkcd:azure', linewidth=4, label='Human, iter=' + str(iteration))
        if iteration%2 == 0:
            axs[0].plot([x[0][0] for x in x_comp], alpha=0.5, label='Robot, iter=' + str(iteration))

    axs[0].plot([x[0][0] for x in x_comp], 'xkcd:bright red', linewidth=4, label='Robot, Final Iteration')
    axs[0].set_title("(a) Trajectories")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot([zR[0][0] for zR in zRs])
    axs[1].set_xlabel("Iteration")
    axs[1].set_title(r'$(b) dJ(\tau_{comp}; \theta_{comp})/dR - dJ(\tau_{human}; \theta_{comp})/dR$')
    axs[1].grid(True)

    # axs[1, 0].plot([Q[0][0] for Q in Qs])
    # axs[1, 0].set_xlabel("Iteration")
    # axs[1, 0].set_title("Q")
    #
    # axs[1, 1].plot([R[0][0] for R in Rs])
    # axs[1, 1].set_xlabel("Iteration")
    # axs[1, 1].set_title("R")

    fig.suptitle("A = {}, B = {}, Qc = {}, Rc = {}, F = {}, cov_v = {}, cov_w = {}, N = {}, "
                 "x0={}\nQh = {}, Rh = {}, num_iterations = {}, eta = {}".format(
        comp.A, comp.B, comp.Q, comp.R, comp.F, comp.cov_dyn,
        comp.cov_ctrl, N, x0, human.Q, human.R, num_iters, eta
    ))

    # plt.grid(True)
    fig.tight_layout()
    plt.show()
