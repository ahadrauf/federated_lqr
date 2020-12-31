import numpy as np
from lqr import LQR
import matplotlib.pyplot as plt


def initialize_computer_LQR():
    A = np.array([[1.]])
    B = np.array([[1.]])
    Q = np.array([[1.]])
    R = np.array([[1.]])
    F = np.array([[1.]])
    var_dyn = 1e-1
    var_ctrl = 1e-1
    return LQR(A, B, Q, R, F, var_dyn, var_ctrl)


def initialize_human_LQR(qh, rh):
    controller = initialize_computer_LQR()
    controller.Q *= qh
    controller.R *= rh
    return controller


if __name__ == "__main__":
    qh, rh = 1, 3

    controller = initialize_computer_LQR()
    human = initialize_human_LQR(qh, rh)

    x0 = np.array([10])

    num_to_sim = 200
    N = 20
    resolution_dyn = 1
    resolution_ctrl = 0.1
    eps = 1e-70

    traj_probs_human = []
    state_transition_probs_human = []
    action_probs_human = []
    traj_probs_comp = []
    state_transition_probs_comp = []
    action_probs_comp = []

    avg_traj_human = []
    avg_traj_comp = []

    fig, axs = plt.subplots(2, 2)
    for _ in range(num_to_sim):
        traj_human, metadata_human = human.generate_trajectory(x0, N)
        traj_comp, metadata_comp = controller.generate_trajectory(x0, N)

        traj_human_plt = controller.extract_trajectory_from_transition_trajectory(traj_human)
        traj_comp_plt = controller.extract_trajectory_from_transition_trajectory(traj_comp)
        idx_human_plt = np.arange(0, len(traj_human_plt))
        idx_comp_plt = np.arange(0, len(traj_comp_plt))

        if len(avg_traj_human) == 0:
            avg_traj_human = traj_human_plt
        else:
            avg_traj_human = [x+y for x, y in zip(traj_human_plt, avg_traj_human)]

        if len(avg_traj_comp) == 0:
            avg_traj_comp = traj_comp_plt
        else:
            avg_traj_comp = [x+y for x, y in zip(traj_comp_plt, avg_traj_comp)]

        axs[0, 0].plot(idx_human_plt, traj_human_plt, 'b', alpha=0.05)
        axs[0, 0].plot(idx_comp_plt, traj_comp_plt, 'r', alpha=0.05)

        # Human trajectory (estimated via computer metadata)
        traj_probs_human.append(controller.log_prob_trajectory(traj_human, metadata_comp, resolution_dyn,
                                                               resolution_ctrl, eps, eps))
        for state, metadatum_comp in zip(traj_human, metadata_comp):
            state_transition_probs_human.append(np.log(controller.prob_state_transition(state, resolution_dyn, eps)))
            action_probs_human.append(np.log(controller.prob_action(state, metadatum_comp[0], resolution_ctrl, eps)))

        # Computer trajectory
        traj_probs_comp.append(controller.log_prob_trajectory(traj_comp, metadata_comp, resolution_dyn,
                                                              resolution_ctrl, eps, eps))
        for state, metadatum_comp, metadatum_human in zip(traj_comp, metadata_comp, metadata_human):
            state_transition_probs_comp.append(np.log(controller.prob_state_transition(state, resolution_dyn, eps)))
            action_probs_comp.append(np.log(controller.prob_action(state, metadatum_comp[0], resolution_ctrl, eps)))

    avg_traj_human = [x/num_to_sim for x in avg_traj_human]
    avg_traj_comp = [x/num_to_sim for x in avg_traj_comp]

    axs[0, 0].plot(idx_human_plt, avg_traj_human, 'xkcd:azure', linewidth=3, label='Average Human Trajectory')
    axs[0, 0].plot(idx_comp_plt, avg_traj_comp, 'xkcd:bright red', linewidth=3, label='Average Computer Trajectory')
    axs[0, 0].set_ylabel('x')
    axs[0, 0].set_title(r'$Trajectories,\ q^h = '+str(qh)+', r^h = '+str(rh)+'$')
    axs[0, 0].legend()

    axs[0, 1].hist(traj_probs_human, 50, density=True, facecolor='b', alpha=0.75, label='Human')
    axs[0, 1].hist(traj_probs_comp, 50, density=True, facecolor='r', alpha=0.75, label='Robot')
    axs[0, 1].set_ylabel('Probability Density')
    axs[0, 1].set_title(r'$log(p(traj)),\ q^h = '+str(qh)+', r^h = '+str(rh)+'$')
    axs[0, 1].legend()

    axs[1, 0].hist(state_transition_probs_human, 50, density=True, facecolor='b', alpha=0.75, label='Human')
    axs[1, 0].hist(state_transition_probs_comp, 50, density=True, facecolor='r', alpha=0.75, label='Robot')
    axs[1, 0].set_ylabel('Probability Density')
    axs[1, 0].set_title(r'$log(p(x_{t+1} | x_t, u_t)),\ q^h = '+str(qh)+', r^h = '+str(rh)+'$')
    axs[1, 0].legend()

    axs[1, 1].hist(action_probs_human, 25, density=True, facecolor='b', alpha=1, label='Human')
    axs[1, 1].hist(action_probs_comp, 25, density=True, facecolor='r', alpha=0.5, label='Robot')
    axs[1, 1].set_ylabel('Probability Density')
    axs[1, 1].set_title(r'$log(p(u_t | x_t)),\ q^h = '+str(qh)+', r^h = '+str(rh)+'$')
    plt.legend()

    fig.suptitle("A = {}, B = {}, Qc = {}, Rc = {}, F = {}, sigma_v = {}, sigma_w = {}, N = {}, x0={}".format(
        controller.A, controller.B, controller.Q, controller.R, controller.F, controller.var_dyn,
        controller.var_ctrl, N, x0
    ))

    plt.grid(True)
    plt.show()
