import numpy as np
from lqr import LQR
import matplotlib.pyplot as plt

import cvxpy as cp
from cvxopt import matrix, solvers


def initialize_computer_LQR():
    A = np.array([[1.]])
    B = np.array([[1.]])
    Q = np.array([[1.]])
    R = np.array([[1.]])
    F = np.array([[1.]])
    var_dyn = np.array([[0]])  # np.array([[1e-1**2]])
    var_ctrl = np.array([[0]])  # np.array([[1e-1**2]])
    return LQR(A, B, Q, R, F, var_dyn, var_ctrl)


# def initialize_computer_LQR():
#     A = np.array([[-0.1922, -0.2490, 1.2347],
#                   [-0.2741, -1.0642, -0.2296],
#                   [1.5301, 1.6035, -1.5062]])
#     B = np.array([[-0.4446],
#                   [-0.1559],
#                   [0.2761]])
#     Q = np.array([[0.0068, -0.0116, -0.0102],
#                   [-0.0116, 0.0197, 0.0174],
#                   [-0.0102, 0.0174, 0.0154]])
#     R = np.eye(1)
#     F = np.zeros((3, 3))
#     var_dyn = np.zeros((3, 3))  # np.array([[1e-1**2]])
#     var_ctrl = np.array([[0]])  # np.array([[1e-1**2]])
#     return LQR(A, B, Q, R, F, var_dyn, var_ctrl)


def ADMM(sys, K0, P0, Q0, R0, rho, num_iters):
    for k in range(num_iters):
        # K step
        


if __name__ == "__main__":
    qh, rh = 1, 3

    comp = initialize_computer_LQR()

    # x0 = np.array([[-25.0136],
    #                [-18.9592],
    #                [-14.8221]])
    x0 = np.array([[10]])

    # num_to_sim = 100
    num_iters = 30
    N = 15
    resolution_dyn = 1
    resolution_ctrl = 0.1
    eps = 1e-70

    fig, axs = plt.subplots(1, 2)
    zQs = []
    zRs = []
    Qs = [comp.Q]
    Rs = [comp.R]
    eta = 0.01
    x_comp, u_comp, metadata_comp = comp.generate_trajectory(x0, N)

    # Section 3 - Inverse Optimal Control in the Noiseless Case
    # Calculate adjoint variables lambda 2...N
    # lambdas = [0]
    # for t in reversed(range(2, N)):
    #     lambdas.insert(0, np.multiply(comp.A.T, lambdas[0]) + np.multiply(comp.Q, x_comp[t]))
    #
    # print(lambdas)
    #
    # print([-np.multiply(comp.B.T, l) for l in lambdas])
    # print(u_comp)

    # Testing cvxpy
    print(x_comp)
    print(u_comp)
    print(metadata_comp)



    # plt.plot(range(N), [x[1][0, 0] for x in metadata_comp])
    # plt.grid(True)
    # plt.xlabel("t")
    # plt.ylabel(r'$J(\tau; Q_{true})$')
    # plt.title(r'Convergence of LQR Cost')
    # plt.show()


