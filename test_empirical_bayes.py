import numpy as np
import matplotlib.pyplot as plt


def wishart(V, n, p):
    # Pensky 1998 notation --> Wikipedia: r -> n, k -> p, Sigma -> V
    assert n > (p - 1)
    ret = np.zeros((p, p))
    for _ in range(n):
        # G = V*np.random.rand(p, 1)
        G = np.vstack([np.random.multivariate_normal(mean=np.zeros(p), cov=V) for _ in range(n)]).T
        ret += np.matmul(G, G.T)
    return ret


if __name__ == '__main__':
    N = 3000
    n = 3
    p = 2
    C_size = 10

    # Generate Vtrue (a positive definite scaling matrix)
    # C = np.random.rand(p, 3)
    # Vtrue = np.matmul(C, C.T)
    mean = 10
    std = 10

    Vhat = np.zeros((p, p))
    Qs = []
    Ws = []
    l2error = []
    for i in range(1, N):
        # C = np.random.normal(scale=Vtrue, size=(p, 3))
        C = np.random.normal(mean, std, size=(p, C_size))
        Q = np.matmul(C, C.T)
        Qs.append(Q)
        Vhat = 1/(n*n)*np.mean(Qs, axis=0)
        Ws.append(wishart(Vhat, n, p))

        l2error.append(np.linalg.norm(np.mean(Qs, axis=0) - np.mean(Ws, axis=0)))
        # l2error.append(np.linalg.norm(Qs[-1] - Ws[-1]))
        # l2error.append(np.linalg.norm(Qs[-1] - np.mean(Ws, axis=0)))
        # print(np.mean(Qs, axis=0))
        # print(np.mean(Ws, axis=0))
        # print(Vhat*n*n)
        # print(Qs[-1])
        # print(Ws[-1])
        # print()

    plt.plot(range(1, N), np.divide(l2error, np.linalg.norm(Ws[-1])))
    plt.grid(True)
    plt.xlabel("Iter")
    plt.ylabel(r'$|| \overline{Q} - \overline{W} || / ||\overline{W}||$')
    plt.title(r'Convergence of Empirical Bayes Model for Wishart Distribution Given $Q = CC^T$, where ' + \
              '$C \sim \operatorname{Normal}(' + str(mean) + ',' + str(std) + ')^{' + str(p) + '\\times ' + str(
        C_size) + '}$')
    plt.show()
