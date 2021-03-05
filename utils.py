import numpy as np


def is_positive_definite(A: np.ndarray):
    return np.all(np.linalg.eigvals(A) > 0)


def is_positive_semidefinite(A: np.ndarray):
    return np.all(np.linalg.eigvals(A) >= 0)


def is_symmetric(A: np.ndarray):
    return np.all(np.equal(A, np.transpose(A)))


def vectorization(A: np.ndarray):
    # Source: https://en.wikipedia.org/wiki/Vectorization_(mathematics)
    return np.ndarray.flatten(A, 'F')  # column-major ("Fortran") order


def duplication_matrix(n: int):
    # Source: https://en.wikipedia.org/wiki/Duplication_and_elimination_matrices
    D_T = np.zeros((n*(n + 1)/2, n*n))  # Transpose of the duplication matrix
    for i in range(n):
        for j in range(i, n):
            u_ij = np.zeros((n*(n + 1)/2, 1))
            u_ij[(j - 1)*n + i - j*(j - 1)/2] = 1
            T_ij = np.zeros((n, n))
            T_ij[i][j] = 1
            T_ij[j][i] = 1
            D_T += np.multiply(u_ij, vectorization(T_ij))

    return np.transpose(D_T)


# def wishart(V, n, p):
#     """
#     Samples from a Wishart distribution
#     Pensky 1998 notation --> Wikipedia: r -> n, k -> p, Sigma -> V
#     This might be implemented wrong? I'm getting E[X] = n^2*V, while it should be n*V
#
#     :param V: Covariance matrix, dimensions pxp
#     :param n: The number of degrees of freedom, n = p is the least informative
#     :param p: The dimension of the output matrix (pxp)
#     :return:
#     """
#     assert n > (p - 1)
#     G = np.vstack([np.random.multivariate_normal(mean=np.zeros(p), cov=V) for _ in range(n)]).T
#     return G@G.T


# Original Author: Prof. Nipun Batra
# nipunbatra.github.io
from math import sqrt
import matplotlib

SPINE_COLOR = 'gray'


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.
    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert (columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0)/2.0  # Aesthetic ratio
        fig_height = fig_width*golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    # NB (bart): default font-size in latex is 11. This should exactly match
    # the font size in the text if the figwidth is set appropriately.
    # Note that this does not hold if you put two figures next to each other using
    # minipage. You need to use subplots.
    params = {'backend': 'ps',
              # 'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 11,  # fontsize for x and y labels (was 12 and before 10)
              'axes.titlesize': 11,
              'font.size': 11,  # was 12 and before 10
              'legend.fontsize': 11,  # was 12 and before 10
              'xtick.labelsize': 11,
              'ytick.labelsize': 11,
              # 'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)
