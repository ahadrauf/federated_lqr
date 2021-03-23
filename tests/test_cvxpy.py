import numpy as np
import cvxpy as cp
import warnings

try:
    import mosek
    print("Using MOSEK solver")
    solver = cp.MOSEK
except:
    warnings.warn("Solver MOSEK is not installed, falling back to SCS.")
    solver = cp.SCS

if __name__ == '__main__':
    m = 2000
    n = 1000
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    for x in range(10):
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A@x - b)),
                          [x >= 0])

        prob.solve(solver=solver)
        print("Solve times:", prob.solver_stats.solve_time, end=' ')

        prob.solve(warm_start=True, solver=solver)
        print(prob.solver_stats.solve_time, end=' ')

        prob.solve(warm_start=True, solver=solver)
        print(prob.solver_stats.solve_time)
