import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Time evolution operator
def carleman_operator(n, params, t):
    a, b, c = params

    # Generate matrix
    A = np.zeros((n, n))

    # Fill upper diagonal
    A += np.diag([i*a for i in range(1, n)], k=1)

    # Fill diagonal
    A += np.diag([i*b for i in range(1, n+1)])

    # Fill lower diagonal
    A += np.diag([i*c for i in range(2, n+1)], k=-1)

    # Exponentiate the matrix
    return la.expm(t*A)

# Carleman embedding
def carleman_embedding(n, params, t):
    a, b, c = params

    # Generate matrix
    A = np.zeros((n, n))

    # Fill upper diagonal
    A += np.diag([i*a for i in range(1, n)], k=1)

    # Fill diagonal
    A += np.diag([i*b for i in range(1, n+1)])

    # Fill lower diagonal
    A += np.diag([i*c for i in range(2, n+1)], k=-1)

    return A

# Carleman numerical solutions
def carleman_numerical(C, g0, n, params, t):
    def f(t, y):
        return C @ y
    
    sol = solve_ivp(f, [t[0], t[-1]], g0, t_eval=t)

    return sol.y[0,:], sol.t

# Initial state preparation
def g0(n, x0):
    return np.array([x0**i for i in range(1, n+1)])

# Time evolution
def time_evolution(C, g0, delta, n_steps):
    print('Time evolving...')
    n = len(g0)
    g = np.zeros((n, n_steps), dtype=complex)
    g[:, 0] = g0
    g_t = g0

    # Time evolution
    for i in tqdm(range(1, n_steps)):
        g_t = C@g_t
        g[:, i] = g_t

    return g[0, :]

# Plotting
def plot_evolution(t, g, params, save=False):
    A, B, C = params
    plt.plot(t, g)
    plt.xlabel('$t$')
    plt.ylabel('$g_1(t)$')
    plt.show()