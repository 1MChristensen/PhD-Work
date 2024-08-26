import numpy as np
import KvN_tools as kvn

# Quadratic parameters
params = (-1,0,0)

# Set up the grid
n_qubits = 10
n_grid = 2**n_qubits
grid_extent = (0,1)
x = np.linspace(*grid_extent, n_grid)

x0 = 1

# Set up time
n_steps = 4000
delta = 0.01
t = np.linspace(0, n_steps*delta, n_steps)

# Initial state
psi = kvn.psi0(x, x0, type='gaussian')

H = kvn.KvN_hamiltonian_vec(x, params, deriv_type='FFT')

psi_FFT = kvn.time_evolution(H, psi, delta, n_steps)

#kvn.plot_evolution(x, psi_store, t, save=True,  vmax=0.05)
#kvn.plot_mode(x, psi_store, t, plot_analytical=True, params=params, save=True, x0=x0)

# Initial state
psi = kvn.psi0(x, x0, type='gaussian')

H = kvn.KvN_hamiltonian_vec(x, params, deriv_type='FD')

psi_FD = kvn.time_evolution(H, psi, delta, n_steps)

kvn.plot_comparison(x,psi_FFT, psi_FD, t, params, x0=1)
