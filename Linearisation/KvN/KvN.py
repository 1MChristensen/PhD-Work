import KvN_tools as kvn
import numpy as np

# Set up the grid
n_qubits = 10
n_grid = 2**n_qubits
grid_extent = (0,2)
x = np.linspace(*grid_extent, n_grid)

# Set up time
n_steps = 3000
delta = 0.01
t = np.linspace(0, n_steps*delta, n_steps)

# Set up the initial state (delta in this case)
psi = kvn.psi0(x, 1, type='gaussian', plot=False)

# Generate the Hamiltonian to solve generic quadratic ODE, 
# i.e. ax^2 + bx + c where a,b,c are in params list
params = (-1,0,0)

H_vec = kvn.KvN_hamiltonian_vec(x, params, deriv_type='FFT')
psi_store = kvn.time_evolution(H_vec, psi, delta, n_steps)
kvn.plot_evolution(x, psi_store, t, save=True,  vmax=0.05)
kvn.plot_mode(x, psi_store, t, plot_analytical=False, params=(-1,0,0), save=True)

#kvn.plot_evolution_mode(x, psi_store, t, save=True, plot_analytical=False, params=(-1,0,0), filename='KvN_delta_FFT')
kvn.plot_initial_evolution_mode(x, psi_store, t, save=True, plot_analytical=False, params=(-1,0,0), filename='KvN_gaussian_FD')