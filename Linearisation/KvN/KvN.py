import KvN_tools as kvn
import numpy as np

# Set up the grid
n_qubits = 11
n_grid = 2**n_qubits
grid_extent = (-2,2)
x = np.linspace(*grid_extent, n_grid)
x0 = 1.5

# Set up time
n_steps = 10000
delta = 0.001
t = np.linspace(0, n_steps*delta, n_steps)

# Set up the initial state (delta in this case)
psi = kvn.psi0(x,x0, type='gaussian', plot=False, std=0.03)

# Generate the Hamiltonian to solve generic quadratic ODE, 
# i.e. x' = ax^2 + bx + c where a,b,c are in params list
params = (-1,0,0)

H_vec = kvn.KvN_hamiltonian_vec(x, params, deriv_type='FFT')
psi_store = kvn.time_evolution(H_vec, psi, delta, n_steps)
#kvn.plot_evolution(x, psi_store, t, save=False,  vmax=0.05)
#kvn.plot_mode(x, psi_store, t, plot_analytical=False, params=(-1,0,0), save=False)

#kvn.plot_evolution_mode(x, psi_store, t, save=True, plot_analytical=False, params=(-1,0,0), filename='KvN_delta_FFT')
#kvn.plot_initial_evolution_mode(x, psi_store, t, save=True, plot_analytical=False, params=(-1,0,0), filename='test')
kvn.plot_std_evolution_mode(x, psi_store, t, save=True, plot_analytical=False, params=params, filename='test', x0=x0)