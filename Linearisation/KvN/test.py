import KvN_tools as kvn
import numpy as np
import time


params = (-1,0,0)

# Setup grid
n_qubits = 12
n_grid = 2**n_qubits
grid_extent = (0,2)
x = np.linspace(*grid_extent, n_grid)

# Set up time
n_steps = 1500
delta = 0.01
t = np.linspace(0, n_steps*delta, n_steps)


H_fast = kvn.KvN_hamiltonian_fast(x, params)
H_old = kvn.KvN_hamiltonian_vec(x,params,deriv_type='FD')

start = time.time()
fast_results = kvn.time_evolution_fast(H_fast, x, delta, n_steps)
end = time.time()

print('Fast method:', end-start)

start = time.time()
slow_results = kvn.time_evolution(H_old, x, delta, n_steps)
end = time.time()

print('Slow method:', end-start)
