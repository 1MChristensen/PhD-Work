import numpy as np
import KvN_tools_md as kvn_md

mu = 0.1

# Set up the grid
nx = ny = 60
x_extent = (-1,1)
y_extent = (-1,1)
x = np.linspace(*x_extent, nx)
y = np.linspace(*y_extent, ny)

x0 = 0.5
y0 = 0


# Set up time
n_steps = 2000
delta = 0.01
t = np.linspace(0, n_steps*delta, n_steps)

# Set up the initial state (delta in this case)
psi = kvn_md.psi0(x, y, x0, y0, type='delta')

# Generate the Hamiltonian
H = kvn_md.KvN_Hamiltonian(x, y, mu)

# Time evolve the state
psi_store = kvn_md.time_evolution(H, psi, t)

# Plot the results
kvn_md.plot_evolution(x, y, psi_store, t, save=True)

x_pred, y_pred, x_sol, y_sol = kvn_md.plot_mode(x,y,psi_store,t, save=True, numerical=True, x0=[x0,y0], mu=mu)

x_pred_length, y_pred_length = kvn_md.find_prediction_length(x_pred, y_pred, x_sol, y_sol)

print(f'Predicted length: {x_pred_length}, {y_pred_length}')