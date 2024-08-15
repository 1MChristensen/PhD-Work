# Tools for KvN project (multi-dimension)

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.integrate import solve_ivp
from tqdm import tqdm
from datetime import datetime

# Multivariate Gaussian (assuming the covariance matrix is identity*cov)
def gaussian(x,y, mu, cov):
    return (2*np.pi*cov)**-1*np.exp(-0.5/cov*((x-mu[0])**2 + (y-mu[1])**2))

# Initial state
def psi0(x, y, x0, y0, type='delta', cov=0.03):
    psi0 = np.zeros((len(x), len(y)))
    mu = (x0, y0)

    if type=='delta':
        psi0[np.argmin(np.abs(y-y0)), np.argmin(np.abs(x - x0))] = 1 
        
        plt.imshow(psi0, aspect='auto', extent=[x[0],x[-1],y[0],y[-1]], origin='lower')
        plt.savefig('initial.pdf')
        plt.show()

        flat_psi0 = psi0.flatten(order='F')

    if type=='gaussian':
        X, Y = np.meshgrid(x,y)

        psi0 = gaussian(X, Y, (x0,y0), cov)
        #psi0 = np.flipud(psi0)
        plt.imshow(psi0, aspect='auto', extent=[x[0],x[-1],y[0],y[-1]], origin='lower')
        plt.savefig('initial.pdf')
        plt.show()

        flat_psi0 = psi0.flatten(order='F')

    return flat_psi0

# X operator
def X(x, y):
    nx = len(x)
    ny = len(y)
    #return np.kron(np.eye(nx), np.diag(x))
    return np.kron(np.diag(x), np.eye(ny))

# Y operator
def Y(x, y):
    nx = len(x)
    ny = len(y)
    #return np.kron(np.diag(y), np.eye(ny))
    return np.kron(np.eye(nx), np.diag(y))

# DFT operator
def DFT(n):
    return np.fft.fft(np.eye(n))

# Inverse DFT operator
def IDFT(n):
    return np.fft.ifft(np.eye(n))

# Momentum operator
def P(x, y):
    nx = len(x)
    ny = len(y)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    kx = np.diag(1j * np.fft.fftfreq(nx, dx) * 2 * np.pi)
    ky = np.diag(1j * np.fft.fftfreq(ny, dy) * 2 * np.pi)

    P_x = -1j*IDFT(nx) @ kx @ DFT(nx)
    P_y = -1j*IDFT(ny) @ ky @ DFT(ny)

    return np.kron(P_x, np.eye(ny)), np.kron(np.eye(nx), P_y)

# Flow operator
def F(x, y, mu):
    nx = len(x)
    ny = len(y)
    X_op = X(x, y)
    Y_op = Y(x, y)
    return Y_op, -X_op + mu*(Y_op - X_op@X_op@Y_op)

# Hamiltonian
def KvN_Hamiltonian(x, y, mu):
    print('Generating the Hamiltonian')

    nx = len(x)
    ny = len(y)

    H = np.zeros((nx*ny, nx*ny), dtype=complex)
    
    P_x, P_y = P(x, y)
    F_x, F_y = F(x, y, mu)

    H = 0.5*(P_x @ F_x + F_x @ P_x + P_y @ F_y + F_y @ P_y)

    return H

# Time evolution
def time_evolution(H, psi0, t):
    n_psi = len(psi0)
    n_t = len(t)
    delta = t[1] - t[0]

    psi_t = np.zeros((n_psi, n_t), dtype=complex)
    psi_t[:, 0] = psi0
    psi = psi0

    print('Exponetiating the Hamiltonian')
    U = la.expm(-1j*H*delta)

    print('Time evolution')
    for i in tqdm(range(1, n_t)):
        psi = U@psi
        psi_t[:, i] = psi

    return psi_t 

# Plotting
def plot_evolution(x, y, psi_t, t, save=False):
    nx = len(x)
    ny = len(y)    

    #psi_t = np.flipud(psi_t)
    psi_t = np.reshape(psi_t, (nx, ny, -1), order='F')

    # Collapse to x
    psi_x = np.sum(psi_t, axis=0)

    # Collapse to y
    psi_y = np.sum(psi_t, axis=1)

    plt.imshow(np.abs(psi_x)**2, aspect='auto',extent=[0, t[-1], x[0], x[-1]], origin='lower')
    plt.colorbar()
    if save:
        plt.savefig('plots/KvN_evolution_x.pdf')
    
    plt.show()

    plt.imshow(np.abs(psi_y)**2, aspect='auto', extent=[0, t[-1], y[0], y[-1]], origin='lower')
    #plt.colorbar()
    if save:
        plt.savefig('plots/KvN_evolution_y.pdf')

    plt.show()

    return None

# Plot the mode
def plot_mode(x, y, psi_t, t, save=False, numerical=False, x0=[-1,1], mu=0.5):
    plt.show()
    nx = len(x)
    ny = len(y)

    #psi_t = np.flipud(psi_t)
    psi_t = np.reshape(psi_t, (nx, ny, -1), order='F')

    rho_t = np.abs(psi_t)**2

    # Collapse to x
    rho_x = np.sum(rho_t, axis=0)

    # Collapse to y
    rho_y = np.sum(rho_t, axis=1)

    x_indices  = np.argmax(rho_x, axis=0)
    y_indices  = np.argmax(rho_y, axis=0)

    plt.clf()
    plt.plot(t, x[x_indices], label='KvN $x$ prediction')
    plt.plot(t, y[y_indices], label='KvN $y$ prediction')

    if numerical:
        sol = solve_ivp(f, [t[0], t[-1]], x0, t_eval=t, args=[mu])
        x_sol, y_sol = sol['y']
        plt.plot(t, x_sol, label='Numerical $x$ solution')
        plt.plot(t, y_sol, label='Numerical $y$ solution')
    plt.legend()
    if save:
        plt.savefig('plots/KvN_mode_xy.pdf')

    plt.show()
    
    if numerical:
        return x[x_indices], y[y_indices], x_sol, y_sol
    
    return None

# Van der Pol oscillator
def f(t, y, mu):
    return np.array([y[1], -y[0] + mu*(1-y[0]**2)*y[1]])

def find_prediction_length(x_pred, y_pred, x_sol, y_sol, t, tol=0.1):
    n = len(x_pred)

    for i in range(n):
        if np.abs(x_pred[i] - x_sol[i]) > tol:
            x_pred_length = t[i]
            break

    for i in range(n):
        if np.abs(y_pred[i] - y_sol[i]) > tol:
            y_pred_length = t[i]
            break

    return x_pred_length, y_pred_length
