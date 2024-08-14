# Tools for KvN project (multi-dimension)

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from tqdm import tqdm
from datetime import datetime

# Initial state
def psi0(x, y, x0, y0):
    psi0 = np.zeros((len(x), len(y)))
    psi0[np.argmin(np.abs(x - x0)), np.argmin(np.abs(y-y0))] = 1 
    flat_psi0 = psi0.flatten(order='F')
    return flat_psi0

# X operator
def X(x, y):
    nx = len(x)
    ny = len(y)
    return np.kron(np.eye(ny), np.diag(x))

# Y operator
def Y(x, y):
    nx = len(x)
    ny = len(y)
    return np.kron(np.diag(y), np.eye(ny))

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

    return np.kron(np.eye(ny), P_x), np.kron(P_y, np.eye(nx))

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

    psi_t = np.flipud(psi_t)
    psi_t = np.reshape(psi_t, (nx, ny, -1), order='F')

    # Collapse to x
    psi_x = np.sum(psi_t, axis=1)

    # Collapse to y
    psi_y = np.sum(psi_t, axis=0)

    plt.imshow(np.abs(psi_x), aspect='auto')
    plt.colorbar()
    plt.show()

    plt.imshow(np.abs(psi_y), aspect='auto')
    plt.colorbar()
    plt.show
    
    if save:
        plt.savefig('KvN_evolution.pdf')

    return None
