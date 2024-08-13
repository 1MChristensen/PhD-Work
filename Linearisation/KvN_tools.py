# Tools for KvN project (1 dimension)

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from tqdm import tqdm

# Discrete Fourier Transform
def DFT(n):
   return np.fft.fft(np.eye(n)) # Returns the DFT matrix

# Inverse Discrete Fourier Transform
def IDFT(n):
    return np.fft.ifft(np.eye(n)) # Returns the inverse DFT matrix

# Momentum operator
def P(psi,k):
    psi_k = np.fft.fft(psi)         # Transform to momentum space
    p_psi_k = 1j  * psi_k * k       # Apply the momentum operator in k-space
    p_psi = np.fft.ifft(p_psi_k)    # Transform back to position space
    return -1j*np.real(p_psi)

# Generic quadratic flow operator
def F(params, x, psi):
    a, b, c = params                            # Unpack parameters
    X = np.diag(x)
    return a*X @ X @ psi + b*X @ psi + c*psi    # Apply the flow operator

# Vectorised momentum operator
def P_vec(k):
    DFT_n = DFT(len(k))           # Apply the DFT matrix
    k_DFT = 1j * np.diag(k) @ DFT_n    # Apply the momentum operator in k-space
    IDFT_n = IDFT(len(k))         # get the inverse DFT matrix
    k_IDFT = IDFT_n @ k_DFT       # Transform back to position space 
    return -1j*k_IDFT

# Vectorised generic quadratic flow operator
def F_vec(params, x):
    a, b, c = params                          # Unpack parameters 
    X = np.diag(x)
    return a*X @ X + b*X + c*np.eye(len(x))   # Apply the flow operator

# Hamiltonian
def hamiltonian(x, psi, params):
    dx= x[1] - x[0]                                                 # Get the step size
    k = 2 * np.pi * np.fft.fftfreq(len(psi), d=dx)                  # Get the k values
    return 0.5*(P(F(params, x, psi),k) + F(params, x, P(psi,k)))    # Return the Hamiltonian

# Vectorised Hamiltonian
def hamiltonian_vec(x, psi, params):
    dx= x[1] - x[0]                                                             # Get the step size
    k = 2 * np.pi * np.fft.fftfreq(len(psi), d=dx)                              # Get the k values
    return 0.5*(P_vec(k)@F_vec(params, x)@psi + F_vec(params, x)@P_vec(k)@psi)  # Return the Hamiltonian


# KvN Hamiltonian
def KvN_hamiltonian(x, params):
    print('Generating the Hamiltonian')
    n = len(x)
    H = np.zeros((n, n), dtype=complex)

    for i in tqdm(range(n)):
        psi_i = np.zeros(n)
        psi_i[i] = 1
        H[:, i] = hamiltonian(x, psi_i, params)

    #psi_i = np.eye(n)
    #H = hamiltonian(x, psi_i, params)

    return H

# Vectorised KvN Hamiltonian
def KvN_hamiltonian_vec(x, params):
    print('Generating the Hamiltonian')
    n = len(x)
    H = np.zeros((n, n), dtype=complex)

    #for i in tqdm(range(n)):
    #    psi_i = np.zeros(n)
    #    psi_i[i] = 1
    #    H[:, i] = hamiltonian(x, psi_i, params)

    psi_i = np.eye(n)
    H = hamiltonian_vec(x, psi_i, params)

    return H


# Gaussian function
def gaussian(x, mu=0, sig=0.25):
    return 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)

# State initialization
def psi0(x, x0, type='delta', n_bins=10, std=0.03, mu=1):
    print('Initializing the state')
    psi = np.zeros(len(x))

    if(type == 'delta'):
      psi[np.argmin(np.abs(x - x0))] = 1

    elif(type == 'gaussian'):
      psi = gaussian(x, mu=x0, sig=std)

      psi = psi / np.linalg.norm(psi)
      plt.plot(x, psi)
      plt.xlim([0.9, 1.1])
      plt.show()

    return psi

# Time evolution
def time_evolution(H, psi0, delta, n):
  n_psi = len(psi0)
  psi_t = np.zeros((n_psi, n), dtype=complex)
  psi_t[:, 0] = psi0
  psi = psi0

  print('Exponetiating the Hamiltonian')
  U = la.expm(-1j * delta * H)

  print('Time evolution')
  for i in tqdm(range(1, n)):
      psi = U@psi
      psi_t[:, i] = psi

  return psi_t

# Plotting
def analytical(t, params):
   A, B, C = params
   return (A-B*np.exp((t+C)*(A-B)))/(1-np.exp((t+C)*(A-B)))


def plot_evolution(x, psi_store, t, save=False):
  # Plot the time evolution
  rho_store = np.abs(psi_store)**2
  rho_store = np.flipud(rho_store)

  n_steps = len(t)
  delta = t[1] - t[0]
  grid_extent = (x[0], x[-1])

  plt.imshow(rho_store, aspect='auto', extent=[0, n_steps*delta, *grid_extent], vmax=0.01)
  plt.colorbar()
  plt.xlabel('$t$')
  plt.ylabel('$x$')
  if save:
    plt.savefig('plots/KvN_evolution.pdf')
  plt.show()

def plot_mode(x, psi_store, t, save=False, plot_analytical=False, params=None):
  # Plot the mode
  rho_store = np.abs(psi_store)**2
  #rho_store = np.flipud(rho_store)
  max_indices = np.argmax(rho_store, axis=0)
    
  plt.plot(t, x[max_indices])
  plt.xlabel('$t$')
  plt.ylabel('$x$')

  if plot_analytical:
      plt.plot(t, analytical(t, params), 'r--')

  if save:
      plt.savefig('plots/KvN_mode.pdf')
  plt.show()

def plot_std(x, psi_store, t, save=False, plot_analytical=False, log=False):
  # Plot the standard deviation
  rho_store = np.abs(psi_store)**2
  #rho_store = np.flipud(rho_store)
  std = np.sqrt(((np.diag(x**2))@rho_store-(np.diag(x)@rho_store)**2)/len(x))
  dx = x[1] - x[0]

  plt.plot(t, std)
  plt.xlabel('$t$')
  plt.ylabel('$\sigma$')
  plt.axhline(y=dx, color='grey', linestyle='--')
  if log:
    plt.yscale('log')
  if plot_analytical:
      plt.plot(t, analytical(t), 'r--')

  if save:
      plt.savefig('plots/KvN_std.pdf')
  plt.show()
  print(std.shape)

