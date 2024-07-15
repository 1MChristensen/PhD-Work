# Tools for KvN project

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from tqdm import tqdm

# Momentum operator
def P(psi, k):
    psi_k = np.fft.fft(psi)  # Transform to momentum space
    p_psi_k = 1j * k * psi_k  # Apply the momentum operator in k-space
    p_psi = np.fft.ifft(p_psi_k)  # Transform back to position space
    return -1j*np.real(p_psi)

# generic quadratic flow operator
def F(params, x, psi):
    a, b, c = params
    X = np.diag(x)
    return a*X @ X @ psi + b*X @ psi + c*psi

# Hamiltonian
def hamiltonian(x, psi, params):
    dx= x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(len(psi), d=dx)
    return 0.5*(P(F(params, x, psi),k) + F(params, x, P(psi,k)))

# KvN Hamiltonian
def KvN_hamiltonian(x, params):
    print('Generating the Hamiltonian')
    n = len(x)
    H = np.zeros((n, n), dtype=complex)

    for i in tqdm(range(n)):
        psi_i = np.zeros(n)
        psi_i[i] = 1
        H[:, i] = hamiltonian(x, psi_i, params)

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
      bin_centers = np.linspace(mu - 3*std, mu + 3*std, n_bins)
      bin_values = gaussian(bin_centers, mu, std)
      bin_intervals = bin_centers[1] - bin_centers[0]

      for i in range(n_bins):
         psi[np.where(np.abs(x - bin_centers[i]) < bin_intervals/2)] = bin_values[i]
      
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

def plot_mode(x, psi_store, t, save=False, analytical=None):
  # Plot the mode
  rho_store = np.abs(psi_store)**2
  #rho_store = np.flipud(rho_store)
  max_indices = np.argmax(rho_store, axis=0)
    
  plt.plot(t, x[max_indices])
  plt.xlabel('$t$')
  plt.ylabel('$x$')

  if analytical is not None:
      plt.plot(t, analytical(t), 'r--')

  if save:
      plt.savefig('plots/KvN_mode.pdf')
  plt.show()

def plot_std(x, psi_store, t, save=False, analytical=None, log=False):
  # Plot the standard deviation
  rho_store = np.abs(psi_store)**2
  #rho_store = np.flipud(rho_store)
  std = np.sqrt((x**2)@rho_store-(x@rho_store)**2)
  dx = x[1] - x[0]



  plt.plot(t, std)
  plt.xlabel('$t$')
  plt.ylabel('$\sigma$')
  plt.axhline(y=dx, color='grey', linestyle='--')
  if log:
    plt.yscale('log')
  if analytical is not None:
      plt.plot(t, analytical(t), 'r--')

  if save:
      plt.savefig('plots/KvN_std.pdf')
  plt.show()