# Tools for KvN project (1 dimension)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.linalg as la
import scipy.sparse as sparse
from scipy.integrate import solve_ivp
from tqdm import tqdm
import pandas as pd

plt.style.use('ggplot')


# Discrete Fourier Transform
def DFT(n):
   return np.fft.fft(np.eye(n)) # Returns the DFT matrix

# Inverse Discrete Fourier Transform
def IDFT(n):
    return np.fft.ifft(np.eye(n)) # Returns the inverse DFT matrix

# Vectorised momentum operator
def P(x, type='FFT'):
    dx = x[1] - x[0]
    if type=='FFT': # FFT based derivative
        k = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)  # Get k values
        DFT_n = DFT(len(k))                           # Apply the DFT matrix
        k_DFT = 1j * np.diag(k) @ DFT_n               # Apply the momentum operator in k-space
        IDFT_n = IDFT(len(k))                         # get the inverse DFT matrix
        k_IDFT = IDFT_n @ k_DFT                       # Transform back to position space 
        P = -1j*k_IDFT


    if type=='FD': # Finite difference derivative
        P = 1.0j/(2*dx)*(np.diag(np.ones(len(x)-1), k=-1) - np.diag(np.ones(len(x)-1), k=1))
    return P

# Vectorised generic quadratic flow operator
def F(params, x, type='quadratic'):
    if type == 'quadratic':
        a, b, c = params                          # Unpack parameters 
        X = np.diag(x)
        return a*X @ X + b*X + c*np.eye(len(x))   # Apply the flow operator
        
# Vectorised Hamiltonian
def hamiltonian_vec(x, psi, params, deriv_type='FFT'):
    dx= x[1] - x[0]                                                                                              # Get the step size
    return 0.5*(P(x, type=deriv_type)@F(params, x)@psi + F(params, x)@P(x,type=deriv_type)@psi)  # Return the Hamiltonian

# Vectorised KvN Hamiltonian
def KvN_hamiltonian(x, params, deriv_type='FFT'):
    print('Generating the Hamiltonian')
    n = len(x)
    H = np.zeros((n, n), dtype=complex)

    #for i in tqdm(range(n)):
    #    psi_i = np.zeros(n)
    #    psi_i[i] = 1
    #    H[:, i] = hamiltonian(x, psi_i, params)

    psi_i = np.eye(n)
    H = hamiltonian_vec(x, psi_i, params, deriv_type=deriv_type)

    return H

# Faster Hamiltonian
def KvN_hamiltonian_fast(x, params):
    print('Generating the Hamiltonian')
    n = len(x)
    H = np.zeros((n, n), dtype=complex)

    F = F(params, x)
    P = P(x, type='FD')

    H = 0.5*(F@P + P@F)

    return sparse.csc_matrix(H)

# Gaussian function
def gaussian(x, mu=0, sig=0.25):
    return 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)

# State initialization
def psi0(x, x0, type='delta', n_bins=10, std=0.03, plot=False):
    print('Initializing the state')
    psi = np.zeros(len(x))

    if(type == 'delta'):
      psi[np.argmin(np.abs(x - x0))] = 1

    elif(type == 'gaussian'):
      psi = gaussian(x, mu=x0, sig=std)

      psi = psi / np.linalg.norm(psi)
    
    else: 
       raise ValueError(f"No regocnised initial conidtion {type}")


    if plot:
      plt.plot(x, psi)
      plt.xlim([x0 - 0.1, x0 + 0.1])
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
      #print(np.linalg.norm(psi)**2)

  return psi_t

# Faster time evolution
def time_evolution_fast(H, psi0, delta, n):
    n_psi = len(psi0)
    psi_t = np.zeros((n_psi, n), dtype=complex)
    psi_t[:, 0] = psi0
    psi = psi0

    print('Exponetiating the Hamiltonian')
    U = sparse.linalg.expm(-1j * delta * H)

    print('Time evolution')
    for i in tqdm(range(1, n)):
      psi = U@psi
      psi_t[:, i] = psi

    return psi_t

# Plotting
def analytical(t, params):
   A, B, C = params
   return (A-B*np.exp((t+C)*(A-B)))/(1-np.exp((t+C)*(A-B)))


def plot_evolution(x, psi_store, t, save=False, filename='test', vmin=0.001, vmax=0.01):
  # Plot the time evolution
  rho_store = np.abs(psi_store)**2
  rho_store = np.flipud(rho_store)

  n_steps = len(t)
  delta = t[1] - t[0]
  grid_extent = (x[0], x[-1])

  plt.imshow(rho_store, aspect='auto', extent=[0, n_steps*delta, *grid_extent], vmin=vmin, vmax=vmax)
  plt.colorbar()
  plt.xlabel('$t$')
  plt.ylabel('$x$')
  plt.grid(False)
  if save:
    plt.savefig(f'plots/{filename}.pdf')
  plt.show()

def plot_mode(x, psi_store, t, save=False, plot_analytical=False, params=(-1,0,0), x0=1):
  # Plot the mode
  rho_store = np.abs(psi_store)**2
  #rho_store = np.flipud(rho_store)
  max_indices = np.argmax(rho_store, axis=0)
  
  plt.clf()
  plt.plot(t, x[max_indices])
  plt.xlabel('$t$')
  plt.ylabel('$x$')

  if plot_analytical:
      
      sol = solve_ivp(numerical, (t[0], t[-1]), y0=[x0], t_eval=t, args=params)
      #print(len(sol['t']))
      plt.plot(sol['t'], sol['y'].T, 'r--')

  if save:
      plt.savefig('plots/KvN_mode.pdf')
  plt.show()

def plot_std(x, psi_store, t, save=False, plot_analytical=False, log=False):
  # Plot the standard deviation
  rho_store = np.abs(psi_store)**2
  #rho_store = np.flipud(rho_store)
  std = np.sqrt((x**2)@rho_store-(x@rho_store)**2)/len(x)
  dx = x[1] - x[0]
  print(std.shape)

  plt.plot(t, std)
  plt.xlabel('$t$')
  plt.ylabel('$\sigma$')
  plt.axhline(y=dx, color='grey', linestyle='--')
  if log:
    plt.yscale('log')
  if plot_analytical:
      x0 = [1]
      sol = solve_ivp(numerical, (t[0], t[-1]), y0=x0, t_eval=t, args=params)
      #print(len(sol['t']))
      plt.plot(sol['t'], sol['y'].T, 'r--')

  if save:
      plt.savefig('plots/KvN_std.pdf')
  plt.show()

def numerical(t, x, a,b,c):
   return a*x**2 + b*x + c

def plot_comparison(x ,psi_fft, psi_fd, t, params, x0=1):
  rho_fft = np.abs(psi_fft)**2
  rho_fd = np.abs(psi_fd)**2
  
  fft_max_idx = np.argmax(rho_fft, axis=0)
  fd_max_idx = np.argmax(rho_fd, axis=0)

  plt.clf()
  plt.plot(t, x[fft_max_idx], label='FFT based derivative', linewidth=4)
  plt.plot(t, x[fd_max_idx], label='FD based derivative', linewidth=2)
  
  sol = solve_ivp(numerical, (t[0], t[-1]), y0=[x0], t_eval=t, args=params)
  plt.plot(sol['t'], sol['y'].T, linestyle=':', color='black', label='Numerical solution')
  plt.xlabel('$t$')
  plt.ylabel('$x$')
  plt.legend()

  plt.savefig('plots/KvN_FFT_vs_FD.pdf')
  plt.show()
  return None

# Plot both evolution and mode on a subplots
def plot_evolution_mode(x, psi_store, t, save=True, plot_analytical=False, params=(-1,0,0), x0=1, filename='test'):
     # Plot the time evolution
    rho_store = np.abs(psi_store)**2
    rho_store = np.flipud(rho_store)

    n_steps = len(t)
    delta = t[1] - t[0]
    where_x0 = np.argmin((x-x0)**2)
    grid_extent = (x[0], x0)

    rho_store = rho_store[where_x0:-1, :]

    fig, ax = plt.subplots(2,1, figsize=(6,6))

    cax = ax[0].imshow(rho_store, aspect='auto', extent=[0, n_steps*delta, *grid_extent], vmax=0.1 ,cmap='plasma')
    cbar = fig.colorbar(cax)
    ax[0].set_xlabel('$t$', fontsize=16)
    ax[0].set_ylabel('$x(t)$', fontsize=16)
    ax[0].grid(False)
    for spine in ax[0].spines.values():
        spine.set_visible(True)  # Turn on the spine
        spine.set_color('grey')  # Set spine color
        spine.set_linewidth(1)    # Set spine line width

    for spine in cbar.ax.spines.values():
        spine.set_visible(True)  # Make the spine visible
        spine.set_color('grey')  # Set the color of the spine
        spine.set_linewidth(1)    # Set the width of the spine


    rho_store = np.flipud(rho_store)
    max_indices = np.argmax(rho_store, axis=0)

    ax[1].plot(t, x[max_indices])
    ax[1].set_xlabel('$t$', fontsize=16)
    ax[1].set_ylabel('$x(t)$', fontsize=16)

    ax[0].text(0,1.05, 'a)', fontsize=16, color='#3C3C3C')
    ax[1].text(-0.5,1.1, 'b)', fontsize=16, color='#3C3C3C')

    #plt.clf()
    #plt.plot(t, x[max_indices])
    #plt.xlabel('$t$')
    #plt.ylabel('$x$')

    #if plot_analytical:
    #    
    #    sol = solve_ivp(numerical, (t[0], t[-1]), y0=[x0], t_eval=t, args=params)
    #    #print(len(sol['t']))
    #    plt.plot(sol['t'], sol['y'].T, 'r--')

    #if save:
    #    plt.savefig('plots/KvN_mode.pdf')
    #plt.show()

    plt.tight_layout()

    if save:
       plt.savefig(f'plots/{filename}.pdf')
    plt.show()



    #plt.imshow(rho_store, aspect='auto', extent=[0, n_steps*delta, *grid_extent], vmin=vmin, vmax=vmax)
    #plt.colorbar()
    #plt.xlabel('$t$')
    #plt.ylabel('$x$')
    #plt.grid(False)
    #if save:
    #  plt.savefig(f'plots/{filename}.pdf')
    #plt.show()

# Plot initial, evolution and mode on a subplots
def plot_initial_evolution_mode(x, psi_store, t, save=True, plot_analytical=False, params=(-1,0,0), x0=1, filename='test'):
     # Plot the time evolution
    rho_store = np.abs(psi_store)**2
    rho_store = np.flipud(rho_store)

    n_steps = len(t)
    delta = t[1] - t[0]
    where_x0 = np.argmin((x-x0)**2)
    grid_extent = (x[0], x0)

    #print(rho_store)

    rho_backup = rho_store
    #rho_store = rho_store[where_x0:-1, :]

    #df = pd.DataFrame(rho_store)
    #df.to_csv('matrix.csv')

    fig, ax = plt.subplots(3,1, figsize=(6,8))

    cax = ax[0].imshow(rho_store, aspect='auto', extent=[0, n_steps*delta, *grid_extent], vmax=0.1,cmap='plasma')
    cbar = fig.colorbar(cax)
    ax[0].set_xlabel('$t$', fontsize=16)
    ax[0].set_ylabel('$x(t)$', fontsize=16)
    ax[0].grid(False)
    #np.savetxt("matrix.txt", rho_store, delimiter=",", fmt="%f")

    for spine in ax[0].spines.values():
        spine.set_visible(True)  # Turn on the spine
        spine.set_color('grey')  # Set spine color
        spine.set_linewidth(1)    # Set spine line width

    for spine in cbar.ax.spines.values():
        spine.set_visible(True)  # Make the spine visible
        spine.set_color('grey')  # Set the color of the spine
        spine.set_linewidth(1)    # Set the width of the spine

    #print(rho_store)

    rho_store = np.flipud(rho_store)
    max_indices = np.argmax(rho_store, axis=0)

    ax[1].plot(t, x[max_indices])
    ax[1].set_xlabel('$t$', fontsize=16)
    ax[1].set_ylabel('$x(t)$', fontsize=16)

    #plt.clf()
    #plt.plot(t, x[max_indices])
    #plt.xlabel('$t$')
    #plt.ylabel('$x$')

    #if plot_analytical:
    #    
    #    sol = solve_ivp(numerical, (t[0], t[-1]), y0=[x0], t_eval=t, args=params)
    #    #print(len(sol['t']))
    #    plt.plot(sol['t'], sol['y'].T, 'r--')

    #if save:
    #    plt.savefig('plots/KvN_mode.pdf')
    #plt.show()

    psi0 = rho_store[:,0]
    #psi0 = rho_backup[:,0]
    
    ax[2].plot(x,psi0)
    ax[2].set_xlabel('$x(t=0)$', fontsize=16)
    ax[2].set_ylabel('$\psi(x,t)$', fontsize=16)
    ax[2].set_xlim([x0-0.2, x0+0.2])

    #ax[0].text(0.6, 1.05, 'a)', fontsize=16, color='#3C3C3C')
    #ax[1].text(-0.5, 1.1, 'b)', fontsize=16, color='#3C3C3C')
    #ax[2].text(0.03, 1.05, 'c)', transform=ax[2].transAxes, fontsize=16, color='#3C3C3C')

    plt.tight_layout()

    if save:
       plt.savefig(f'plots/{filename}.pdf')
    plt.show()



    #plt.imshow(rho_store, aspect='auto', extent=[0, n_steps*delta, *grid_extent], vmin=vmin, vmax=vmax)
    #plt.colorbar()
    #plt.xlabel('$t$')
    #plt.ylabel('$x$')
    #plt.grid(False)
    #if save:
    #  plt.savefig(f'plots/{filename}.pdf')
    #plt.show()

# Plot std, evolution and mode on a subplots
def plot_std_evolution_mode(x, psi_store, t, save=True, plot_analytical=False, params=(-1,0,0), x0=1, filename='test'):
     # Plot the time evolution
    rho_store = np.abs(psi_store)**2
    rho_store = np.flipud(rho_store)

    n_steps = len(t)
    delta = t[1] - t[0]
    where_x0 = np.argmin((x-x0)**2)
    #grid_extent = (x[0], x0)
    grid_extent = (x[0], x[-1])
    #print(rho_store)

    rho_backup = rho_store.copy()
    #rho_store = rho_store[where_x0:-1, :]

    #df = pd.DataFrame(rho_store)
    #df.to_csv('matrix.csv')

    fig, ax = plt.subplots(3,1, figsize=(6,8))

    minimum_val = 1e-8

    rho_backup[rho_backup<=minimum_val] = minimum_val

    cax = ax[0].imshow(rho_backup, aspect='auto', extent=[0, n_steps*delta, *grid_extent], norm=LogNorm(vmin=minimum_val, vmax=rho_backup.max()),cmap='plasma')
    cbar = fig.colorbar(cax)
    ax[0].set_xlabel('$t$', fontsize=16)
    ax[0].set_ylabel('$x(t)$', fontsize=16)
    ax[0].grid(False)
    #np.savetxt("matrix.txt", rho_store, delimiter=",", fmt="%f")

    for spine in ax[0].spines.values():
        spine.set_visible(True)  # Turn on the spine
        spine.set_color('grey')  # Set spine color
        spine.set_linewidth(1)    # Set spine line width

    for spine in cbar.ax.spines.values():
        spine.set_visible(True)  # Make the spine visible
        spine.set_color('grey')  # Set the color of the spine
        spine.set_linewidth(1)    # Set the width of the spine

    #print(rho_store)

    #rho_store = rho_backup
    rho_store = np.flipud(rho_store)
    max_indices = np.argmax(rho_store, axis=0)

    ax[1].plot(t, x[max_indices], label='KvN solution')
    ax[1].set_xlabel('$t$', fontsize=16)
    ax[1].set_ylabel('$x(t)$', fontsize=16)

    sol = solve_ivp(numerical, (t[0], t[-1]), y0=[x0], t_eval=t, args=params)
    ax[1].plot(sol.t, sol.y.T, linestyle='--', label='Numerical solution')
    ax[1].legend()
    psi0 = rho_store[:,0]

    dx = x[1] - x[0]

    X = np.diag(x)
    
    term_1 = X@X@rho_store
    term_2 = X@rho_store

    std = np.sqrt(np.sum(term_1, axis=0) - (np.sum(term_2, axis=0))**2)
    
    epsilon = 1e-10

    nyquist_condition = np.pi * std / np.sqrt(np.log(2*np.pi*std**2/epsilon))

    print(f'Min. Nyquist condition: {np.min(nyquist_condition)}')

    #std = np.std(rho_store, axis=0)
    #print(std.shape)
    ax[2].plot(t, std, label='STD')
    ax[2].plot(t, nyquist_condition, label='Nyquist')
    ax[2].axhline(y=dx, color='black', linestyle='--')
    ax[2].set_xlabel('$t$')
    ax[2].set_ylabel('STD')
    ax[2].set_yscale('log')

    #print(rho_store.shape)
    print('Min. STD:', np.min(std))
    print('dx:', dx)

    #print(idx)
    #print(f'Crossover point: {t[idx]}')

    if len(np.where(nyquist_condition < dx)[0]) != 0:
        # Plot Nyquist cutoff
        idx = np.min(np.where(nyquist_condition < dx)[0])
        ax[0].axvline(t[idx])
    if len(np.where(std < dx)[0]) !=0:
        # Plot STD cutoff
        idx = np.min(np.where(std < dx)[0])
        ax[0].axvline(t[idx])



    plt.tight_layout()

    if save:
       plt.savefig(f'plots/{filename}.pdf')
    plt.show()

# Get sig0
def sig0(x, psi0):
    X = np.diag(x)

    rho = np.abs(psi0)**2

    term_1 = X@X@rho
    term_2 = X@rho

    std = np.sqrt(np.sum(term_1, axis=0) - (np.sum(term_2, axis=0))**2)

    return std


# Time evolve the STD of the system
def sigma(x, psi0, sig0, H, N, delta):
   
    # Time evolution operators
    U1 = la.expm(-1.0j*H*delta)
    U2 = la.expm(1.0j*H.conjugate()*delta)

    U1_n = U1.copy()
    U2_n = U2.copy()

    I_n = np.eye(U1.shape[0])

    X = np.diag(x)

    sigs = np.zeros(N, dtype=complex)

    sigs[0] = sig0

    T1 = delta**(-1) * (U1 - I_n)@U1
    T2 = delta**(-1) * (U2 - I_n)@U2

    for n in tqdm(range(1, N), desc='Sigma evolution'):
        # First determine S
        T1 = T1@U1
        T2 = T2@U2

        S1 = (T1 @ psi0).T @ X @ X @ (U2_n @ psi0) + (U1_n @ psi0).T @ X @ X @ (T2 @ psi0)
        S2 = (T1 @ psi0).T @ X @ (U2_n @ psi0) + (U1_n @ psi0).T @ X @ (T2 @ psi0)

        # Next determine D        
        D = (U1_n @ psi0).T @ X @ (U2_n @ psi0)

        if(n==1):
           print(f'S1: {S1}, S2: {S2}, D: {D}')

        sigs[n] = sigs[n-1] + 0.5 * delta * sigs[n-1]**(-1) * (S1 - 2*D*S2)

        U1_n = U1_n@U1
        U2_n = U2_n@U2

    return sigs

# Plot the sigmas for comparison
def plot_sigma(x, psi_store, t, sig_num):
    
    rho_store = np.abs(psi_store)**2
    rho_store = np.flipud(rho_store)

    X = np.diag(x)

    term_1 = X@X@rho_store
    term_2 = X@rho_store

    std = np.sqrt(np.sum(term_1, axis=0) - (np.sum(term_2, axis=0))**2)

    plt.plot(t, std)
    plt.plot(t, sig_num)
    plt.savefig('plots/test.pdf')
    plt.show()

