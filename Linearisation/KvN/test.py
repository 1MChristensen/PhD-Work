import numpy as np
import matplotlib.pyplot as plt


# Setup grid
n_qubits = 10
n_grid = 2**n_qubits
grid_extent = (-5,5)
x = np.linspace(*grid_extent, n_grid)

dx = x[1] - x[0]

g_store = np.zeros((n_grid,2))

def gaussian(x, mu=0, sigma=0.1):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

for t in range(2):
    g_store[:,t] = gaussian(x)

# Get the maximum Fourier frequency
def check_nyquist(x, psi_store, t, epsilon=0.1):
    dx = x[1] - x[0]

    nt = psi_store.shape[1]

    freq = np.fft.fftshift(np.fft.fftfreq(len(x), dx))

    B = np.zeros(nt)

    for t in range(nt):
        fft = np.abs(np.fft.fftshift(np.fft.fft(psi_store[:,t])))
        if len(freq[fft < epsilon]) == 0:
            break
        B[t] = np.min(np.abs(freq[fft < epsilon]))

    return B

B = check_nyquist(x, g_store, 1)

print(f'B: {B}, dx: {dx}, 1/2B: {1/(2*B)}')


freq = np.fft.fftshift(np.fft.fftfreq(len(x), dx))
fft = np.fft.fftshift(np.fft.fft(g_store[:,0]))

#plt.plot(x, gaussian(x))
plt.plot(freq, np.abs(fft))
plt.axvline(x = B[0], color='red')
plt.savefig('plots/test.pdf')
plt.show()