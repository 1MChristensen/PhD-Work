import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Set print options for numpy
np.set_printoptions(linewidth=np.inf)

# Set parameter of problem
k0 = 0.5
alpha = 1e-3

# Order of expansion
n = 99

# Upper and lower diagonals
a = c = np.array([(i+1)**0.5 for i in range(n)])

# Add collision term -- WIP
B = 0.0125
b = np.array([-1j*B*i/k0 for i in range(n+1)])

# Generate base matrix
matrix = np.diag(a,1) + np.diag(c,-1) + np.diag(b)

# Alter the first upper and lower diagonal entries
matrix[0,1] = matrix[1,0] = (1 + k0**(-2))**0.5

# Calculate eigenvalues and eigenvectors
eigval = np.linalg.eig(matrix)[0]
eigvec = np.linalg.eig(matrix)[1]

# Calculate total field
tot = 0

t = np.linspace(0,30,1000)

for i in range(n+1):
    #tot += (1j*alpha)/(2*k0)*eigvec[:,i][0]**2 *np.exp(1j*k0*eigval[i]*t)
    r_val = np.real(eigval[i]); i_val = np.imag(eigval[i])
    r_vec = np.real(eigvec[:,i])[0]; i_vec = np.imag(eigvec[:,i])[0]
    tot += (1j*alpha)/(2*k0)*(r_vec*np.cos(i_val*t) - i_vec*np.sin(i_val*t))*np.exp(r_val*t)

# Find the peaks in the oscillations
peaks, _ = find_peaks(abs(tot))

# Fit a line to the log of the peaks
coefficients = np.polyfit(t[peaks], np.log(abs(tot)[peaks]), 1)

print('Coefficients', coefficients)

# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(t, abs(tot))
axs[0].plot(t[peaks], abs(tot)[peaks], 'x')
axs[0].set_xlabel('t')
axs[0].set_ylabel('E')
axs[0].set_title('Plot of E against t')

axs[1].plot(t[peaks], np.log(abs(tot)[peaks]), 'x')
axs[1].plot(t, coefficients[0]*t + coefficients[1])
axs[1].set_xlabel('t')
axs[1].set_ylabel('log(E)')
axs[1].set_title(f'Decay constant: {-coefficients[0]}')
plt.tight_layout()

plt.savefig('subplot.pdf')
plt.show()