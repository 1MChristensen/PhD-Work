# Tools for KvN project (multi-dimension)

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from tqdm import tqdm

# Momentum operator
def P(psi, k):
    psi_k = np.fft.fftn(psi)  # Transform to momentum space
    p_psi_k = 1j * k * psi_k  # Apply the momentum operator in k-space
    p_psi = np.fft.ifftn(p_psi_k)  # Transform back to position space
    return -1j*np.real(p_psi)

