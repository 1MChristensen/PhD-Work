import numpy as np
import matplotlib.pyplot as plt 
from scipy.special import hermite, roots_hermite, factorial
import time

def evaluate_coefficients(func, degree):
    # Generate the Gauss-Hermite quadrature points and weights
    points, weights = roots_hermite(degree)

    # Initialize the coefficients array
    coefficients = np.zeros(degree + 1)

    # Evaluate the coefficients using the quadrature formula
    for i in range(degree + 1):
        coefficients[i] = (np.sqrt(np.pi)*2**i*factorial(i))**-1 * np.sum(func(points) * hermite(i)(points) * weights)

    return coefficients

# Define the function to be expandeds
n_p = 0.9
n_b = 0.2
v_b = 4.5
v_t = 0.5

def f(v):
    # Define the function to be expanded
    return (n_p/np.sqrt(2*np.pi))*np.exp(-0.5*v**2) + (n_b/np.sqrt(2*np.pi))*np.exp(-0.5*(v-v_b)**2/v_t**2) 
 
# Set the degree of the expansion
degree = 170

# Evaluate the coefficients
coefficients = evaluate_coefficients(f, degree)

print("Coefficients:", coefficients)

f_exp = 0

v = np.linspace(-3*np.pi, 3*np.pi, 1000)

# Start the timer
start_time = time.time()

# Expand the function in terms of Hermite polynomials
for i in range(degree + 1):
    f_exp += coefficients[i] * hermite(i)(v)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

plt.plot(v, f_exp)
#plt.plot(v, f(v))
plt.show()