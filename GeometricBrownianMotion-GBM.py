import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm(S0, mu, sigma, T, dt, n_paths):
    """
    Simulate multiple paths of a Geometric Brownian Motion.

    Parameters:
    S0      : float - initial asset price.
    mu      : float - drift coefficient.
    sigma   : float - volatility coefficient.
    T       : float - total time in years.
    dt      : float - time increment.
    n_paths : int   - number of simulation paths.

    Returns:
    t       : numpy.ndarray - array of time points.
    paths   : numpy.ndarray - simulated asset price paths.
    """
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)
    paths = np.zeros((n_steps, n_paths))
    paths[0] = S0
    for i in range(1, n_steps):
        # Generate random standard normal numbers for each path
        Z = np.random.standard_normal(n_paths)
        # Update the asset price using the GBM formula
        paths[i] = paths[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return t, paths

# Parameters for the GBM
S0 = 100       # Initial asset price
mu = 0.05      # Drift (5% annual return)
sigma = 0.2    # Volatility (20% annual volatility)
T = 1          # Total time in years
dt = 1/252     # Daily time steps (assuming 252 trading days in a year)
n_paths = 10   # Number of simulation paths

# Simulate the GBM paths
t, paths = simulate_gbm(S0, mu, sigma, T, dt, n_paths)

# Plot the simulated GBM paths
plt.figure(figsize=(10, 6))
for i in range(n_paths):
    plt.plot(t, paths[:, i], lw=1)
plt.xlabel("Time (Years)")
plt.ylabel("Asset Price")
plt.title("Geometric Brownian Motion Simulation")
plt.grid(True)
plt.show()
