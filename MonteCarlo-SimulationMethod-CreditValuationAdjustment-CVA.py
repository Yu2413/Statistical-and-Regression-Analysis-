import numpy as np


def simulate_exposures(S0, mu, sigma, T, dt, n_sim):
    """
    Simulate exposure paths using geometric Brownian motion.

    Parameters:
    S0    : float - initial exposure level
    mu    : float - drift (annualized)
    sigma : float - volatility (annualized)
    T     : float - total time horizon in years
    dt    : float - time step in years
    n_sim : int   - number of simulation paths

    Returns:
    exposures : numpy.ndarray of shape (n_sim, n_steps)
    """
    n_steps = int(T / dt)
    exposures = np.zeros((n_sim, n_steps))

    for i in range(n_sim):
        S = S0
        for t in range(n_steps):
            # Update exposure using geometric Brownian motion dynamics
            S *= np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn())
            exposures[i, t] = max(S, 0)  # Ensure exposure remains non-negative
    return exposures


# Parameters for exposure simulation
S0 = 100  # initial exposure value
mu = 0.05  # drift (5% annualized)
sigma = 0.2  # volatility (20% annualized)
T = 5  # simulation for 5 years
dt = 1  # yearly time steps
n_sim = 10000  # number of simulation paths

# Run the simulation
exposure_paths = simulate_exposures(S0, mu, sigma, T, dt, n_sim)

# Calculate the Expected Exposure (EE) at each time step (average over simulation paths)
EE = exposure_paths.mean(axis=0)

# Parameters for CVA calculation
r = 0.03  # risk-free rate (3%)
LGD = 0.6  # Loss Given Default (60%)
PD = 0.02  # Annual default probability (2%) per time step

# Time points for each simulation step (yearly)
time_points = np.arange(dt, T + dt, dt)
# Calculate discount factors for each time step
discount_factors = np.exp(-r * time_points)

# Compute CVA as the sum of discounted expected losses over time:
# CVA = Σ (discount_factor(t) × LGD × PD × EE(t))
CVA = np.sum(discount_factors * LGD * PD * EE)

print("Calculated CVA: {:.2f}".format(CVA))
