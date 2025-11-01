import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
phi = 0.9         # AR coefficient for Phi_t
mu = 0.8          # Mean level for Phi_t
sigma_eps = 0.5   # Variance for epsilon_t
sigma_zeta = 0.1  # Variance for zeta_t
T = 200           # Length of simulation

# --- Initialize arrays ---
Phi = np.zeros(T)
Y = np.zeros(T)

# Initial values
Phi[0] = mu
Y[0] = np.random.normal(0, 1)

# --- Simulate the model ---
for t in range(1, T):
    Phi[t] = phi * Phi[t-1] + mu * (1 - phi) + np.random.normal(0, sigma_zeta)
    Y[t] = Phi[t] * Y[t-1] + np.random.normal(0, sigma_eps)

# --- Plot results ---
plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(Phi, label=r'$\Phi_t$ (time-varying coefficient)', color='orange')
plt.legend(); plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(Y, label=r'$Y_t$ (observed process)', color='blue')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
phi, mu = 0.9, 0.8
sigma_eps, sigma_zeta = 0.5, 0.1
T = 200

# Arrays
Phi, Y = np.zeros(T), np.zeros(T)
Phi[0], Y[0] = mu, np.random.normal(0, 1)

for t in range(1, T):
    Phi[t] = phi * Phi[t-1] + mu * (1 - phi) + np.random.normal(0, sigma_zeta)
    Y[t] = Phi[t] * Y[t-1] + np.random.normal(0, sigma_eps)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(Phi, color='orange', label=r'$\Phi_t$ (time-varying coefficient)')
plt.legend(); plt.grid(True)

plt.subplot(2,1,2)
plt.plot(Y, color='blue', label=r'$Y_t$ (observed process)')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()
