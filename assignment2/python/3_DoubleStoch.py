import numpy as np
import matplotlib.pyplot as plt


T = 200
delay = 1
threshold = 0.0

phi1 = 0.6
c1 = 2.0
sigma1 = 0.8

phi2 = -0.4 
c2 = -2.0 
sigma2 = 1.0  

Y = np.zeros(T)
Y[0] = np.random.normal(0, 1)

# --- Simulate the SETAR process ---
for t in range(1, T):
    lag_idx = max(0, t - delay)
    if Y[lag_idx] <= threshold:
        eps = np.random.normal(0, sigma1)
        Y[t] = c1 + phi1 * Y[t-1] + eps
    else:
        eps = np.random.normal(0, sigma2)
        Y[t] = c2 + phi2 * Y[t-1] + eps


regime = np.where(Y[:-delay] <= threshold, 1, 2)
regime = np.concatenate(([1], regime))

# --- Plot results ---
plt.figure(figsize=(10, 7))

plt.subplot(2, 1, 1)
plt.plot(Y, label=r'$Y_t$ (observed process)', color='blue')
plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
plt.legend(); plt.grid(True)
plt.title('Simulated SETAR(1) Process')

plt.subplot(2, 1, 2)
plt.plot(regime, drawstyle='steps-post', color='orange')
plt.yticks([1, 2], ['Regime 1', 'Regime 2'])
plt.xlabel('Time')
plt.ylabel('Active Regime')
plt.grid(True)
plt.tight_layout()
plt.show()
