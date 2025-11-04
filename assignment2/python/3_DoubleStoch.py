import numpy as np
import matplotlib.pyplot as plt

def simulate_doubly_stochastic_setar(n, a0, b0, a1, b1, sigma_eps, sigma_zeta, c, gamma, seed=42):
    """
    Simulate a doubly stochastic SETAR(2,1,1) model.
    
    X_t = a_{t-1} + b_{t-1}*X_{t-1} + phi_t + eps_t
    phi_t = c*phi_{t-1} + zeta_t
    
    where a_{t-1} and b_{t-1} depend on whether X_{t-1} <= gamma
    """
    np.random.seed(seed)
    
    # Initialize
    X = np.zeros(n)
    phi = np.zeros(n)
    
    # Generate noise
    eps = np.random.normal(0, sigma_eps, n)
    zeta = np.random.normal(0, sigma_zeta, n)
    
    # Simulate
    for t in range(1, n):
        # Update latent component
        phi[t] = c * phi[t-1] + zeta[t]
        
        # Determine regime
        if X[t-1] <= gamma:
            a_t = a0
            b_t = b0
        else:
            a_t = a1
            b_t = b1
        
        # Update observation
        X[t] = a_t + b_t * X[t-1] + phi[t] + eps[t]
    
    return X, phi

def plot_simulation(X, phi, c, gamma, ax_ts, ax_lag):
    """Plot time series and lag plot for one simulation"""
    n = len(X)
    t = np.arange(n)
    
    # Time series plot
    ax_ts.plot(t, X, label=f'Observed process $Y_t$', linewidth=0.8, color='steelblue')
    ax_ts.plot(t, phi, label=f'Latent component $\\phi_t$ (c={c})', linewidth=0.8, color='darkorange')
    ax_ts.axhline(gamma, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_ts.set_xlabel('t')
    ax_ts.set_ylabel('Value')
    ax_ts.legend(loc='upper right', fontsize=9)
    ax_ts.grid(True, alpha=0.3)
    ax_ts.set_title(f'Doubly Stochastic SETAR(2,1,1) simulation')
    
    # Lag plot
    X_lag = X[:-1]
    X_curr = X[1:]
    
    # Color by regime
    regime = X_lag <= gamma
    ax_lag.scatter(X_lag[regime], X_curr[regime], c='steelblue', s=10, alpha=0.6, label='Regime 1')
    ax_lag.scatter(X_lag[~regime], X_curr[~regime], c='darkorange', s=10, alpha=0.6, label='Regime 2')
    ax_lag.axvline(gamma, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_lag.axhline(gamma, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_lag.set_xlabel('$Y_{t-1}$ (t-1)')
    ax_lag.set_ylabel('$Y_t$')
    ax_lag.legend(loc='best', fontsize=9)
    ax_lag.grid(True, alpha=0.3)
    ax_lag.set_title(f'Lag Plot ($Y_t$ vs $Y_{{t-1}}$) with c={c}')

# Parameters
n = 1000
a0, b0 = 0.5, 0.4
a1, b1 = 0.5, -0.5
sigma_eps = 0.3
sigma_zeta = 0.2
gamma = 0

# Different persistence values
c_values = [0.2, 0.6, 0.9]

# Create figure
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Doubly Stochastic SETAR(2,1,1) - Effect of Latent Persistence', fontsize=14, y=0.995)

for i, c in enumerate(c_values):
    X, phi = simulate_doubly_stochastic_setar(
        n, a0, b0, a1, b1, sigma_eps, sigma_zeta, c, gamma, seed=42+i
    )
    
    plot_simulation(X, phi, c, gamma, axes[i, 0], axes[i, 1])
    
    # Add simulation number
    axes[i, 0].text(0.02, 0.98, f'Simulation {i+1}', 
                    transform=axes[i, 0].transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n--- Doubly Stochastic SETAR(2,1,1) Simulation Summary ---")
print(f"Number of observations: {n}")
print(f"Parameters: a0={a0}, b0={b0}, a1={a1}, b1={b1}")
print(f"Noise: σ_ε={sigma_eps}, σ_ζ={sigma_zeta}")
print(f"Threshold: γ={gamma}")
print("\nLatent persistence values tested:")
for i, c in enumerate(c_values):
    X, phi = simulate_doubly_stochastic_setar(
        n, a0, b0, a1, b1, sigma_eps, sigma_zeta, c, gamma, seed=42+i
    )
    regime_1_pct = np.mean(X[:-1] <= gamma) * 100
    print(f"  Simulation {i+1}: c={c:.1f}, Regime 1: {regime_1_pct:.1f}%, Regime 2: {100-regime_1_pct:.1f}%")