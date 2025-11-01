import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------
# Simulation function
#----------------------------------------------------------------
def simulate_model(T=50, a=0.4, sigma_v=1.0, sigma_e=1.0):
    """
    Simulate the state-space model:
      x_{t+1} = a * x_t + v_t
      y_t = x_t + e_t
    """
    x = np.zeros(T)
    y = np.zeros(T)
    for t in range(1, T):
        x[t] = a * x[t-1] + np.random.normal(0, sigma_v)
    y = x + np.random.normal(0, sigma_e, T)
    return x, y


#----------------------------------------------------------------
# EKF estimation function
#----------------------------------------------------------------
def EKF(y, aInit=0.5, aVarInit=1, sigma_v=1, sigma_e=1):
    """
    Extended Kalman Filter for parameter estimation of a in:
      x_{t+1} = a * x_t + v_t
      y_t = x_t + e_t
    """
    n = len(y)
    
    # Initialize
    zt = np.array([0.0, aInit])                    # state vector [X, a]
    Rv = np.array([[sigma_v**2, 0], [0, 0]])       # process noise covariance
    Re = sigma_e                                   # measurement noise variance
    Pt = np.array([[Re, 0], [0, aVarInit]])        # initial covariance
    Ht = np.array([[1.0, 0.0]])                    # observation Jacobian
    
    # Storage
    Z = np.zeros((n, 2))
    Z[0, :] = zt
    aVar = np.full(n, np.nan)
    aVar[0] = aVarInit

    # Kalman filtering
    for t in range(n-1):
        # Jacobian of f(z)
        Ft = np.array([[zt[1], 0], [zt[0], 1]])  # F_t-1
        
        # Prediction step
        zt = np.array([zt[1]*zt[0], zt[1]])      # f(z_{t-1})
        Pt = Ft @ Pt @ Ft.T + Rv
        
        # Update step
        res = y[t] - zt[0]                       # residual
        St = Ht @ Pt @ Ht.T + Re
        Kt = Pt @ Ht.T @ np.linalg.inv(St)
        zt = zt + (Kt.flatten() * res)
        Pt = (np.eye(2) - Kt @ Ht) @ Pt
        
        # Store
        Z[t+1, :] = zt
        aVar[t+1] = Pt[1, 1]
        
    return Z, aVar


#----------------------------------------------------------------
# Part 4a + 4b: Multiple simulations
#----------------------------------------------------------------
np.random.seed(0)
nSim = 50
T = 30
a_true = 0.4
sigma_v_true = 1
sigma_e_true = 1

# Four filter setups
filter_cases = [
    {"sigma_v": np.sqrt(10), "aVarInit": 1, "label": "σv²=10, Var(a0)=1"},
    {"sigma_v": np.sqrt(1),  "aVarInit": 1, "label": "σv²=1, Var(a0)=1"},
    {"sigma_v": np.sqrt(10), "aVarInit": 10, "label": "σv²=10, Var(a0)=10"},
    {"sigma_v": np.sqrt(1),  "aVarInit": 10, "label": "σv²=1, Var(a0)=10"},
]

# Plot setup
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i, params in enumerate(filter_cases):
    ax = axs[i]
    
    for sim in range(nSim):
        # Simulate data
        _, y = simulate_model(T=T, a=a_true, sigma_v=sigma_v_true, sigma_e=sigma_e_true)
        
        # EKF with two different initial a values
        Z_pos, _ = EKF(y, aInit=0.5, aVarInit=params["aVarInit"], sigma_v=params["sigma_v"], sigma_e=sigma_e_true)
        Z_neg, _ = EKF(y, aInit=-0.5, aVarInit=params["aVarInit"], sigma_v=params["sigma_v"], sigma_e=sigma_e_true)
        
        ax.plot(Z_pos[:,1], color='blue', alpha=0.4)
        ax.plot(Z_neg[:,1], color='red', alpha=0.4)
    
    ax.axhline(a_true, color='black', linestyle='--', label='True a')
    ax.set_title(params["label"])
    ax.set_xlabel("Time")
    ax.set_ylabel("a estimate")
    ax.legend()

plt.tight_layout()
plt.show()
