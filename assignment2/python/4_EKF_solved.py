import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------
# Simulation function
#----------------------------------------------------------------
def simulate_model(T=50, a=0.4, sigma_v=1.0, sigma_e=1.0):
    """Simulate the linear state-space model."""
    x = np.zeros(T)
    y = np.zeros(T)
    for t in range(1, T):
        x[t] = a * x[t-1] + np.random.normal(0, sigma_v)
    y = x + np.random.normal(0, sigma_e, T)
    return x, y


#----------------------------------------------------------------
# Stabilized EKF estimation function
#----------------------------------------------------------------
def EKF_stable(y, aInit=0.5, aVarInit=1.0, sigma_v=1.0, sigma_e=1.0, q_a=1e-3, clip_a=(-1.5, 1.5)):
    """
    Stabilized EKF for joint estimation of state and parameter a.
    Adds small process noise on a_t and optional clipping.
    """
    n = len(y)
    zt = np.array([0.0, aInit])                            # [x, a]
    Rv = np.array([[sigma_v**2, 0], [0, q_a]])             # small process noise on a
    Re = sigma_e**2                                         # measurement noise variance
    Pt = np.diag([sigma_e**2, aVarInit])                    # initial covariance
    Ht = np.array([[1.0, 0.0]])                             # observation matrix

    # Storage
    Z = np.zeros((n, 2))
    Z[0, :] = zt

    for t in range(n-1):
        # Jacobian
        Ft = np.array([[zt[1], zt[0]], [0, 1]])            # df/dz

        # Predict
        z_pred = np.array([zt[1]*zt[0], zt[1]])            # f(z_t)
        P_pred = Ft @ Pt @ Ft.T + Rv

        # Update
        y_pred = z_pred[0]
        res = y[t] - y_pred
        St = Ht @ P_pred @ Ht.T + Re
        Kt = P_pred @ Ht.T / St
        zt = z_pred + (Kt.flatten() * res)
        Pt = (np.eye(2) - Kt @ Ht) @ P_pred

        # Optional stability clip
        zt[1] = np.clip(zt[1], clip_a[0], clip_a[1])

        # Store
        Z[t+1, :] = zt

    return Z


#----------------------------------------------------------------
# Multiple simulations for visualization
#----------------------------------------------------------------
np.random.seed(0)
nSim = 50
T = 30
a_true = 0.4
sigma_v_true = 1.0
sigma_e_true = 1.0

filter_cases = [
    {"sigma_v": np.sqrt(10), "aVarInit": 1, "label": "σv²=10, Var(a0)=1"},
    {"sigma_v": np.sqrt(1),  "aVarInit": 1, "label": "σv²=1, Var(a0)=1"},
    {"sigma_v": np.sqrt(10), "aVarInit": 10, "label": "σv²=10, Var(a0)=10"},
    {"sigma_v": np.sqrt(1),  "aVarInit": 10, "label": "σv²=1, Var(a0)=10"},
]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i, params in enumerate(filter_cases):
    ax = axs[i]
    for sim in range(nSim):
        _, y = simulate_model(T=T, a=a_true, sigma_v=sigma_v_true, sigma_e=sigma_e_true)

        # Two different initial a values
        Z_pos = EKF_stable(y, aInit=0.5, aVarInit=params["aVarInit"], sigma_v=params["sigma_v"], sigma_e=sigma_e_true)
        Z_neg = EKF_stable(y, aInit=-0.5, aVarInit=params["aVarInit"], sigma_v=params["sigma_v"], sigma_e=sigma_e_true)

        ax.plot(Z_pos[:, 1], color='blue', alpha=0.4)
        ax.plot(Z_neg[:, 1], color='red', alpha=0.4)

    ax.axhline(a_true, color='black', linestyle='--', label='True a')
    ax.set_title(params["label"])
    ax.set_xlabel("Time")
    ax.set_ylabel("a estimate")
    ax.legend()

plt.tight_layout()
plt.show()

