import numpy as np

#---------------------------------------------------------------
# EKF algorithm (Python translation)
# For Advanced Time Series Analysis - Part 4
#---------------------------------------------------------------

#---------------------------------------------------------------
# Simulated data (replace this with your own simulation)
#---------------------------------------------------------------
# Example: create dummy observations
y = np.random.randn(100)

#---------------------------------------------------------------
# EKF parameter initialization
#---------------------------------------------------------------
aInit = 0.5       # initial AR coefficient estimate
aVarInit = 1.0    # initial variance of a estimate
sigma_v = 1.0     # system noise std. dev.
Re = 1.0          # measurement noise variance

#---------------------------------------------------------------
# Initialization
#---------------------------------------------------------------
zt = np.array([0.0, aInit])           # state vector [X, a]
Rv = np.array([[sigma_v**2, 0],
               [0, 0]])               # process noise covariance
Pt = np.array([[Re, 0],
               [0, aVarInit]])        # initial state covariance
Ht = np.array([[1.0, 0.0]])           # observation model

aVar = np.full(len(y), np.nan)        # store a variance estimates
Z = np.full((len(y), 2), np.nan)      # store state estimates
Z[0, :] = zt

#---------------------------------------------------------------
# Kalman filtering loop
#---------------------------------------------------------------
for t in range(len(y) - 1):
    # Jacobian matrix (Ft)
    Ft = np.array([[zt[1], 0.0],
                   [zt[0], 1.0]])

    # Prediction step
    zt = np.array([zt[1] * zt[0], zt[1]])  # state prediction
    Pt = Ft @ Pt @ Ft.T + Rv                # covariance prediction

    # Update step
    res = y[t] - zt[0]                      # residual
    St = Ht @ Pt @ Ht.T + Re                # innovation covariance
    Kt = Pt @ Ht.T @ np.linalg.inv(St)      # Kalman gain

    zt = zt + (Kt.flatten() * res)          # state update
    Pt = (np.eye(2) - Kt @ Ht) @ Pt         # covariance update

    # Store results
    Z[t + 1, :] = zt
    aVar[t + 1] = Pt[1, 1]

#---------------------------------------------------------------
# Results
#---------------------------------------------------------------
print("Final state estimate:", zt)
print("Final covariance matrix:\n", Pt)
