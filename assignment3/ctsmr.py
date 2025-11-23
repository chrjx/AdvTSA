import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm

class LinearReservoirKF:
    """
    Implements ctsm in python.
    Linear rainfall-runoff model with exact Kalman filtering.
    
    System equations for m states (n = m-1):
      dX1 = (A*U - (n/K)*X1) dt + sigma*dW1
      dX2 = ((n/K)*X1 - (n/K)*X2) dt + sigma*dW2
      ...
      dXm = (n/K)*X_{m-1} dt + sigma*dWm  [NO outflow from last state]
    
    Observation: Y = Xm + noise
    """

    def __init__(self, t, y, u, m):
        self.t = t
        self.y = np.asarray(y)
        self.u = np.asarray(u)
        self.m = m  # number of states
        self.n = m - 1  # n in the equations
        self.dt = np.diff(t, prepend=t[0])
####################
# GIVEN MODEL
    def system_matrices(self, K, A, sigma):
        """
        Construct continuous-time system matrices:
        dX = F*X dt + G*U dt + L*dW
        """
        m, n = self.m, self.n
        rate = n / K
        
        # State transition matrix F
        F = np.zeros((m, m))
        F[0, 0] = -rate  # First state drains out
        for i in range(1, m-1):
            F[i, i-1] = rate   # Inflow from previous
            F[i, i] = -rate    # Outflow to next
        # Last state (i = m-1): only receives inflow, no outflow
        F[m-1, m-2] = rate
        F[m-1, m-1] = 0  # NO outflow term
        
        # Input matrix G (rainfall enters first state only)
        G = np.zeros((m, 1))
        G[0, 0] = A
        
        # Diffusion matrix L
        L = np.eye(m) * sigma
        
        return F, G, L

    def negloglik(self, params, return_full=False):
        """
        Compute negative log-likelihood using Kalman filter.
        params: [K, A, sigma, S, X1_0, ..., Xm_0]
        """
        K, A, sigma, S = params[:4]
        x0 = np.array(params[4:4 + self.m])



        # Parameter constraints
        if K <= 0 or A <= 0 or sigma <= 0 or S <= 0:
            return 1e10 if not return_full else (1e10, None, None)

        # Get system matrices
        F, G, L = self.system_matrices(K, A, sigma)
        
        # Observation setup
        R = S ** 2  # Observation noise variance
        H = np.zeros((1, self.m))
        H[0, -1] = 1.0  # Observe last state only

        # Initialize
        x = x0.copy()
        P = np.eye(self.m) * 10.0

        loglik = 0.0
        x_hist = []
        resids = []

        for i in range(1, len(self.t)):
            dt = self.dt[i]
            u_t = self.u[i - 1]
            
            # Exact discretization using matrix exponential
            Phi = expm(F * dt)  # State transition matrix
            
            ################
            # Compute discrete-time input effect B
            # B = (∫_0^dt exp(F s) ds) G * u_t
            # Use analytic formula B = F^{-1}(Phi - I)G when F is invertible;
            # otherwise fall back to numerical quadrature.
            ################

            if self.m == 2 and abs(F[1,1]) < 1e-10:
                # Special case: 2-state model where F is singular
                # Use numerical integration
                B = np.zeros(self.m)
                n_steps = 10
                for k in range(n_steps):
                    s = (k + 0.5) * dt / n_steps
                    B += (expm(F * s) @ G).ravel() * u_t * (dt / n_steps)
            else:
                try:
                    F_inv = np.linalg.inv(F)
                    B = (F_inv @ (Phi - np.eye(self.m)) @ G).ravel() * u_t
                except np.linalg.LinAlgError:
                    # Fallback to numerical integration
                    B = np.zeros(self.m)
                    n_steps = 10
                    for k in range(n_steps):
                        s = (k + 0.5) * dt / n_steps
                        B += (expm(F * s) @ G).ravel() * u_t * (dt / n_steps)
            
            # Process noise cov
            Q = (sigma ** 2 * dt) * np.eye(self.m)

            ####################
            # KALMAN FILTERING
            ####################

            # PREDICTION STEP
            x_pred = Phi @ x + B.ravel()
            P_pred = Phi @ P @ Phi.T + Q
            
            # INNOVATION
            y_pred = H @ x_pred
            innov = self.y[i] - y_pred
            S_inn = H @ P_pred @ H.T + R
            S_inn = np.atleast_2d(S_inn)
            
            # Kalman gain
            K_gain = P_pred @ H.T @ np.linalg.inv(S_inn)
            
            # UPDATE STEP
            x = x_pred + (K_gain @ innov).ravel()
            P = (np.eye(self.m) - K_gain @ H) @ P_pred
            
            # Log-likelihood contribution
            ll = -0.5 * (np.log(np.linalg.det(S_inn)) +
                         innov.T @ np.linalg.inv(S_inn) @ innov + 
                         np.log(2 * np.pi))
            loglik += ll.item()
            
            # Store results
            x_hist.append(x.copy())
            resids.append(float(innov / np.sqrt(S_inn)))

        if return_full:
            return -loglik, np.array(x_hist), np.array(resids)
        return -loglik

    def fit(self):
        """Estimate parameters using maximum likelihood."""
        x0 = [2.0, 50.0, 0.01, 0.01] + [0.0] * self.m
        bounds = [(0.01, 200), (0.1, 1000), (1e-6, 1), (1e-6, 1)] + [(-5, 5)] * self.m
        res = minimize(self.negloglik, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 1000})
        return res