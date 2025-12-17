### Cell: kalman_arx_ar1.py
# ------------------------------------------------------------
# Kalman-filtered adaptive regression with AR(1) observation noise
# ------------------------------------------------------------
# This module implements:
#
#   y_t = x_t' β_t + ε_t
#   β_t = β_{t-1} + w_t        , w_t ~ N(0, Q)
#   ε_t = φ ε_{t-1} + u_t      , u_t ~ N(0, R)
#
# Key idea (from the book & your diagnostics):
# - Remaining autocorrelation means ε_t is NOT white noise
# - We explicitly model AR(1) noise in the observation equation
# - This is done by *state augmentation* (textbook approach)
#
# This file is designed to be:
# - readable
# - modular
# - easy to call from a Jupyter notebook
#
# You can import it as:
#   from kalman_arx_ar1 import KalmanARX_AR1
# ------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any


# ============================================================
# Helper metrics (kept minimal & explicit)
# ============================================================
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


# ============================================================
# Kalman ARX with AR(1) observation noise
# ============================================================
class KalmanARX_AR1:
    """
    Dynamic linear regression with:
      - time-varying regression coefficients (random walk)
      - AR(1) observation noise

    Model:
        y_t = x_t' β_t + ε_t
        β_t = β_{t-1} + w_t
        ε_t = φ ε_{t-1} + u_t

    State vector:
        s_t = [ β_t
                ε_t ]

    This is the *canonical* way to handle autocorrelated residuals
    in a Kalman / state-space framework.
    """

    def __init__(
        self,
        phi: float,
        q_beta: float,
        r_eps: float,
        beta0: Optional[np.ndarray] = None,
        P0_scale: float = 1e4,
    ):
        """
        Parameters
        ----------
        phi : float
            AR(1) coefficient for observation noise (|phi| < 1)
        q_beta : float
            Process noise variance for regression coefficients
            (Q_beta = q_beta * I)
        r_eps : float
            Innovation variance of AR(1) noise u_t
        beta0 : np.ndarray, optional
            Initial regression coefficients
        P0_scale : float
            Initial covariance scale for the state
        """
        self.phi = float(phi)
        self.q_beta = float(q_beta)
        self.r_eps = float(r_eps)
        self.beta0 = beta0
        self.P0_scale = float(P0_scale)

    # --------------------------------------------------------
    # Core Kalman filter
    # --------------------------------------------------------
    def fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        clip: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Run Kalman filter on full sequence and return predictions.

        Parameters
        ----------
        X : (n, p) array
            Design matrix
        y : (n,) array
            Target variable
        clip : (lo, hi), optional
            Clip predictions to physical bounds

        Returns
        -------
        dict with keys:
            - y_pred : one-step-ahead predictions
            - beta_trace : time-varying regression coefficients
            - eps_trace : estimated AR(1) noise state
            - P_trace : diagonal of state covariance
            - residuals
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n, p = X.shape

        # ---------------------------
        # State definition
        # s_t = [β_t (p), ε_t (1)]
        # ---------------------------
        dim_state = p + 1

        # Initial state
        beta_init = np.zeros(p) if self.beta0 is None else self.beta0.copy()
        s = np.zeros(dim_state)
        s[:p] = beta_init
        s[p] = 0.0  # initial ε_0

        # Initial covariance
        P = self.P0_scale * np.eye(dim_state)

        # ---------------------------
        # System matrices
        # ---------------------------

        # State transition matrix F
        F = np.eye(dim_state)
        F[p, p] = self.phi  # AR(1) noise transition

        # Process noise covariance Q
        Q = np.zeros((dim_state, dim_state))
        Q[:p, :p] = self.q_beta * np.eye(p)  # coefficient drift
        Q[p, p] = self.r_eps                 # AR(1) innovation variance

        # Observation noise is zero because noise is in the state
        R = 0.0

        # Storage
        y_pred = np.zeros(n)
        beta_trace = np.zeros((n, p))
        eps_trace = np.zeros(n)
        P_trace = np.zeros((n, dim_state))

        I = np.eye(dim_state)

        # ---------------------------
        # Kalman recursion
        # ---------------------------
        for t in range(n):
            x = X[t]

            # Observation matrix H_t = [x_t', 1]
            H = np.zeros((1, dim_state))
            H[0, :p] = x
            H[0, p] = 1.0

            # ---- Prediction
            s_pred = F @ s
            P_pred = F @ P @ F.T + Q

            # ---- Predict observation
            yhat = float(H @ s_pred)
            y_pred[t] = yhat

            # ---- Update
            innov = y[t] - yhat
            S = float(H @ P_pred @ H.T) + R
            K = (P_pred @ H.T) / S

            s = s_pred + (K.flatten() * innov)
            P = (I - K @ H) @ P_pred

            # Store traces
            beta_trace[t] = s[:p]
            eps_trace[t] = s[p]
            P_trace[t] = np.diag(P)

        if clip is not None:
            lo, hi = clip
            y_pred = np.clip(y_pred, lo, hi)

        residuals = y - y_pred

        return {
            "y_pred": y_pred,
            "beta_trace": beta_trace,
            "eps_trace": eps_trace,
            "P_trace": P_trace,
            "residuals": residuals,
        }


# ============================================================
# Convenience wrapper for train/test usage
# ============================================================
def fit_kalman_arx_ar1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    phi: float,
    q_beta: float,
    r_eps: float,
    clip: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Typical workflow helper:
    - Initialize β₀ from OLS on training data
    - Run Kalman filter on concatenated train+test
    - Return test-period predictions and traces
    """

    # OLS init for stability
    beta0 = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

    kf = KalmanARX_AR1(
        phi=phi,
        q_beta=q_beta,
        r_eps=r_eps,
        beta0=beta0,
        P0_scale=1e2,
    )

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    out = kf.fit_predict(X_all, y_all, clip=clip)

    ntr = len(y_train)

    return {
        "y_pred": out["y_pred"][ntr:],
        "y_true": y_test,
        "beta_trace": out["beta_trace"][ntr:],
        "eps_trace": out["eps_trace"][ntr:],
        "residuals": out["residuals"][ntr:],
        "rmse": rmse(y_test, out["y_pred"][ntr:]),
        "mae": mae(y_test, out["y_pred"][ntr:]),
    }
