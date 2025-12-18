from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from statsmodels.stats.diagnostic import acorr_ljungbox


# =============================================================================
# Diagnostics helpers
# =============================================================================
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def lb_pvalue(x: np.ndarray, lag: int = 48) -> float:
    """
    Ljung–Box p-value for remaining autocorrelation.

    Note: for Kalman filtering, the *standardized innovations* are the
    residual-like object that theory expects to be white (if the model is correct).
    """
    x = np.asarray(x, dtype=float)
    out = acorr_ljungbox(x, lags=[lag], return_df=True)
    return float(out["lb_pvalue"].iloc[0])


def clip_pred(pred, lo, hi):
    return np.clip(np.asarray(pred, dtype=float), lo, hi)


@dataclass
class ResidualBundle:
    """
    Carry the different "residual-like" sequences you may want to diagnose.

    - resid: plain one-step-ahead forecast error (y - yhat)
    - innov: innovation v_t computed at KF prediction step (same as resid if 1-step ahead)
    - std_innov: standardized innovation v_t / sqrt(S_t) (should be ~white, ~N(0,1) if model is right)
    """
    resid: np.ndarray
    innov: Optional[np.ndarray] = None
    std_innov: Optional[np.ndarray] = None


# =============================================================================
# Base interface
# =============================================================================
class BaseModel:
    def fit_predict(self, h: int, data_mgr, split, Xy: Dict[str, Tuple[pd.DataFrame, np.ndarray]]) -> Dict[str, Any]:
        raise NotImplementedError

class StaticARX(BaseModel):
    """
    Simple static ARX estimated by OLS.
    Baseline benchmark model.
    """

    def fit_predict(self, h, data_mgr, split, Xy):
        p_max = data_mgr.p_max

        Xtr, ytr = Xy["train_df"]
        Xte, yte = Xy["test_df"]

        beta = np.linalg.lstsq(Xtr.values, ytr, rcond=None)[0]
        pred = Xte.values @ beta
        pred = clip_pred(pred, 0.0, p_max)

        resid = yte - pred

        return {
            "pred": pred,
            "y": yte,
            "resid": resid,
            "rb": ResidualBundle(resid=resid),
            "beta": beta,
            "names": Xte.columns.tolist(),
            "config": {"type": "Static ARX"},
        }

class TAR_ARX(BaseModel):
    """
    Threshold ARX using wind speed as regime variable.
    """

    def __init__(self, qtiles=(0.3, 0.5, 0.7)):
        self.qtiles = qtiles
        self.configs_ = {}

    def fit_predict(self, h, data_mgr, split, Xy):
        p_max = data_mgr.p_max

        Xtr, ytr = Xy["train_df"]
        Xte, yte = Xy["test_df"]

        Ws_tr = split.train_df[f"Ws{h}"].values
        Ws_te = split.test_df[f"Ws{h}"].values

        best = None

        for q in self.qtiles:
            c = np.quantile(Ws_tr, q)

            idx_L = Ws_tr <= c
            idx_H = Ws_tr > c

            if idx_L.sum() < 50 or idx_H.sum() < 50:
                continue

            beta_L = np.linalg.lstsq(Xtr.values[idx_L], ytr[idx_L], rcond=None)[0]
            beta_H = np.linalg.lstsq(Xtr.values[idx_H], ytr[idx_H], rcond=None)[0]

            pred = np.where(
                Ws_te <= c,
                Xte.values @ beta_L,
                Xte.values @ beta_H,
            )
            pred = clip_pred(pred, 0.0, p_max)

            rmse_val = rmse(yte, pred)

            if best is None or rmse_val < best["rmse"]:
                best = {
                    "c": float(c),
                    "rmse": rmse_val,
                    "beta_L": beta_L,
                    "beta_H": beta_H,
                }

        assert best is not None
        self.configs_[h] = best

        pred = np.where(
            Ws_te <= best["c"],
            Xte.values @ best["beta_L"],
            Xte.values @ best["beta_H"],
        )
        pred = clip_pred(pred, 0.0, p_max)
        resid = yte - pred

        return {
            "pred": pred,
            "y": yte,
            "resid": resid,
            "rb": ResidualBundle(resid=resid),
            "names": Xte.columns.tolist(),
            "config": best,
        }


# =============================================================================
# 1) Kalman ARX with white observation noise (time-varying coefficients)
# =============================================================================
class KalmanARXWhite(BaseModel):
    def __init__(
        self,
        q_grid: List[float] = (1e-6, 1e-5, 1e-4, 1e-3),
        r_scale_grid: List[float] = (0.25, 0.5, 1.0, 2.0),
        lb_lag: int = 48,
        alpha_lb_penalty: float = 0.15,
        P0_scale: float = 1e2,
    ):
        self.q_grid = list(q_grid)
        self.r_scale_grid = list(r_scale_grid)
        self.lb_lag = int(lb_lag)
        self.alpha_lb_penalty = float(alpha_lb_penalty)
        self.P0_scale = float(P0_scale)

        self.configs_: Dict[int, Dict[str, Any]] = {}

    def _score(self, y_true, pred, std_innov=None) -> Dict[str, float]:
        """
        Penalize RMSE and (optionally) autocorrelation in standardized innovations.
        """
        r = rmse(y_true, pred)
        if std_innov is None:
            p = lb_pvalue(np.asarray(y_true) - np.asarray(pred), lag=self.lb_lag)
        else:
            p = lb_pvalue(std_innov, lag=self.lb_lag)

        score = r + self.alpha_lb_penalty * (-math.log(max(p, 1e-12)))
        return {"rmse": r, "lb_p": p, "score": score}

    def _kf_run(self, X: np.ndarray, y: np.ndarray, q: float, r: float, beta0: Optional[np.ndarray] = None):
        """
        Random walk beta:
            beta_t = beta_{t-1} + w_t,   w_t ~ N(0, q I)
        Observation:
            y_t = x_t^T beta_t + e_t,    e_t ~ N(0, r)

        Returns:
          yhat: filtered one-step-ahead predictions
          beta_trace: filtered beta_t
          innov: v_t
          std_innov: v_t / sqrt(S_t)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        Q = float(q) * np.eye(p)
        R = float(r)
        I = np.eye(p)

        beta = np.zeros(p) if beta0 is None else np.asarray(beta0, dtype=float).copy()
        P = self.P0_scale * np.eye(p)

        yhat = np.zeros(n)
        beta_trace = np.zeros((n, p))
        innov = np.zeros(n)
        std_innov = np.zeros(n)

        for t in range(n):
            x = X[t].reshape(p, 1)

            # predict
            beta_pred = beta
            P_pred = P + Q

            # innovation
            yhat[t] = float(beta_pred.reshape(1, -1) @ x)
            v = float(y[t] - yhat[t])
            S = float(x.T @ P_pred @ x) + R

            innov[t] = v
            std_innov[t] = v / math.sqrt(max(S, 1e-12))

            # update
            K = (P_pred @ x) / max(S, 1e-12)
            beta = beta_pred + (K.flatten() * v)
            P = (I - K @ x.T) @ P_pred

            beta_trace[t] = beta

        return yhat, beta_trace, innov, std_innov

    def fit_predict(self, h: int, data_mgr, split, Xy: Dict[str, Tuple[pd.DataFrame, np.ndarray]]) -> Dict[str, Any]:
        p_max = float(data_mgr.p_max)
        yvar = float(split.train_in["p"].var())

        Xtr, ytr = Xy["train_in"]
        Xva, yva = Xy["val_in"]
        Xtrain_full, ytrain_full = Xy["train_df"]
        Xte, yte = Xy["test_df"]

        best = None

        # tune on (train_in -> val_in) using innovations whiteness
        for q in self.q_grid:
            for rs in self.r_scale_grid:
                r = max(1e-10, float(rs) * yvar)
                beta0 = np.linalg.lstsq(Xtr.values, ytr, rcond=None)[0]

                X_all = np.vstack([Xtr.values, Xva.values])
                y_all = np.concatenate([ytr, yva])

                yhat_all, _, _, std_innov_all = self._kf_run(X_all, y_all, q=q, r=r, beta0=beta0)

                pred_val = clip_pred(yhat_all[len(ytr):], 0.0, p_max)
                std_innov_val = std_innov_all[len(ytr):]  # correct object for LB
                s = self._score(yva, pred_val, std_innov=std_innov_val)

                cand = {"q": float(q), "r": float(r), **s}
                if best is None or cand["score"] < best["score"]:
                    best = cand

        assert best is not None
        self.configs_[h] = best

        # fit on full train_df -> test_df
        beta0 = np.linalg.lstsq(Xtrain_full.values, ytrain_full, rcond=None)[0]
        X_all = np.vstack([Xtrain_full.values, Xte.values])
        y_all = np.concatenate([ytrain_full, yte])

        yhat_all, beta_trace_all, innov_all, std_innov_all = self._kf_run(
            X_all, y_all, q=best["q"], r=best["r"], beta0=beta0
        )

        pred = clip_pred(yhat_all[len(ytrain_full):], 0.0, p_max)
        resid = yte - pred

        # slice traces to test portion
        beta_trace_test = beta_trace_all[len(ytrain_full):]
        innov_test = innov_all[len(ytrain_full):]
        std_innov_test = std_innov_all[len(ytrain_full):]

        return {
            "pred": pred,
            "y": yte,
            "resid": resid,
            "rb": ResidualBundle(resid=resid, innov=innov_test, std_innov=std_innov_test),
            "beta_trace": beta_trace_test,
            "names": Xte.columns.tolist(),
            "config": best,
        }


# =============================================================================
# 2) Regime-dependent AR(1) observation noise (state augmentation)
# =============================================================================
@dataclass
class RegimeSpec:
    """
    Define regimes and time-varying AR(1) noise parameters for u_t.

    u_t = phi_t * u_{t-1} + w_t,  w_t ~ N(0, sigma_u2_t)

    Currently implemented:
      - "month": piecewise-constant sigma_u2 by calendar month
      - "dir": two-regime split by wind direction half-circle
    """
    mode: str = "month"             # "month" | "dir"
    cut_deg: float = 0.0            # used if mode == "dir"
    phi_A: float = 0.8
    phi_B: float = 0.2
    sigma2_A: float = 1.0
    sigma2_B: float = 4.0

    def build(self, df_all: pd.DataFrame, h: int) -> Dict[str, np.ndarray]:
        n = len(df_all)

        if self.mode == "month":
            # robust month extraction
            if "t" in df_all.columns and np.issubdtype(df_all["t"].dtype, np.datetime64):
                month = pd.to_datetime(df_all["t"]).dt.month.values
            else:
                # fallback: expect a datetime index
                month = pd.to_datetime(df_all.index).month.values

            # compute monthwise variance proxy from p-diff (simple, stable)
            p = df_all["p"].astype(float).values
            dp = np.diff(p, prepend=p[0])
            s2_by_month = {}
            for m in range(1, 13):
                idx = month == m
                if idx.sum() < 50:
                    continue
                s2_by_month[m] = float(np.nanvar(dp[idx]))
            global_s2 = float(np.nanvar(dp))

            sigma2_t = np.array([s2_by_month.get(int(m), global_s2) for m in month], dtype=float)
            sigma2_t = np.maximum(1e-10, sigma2_t)

            phi_t = np.full(n, float(self.phi_A), dtype=float)
            regime_mask = np.ones(n, dtype=bool)

            return {"phi_t": phi_t, "sigma2_t": sigma2_t, "regime_A": regime_mask}

        if self.mode == "dir":
            wd_col = f"Wd{h}"
            if wd_col not in df_all.columns:
                raise KeyError(f"RegimeSpec(mode='dir') needs column {wd_col} in df_all.")
            a = (df_all[wd_col].astype(float).values % 360.0)
            c = float(self.cut_deg) % 360.0
            d = (a - c) % 360.0
            A = d < 180.0

            phi_t = np.where(A, float(self.phi_A), float(self.phi_B)).astype(float)
            sigma2_t = np.where(A, float(self.sigma2_A), float(self.sigma2_B)).astype(float)
            sigma2_t = np.maximum(1e-10, sigma2_t)

            return {"phi_t": phi_t, "sigma2_t": sigma2_t, "regime_A": A}

        raise ValueError(f"Unknown RegimeSpec.mode={self.mode!r}")


class KalmanARXRegimeARNoise(BaseModel):
    """
    State: x_t = [beta_t; u_t]
      beta_t: ARX coefficients (random walk)
      u_t: AR(1) correlated observation noise with regime-dependent (phi_t, sigma_u2_t)

    Transition:
      beta_t = beta_{t-1} + w_beta,   w_beta ~ N(0, q_beta I)
      u_t    = phi_t * u_{t-1} + w_u, w_u    ~ N(0, sigma_u2_t)

    Observation:
      y_t = X_t beta_t + u_t + e_t,   e_t ~ N(0, r_y)
    """

    def __init__(
        self,
        q_beta: float = 1e-5,
        r_y: float = 0.5,
        regime_spec: Optional[RegimeSpec] = None,
        lb_lag: int = 48,
        alpha_lb_penalty: float = 0.15,
        P0_scale: float = 1e2,
    ):
        self.q_beta = float(q_beta)
        self.r_y = float(r_y)
        self.regime_spec = regime_spec if regime_spec is not None else RegimeSpec(mode="month")
        self.lb_lag = int(lb_lag)
        self.alpha_lb_penalty = float(alpha_lb_penalty)
        self.P0_scale = float(P0_scale)

        self.configs_: Dict[int, Dict[str, Any]] = {}

    def _score(self, y_true, pred, std_innov=None) -> Dict[str, float]:
        r = rmse(y_true, pred)
        if std_innov is None:
            p = lb_pvalue(np.asarray(y_true) - np.asarray(pred), lag=self.lb_lag)
        else:
            p = lb_pvalue(std_innov, lag=self.lb_lag)
        score = r + self.alpha_lb_penalty * (-math.log(max(p, 1e-12)))
        return {"rmse": r, "lb_p": p, "score": score}

    def _run_kf_timevarying(
        self,
        X: np.ndarray,
        y: np.ndarray,
        phi_t: np.ndarray,
        sigma2_t: np.ndarray,
        q_beta: float,
        r_y: float,
        beta0: Optional[np.ndarray] = None,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        phi_t = np.asarray(phi_t, dtype=float)
        sigma2_t = np.asarray(sigma2_t, dtype=float)

        n, p = X.shape
        dim = p + 1  # beta + u

        x = np.zeros(dim)
        if beta0 is not None:
            x[:p] = np.asarray(beta0, dtype=float)
        P = self.P0_scale * np.eye(dim)

        yhat = np.zeros(n)
        beta_trace = np.zeros((n, p))
        u_trace = np.zeros(n)

        innov = np.zeros(n)
        std_innov = np.zeros(n)

        I = np.eye(dim)

        for t in range(n):
            A = np.eye(dim)
            A[p, p] = float(phi_t[t])

            Q = np.zeros((dim, dim))
            Q[:p, :p] = float(q_beta) * np.eye(p)
            Q[p, p] = float(sigma2_t[t])

            # predict
            x_pred = A @ x
            P_pred = A @ P @ A.T + Q

            # observation
            H = np.zeros((1, dim))
            H[0, :p] = X[t]
            H[0, p] = 1.0

            yhat[t] = float(H @ x_pred)
            v = float(y[t] - yhat[t])
            S = float(H @ P_pred @ H.T) + float(r_y)

            innov[t] = v
            std_innov[t] = v / math.sqrt(max(S, 1e-12))

            # update
            K = (P_pred @ H.T) / max(S, 1e-12)
            x = x_pred + (K.flatten() * v)
            P = (I - K @ H) @ P_pred

            beta_trace[t] = x[:p]
            u_trace[t] = x[p]

        return yhat, beta_trace, u_trace, innov, std_innov

    def fit_predict(self, h: int, data_mgr, split, Xy: Dict[str, Tuple[pd.DataFrame, np.ndarray]]) -> Dict[str, Any]:
        p_max = float(data_mgr.p_max)

        Xtrain_full, ytrain_full = Xy["train_df"]
        Xte, yte = Xy["test_df"]

        df_all = pd.concat([split.train_df, split.test_df], axis=0)
        regime = self.regime_spec.build(df_all, h=h)
        phi_t = regime["phi_t"]
        sigma2_t = regime["sigma2_t"]

        beta0 = np.linalg.lstsq(Xtrain_full.values, ytrain_full, rcond=None)[0]

        X_all = np.vstack([Xtrain_full.values, Xte.values])
        y_all = np.concatenate([ytrain_full, yte])

        yhat_all, beta_trace_all, u_trace_all, innov_all, std_innov_all = self._run_kf_timevarying(
            X=X_all,
            y=y_all,
            phi_t=phi_t,
            sigma2_t=sigma2_t,
            q_beta=self.q_beta,
            r_y=self.r_y,
            beta0=beta0,
        )

        pred = clip_pred(yhat_all[len(ytrain_full):], 0.0, p_max)
        resid = yte - pred

        innov_test = innov_all[len(ytrain_full):]
        std_innov_test = std_innov_all[len(ytrain_full):]
        beta_trace_test = beta_trace_all[len(ytrain_full):]
        u_trace_test = u_trace_all[len(ytrain_full):]

        cfg = {
            "q_beta": self.q_beta,
            "r_y": self.r_y,
            "regime_spec": self.regime_spec,
            "phi_summary": {"min": float(np.min(phi_t)), "max": float(np.max(phi_t))},
            "sigma2_summary": {"min": float(np.min(sigma2_t)), "max": float(np.max(sigma2_t))},
        }

        return {
            "pred": pred,
            "y": yte,
            "resid": resid,
            "rb": ResidualBundle(resid=resid, innov=innov_test, std_innov=std_innov_test),
            "beta_trace": beta_trace_test,
            "u_trace": u_trace_test,
            "names": Xte.columns.tolist(),
            "config": cfg,
        }


@dataclass
class TARObsVarSpec:
    """
    Threshold (TAR-like) observation variance specification.

    R_t = r_low  if z_t <= c
        = r_high if z_t >  c

    driver:
      - "Ws": use Ws{h}
      - "abs_dWs": use abs(dWs{h})
      - "dWs": use dWs{h}
    """
    driver: str = "Ws"           # "Ws" | "abs_dWs" | "dWs"
    c_grid: Optional[List[float]] = None   # if None, constructed from quantiles on training z


class KalmanARXWhiteTARObsVar(BaseModel):
    def __init__(
        self,
        tar_spec: TARObsVarSpec = TARObsVarSpec(driver="Ws", c_grid=None),
        q_grid: List[float] = (1e-6, 1e-5, 1e-4, 1e-3),
        r_low_scale_grid: List[float] = (0.25, 0.5, 1.0),
        r_high_scale_grid: List[float] = (1.0, 2.0, 4.0),
        lb_lag: int = 48,
        alpha_lb_penalty: float = 0.15,
        P0_scale: float = 1e2,
    ):
        self.tar_spec = tar_spec
        self.q_grid = list(q_grid)
        self.r_low_scale_grid = list(r_low_scale_grid)
        self.r_high_scale_grid = list(r_high_scale_grid)
        self.lb_lag = int(lb_lag)
        self.alpha_lb_penalty = float(alpha_lb_penalty)
        self.P0_scale = float(P0_scale)

        self.configs_: Dict[int, Dict[str, Any]] = {}

    def _score(self, y_true, pred, std_innov=None) -> Dict[str, float]:
        r = rmse(y_true, pred)
        if std_innov is None:
            p = lb_pvalue(np.asarray(y_true) - np.asarray(pred), lag=self.lb_lag)
        else:
            p = lb_pvalue(std_innov, lag=self.lb_lag)
        score = r + self.alpha_lb_penalty * (-math.log(max(p, 1e-12)))
        return {"rmse": r, "lb_p": p, "score": score}

    def _extract_driver(self, df: pd.DataFrame, h: int) -> np.ndarray:
        if self.tar_spec.driver == "Ws":
            col = f"Ws{h}"
        elif self.tar_spec.driver == "abs_dWs":
            col = f"dWs{h}"
        elif self.tar_spec.driver == "dWs":
            col = f"dWs{h}"
        else:
            raise ValueError(f"Unknown TARObsVarSpec.driver={self.tar_spec.driver!r}")

        if col not in df.columns:
            raise KeyError(
                f"TARObsVarSpec(driver={self.tar_spec.driver!r}) needs column '{col}' in df. "
                f"(Did you add wind-change features to data_prep.py?)"
            )
        z = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if self.tar_spec.driver == "abs_dWs":
            z = np.abs(z)
        return z

    def _make_R_t(self, z: np.ndarray, c: float, r_low: float, r_high: float) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        R_t = np.where(z <= float(c), float(r_low), float(r_high)).astype(float)
        return np.maximum(1e-12, R_t)

    def _kf_run(self, X: np.ndarray, y: np.ndarray, q: float, R_t: np.ndarray, beta0: Optional[np.ndarray] = None):
        """
        Same as KalmanARXWhite._kf_run but with time-varying observation variance R_t.

        Returns:
          yhat, beta_trace, innov, std_innov
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        R_t = np.asarray(R_t, dtype=float)

        n, p = X.shape
        if R_t.shape[0] != n:
            raise ValueError(f"R_t length {R_t.shape[0]} must match n={n}")

        Q = float(q) * np.eye(p)
        I = np.eye(p)

        beta = np.zeros(p) if beta0 is None else np.asarray(beta0, dtype=float).copy()
        P = self.P0_scale * np.eye(p)

        yhat = np.zeros(n)
        beta_trace = np.zeros((n, p))
        innov = np.zeros(n)
        std_innov = np.zeros(n)

        for t in range(n):
            x = X[t].reshape(p, 1)

            # predict
            beta_pred = beta
            P_pred = P + Q

            # innovation
            yhat[t] = float(beta_pred.reshape(1, -1) @ x)
            v = float(y[t] - yhat[t])
            S = float(x.T @ P_pred @ x) + float(R_t[t])

            innov[t] = v
            std_innov[t] = v / math.sqrt(max(S, 1e-12))

            # update
            K = (P_pred @ x) / max(S, 1e-12)
            beta = beta_pred + (K.flatten() * v)
            P = (I - K @ x.T) @ P_pred

            beta_trace[t] = beta

        return yhat, beta_trace, innov, std_innov

    def fit_predict(self, h: int, data_mgr, split, Xy: Dict[str, Tuple[pd.DataFrame, np.ndarray]]) -> Dict[str, Any]:
        p_max = float(data_mgr.p_max)
        yvar = float(split.train_in["p"].var())

        Xtr, ytr = Xy["train_in"]
        Xva, yva = Xy["val_in"]
        Xtrain_full, ytrain_full = Xy["train_df"]
        Xte, yte = Xy["test_df"]

        # driver series aligned with each split
        z_tr = self._extract_driver(split.train_in, h=h)
        z_va = self._extract_driver(split.val_in, h=h)
        z_train_full = self._extract_driver(split.train_df, h=h)
        z_te = self._extract_driver(split.test_df, h=h)

        # build c grid (thresholds) if not provided
        if self.tar_spec.c_grid is None:
            # quantiles on training driver (robust & exam-friendly)
            qtiles = [0.2, 0.35, 0.5, 0.65, 0.8]
            c_grid = [float(np.nanquantile(z_tr, q)) for q in qtiles]
            c_grid = sorted(list({c for c in c_grid if np.isfinite(c)}))
            if len(c_grid) == 0:
                c_grid = [float(np.nanmedian(z_tr))]
        else:
            c_grid = [float(c) for c in self.tar_spec.c_grid]

        best = None

        # tune on (train_in -> val_in)
        for q in self.q_grid:
            for c in c_grid:
                for rls in self.r_low_scale_grid:
                    for rhs in self.r_high_scale_grid:
                        # keep ordering sensible
                        r_low = max(1e-12, float(rls) * yvar)
                        r_high = max(1e-12, float(rhs) * yvar)
                        if r_high < r_low:
                            continue

                        beta0 = np.linalg.lstsq(Xtr.values, ytr, rcond=None)[0]

                        X_all = np.vstack([Xtr.values, Xva.values])
                        y_all = np.concatenate([ytr, yva])
                        z_all = np.concatenate([z_tr, z_va])

                        R_all = self._make_R_t(z_all, c=c, r_low=r_low, r_high=r_high)

                        yhat_all, _, _, std_innov_all = self._kf_run(
                            X_all, y_all, q=float(q), R_t=R_all, beta0=beta0
                        )

                        pred_val = clip_pred(yhat_all[len(ytr):], 0.0, p_max)
                        std_innov_val = std_innov_all[len(ytr):]
                        s = self._score(yva, pred_val, std_innov=std_innov_val)

                        cand = {
                            "q": float(q),
                            "c": float(c),
                            "r_low": float(r_low),
                            "r_high": float(r_high),
                            "driver": self.tar_spec.driver,
                            **s,
                        }
                        if best is None or cand["score"] < best["score"]:
                            best = cand

        assert best is not None
        self.configs_[h] = best

        # fit on full train_df -> test_df
        beta0 = np.linalg.lstsq(Xtrain_full.values, ytrain_full, rcond=None)[0]

        X_all = np.vstack([Xtrain_full.values, Xte.values])
        y_all = np.concatenate([ytrain_full, yte])
        z_all = np.concatenate([z_train_full, z_te])

        R_all = self._make_R_t(z_all, c=best["c"], r_low=best["r_low"], r_high=best["r_high"])

        yhat_all, beta_trace_all, innov_all, std_innov_all = self._kf_run(
            X_all, y_all, q=best["q"], R_t=R_all, beta0=beta0
        )

        pred = clip_pred(yhat_all[len(ytrain_full):], 0.0, p_max)
        resid = yte - pred

        beta_trace_test = beta_trace_all[len(ytrain_full):]
        innov_test = innov_all[len(ytrain_full):]
        std_innov_test = std_innov_all[len(ytrain_full):]

        return {
            "pred": pred,
            "y": yte,
            "resid": resid,
            "rb": ResidualBundle(resid=resid, innov=innov_test, std_innov=std_innov_test),
            "beta_trace": beta_trace_test,
            "names": Xte.columns.tolist(),
            "config": best,
        }
