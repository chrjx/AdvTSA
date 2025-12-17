### Cell: modular_comparison_framework.py
# -----------------------------------------------------------------------------
# Goal: user-friendly, non-duplicated comparison of 3 models end-to-end:
#   1) RLS adaptive ARX (with optional direction TARSO at h=1)
#   2) Kalman adaptive ARX (white observation noise)
#   3) Kalman adaptive ARX + AR(1) observation noise (state augmentation)
#
# Design:
# - Read/prepare data ONCE
# - Build features ONCE per horizon (and reuse for all models)
# - Do ONE time-split that all models share
# - Each model is a thin wrapper implementing fit_predict(h, split, features)
# - Runner computes RMSE + residual diagnostics consistently
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


# =============================================================================
# Shared utilities
# =============================================================================
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def clip_pred(pred, lo, hi):
    return np.clip(np.asarray(pred, dtype=float), lo, hi)

def lb_pvalue(resid: np.ndarray, lag: int = 48) -> float:
    resid = np.asarray(resid, dtype=float)
    out = acorr_ljungbox(resid, lags=[lag], return_df=True)
    return float(out["lb_pvalue"].iloc[0])

def hinge_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    cols = [x]
    for k in knots:
        cols.append(np.maximum(0.0, x - float(k)))
    return np.vstack(cols).T

def half_circle_regime(angle_deg: np.ndarray, cut_deg: float) -> np.ndarray:
    a = np.asarray(angle_deg, dtype=float) % 360.0
    c = float(cut_deg) % 360.0
    d = (a - c) % 360.0
    return d < 180.0


# =============================================================================
# Data + Features (ONE place)
# =============================================================================
@dataclass
class Split:
    train_df: pd.DataFrame
    train_in: pd.DataFrame
    val_in: pd.DataFrame
    test_df: pd.DataFrame

class WindDataManager:
    def __init__(
        self,
        data_path: str,
        time_col: str = "t",
        target_col: str = "p",
        toy_col: str = "toy",
        max_lag: int = 200,
        train_frac: float = 0.7,
        val_frac_within_train: float = 0.2,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.time_col = time_col
        self.target_col = target_col
        self.toy_col = toy_col
        self.max_lag = int(max_lag)
        self.train_frac = float(train_frac)
        self.val_frac = float(val_frac_within_train)
        self.seed = int(seed)
        np.random.seed(self.seed)

        self.df: Optional[pd.DataFrame] = None
        self.knots: Optional[np.ndarray] = None
        self.p_max: Optional[float] = None

        # cache features per horizon
        self._Xy_cache: Dict[int, Tuple[pd.DataFrame, np.ndarray]] = {}

    def load_and_prepare(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, parse_dates=[self.time_col])
        df = df.sort_values(self.time_col).reset_index(drop=True)

        # time features
        df["hour"] = df[self.time_col].dt.hour.astype(int)
        YEAR = 365.25
        df["toy_rad"] = 2 * np.pi * (df[self.toy_col] % YEAR) / YEAR
        df["hour_rad"] = 2 * np.pi * df["hour"] / 24.0

        df["time_idx"] = np.arange(len(df))
        df["week_rad"] = 2 * np.pi * (df["time_idx"] % (24 * 7)) / (24 * 7)

        def add_fourier(df_in, rad_col, K, prefix):
            for k in range(1, K + 1):
                df_in[f"{prefix}_sin{k}"] = np.sin(k * df_in[rad_col])
                df_in[f"{prefix}_cos{k}"] = np.cos(k * df_in[rad_col])
            return df_in

        # stable seasonal basis (as per our chat)
        df = add_fourier(df, "toy_rad",  K=1, prefix="toy")
        df = add_fourier(df, "hour_rad", K=2, prefix="hr")
        df = add_fourier(df, "week_rad", K=1, prefix="wk")

        for h in [1, 2, 3]:
            rad = np.deg2rad(df[f"Wd{h}"] % 360.0)
            df[f"wd{h}_sin"] = np.sin(rad)
            df[f"wd{h}_cos"] = np.cos(rad)
            df[f"T{h}_c"] = df[f"T{h}"] - df[f"T{h}"].mean()

        # lags
        missing_lags = [k for k in range(1, self.max_lag + 1) if f"p_lag{k}" not in df.columns]
        if missing_lags:
            lag_df = pd.concat(
                {f"p_lag{k}": df[self.target_col].shift(k) for k in missing_lags},
                axis=1,
            )
            df = pd.concat([df, lag_df], axis=1)

        df = df.copy()

        # --- explicit two-stage drop (instead of df.dropna())
        essential_cols = (
            [self.target_col]
            + [f"Ws{h}" for h in [1, 2, 3]]
            + [f"Wd{h}" for h in [1, 2, 3]]
            + [f"T{h}"  for h in [1, 2, 3]]
        )
        df = df.dropna(subset=essential_cols)

        lag_cols = [f"p_lag{k}" for k in range(1, self.max_lag + 1)]
        df = df.dropna(subset=lag_cols).reset_index(drop=True)

        # knots from Ws1 quantiles (few hinges for stability)
        knots = df["Ws1"].quantile([0.2, 0.5, 0.8]).values.astype(float)
        knots = np.unique(np.round(knots, 3))

        self.df = df
        self.knots = knots
        self.p_max = float(df[self.target_col].max())
        self._Xy_cache.clear()
        return df

    def split(self) -> Split:
        assert self.df is not None
        df = self.df
        n = len(df)
        cut = int(n * self.train_frac)
        train_df = df.iloc[:cut].copy()
        test_df = df.iloc[cut:].copy()

        ntr = len(train_df)
        cut2 = int(ntr * (1 - self.val_frac))
        train_in = train_df.iloc[:cut2].copy()
        val_in = train_df.iloc[cut2:].copy()
        return Split(train_df=train_df, train_in=train_in, val_in=val_in, test_df=test_df)

    def build_Xy(self, df: pd.DataFrame, h: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Single authoritative feature builder used by ALL models.
        """
        assert self.knots is not None
        out = df.copy()
        knots = self.knots

        ws = out[f"Ws{h}"].astype(float).values
        ws_b = hinge_basis(ws, knots)

        cols: Dict[str, np.ndarray] = {}
        cols["const"] = np.ones(len(out))

        # short lags
        for L in [h, h + 1, h + 2]:
            if L <= self.max_lag:
                cols[f"p_lag{L}"] = out[f"p_lag{L}"].values

        # daily/weekly memory
        for L in [h + 24, h + 48, h + 168]:
            if L <= self.max_lag:
                cols[f"p_lag{L}"] = out[f"p_lag{L}"].values

        # weather
        cols[f"T{h}_c"] = out[f"T{h}_c"].values
        cols[f"wd{h}_sin"] = out[f"wd{h}_sin"].values
        cols[f"wd{h}_cos"] = out[f"wd{h}_cos"].values

        # nonlinear ws
        cols[f"ws{h}"] = ws_b[:, 0]
        for i, k in enumerate(knots, start=1):
            cols[f"ws{h}_hinge_{k:g}"] = ws_b[:, i]

        # stable interaction
        cols[f"ws{h}_x_sin"] = cols[f"ws{h}"] * cols[f"wd{h}_sin"]
        cols[f"ws{h}_x_cos"] = cols[f"ws{h}"] * cols[f"wd{h}_cos"]

        # seasonality
        cols["toy_sin1"] = out["toy_sin1"].values
        cols["toy_cos1"] = out["toy_cos1"].values
        for k in [1, 2]:
            cols[f"hr_sin{k}"] = out[f"hr_sin{k}"].values
            cols[f"hr_cos{k}"] = out[f"hr_cos{k}"].values
        cols["wk_sin1"] = out["wk_sin1"].values
        cols["wk_cos1"] = out["wk_cos1"].values

        X = pd.DataFrame(cols, index=out.index)
        y = pd.Series(out["p"].values.astype(float), index=out.index, name="y")
        return X, y


    def get_cached_Xy(self, h: int, split: Split) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Compute X/y for train_df and test_df ONCE per horizon,
        then serve slices for train_in/val_in without rebuilding matrices.
        Ensures strict X/y alignment via index-based slicing.
        """
        assert self.df is not None

        key = (h, "train_df")
        if key not in self._Xy_cache:
            Xtr, ytr = self.build_Xy(split.train_df, h)   # ytr is a Series
            self._Xy_cache[key] = (Xtr, ytr)

        key = (h, "test_df")
        if key not in self._Xy_cache:
            Xte, yte = self.build_Xy(split.test_df, h)    # yte is a Series
            self._Xy_cache[key] = (Xte, yte)

        X_train_df, y_train_df = self._Xy_cache[(h, "train_df")]
        X_test_df,  y_test_df  = self._Xy_cache[(h, "test_df")]

        idx_train_in = split.train_in.index
        idx_val_in   = split.val_in.index

        X_train_in = X_train_df.loc[idx_train_in]
        y_train_in = y_train_df.loc[idx_train_in].values  # numpy for downstream models
        X_val_in   = X_train_df.loc[idx_val_in]
        y_val_in   = y_train_df.loc[idx_val_in].values

        return {
            "train_df": (X_train_df, y_train_df.values),
            "train_in": (X_train_in, y_train_in),
            "val_in":   (X_val_in,   y_val_in),
            "test_df":  (X_test_df,  y_test_df.values),
        }


# =============================================================================
# Model wrappers (keep functionality, no duplicated prep)
# =============================================================================
class RLS_ARX_Model:
    """
    Wraps your RLS pipeline logic but consumes prepared X/y from WindDataManager.
    Supports optional TARSO direction split at h=1.
    """

    def __init__(
        self,
        lam_grid: List[float] = [0.9995, 0.999, 0.997, 0.995],
        cut_grid: List[float] = list(range(0, 360, 30)),
        lb_lag: int = 48,
        alpha_lb_penalty: float = 0.15,
        min_regime_n: int = 500,
        delta: float = 1e4,
        tarso_h1: bool = True,
    ):
        self.lam_grid = lam_grid
        self.cut_grid = cut_grid
        self.lb_lag = lb_lag
        self.alpha_lb_penalty = alpha_lb_penalty
        self.min_regime_n = min_regime_n
        self.delta = float(delta)
        self.tarso_h1 = bool(tarso_h1)

        self.configs_: Dict[int, Dict[str, Any]] = {}

    def _score(self, y_true, y_pred) -> Dict[str, float]:
        r = rmse(y_true, y_pred)
        p = lb_pvalue(np.asarray(y_true) - np.asarray(y_pred), lag=self.lb_lag)
        score = r + self.alpha_lb_penalty * (-math.log(max(p, 1e-12)))
        return {"rmse": r, "lb_p": p, "score": score}

    def _rls_predict_update(self, X_train, y_train, X_test, y_test, lam: float):
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = np.asarray(y_train, dtype=float)
        y_test = np.asarray(y_test, dtype=float)

        p = X_train.shape[1]
        theta = np.zeros(p)
        P = self.delta * np.eye(p)

        # burn-in
        for t in range(len(X_train)):
            x = X_train.iloc[t].values.reshape(-1, 1)
            y = float(y_train[t])
            denom = lam + float(x.T @ P @ x)
            K = (P @ x) / denom
            err = y - float(theta.reshape(1, -1) @ x)
            theta = theta + (K.flatten() * err)
            P = (P - K @ x.T @ P) / lam

        # test predict+update
        pred = np.zeros(len(X_test))
        trace = np.zeros((len(X_test), p))
        for t in range(len(X_test)):
            x = X_test.iloc[t].values.reshape(-1, 1)
            pred[t] = float(theta.reshape(1, -1) @ x)
            trace[t] = theta

            denom = lam + float(x.T @ P @ x)
            K = (P @ x) / denom
            err = float(y_test[t]) - pred[t]
            theta = theta + (K.flatten() * err)
            P = (P - K @ x.T @ P) / lam

        return pred, trace

    def fit_predict(
        self,
        h: int,
        data_mgr: WindDataManager,
        split: Split,
        Xy: Dict[str, Tuple[pd.DataFrame, np.ndarray]],
    ) -> Dict[str, Any]:
        p_max = data_mgr.p_max

        Xtr, ytr = Xy["train_in"]
        Xva, yva = Xy["val_in"]
        Xtrain_full, ytrain_full = Xy["train_df"]
        Xte, yte = Xy["test_df"]

        # choose config (lam and optional cut)
        best = None

        # --- single
        for lam in self.lam_grid:
            pred_val, _ = self._rls_predict_update(Xtr, ytr, Xva, yva, lam)
            pred_val = clip_pred(pred_val, 0.0, p_max)
            s = self._score(yva, pred_val)
            cand = {"mode": "single", "lam": lam, "cut_deg": None, **s}
            if best is None or cand["score"] < best["score"]:
                best = cand

        # --- TARSO only for h=1 (and if enabled)
        if h == 1 and self.tarso_h1:
            for lam in self.lam_grid:
                for cut in self.cut_grid:
                    trA = half_circle_regime(split.train_in["Wd1"].values, cut)
                    vaA = half_circle_regime(split.val_in["Wd1"].values, cut)

                    train_A = split.train_in.loc[split.train_in.index[trA]]
                    train_B = split.train_in.loc[split.train_in.index[~trA]]
                    val_A = split.val_in.loc[split.val_in.index[vaA]]
                    val_B = split.val_in.loc[split.val_in.index[~vaA]]

                    if min(len(train_A), len(train_B), len(val_A), len(val_B)) < self.min_regime_n:
                        continue

                    XA_tr, yA_tr = data_mgr.build_Xy(train_A, 1)
                    XB_tr, yB_tr = data_mgr.build_Xy(train_B, 1)
                    XA_va, yA_va = data_mgr.build_Xy(val_A, 1)
                    XB_va, yB_va = data_mgr.build_Xy(val_B, 1)

                    predA, _ = self._rls_predict_update(XA_tr, yA_tr, XA_va, yA_va, lam)
                    predB, _ = self._rls_predict_update(XB_tr, yB_tr, XB_va, yB_va, lam)

                    pred_full = pd.Series(index=split.val_in.index, dtype=float)
                    pred_full.loc[val_A.index] = predA
                    pred_full.loc[val_B.index] = predB
                    pred_full = clip_pred(pred_full.sort_index().values, 0.0, p_max)

                    s = self._score(yva, pred_full)
                    cand = {"mode": "tarso_dir", "lam": lam, "cut_deg": float(cut), **s}
                    if best is None or cand["score"] < best["score"]:
                        best = cand

        assert best is not None
        self.configs_[h] = best

        # fit on full train (train_df) and predict test
        if best["mode"] == "single":
            pred_test, trace = self._rls_predict_update(Xtrain_full, ytrain_full, Xte, yte, best["lam"])
            pred_test = clip_pred(pred_test, 0.0, p_max)
            resid = yte - pred_test
            return {
                "pred": pred_test,
                "y": yte,
                "resid": resid,
                "trace": trace,
                "names": Xte.columns.tolist(),
                "config": best,
            }

        # TARSO test
        cut = best["cut_deg"]
        teA = half_circle_regime(split.test_df["Wd1"].values, cut)
        trainA = split.train_df.loc[half_circle_regime(split.train_df["Wd1"].values, cut)]
        trainB = split.train_df.loc[~half_circle_regime(split.train_df["Wd1"].values, cut)]
        testA = split.test_df.loc[split.test_df.index[teA]]
        testB = split.test_df.loc[split.test_df.index[~teA]]

        XA_tr, yA_tr = data_mgr.build_Xy(trainA, 1)
        XB_tr, yB_tr = data_mgr.build_Xy(trainB, 1)
        XA_te, yA_te = data_mgr.build_Xy(testA, 1)
        XB_te, yB_te = data_mgr.build_Xy(testB, 1)

        predA, traceA = self._rls_predict_update(XA_tr, yA_tr, XA_te, yA_te, best["lam"])
        predB, traceB = self._rls_predict_update(XB_tr, yB_tr, XB_te, yB_te, best["lam"])

        pred_full = pd.Series(index=split.test_df.index, dtype=float)
        pred_full.loc[testA.index] = predA
        pred_full.loc[testB.index] = predB
        pred_full = clip_pred(pred_full.sort_index().values, 0.0, p_max)

        yte_full = split.test_df["p"].values.astype(float)
        resid = yte_full - pred_full

        return {
            "pred": pred_full,
            "y": yte_full,
            "resid": resid,
            "trace_A": traceA,
            "trace_B": traceB,
            "names_A": XA_te.columns.tolist(),
            "names_B": XB_te.columns.tolist(),
            "config": best,
        }


class Kalman_White_ARX_Model:
    """
    Wraps your Kalman (white noise) model in a thin interface.
    Uses q,r grid search (same as your pipeline) but reuses features from DataManager.
    """

    def __init__(
        self,
        q_grid: List[float] = [1e-6, 1e-5, 1e-4, 1e-3],
        r_scale_grid: List[float] = [0.25, 0.5, 1.0, 2.0],
        lb_lag: int = 48,
        alpha_lb_penalty: float = 0.15,
        tarso_h1: bool = True,
        cut_grid: List[float] = list(range(0, 360, 30)),
        min_regime_n: int = 800,
    ):
        self.q_grid = q_grid
        self.r_scale_grid = r_scale_grid
        self.lb_lag = lb_lag
        self.alpha_lb_penalty = alpha_lb_penalty
        self.tarso_h1 = bool(tarso_h1)
        self.cut_grid = cut_grid
        self.min_regime_n = min_regime_n

        self.configs_: Dict[int, Dict[str, Any]] = {}

    class _KF:
        def __init__(self, q, r, beta0=None, P0_scale=1e2):
            self.q = float(q)
            self.r = float(r)
            self.beta0 = beta0
            self.P0_scale = float(P0_scale)

        def run(self, X, y):
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            n, p = X.shape
            Q = self.q * np.eye(p)
            R = self.r

            beta = np.zeros(p) if self.beta0 is None else np.asarray(self.beta0, dtype=float).copy()
            P = self.P0_scale * np.eye(p)

            yhat = np.zeros(n)
            beta_trace = np.zeros((n, p))
            Pdiag = np.zeros((n, p))
            I = np.eye(p)

            for t in range(n):
                x = X[t].reshape(p, 1)
                beta_pred = beta
                P_pred = P + Q
                yhat[t] = float(beta_pred.reshape(1, -1) @ x)
                S = float(x.T @ P_pred @ x) + R
                K = (P_pred @ x) / S
                innov = y[t] - yhat[t]
                beta = beta_pred + (K.flatten() * innov)
                P = (I - K @ x.T) @ P_pred
                beta_trace[t] = beta
                Pdiag[t] = np.diag(P)

            return yhat, beta_trace, Pdiag

    def _score(self, y_true, y_pred):
        r = rmse(y_true, y_pred)
        p = lb_pvalue(np.asarray(y_true) - np.asarray(y_pred), lag=self.lb_lag)
        score = r + self.alpha_lb_penalty * (-math.log(max(p, 1e-12)))
        return {"rmse": r, "lb_p": p, "score": score}

    def fit_predict(self, h: int, data_mgr: WindDataManager, split: Split, Xy: Dict[str, Tuple[pd.DataFrame, np.ndarray]]) -> Dict[str, Any]:
        p_max = data_mgr.p_max
        yvar = float(split.train_in["p"].var())

        Xtr, ytr = Xy["train_in"]
        Xva, yva = Xy["val_in"]
        Xtrain_full, ytrain_full = Xy["train_df"]
        Xte, yte = Xy["test_df"]

        best = None

        # single (and TARSO for h=1 optional)
        for q in self.q_grid:
            for rs in self.r_scale_grid:
                r = max(1e-8, rs * yvar)
                beta0 = np.linalg.lstsq(Xtr.values, ytr, rcond=None)[0]
                kf = self._KF(q=q, r=r, beta0=beta0, P0_scale=1e2)

                yhat_all, _, _ = kf.run(np.vstack([Xtr.values, Xva.values]), np.concatenate([ytr, yva]))
                pred_val = clip_pred(yhat_all[len(ytr):], 0.0, p_max)
                s = self._score(yva, pred_val)
                cand = {"mode": "single", "q": q, "r": r, "cut_deg": None, **s}
                if best is None or cand["score"] < best["score"]:
                    best = cand

        if h == 1 and self.tarso_h1:
            for q in self.q_grid:
                for rs in self.r_scale_grid:
                    r = max(1e-8, rs * yvar)
                    for cut in self.cut_grid:
                        trA = half_circle_regime(split.train_in["Wd1"].values, cut)
                        vaA = half_circle_regime(split.val_in["Wd1"].values, cut)

                        train_A = split.train_in.loc[split.train_in.index[trA]]
                        train_B = split.train_in.loc[split.train_in.index[~trA]]
                        val_A = split.val_in.loc[split.val_in.index[vaA]]
                        val_B = split.val_in.loc[split.val_in.index[~vaA]]

                        if min(len(train_A), len(train_B), len(val_A), len(val_B)) < self.min_regime_n:
                            continue

                        XA_tr, yA_tr = data_mgr.build_Xy(train_A, 1)
                        XB_tr, yB_tr = data_mgr.build_Xy(train_B, 1)
                        XA_va, yA_va = data_mgr.build_Xy(val_A, 1)
                        XB_va, yB_va = data_mgr.build_Xy(val_B, 1)

                        beta0A = np.linalg.lstsq(XA_tr.values, yA_tr, rcond=None)[0]
                        beta0B = np.linalg.lstsq(XB_tr.values, yB_tr, rcond=None)[0]
                        kfA = self._KF(q=q, r=r, beta0=beta0A, P0_scale=1e2)
                        kfB = self._KF(q=q, r=r, beta0=beta0B, P0_scale=1e2)

                        yhatA, _, _ = kfA.run(np.vstack([XA_tr.values, XA_va.values]), np.concatenate([yA_tr, yA_va]))
                        yhatB, _, _ = kfB.run(np.vstack([XB_tr.values, XB_va.values]), np.concatenate([yB_tr, yB_va]))

                        predA = yhatA[len(yA_tr):]
                        predB = yhatB[len(yB_tr):]

                        pred_full = pd.Series(index=split.val_in.index, dtype=float)
                        pred_full.loc[val_A.index] = predA
                        pred_full.loc[val_B.index] = predB
                        pred_full = clip_pred(pred_full.sort_index().values, 0.0, p_max)

                        s = self._score(yva, pred_full)
                        cand = {"mode": "tarso_dir", "q": q, "r": r, "cut_deg": float(cut), **s}
                        if best is None or cand["score"] < best["score"]:
                            best = cand

        assert best is not None
        self.configs_[h] = best

        # fit on full train_df -> test_df
        if best["mode"] == "single":
            beta0 = np.linalg.lstsq(Xtrain_full.values, ytrain_full, rcond=None)[0]
            kf = self._KF(q=best["q"], r=best["r"], beta0=beta0, P0_scale=1e2)
            yhat_all, beta_trace_all, _ = kf.run(np.vstack([Xtrain_full.values, Xte.values]), np.concatenate([ytrain_full, yte]))
            pred = clip_pred(yhat_all[len(ytrain_full):], 0.0, p_max)
            resid = yte - pred
            return {"pred": pred, "y": yte, "resid": resid, "beta_trace": beta_trace_all[len(ytrain_full):], "names": Xte.columns.tolist(), "config": best}

        # TARSO test
        cut = best["cut_deg"]
        trA = half_circle_regime(split.train_df["Wd1"].values, cut)
        teA = half_circle_regime(split.test_df["Wd1"].values, cut)
        trainA = split.train_df.loc[split.train_df.index[trA]]
        trainB = split.train_df.loc[split.train_df.index[~trA]]
        testA = split.test_df.loc[split.test_df.index[teA]]
        testB = split.test_df.loc[split.test_df.index[~teA]]

        XA_tr, yA_tr = data_mgr.build_Xy(trainA, 1)
        XB_tr, yB_tr = data_mgr.build_Xy(trainB, 1)
        XA_te, yA_te = data_mgr.build_Xy(testA, 1)
        XB_te, yB_te = data_mgr.build_Xy(testB, 1)

        beta0A = np.linalg.lstsq(XA_tr.values, yA_tr, rcond=None)[0]
        beta0B = np.linalg.lstsq(XB_tr.values, yB_tr, rcond=None)[0]
        kfA = self._KF(q=best["q"], r=best["r"], beta0=beta0A, P0_scale=1e2)
        kfB = self._KF(q=best["q"], r=best["r"], beta0=beta0B, P0_scale=1e2)

        yhatA, betaA, _ = kfA.run(np.vstack([XA_tr.values, XA_te.values]), np.concatenate([yA_tr, yA_te]))
        yhatB, betaB, _ = kfB.run(np.vstack([XB_tr.values, XB_te.values]), np.concatenate([yB_tr, yB_te]))
        predA = yhatA[len(yA_tr):]
        predB = yhatB[len(yB_tr):]

        pred_full = pd.Series(index=split.test_df.index, dtype=float)
        pred_full.loc[testA.index] = predA
        pred_full.loc[testB.index] = predB
        pred_full = clip_pred(pred_full.sort_index().values, 0.0, p_max)

        y = split.test_df["p"].values.astype(float)
        resid = y - pred_full
        return {"pred": pred_full, "y": y, "resid": resid, "beta_A": betaA[len(yA_tr):], "beta_B": betaB[len(yB_tr):], "config": best}


# You already have kalman_arx_ar1.py with fit_kalman_arx_ar1().
# We wrap it here so it plugs into the same comparison harness.
class Kalman_AR1_Noise_Model:
    def __init__(
        self,
        phi: float = 0.6,
        q_beta: float = 1e-5,
        r_eps: float = 0.5,
    ):
        from kalman_arx_ar1 import fit_kalman_arx_ar1  # external module you created
        self._fit = fit_kalman_arx_ar1
        self.phi = float(phi)
        self.q_beta = float(q_beta)
        self.r_eps = float(r_eps)

    def fit_predict(self, h: int, data_mgr: WindDataManager, split: Split, Xy: Dict[str, Tuple[pd.DataFrame, np.ndarray]]) -> Dict[str, Any]:
        p_max = data_mgr.p_max
        Xtrain_full, ytrain_full = Xy["train_df"]
        Xte, yte = Xy["test_df"]

        out = self._fit(
            X_train=Xtrain_full.values,
            y_train=ytrain_full,
            X_test=Xte.values,
            y_test=yte,
            phi=self.phi,
            q_beta=self.q_beta,
            r_eps=self.r_eps,
            clip=(0.0, p_max),
        )
        pred = out["y_pred"]
        resid = out["residuals"]
        return {
            "pred": pred,
            "y": yte,
            "resid": resid,
            "beta_trace": out["beta_trace"],
            "eps_trace": out["eps_trace"],
            "config": {"phi": self.phi, "q_beta": self.q_beta, "r_eps": self.r_eps},
        }


# =============================================================================
# Comparison runner (ONE place for RMSE + residual analysis)
# =============================================================================
class ComparisonRunner:
    def __init__(self, data_mgr: WindDataManager, lb_lag: int = 48):
        self.data_mgr = data_mgr
        self.lb_lag = int(lb_lag)

    def run(self, models: Dict[str, Any], horizons: List[int] = [1, 2, 3]) -> Tuple[pd.DataFrame, Dict[str, Dict[int, Dict[str, Any]]]]:
        """
        Returns:
          results_df: model x horizon summary (RMSE, MAE, LB p-value, persistence RMSE, skill)
          traces: traces[model_name][h] = dict(pred,y,resid,config, ...)
        """
        split = self.data_mgr.split()
        traces: Dict[str, Dict[int, Dict[str, Any]]] = {}
        rows = []

        for h in horizons:
            Xy = self.data_mgr.get_cached_Xy(h, split)

            # baseline persistence (availability-correct)
            y_test = Xy["test_df"][1]
            y_persist = split.test_df.loc[Xy["test_df"][0].index, f"p_lag{h}"].values.astype(float)
            base_rmse = rmse(y_test, y_persist)

            for name, model in models.items():
                out = model.fit_predict(h=h, data_mgr=self.data_mgr, split=split, Xy=Xy)
                pred = out["pred"]
                y = out["y"]
                resid = out["resid"]

                rows.append({
                    "model": name,
                    "h": h,
                    "RMSE": rmse(y, pred),
                    "MAE": mae(y, pred),
                    f"LB_pvalue_lag{self.lb_lag}": lb_pvalue(resid, lag=self.lb_lag),
                    "RMSE_persistence": base_rmse,
                    "Skill_vs_persistence": 1.0 - (rmse(y, pred) / base_rmse),
                })

                traces.setdefault(name, {})[h] = out

        results_df = pd.DataFrame(rows).sort_values(["h", "RMSE"]).reset_index(drop=True)
        return results_df, traces

    @staticmethod
    def residual_plots(resid: np.ndarray, title_prefix: str = "", lags: int = 48):
        resid = np.asarray(resid, dtype=float)

        plt.figure(figsize=(12, 3))
        plt.plot(resid)
        plt.title(f"{title_prefix} Residuals")
        plt.xlabel("Index")
        plt.ylabel("Residual")
        plt.show()

        plt.figure()
        plot_acf(resid, lags=lags)
        plt.title(f"{title_prefix} Residual ACF")
        plt.show()

        plt.figure()
        plot_pacf(resid, lags=min(lags, 30))
        plt.title(f"{title_prefix} Residual PACF")
        plt.show()

        lb = acorr_ljungbox(resid, lags=[12, 24, 48], return_df=True)
        print(lb)
