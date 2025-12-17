### Cell: wind_power_pipeline.py (write a full callable pipeline as a single-class module)
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


# ============================================================
# Utilities
# ============================================================
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def _clip_pred(pred: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=float), lo, hi)

def _lb_pvalue(resid: np.ndarray, lag: int = 48) -> float:
    resid = np.asarray(resid, dtype=float)
    out = acorr_ljungbox(resid, lags=[lag], return_df=True)
    return float(out["lb_pvalue"].iloc[0])

def _hinge_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Columns: [x, max(0, x-k1), max(0, x-k2), ...]
    """
    x = np.asarray(x, dtype=float)
    Xb = [x]
    for k in knots:
        Xb.append(np.maximum(0.0, x - float(k)))
    return np.vstack(Xb).T


def _half_circle_regime(angle_deg: np.ndarray, cut_deg: float) -> np.ndarray:
    """
    Two regimes separated by cut direction c:
      Regime A: angles within [c, c+180)
      Regime B: the opposite half-circle
    Returns boolean mask for Regime A.
    """
    a = np.asarray(angle_deg, dtype=float) % 360.0
    c = float(cut_deg) % 360.0
    d = (a - c) % 360.0
    return d < 180.0


@dataclass
class HorizonConfig:
    h: int
    mode: str  # "tarso_dir" or "single"
    lam: float
    cut_deg: Optional[float] = None  # only for tarso_dir
    clip: bool = True


class WindPowerAdaptivePipeline:
    """
    Final model for wind power forecasting (1–3h ahead) based on this chat:
      - Nonlinear ARX with hinge basis for wind speed
      - Circular wind direction (sin/cos)
      - Seasonality (annual+diurnal Fourier) + optional weekly Fourier
      - Seasonal AR lags (daily/weekly) availability-correct: p_lag(h+24), p_lag(h+48), p_lag(h+168)
      - Adaptive estimation via RLS with forgetting factor
      - Regime switching by wind direction (TARSO-like) ONLY for h=1 by default
        because h=2 was borderline and h=3 was unstable with hard regimes.

    Design choices (from your diagnostics):
      - h=1: TARSO-dir (2 regimes, half-circle split) is often beneficial and stable enough.
      - h=2: single-regime adaptive ARX (avoid instability from splitting).
      - h=3: single-regime adaptive ARX with extra care on stability (stronger regularization via lam ~ 0.999+).
    """

    def __init__(
        self,
        data_path: str,
        time_col: str = "t",
        target_col: str = "p",
        toy_col: str = "toy",
        train_frac: float = 0.7,
        val_frac_within_train: float = 0.2,
        max_lag: int = 200,
        lb_lag: int = 48,
        alpha_lb_penalty: float = 0.15,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.time_col = time_col
        self.target_col = target_col
        self.toy_col = toy_col
        self.train_frac = train_frac
        self.val_frac = val_frac_within_train
        self.max_lag = max_lag
        self.lb_lag = lb_lag
        self.alpha_lb_penalty = alpha_lb_penalty
        self.seed = seed

        self.df_: Optional[pd.DataFrame] = None
        self.knots_: Optional[np.ndarray] = None
        self.p_max_: Optional[float] = None

        self.configs_: Dict[int, HorizonConfig] = {}
        self.fitted_: bool = False

        # Stored artifacts after fit
        self._trace_: Dict[str, Any] = {}

        np.random.seed(self.seed)

    # ---------------------------
    # Data preparation
    # ---------------------------
    def load_and_prepare(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, parse_dates=[self.time_col])
        df = df.sort_values(self.time_col).reset_index(drop=True)

        # time features
        df["hour"] = df[self.time_col].dt.hour.astype(int)

        YEAR = 365.25
        df["toy_rad"] = 2 * np.pi * (df[self.toy_col] % YEAR) / YEAR
        df["hour_rad"] = 2 * np.pi * df["hour"] / 24.0

        # weekly phase from index (simple + robust)
        df["time_idx"] = np.arange(len(df))
        df["week_rad"] = 2 * np.pi * (df["time_idx"] % (24 * 7)) / (24 * 7)

        # Fourier helpers
        def add_fourier(df_in, rad_col, K, prefix):
            out = df_in.copy()
            for k in range(1, K + 1):
                out[f"{prefix}_sin{k}"] = np.sin(k * out[rad_col])
                out[f"{prefix}_cos{k}"] = np.cos(k * out[rad_col])
            return out

        df = add_fourier(df, "toy_rad",  K=1, prefix="toy")   # keep only k=1 for stability in final model
        df = add_fourier(df, "hour_rad", K=2, prefix="hr")    # diurnal: k=1..2
        df = add_fourier(df, "week_rad", K=1, prefix="wk")    # weekly harmonic (optional but cheap)

        # direction sin/cos + temperature centered per horizon
        for h in [1, 2, 3]:
            rad = np.deg2rad(df[f"Wd{h}"] % 360.0)
            df[f"wd{h}_sin"] = np.sin(rad)
            df[f"wd{h}_cos"] = np.cos(rad)
            df[f"T{h}_c"] = df[f"T{h}"] - df[f"T{h}"].mean()

        # lags
        for k in range(1, self.max_lag + 1):
            df[f"p_lag{k}"] = df[self.target_col].shift(k)

        df = df.dropna().reset_index(drop=True)

        # knots from Ws1 quantiles (fixed across horizons for comparability)
        knots = df["Ws1"].quantile([0.2, 0.5, 0.8]).values  # fewer hinges to avoid instability
        knots = np.unique(np.round(knots.astype(float), 3))

        self.df_ = df
        self.knots_ = knots
        self.p_max_ = float(df[self.target_col].max())
        return df

    def split_time(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        cut = int(n * self.train_frac)
        train_df = df.iloc[:cut].copy()
        test_df = df.iloc[cut:].copy()

        ntr = len(train_df)
        cut2 = int(ntr * (1 - self.val_frac))
        train_in = train_df.iloc[:cut2].copy()
        val_in = train_df.iloc[cut2:].copy()
        return train_in, val_in, test_df

    # ---------------------------
    # Feature matrix for final model
    # ---------------------------
    def build_Xy(self, df: pd.DataFrame, h: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Stabilized feature set (availability-correct):
          - const
          - AR lags: p_lag(h), p_lag(h+1), p_lag(h+2)
          - seasonal AR lags: p_lag(h+24), p_lag(h+48), p_lag(h+168) if within max_lag
          - nonlinear wind speed: hinge basis with few knots
          - direction sin/cos
          - interaction: ws * sin/cos only
          - seasonality: toy_sin1/cos1, hr_sin/cos 1..2, weekly sin/cos1
          - temperature centered
        """
        assert self.knots_ is not None, "Call load_and_prepare() first"

        out = df.copy()
        knots = self.knots_

        ws = out[f"Ws{h}"].astype(float).values
        ws_b = _hinge_basis(ws, knots)

        cols: Dict[str, np.ndarray] = {}
        cols["const"] = np.ones(len(out))

        # short horizon-available lags
        for L in [h, h + 1, h + 2]:
            if L <= self.max_lag:
                cols[f"p_lag{L}"] = out[f"p_lag{L}"].values

        # seasonal lags
        for L in [h + 24, h + 48, h + 168]:
            if L <= self.max_lag:
                cols[f"p_lag{L}"] = out[f"p_lag{L}"].values

        # weather / direction / temp
        cols[f"T{h}_c"] = out[f"T{h}_c"].values
        cols[f"wd{h}_sin"] = out[f"wd{h}_sin"].values
        cols[f"wd{h}_cos"] = out[f"wd{h}_cos"].values

        # nonlinear Ws (few hinges)
        cols[f"ws{h}"] = ws_b[:, 0]
        for i, k in enumerate(knots, start=1):
            cols[f"ws{h}_hinge_{k:g}"] = ws_b[:, i]

        # limited interaction (stability)
        cols[f"ws{h}_x_sin"] = cols[f"ws{h}"] * cols[f"wd{h}_sin"]
        cols[f"ws{h}_x_cos"] = cols[f"ws{h}"] * cols[f"wd{h}_cos"]

        # seasonality in mean
        cols["toy_sin1"] = out["toy_sin1"].values
        cols["toy_cos1"] = out["toy_cos1"].values
        for k in [1, 2]:
            cols[f"hr_sin{k}"] = out[f"hr_sin{k}"].values
            cols[f"hr_cos{k}"] = out[f"hr_cos{k}"].values
        cols["wk_sin1"] = out["wk_sin1"].values
        cols["wk_cos1"] = out["wk_cos1"].values

        X = pd.DataFrame(cols, index=out.index)
        y = out[self.target_col].values.astype(float)
        return X, y

    # ---------------------------
    # Adaptive estimation: RLS with forgetting
    # ---------------------------
    def rls_predict_update(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        lam: float,
        delta: float = 1e4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Burn-in on train, then online predict+update on test.
        Returns: (pred_test, theta_trace_test)
        """
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = np.asarray(y_train, dtype=float)
        y_test = np.asarray(y_test, dtype=float)

        n_params = X_train.shape[1]
        theta = np.zeros(n_params, dtype=float)
        P = delta * np.eye(n_params, dtype=float)

        # burn-in
        for t in range(len(X_train)):
            x = X_train.iloc[t].values.reshape(-1, 1)
            y = float(y_train[t])
            denom = lam + float(x.T @ P @ x)
            K = (P @ x) / denom
            err = y - float(theta.reshape(1, -1) @ x)
            theta = theta + (K.flatten() * err)
            P = (P - K @ x.T @ P) / lam

        # online predict+update
        preds = np.zeros(len(X_test), dtype=float)
        trace = np.zeros((len(X_test), n_params), dtype=float)

        for t in range(len(X_test)):
            x = X_test.iloc[t].values.reshape(-1, 1)
            y = float(y_test[t])

            preds[t] = float(theta.reshape(1, -1) @ x)
            trace[t] = theta

            denom = lam + float(x.T @ P @ x)
            K = (P @ x) / denom
            err = y - preds[t]
            theta = theta + (K.flatten() * err)
            P = (P - K @ x.T @ P) / lam

        return preds, trace

    # ---------------------------
    # Model selection (validation): choose lam and (for h=1) TARSO cut
    # ---------------------------
    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        resid = y_true - y_pred
        r = _rmse(y_true, y_pred)
        p = _lb_pvalue(resid, lag=self.lb_lag)
        penalty = -math.log(max(p, 1e-12))
        score = r + self.alpha_lb_penalty * penalty
        return {"rmse": r, "lb_p": p, "score": score}

    def select_configs(
        self,
        train_in: pd.DataFrame,
        val_in: pd.DataFrame,
        lam_grid: List[float] = [0.9995, 0.999, 0.997, 0.995],
        cut_grid: List[float] = list(range(0, 360, 30)),
        min_regime_n: int = 500,
    ) -> Dict[int, HorizonConfig]:
        assert self.p_max_ is not None

        configs: Dict[int, HorizonConfig] = {}

        # ---- h=1: allow TARSO-dir or single; choose best
        h = 1
        best: Optional[Dict[str, Any]] = None

        # single candidates
        Xtr, ytr = self.build_Xy(train_in, h)
        Xva, yva = self.build_Xy(val_in, h)

        for lam in lam_grid:
            pred, _ = self.rls_predict_update(Xtr, ytr, Xva, yva, lam=lam)
            pred = _clip_pred(pred, 0.0, self.p_max_)
            s = self._score(yva, pred)
            cand = {"mode": "single", "lam": lam, "cut": None, **s}
            if best is None or cand["score"] < best["score"]:
                best = cand

        # TARSO-dir candidates
        for lam in lam_grid:
            for cut in cut_grid:
                trA = _half_circle_regime(train_in["Wd1"].values, cut)
                vaA = _half_circle_regime(val_in["Wd1"].values, cut)

                if trA.mean() < 0.1 or trA.mean() > 0.9:
                    continue
                if vaA.mean() < 0.1 or vaA.mean() > 0.9:
                    continue

                train_A = train_in.loc[train_in.index[trA]].copy()
                train_B = train_in.loc[train_in.index[~trA]].copy()
                val_A = val_in.loc[val_in.index[vaA]].copy()
                val_B = val_in.loc[val_in.index[~vaA]].copy()

                if min(len(train_A), len(train_B), len(val_A), len(val_B)) < min_regime_n:
                    continue

                XA_tr, yA_tr = self.build_Xy(train_A, h)
                XB_tr, yB_tr = self.build_Xy(train_B, h)
                XA_va, yA_va = self.build_Xy(val_A, h)
                XB_va, yB_va = self.build_Xy(val_B, h)

                predA, _ = self.rls_predict_update(XA_tr, yA_tr, XA_va, yA_va, lam=lam)
                predB, _ = self.rls_predict_update(XB_tr, yB_tr, XB_va, yB_va, lam=lam)

                pred_full = pd.Series(index=val_in.index, dtype=float)
                pred_full.loc[val_A.index] = predA
                pred_full.loc[val_B.index] = predB
                pred_full = pred_full.sort_index().values
                pred_full = _clip_pred(pred_full, 0.0, self.p_max_)

                s = self._score(yva, pred_full)
                cand = {"mode": "tarso_dir", "lam": lam, "cut": float(cut), **s}
                if best is None or cand["score"] < best["score"]:
                    best = cand

        assert best is not None
        configs[1] = HorizonConfig(h=1, mode=best["mode"], lam=float(best["lam"]), cut_deg=best["cut"], clip=True)

        # ---- h=2 and h=3: single only (based on your stability findings)
        for h in [2, 3]:
            best = None
            Xtr, ytr = self.build_Xy(train_in, h)
            Xva, yva = self.build_Xy(val_in, h)
            for lam in lam_grid:
                pred, _ = self.rls_predict_update(Xtr, ytr, Xva, yva, lam=lam)
                pred = _clip_pred(pred, 0.0, self.p_max_)
                s = self._score(yva, pred)
                cand = {"lam": lam, **s}
                if best is None or cand["score"] < best["score"]:
                    best = cand
            configs[h] = HorizonConfig(h=h, mode="single", lam=float(best["lam"]), cut_deg=None, clip=True)

        self.configs_ = configs
        return configs

    # ---------------------------
    # Fit + predict on test
    # ---------------------------
    def fit(self) -> "WindPowerAdaptivePipeline":
        df = self.load_and_prepare()
        train_in, val_in, test_df = self.split_time(df)
        configs = self.select_configs(train_in, val_in)

        # Refit on full train (train_in + val_in) and evaluate on test_df
        train_df = pd.concat([train_in, val_in], axis=0)

        results = {}
        traces = {}

        for h, cfg in configs.items():
            if cfg.mode == "single":
                Xtr, ytr = self.build_Xy(train_df, h)
                Xte, yte = self.build_Xy(test_df, h)
                pred, trace = self.rls_predict_update(Xtr, ytr, Xte, yte, lam=cfg.lam)
                pred = _clip_pred(pred, 0.0, self.p_max_) if cfg.clip else pred
                resid = yte - pred

                # persistence baseline (availability-correct)
                y_persist = test_df[f"p_lag{h}"].values.astype(float)

                results[h] = {
                    "h": h,
                    "mode": cfg.mode,
                    "lam": cfg.lam,
                    "cut_deg": None,
                    "MAE": _mae(yte, pred),
                    "RMSE": _rmse(yte, pred),
                    "LB_pvalue_lag{}".format(self.lb_lag): _lb_pvalue(resid, lag=self.lb_lag),
                    "RMSE_persistence": _rmse(yte, y_persist),
                    "Skill_vs_persistence": 1.0 - (_rmse(yte, pred) / _rmse(yte, y_persist)),
                }
                traces[h] = {
                    "t": test_df[self.time_col].values,
                    "y": yte,
                    "pred": pred,
                    "resid": resid,
                    "names": Xte.columns.tolist(),
                    "trace": trace,
                }

            elif cfg.mode == "tarso_dir":
                assert cfg.cut_deg is not None
                cut = float(cfg.cut_deg)

                trA = _half_circle_regime(train_df[f"Wd{h}"].values, cut)
                teA = _half_circle_regime(test_df[f"Wd{h}"].values, cut)

                train_A = train_df.loc[train_df.index[trA]].copy()
                train_B = train_df.loc[train_df.index[~trA]].copy()
                test_A = test_df.loc[test_df.index[teA]].copy()
                test_B = test_df.loc[test_df.index[~teA]].copy()

                XA_tr, yA_tr = self.build_Xy(train_A, h)
                XB_tr, yB_tr = self.build_Xy(train_B, h)
                XA_te, yA_te = self.build_Xy(test_A, h)
                XB_te, yB_te = self.build_Xy(test_B, h)

                predA, traceA = self.rls_predict_update(XA_tr, yA_tr, XA_te, yA_te, lam=cfg.lam)
                predB, traceB = self.rls_predict_update(XB_tr, yB_tr, XB_te, yB_te, lam=cfg.lam)

                pred_full = pd.Series(index=test_df.index, dtype=float)
                pred_full.loc[test_A.index] = predA
                pred_full.loc[test_B.index] = predB
                pred_full = pred_full.sort_index().values
                pred_full = _clip_pred(pred_full, 0.0, self.p_max_) if cfg.clip else pred_full

                yte = test_df[self.target_col].values.astype(float)
                resid = yte - pred_full
                y_persist = test_df[f"p_lag{h}"].values.astype(float)

                results[h] = {
                    "h": h,
                    "mode": cfg.mode,
                    "lam": cfg.lam,
                    "cut_deg": cut,
                    "MAE": _mae(yte, pred_full),
                    "RMSE": _rmse(yte, pred_full),
                    "LB_pvalue_lag{}".format(self.lb_lag): _lb_pvalue(resid, lag=self.lb_lag),
                    "RMSE_persistence": _rmse(yte, y_persist),
                    "Skill_vs_persistence": 1.0 - (_rmse(yte, pred_full) / _rmse(yte, y_persist)),
                }
                traces[h] = {
                    "t": test_df[self.time_col].values,
                    "y": yte,
                    "pred": pred_full,
                    "resid": resid,
                    "cut_deg": cut,
                    "names_A": XA_te.columns.tolist(),
                    "trace_A": traceA,
                    "names_B": XB_te.columns.tolist(),
                    "trace_B": traceB,
                }

            else:
                raise ValueError(f"Unknown mode: {cfg.mode}")

        self._trace_ = traces
        self.results_ = results
        self.fitted_ = True
        return self

    # ---------------------------
    # Public getters
    # ---------------------------
    def results_table(self) -> pd.DataFrame:
        assert self.fitted_, "Call fit() first"
        return pd.DataFrame(list(self.results_.values())).sort_values("h").reset_index(drop=True)

    # ---------------------------
    # Plotting utilities (no data exploration, just diagnostics + traces)
    # ---------------------------
    def plot_forecast(self, h: int, n: int = 500):
        assert self.fitted_
        pack = self._trace_[h]
        t = pack["t"]
        y = pack["y"]
        pred = pack["pred"]

        if n is not None and n < len(y):
            t = t[-n:]
            y = y[-n:]
            pred = pred[-n:]

        plt.figure(figsize=(12, 4))
        plt.plot(t, y, label="True")
        plt.plot(t, pred, label="Pred")
        plt.title(f"Forecast vs True (h={h}h) | mode={self.configs_[h].mode}")
        plt.xlabel("Time")
        plt.ylabel("Power")
        plt.legend()
        plt.show()

    def plot_residual_diagnostics(self, h: int, lags: int = 48):
        assert self.fitted_
        pack = self._trace_[h]
        resid = np.asarray(pack["resid"], dtype=float)

        plt.figure(figsize=(12, 3))
        plt.plot(resid)
        plt.title(f"Residuals (h={h}h)")
        plt.xlabel("Index")
        plt.ylabel("Residual")
        plt.show()

        plt.figure()
        plot_acf(resid, lags=lags)
        plt.title(f"Residual ACF (h={h}h)")
        plt.show()

        plt.figure()
        plot_pacf(resid, lags=min(lags, 30))
        plt.title(f"Residual PACF (h={h}h)")
        plt.show()

        lb = acorr_ljungbox(resid, lags=[12, 24, 48], return_df=True)
        print(lb)

    def plot_param_traces(self, h: int, top_k: int = 10):
        assert self.fitted_
        cfg = self.configs_[h]
        pack = self._trace_[h]

        def _plot(trace: np.ndarray, names: List[str], title: str):
            df_tr = pd.DataFrame(trace, columns=names)
            pick = df_tr.std().sort_values(ascending=False).head(top_k).index.tolist()
            plt.figure(figsize=(12, 5))
            for c in pick:
                plt.plot(df_tr[c].values, label=c)
            plt.title(title)
            plt.xlabel("Time index")
            plt.ylabel("Coefficient")
            plt.legend(fontsize=8, ncol=2)
            plt.show()

        if cfg.mode == "single":
            _plot(pack["trace"], pack["names"], f"Param traces (h={h}h) top {top_k} varying | single")
        else:
            cut = pack.get("cut_deg", cfg.cut_deg)
            _plot(pack["trace_A"], pack["names_A"], f"Param traces Regime A (cut={cut}°) (h={h}h) top {top_k}")
            _plot(pack["trace_B"], pack["names_B"], f"Param traces Regime B (cut={cut}°) (h={h}h) top {top_k}")


# ============================================================
# Utilities
# ============================================================
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
    Xb = [x]
    for k in knots:
        Xb.append(np.maximum(0.0, x - float(k)))
    return np.vstack(Xb).T

def half_circle_regime(angle_deg: np.ndarray, cut_deg: float) -> np.ndarray:
    a = np.asarray(angle_deg, dtype=float) % 360.0
    c = float(cut_deg) % 360.0
    d = (a - c) % 360.0
    return d < 180.0

def selection_score(rmse_val: float, lb_p: float, alpha: float = 0.15, eps: float = 1e-12) -> float:
    # Lower is better; penalize small p-values.
    return float(rmse_val + alpha * (-np.log(max(lb_p, eps))))


@dataclass
class KFConfig:
    h: int
    mode: str               # "single" or "tarso_dir" (only recommended for h=1)
    q: float                # process noise scale (Q = q * I)
    r: float                # measurement noise variance (R = r)
    cut_deg: Optional[float] = None
    clip: bool = True


class KalmanAdaptiveARX:
    """
    Dynamic linear regression with time-varying coefficients (random walk):
        y_t = x_t' beta_t + e_t      , e_t ~ N(0, R)
        beta_t = beta_{t-1} + w_t    , w_t ~ N(0, Q)
    We use:
        Q = q * I   (isotropic)
        R = r       (scalar)
    """

    def __init__(self, q: float, r: float, beta0: Optional[np.ndarray] = None, P0_scale: float = 1e4):
        self.q = float(q)
        self.r = float(r)
        self.beta0 = beta0
        self.P0_scale = float(P0_scale)

    def filter_predict_update(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Kalman filter and produce one-step-ahead predictions.
        Returns:
          yhat[t] = E[y_t | y_0..y_{t-1}]  (prior prediction)
          beta_filt[t] = beta_{t|t}
          P_filt[t] = diagonal of P_{t|t} (for quick inspection)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        Q = self.q * np.eye(p)
        R = self.r

        beta = np.zeros(p) if self.beta0 is None else np.asarray(self.beta0, dtype=float).copy()
        P = self.P0_scale * np.eye(p)

        yhat = np.zeros(n)
        beta_filt = np.zeros((n, p))
        P_diag = np.zeros((n, p))

        I = np.eye(p)

        for t in range(n):
            x = X[t].reshape(p, 1)

            # 1) Predict state
            beta_pred = beta
            P_pred = P + Q

            # 2) Predict observation
            yhat[t] = float(beta_pred.reshape(1, -1) @ x)
            S = float(x.T @ P_pred @ x) + R

            # 3) Update with observation
            K = (P_pred @ x) / S  # (p,1)
            innov = y[t] - yhat[t]

            beta = beta_pred + (K.flatten() * innov)
            P = (I - K @ x.T) @ P_pred

            beta_filt[t] = beta
            P_diag[t] = np.diag(P)

        return yhat, beta_filt, P_diag


class WindPowerKalmanPipeline:
    """
    Final Kalman-filtered model based on our chat findings:
      - Use stabilized feature set to avoid coefficient explosions (esp. h=3).
      - Use Kalman filter as the adaptive mechanism (time-varying coefficients).
      - Only allow direction TARSO (2 regimes) at h=1 by default; h=2/h=3 single regime.
      - Tune (q, r) per horizon by validation using RMSE + Ljung–Box penalty.
    """

    def __init__(
        self,
        data_path: str,
        time_col: str = "t",
        target_col: str = "p",
        toy_col: str = "toy",
        train_frac: float = 0.7,
        val_frac_within_train: float = 0.2,
        max_lag: int = 200,
        lb_lag: int = 48,
        alpha_lb_penalty: float = 0.15,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.time_col = time_col
        self.target_col = target_col
        self.toy_col = toy_col
        self.train_frac = float(train_frac)
        self.val_frac = float(val_frac_within_train)
        self.max_lag = int(max_lag)
        self.lb_lag = int(lb_lag)
        self.alpha_lb_penalty = float(alpha_lb_penalty)
        self.seed = int(seed)
        np.random.seed(self.seed)

        self.df_: Optional[pd.DataFrame] = None
        self.knots_: Optional[np.ndarray] = None
        self.p_max_: Optional[float] = None

        self.configs_: Dict[int, KFConfig] = {}
        self.results_: Dict[int, Dict[str, Any]] = {}
        self.trace_: Dict[int, Dict[str, Any]] = {}
        self.fitted_: bool = False

    # ---------------------------
    # Data preparation (same stabilized setup)
    # ---------------------------
    def load_and_prepare(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, parse_dates=[self.time_col])
        df = df.sort_values(self.time_col).reset_index(drop=True)

        df["hour"] = df[self.time_col].dt.hour.astype(int)
        YEAR = 365.25
        df["toy_rad"] = 2 * np.pi * (df[self.toy_col] % YEAR) / YEAR
        df["hour_rad"] = 2 * np.pi * df["hour"] / 24.0

        df["time_idx"] = np.arange(len(df))
        df["week_rad"] = 2 * np.pi * (df["time_idx"] % (24 * 7)) / (24 * 7)

        def add_fourier(df_in, rad_col, K, prefix):
            out = df_in.copy()
            for k in range(1, K + 1):
                out[f"{prefix}_sin{k}"] = np.sin(k * out[rad_col])
                out[f"{prefix}_cos{k}"] = np.cos(k * out[rad_col])
            return out

        # keep annual harmonic k=1 only (stability)
        df = add_fourier(df, "toy_rad", K=1, prefix="toy")
        # diurnal k=1..2
        df = add_fourier(df, "hour_rad", K=2, prefix="hr")
        # weekly k=1
        df = add_fourier(df, "week_rad", K=1, prefix="wk")

        for h in [1, 2, 3]:
            rad = np.deg2rad(df[f"Wd{h}"] % 360.0)
            df[f"wd{h}_sin"] = np.sin(rad)
            df[f"wd{h}_cos"] = np.cos(rad)
            df[f"T{h}_c"] = df[f"T{h}"] - df[f"T{h}"].mean()

        for k in range(1, self.max_lag + 1):
            df[f"p_lag{k}"] = df[self.target_col].shift(k)

        df = df.dropna().reset_index(drop=True)

        # fewer knots -> fewer collinearity issues (per our parameter trace findings)
        knots = df["Ws1"].quantile([0.2, 0.5, 0.8]).values
        knots = np.unique(np.round(knots.astype(float), 3))

        self.df_ = df
        self.knots_ = knots
        self.p_max_ = float(df[self.target_col].max())
        return df

    def split_time(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        cut = int(n * self.train_frac)
        train_df = df.iloc[:cut].copy()
        test_df = df.iloc[cut:].copy()

        ntr = len(train_df)
        cut2 = int(ntr * (1 - self.val_frac))
        train_in = train_df.iloc[:cut2].copy()
        val_in = train_df.iloc[cut2:].copy()
        return train_df, train_in, val_in, test_df

    # ---------------------------
    # Feature matrix (stabilized)
    # ---------------------------
    def build_Xy(self, df: pd.DataFrame, h: int) -> Tuple[pd.DataFrame, np.ndarray]:
        assert self.knots_ is not None
        out = df.copy()
        knots = self.knots_

        ws = out[f"Ws{h}"].astype(float).values
        ws_b = hinge_basis(ws, knots)

        cols: Dict[str, np.ndarray] = {}
        cols["const"] = np.ones(len(out))

        for L in [h, h + 1, h + 2]:
            if L <= self.max_lag:
                cols[f"p_lag{L}"] = out[f"p_lag{L}"].values

        for L in [h + 24, h + 48, h + 168]:
            if L <= self.max_lag:
                cols[f"p_lag{L}"] = out[f"p_lag{L}"].values

        cols[f"T{h}_c"] = out[f"T{h}_c"].values
        cols[f"wd{h}_sin"] = out[f"wd{h}_sin"].values
        cols[f"wd{h}_cos"] = out[f"wd{h}_cos"].values

        cols[f"ws{h}"] = ws_b[:, 0]
        for i, k in enumerate(knots, start=1):
            cols[f"ws{h}_hinge_{k:g}"] = ws_b[:, i]

        # only ws * sin/cos interaction (avoid hinge*dir interactions)
        cols[f"ws{h}_x_sin"] = cols[f"ws{h}"] * cols[f"wd{h}_sin"]
        cols[f"ws{h}_x_cos"] = cols[f"ws{h}"] * cols[f"wd{h}_cos"]

        # seasonality / diurnal / weekly
        cols["toy_sin1"] = out["toy_sin1"].values
        cols["toy_cos1"] = out["toy_cos1"].values
        for k in [1, 2]:
            cols[f"hr_sin{k}"] = out[f"hr_sin{k}"].values
            cols[f"hr_cos{k}"] = out[f"hr_cos{k}"].values
        cols["wk_sin1"] = out["wk_sin1"].values
        cols["wk_cos1"] = out["wk_cos1"].values

        X = pd.DataFrame(cols, index=out.index)
        y = out[self.target_col].values.astype(float)
        return X, y

    # ---------------------------
    # Fit/predict helpers
    # ---------------------------
    def _fit_predict_single(self, train_df: pd.DataFrame, test_df: pd.DataFrame, h: int, q: float, r: float) -> Dict[str, Any]:
        Xtr, ytr = self.build_Xy(train_df, h)
        Xte, yte = self.build_Xy(test_df, h)

        # Initialize beta0 from OLS on train for stability
        beta0 = np.linalg.lstsq(Xtr.values, ytr, rcond=None)[0]

        # Filter on train to get last state, then continue filter on test
        kf = KalmanAdaptiveARX(q=q, r=r, beta0=beta0, P0_scale=1e2)

        # Run filter on concatenated series but record test period outputs cleanly
        X_all = np.vstack([Xtr.values, Xte.values])
        y_all = np.concatenate([ytr, yte])

        yhat_all, beta_all, Pdiag_all = kf.filter_predict_update(X_all, y_all)

        # yhat_all is prior prediction for each t; keep test slice
        ntr = len(ytr)
        pred = yhat_all[ntr:]
        beta_trace = beta_all[ntr:]
        Pdiag = Pdiag_all[ntr:]

        if self.p_max_ is not None:
            pred = clip_pred(pred, 0.0, self.p_max_)

        resid = yte - pred
        return {
            "pred": pred,
            "y": yte,
            "t": test_df[self.time_col].values,
            "beta_trace": beta_trace,
            "Pdiag": Pdiag,
            "names": Xte.columns.tolist(),
            "resid": resid,
        }

    def _fit_predict_tarso_dir(self, train_df: pd.DataFrame, test_df: pd.DataFrame, h: int, q: float, r: float, cut_deg: float) -> Dict[str, Any]:
        # split by direction half-circle
        trA = half_circle_regime(train_df[f"Wd{h}"].values, cut_deg)
        teA = half_circle_regime(test_df[f"Wd{h}"].values, cut_deg)

        train_A = train_df.loc[train_df.index[trA]].copy()
        train_B = train_df.loc[train_df.index[~trA]].copy()
        test_A = test_df.loc[test_df.index[teA]].copy()
        test_B = test_df.loc[test_df.index[~teA]].copy()

        outA = self._fit_predict_single(train_A, test_A, h, q, r)
        outB = self._fit_predict_single(train_B, test_B, h, q, r)

        pred_full = pd.Series(index=test_df.index, dtype=float)
        pred_full.loc[test_A.index] = outA["pred"]
        pred_full.loc[test_B.index] = outB["pred"]
        pred_full = pred_full.sort_index().values

        y = test_df[self.target_col].values.astype(float)
        pred_full = clip_pred(pred_full, 0.0, self.p_max_) if self.p_max_ is not None else pred_full
        resid = y - pred_full

        return {
            "pred": pred_full,
            "y": y,
            "t": test_df[self.time_col].values,
            "resid": resid,
            "cut_deg": float(cut_deg),
            "A": outA,
            "B": outB,
        }

    # ---------------------------
    # Validation selection (q, r, and for h=1 possibly cut)
    # ---------------------------
    def select_configs(
        self,
        train_in: pd.DataFrame,
        val_in: pd.DataFrame,
        q_grid: List[float] = [1e-6, 1e-5, 1e-4, 1e-3],
        r_scale_grid: List[float] = [0.25, 0.5, 1.0, 2.0],
        cut_grid: List[float] = list(range(0, 360, 30)),
        min_regime_n: int = 800,
    ) -> Dict[int, KFConfig]:
        assert self.p_max_ is not None

        # baseline measurement variance estimate from train residuals (simple & robust)
        # Use y variance as proxy if needed.
        yvar = float(train_in[self.target_col].var())

        configs: Dict[int, KFConfig] = {}

        # ---- h=1: allow single and TARSO-dir
        h = 1
        best: Optional[Dict[str, Any]] = None

        # single candidates
        for q in q_grid:
            for rs in r_scale_grid:
                r = max(1e-8, rs * yvar)
                out = self._fit_predict_single(train_in, val_in, h, q, r)
                s = {
                    "rmse": rmse(out["y"], out["pred"]),
                    "lb_p": lb_pvalue(out["resid"], lag=self.lb_lag),
                }
                sc = selection_score(s["rmse"], s["lb_p"], alpha=self.alpha_lb_penalty)
                cand = {"mode": "single", "q": q, "r": r, "cut": None, "score": sc, **s}
                if best is None or cand["score"] < best["score"]:
                    best = cand

        # TARSO-dir candidates
        for q in q_grid:
            for rs in r_scale_grid:
                r = max(1e-8, rs * yvar)
                for cut in cut_grid:
                    trA = half_circle_regime(train_in["Wd1"].values, cut)
                    vaA = half_circle_regime(val_in["Wd1"].values, cut)

                    # balance checks
                    if trA.mean() < 0.1 or trA.mean() > 0.9:
                        continue
                    if vaA.mean() < 0.1 or vaA.mean() > 0.9:
                        continue

                    train_A = train_in.loc[train_in.index[trA]].copy()
                    train_B = train_in.loc[train_in.index[~trA]].copy()
                    val_A = val_in.loc[val_in.index[vaA]].copy()
                    val_B = val_in.loc[val_in.index[~vaA]].copy()
                    if min(len(train_A), len(train_B), len(val_A), len(val_B)) < min_regime_n:
                        continue

                    out = self._fit_predict_tarso_dir(train_in, val_in, h, q, r, cut)
                    s = {
                        "rmse": rmse(out["y"], out["pred"]),
                        "lb_p": lb_pvalue(out["resid"], lag=self.lb_lag),
                    }
                    sc = selection_score(s["rmse"], s["lb_p"], alpha=self.alpha_lb_penalty)
                    cand = {"mode": "tarso_dir", "q": q, "r": r, "cut": float(cut), "score": sc, **s}
                    if best is None or cand["score"] < best["score"]:
                        best = cand

        assert best is not None
        configs[1] = KFConfig(h=1, mode=best["mode"], q=float(best["q"]), r=float(best["r"]), cut_deg=best["cut"], clip=True)

        # ---- h=2, h=3: single only (stability findings from chat)
        for h in [2, 3]:
            best = None
            for q in q_grid:
                for rs in r_scale_grid:
                    r = max(1e-8, rs * yvar)
                    out = self._fit_predict_single(train_in, val_in, h, q, r)
                    s = {
                        "rmse": rmse(out["y"], out["pred"]),
                        "lb_p": lb_pvalue(out["resid"], lag=self.lb_lag),
                    }
                    sc = selection_score(s["rmse"], s["lb_p"], alpha=self.alpha_lb_penalty)
                    cand = {"q": q, "r": r, "score": sc, **s}
                    if best is None or cand["score"] < best["score"]:
                        best = cand
            configs[h] = KFConfig(h=h, mode="single", q=float(best["q"]), r=float(best["r"]), cut_deg=None, clip=True)

        self.configs_ = configs
        return configs

    # ---------------------------
    # Fit pipeline end-to-end
    # ---------------------------
    def fit(self) -> "WindPowerKalmanPipeline":
        df = self.load_and_prepare()
        train_df, train_in, val_in, test_df = self.split_time(df)

        # pick configs
        self.select_configs(train_in, val_in)

        # refit on full train_df and evaluate on test_df
        self.results_ = {}
        self.trace_ = {}

        for h, cfg in self.configs_.items():
            if cfg.mode == "single":
                out = self._fit_predict_single(train_df, test_df, h, cfg.q, cfg.r)
                pred = out["pred"]
                y = out["y"]
                resid = out["resid"]

            else:
                out = self._fit_predict_tarso_dir(train_df, test_df, h, cfg.q, cfg.r, cfg.cut_deg)
                pred = out["pred"]
                y = out["y"]
                resid = out["resid"]

            y_persist = test_df[f"p_lag{h}"].values.astype(float)
            self.results_[h] = {
                "h": h,
                "mode": cfg.mode,
                "q": cfg.q,
                "r": cfg.r,
                "cut_deg": cfg.cut_deg,
                "MAE": mae(y, pred),
                "RMSE": rmse(y, pred),
                f"LB_pvalue_lag{self.lb_lag}": lb_pvalue(resid, lag=self.lb_lag),
                "RMSE_persistence": rmse(y, y_persist),
                "Skill_vs_persistence": 1.0 - (rmse(y, pred) / rmse(y, y_persist)),
            }
            self.trace_[h] = out

        self.fitted_ = True
        return self

    def results_table(self) -> pd.DataFrame:
        assert self.fitted_
        return pd.DataFrame(list(self.results_.values())).sort_values("h").reset_index(drop=True)

    # ---------------------------
    # Plotting (diagnostics + coefficient traces)
    # ---------------------------
    def plot_forecast(self, h: int, n: int = 500):
        assert self.fitted_
        pack = self.trace_[h]
        t = pack["t"]
        y = pack["y"]
        pred = pack["pred"]

        if n is not None and n < len(y):
            t = t[-n:]
            y = y[-n:]
            pred = pred[-n:]

        plt.figure(figsize=(12, 4))
        plt.plot(t, y, label="True")
        plt.plot(t, pred, label="Pred")
        plt.title(f"Kalman DLM — Forecast vs True (h={h}h) | mode={self.configs_[h].mode}")
        plt.xlabel("Time")
        plt.ylabel("Power")
        plt.legend()
        plt.show()

    def plot_residual_diagnostics(self, h: int, lags: int = 48):
        assert self.fitted_
        resid = np.asarray(self.trace_[h]["resid"], dtype=float)

        plt.figure(figsize=(12, 3))
        plt.plot(resid)
        plt.title(f"Residuals (h={h}h)")
        plt.xlabel("Index")
        plt.ylabel("Residual")
        plt.show()

        plt.figure()
        plot_acf(resid, lags=lags)
        plt.title(f"Residual ACF (h={h}h)")
        plt.show()

        plt.figure()
        plot_pacf(resid, lags=min(lags, 30))
        plt.title(f"Residual PACF (h={h}h)")
        plt.show()

        lb = acorr_ljungbox(resid, lags=[12, 24, 48], return_df=True)
        display(lb)

    def plot_beta_traces(self, h: int, top_k: int = 10):
        """
        For single mode: plot top varying beta_t coefficients over test.
        For tarso_dir: plot within regime A/B separately.
        """
        assert self.fitted_
        cfg = self.configs_[h]
        pack = self.trace_[h]

        def _plot(trace: np.ndarray, names: List[str], title: str):
            df_tr = pd.DataFrame(trace, columns=names)
            pick = df_tr.std().sort_values(ascending=False).head(top_k).index.tolist()
            plt.figure(figsize=(12, 5))
            for c in pick:
                plt.plot(df_tr[c].values, label=c)
            plt.title(title)
            plt.xlabel("Time index")
            plt.ylabel("Coefficient")
            plt.legend(fontsize=8, ncol=2)
            plt.show()

        if cfg.mode == "single":
            _plot(pack["beta_trace"], pack["names"], f"Beta traces (h={h}h) top {top_k} varying | single")
        else:
            cut = cfg.cut_deg
            _plot(pack["A"]["beta_trace"], pack["A"]["names"], f"Beta traces Regime A (cut={cut}°) (h={h}h)")
            _plot(pack["B"]["beta_trace"], pack["B"]["names"], f"Beta traces Regime B (cut={cut}°) (h={h}h)")
