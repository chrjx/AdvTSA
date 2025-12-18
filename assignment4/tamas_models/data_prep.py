from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def hinge_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    cols = [x]
    for k in knots:
        cols.append(np.maximum(0.0, x - float(k)))
    return np.vstack(cols).T


@dataclass
class Split:
    train_df: pd.DataFrame
    train_in: pd.DataFrame
    val_in: pd.DataFrame
    test_df: pd.DataFrame


class WindDataManager:
    """
    Clean, leakage-safe data preparation for wind power forecasting.

    Philosophy:
      - Only construct features that are actually used
      - No global NA dropping
      - No future information leakage
      - Physically interpretable transformations
    """

    def __init__(
        self,
        data_path: str,
        time_col: str = "t",
        target_col: str = "p",
        toy_col: str = "toy",
        train_frac: float = 0.7,
        val_frac_within_train: float = 0.2,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.time_col = time_col
        self.target_col = target_col
        self.toy_col = toy_col
        self.train_frac = float(train_frac)
        self.val_frac = float(val_frac_within_train)
        self.seed = int(seed)

        self.df: Optional[pd.DataFrame] = None
        self.knots: Optional[np.ndarray] = None
        self.p_max: Optional[float] = None

        np.random.seed(self.seed)

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _add_fourier(df: pd.DataFrame, rad: pd.Series, K: int, prefix: str):
        for k in range(1, K + 1):
            df[f"{prefix}_sin{k}"] = np.sin(k * rad)
            df[f"{prefix}_cos{k}"] = np.cos(k * rad)

    @staticmethod
    def _angdiff_deg(a: pd.Series, b: pd.Series) -> pd.Series:
        a = a.astype(float)
        b = b.astype(float)
        return (a - b + 180.0) % 360.0 - 180.0

    # ------------------------------------------------------------------
    # Main preparation
    # ------------------------------------------------------------------
    def load_and_prepare(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, parse_dates=[self.time_col])
        df = df.sort_values(self.time_col).reset_index(drop=True)

        # -------------------------------
        # Time features
        # -------------------------------
        df["hour"] = df[self.time_col].dt.hour.astype(int)

        year = 365.25
        toy_rad = 2 * np.pi * (df[self.toy_col] % year) / year
        hr_rad = 2 * np.pi * df["hour"] / 24.0

        self._add_fourier(df, toy_rad, K=1, prefix="toy")   # annual
        self._add_fourier(df, hr_rad,  K=2, prefix="hr")    # diurnal

        # -------------------------------
        # Direction encoding + temperature
        # -------------------------------
        for h in (1, 2, 3):
            rad = np.deg2rad(df[f"Wd{h}"] % 360.0)
            df[f"wd{h}_sin"] = np.sin(rad)
            df[f"wd{h}_cos"] = np.cos(rad)

        # temperature centering (done later on train only)
        for h in (1, 2, 3):
            df[f"T{h}"] = df[f"T{h}"].astype(float)

        # -------------------------------
        # Wind ramp features (one per horizon)
        # -------------------------------
        for h in (1, 2):
            df[f"dWs{h}"] = df[f"Ws{h}"].astype(float) - df[f"Ws{h+1}"].astype(float)

        # absolute direction change only (robust)
        for h in (1, 2):
            dwd = self._angdiff_deg(df[f"Wd{h}"], df[f"Wd{h+1}"])
            df[f"abs_dWd{h}"] = dwd.abs()

        # -------------------------------
        # Power lags (only those used later)
        # -------------------------------
        used_lags = {1, 2, 3, 24, 48, 168}
        for L in used_lags:
            df[f"p_lag{L}"] = df[self.target_col].shift(L)

        # -------------------------------
        # Essential NA handling
        # -------------------------------
        essential = (
            [self.target_col]
            + [f"Ws{h}" for h in (1, 2, 3)]
            + [f"Wd{h}" for h in (1, 2, 3)]
            + [f"T{h}" for h in (1, 2, 3)]
            + [f"p_lag{L}" for L in used_lags]
            + ["toy_sin1", "toy_cos1", "hr_sin1", "hr_cos1", "hr_sin2", "hr_cos2"]
        )

        df = df.dropna(subset=essential).reset_index(drop=True)

        # -------------------------------
        # Train / test split (for leakage-safe stats)
        # -------------------------------
        n = len(df)
        cut = int(n * self.train_frac)
        train_df = df.iloc[:cut]

        # temperature centering (training mean only)
        for h in (1, 2, 3):
            mu = train_df[f"T{h}"].mean()
            df[f"T{h}_c"] = df[f"T{h}"] - mu

        # hinge knots from training data only
        knots = train_df["Ws1"].quantile([0.2, 0.5, 0.8]).to_numpy(dtype=float)
        self.knots = np.unique(np.round(knots, 3))

        self.df = df
        self.p_max = float(train_df[self.target_col].max())

        return df

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
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

        return Split(
            train_df=train_df,
            train_in=train_in,
            val_in=val_in,
            test_df=test_df,
        )


    def build_Xy(self, df: pd.DataFrame, h: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Feature construction consistent with cleaned data preparation.

        Model structure:
          y_t ≈ X_t' β_t

        Nonlinear in wind speed via hinge basis.
        Linear in all other regressors.
        """
        assert self.knots is not None
        assert self.df is not None

        out = df.copy()
        n = len(out)

        cols: Dict[str, np.ndarray] = {}
        cols["const"] = np.ones(n)

        # -------------------------------------------------
        # Power lags (horizon-consistent)
        # -------------------------------------------------
        for L in (h, h + 1, h + 2, 24, 48, 168):
            if f"p_lag{L}" in out.columns:
                cols[f"p_lag{L}"] = out[f"p_lag{L}"].to_numpy(dtype=float)

        # -------------------------------------------------
        # Temperature
        # -------------------------------------------------
        cols[f"T{h}_c"] = out[f"T{h}_c"].to_numpy(dtype=float)

        # -------------------------------------------------
        # Wind direction (sin / cos)
        # -------------------------------------------------
        cols[f"wd{h}_sin"] = out[f"wd{h}_sin"].to_numpy(dtype=float)
        cols[f"wd{h}_cos"] = out[f"wd{h}_cos"].to_numpy(dtype=float)

        # -------------------------------------------------
        # Wind ramp + direction change (if available)
        # -------------------------------------------------
        if h in (1, 2):
            cols[f"dWs{h}"] = out[f"dWs{h}"].to_numpy(dtype=float)
            cols[f"abs_dWd{h}"] = out[f"abs_dWd{h}"].to_numpy(dtype=float)

            # interaction: ramp magnitude scales with wind
            cols[f"Ws{h}_x_dWs{h}"] = (
                out[f"Ws{h}"].to_numpy(dtype=float)
                * out[f"dWs{h}"].to_numpy(dtype=float)
            )

        # -------------------------------------------------
        # Wind speed hinge basis
        # -------------------------------------------------
        ws = out[f"Ws{h}"].to_numpy(dtype=float)
        ws_b = hinge_basis(ws, self.knots)

        cols[f"Ws{h}"] = ws_b[:, 0]
        for i, k in enumerate(self.knots, start=1):
            cols[f"Ws{h}_hinge_{k:g}"] = ws_b[:, i]

        # -------------------------------------------------
        # Direction × wind speed interactions
        # -------------------------------------------------
        cols[f"Ws{h}_x_sin"] = cols[f"Ws{h}"] * cols[f"wd{h}_sin"]
        cols[f"Ws{h}_x_cos"] = cols[f"Ws{h}"] * cols[f"wd{h}_cos"]

        # -------------------------------------------------
        # Seasonality (Fourier)
        # -------------------------------------------------
        for name in (
            "toy_sin1", "toy_cos1",
            "hr_sin1", "hr_cos1",
            "hr_sin2", "hr_cos2",
        ):
            cols[name] = out[name].to_numpy(dtype=float)

        X = pd.DataFrame(cols, index=out.index)
        y = pd.Series(
            out[self.target_col].to_numpy(dtype=float),
            index=out.index,
            name="y",
        )

        return X, y


    def get_cached_Xy(
        self,
        h: int,
        split: Split,
    ) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Build X/y ONCE per horizon and dataset.
        Sub-slice using indices to avoid misalignment.
        """

        key_tr = (h, "train_df")
        key_te = (h, "test_df")

        if key_tr not in self._Xy_cache:
            Xtr, ytr = self.build_Xy(split.train_df, h)
            self._Xy_cache[key_tr] = (Xtr, ytr)

        if key_te not in self._Xy_cache:
            Xte, yte = self.build_Xy(split.test_df, h)
            self._Xy_cache[key_te] = (Xte, yte)

        X_train_df, y_train_df = self._Xy_cache[key_tr]
        X_test_df, y_test_df = self._Xy_cache[key_te]

        # inner split via index alignment
        idx_train_in = split.train_in.index
        idx_val_in = split.val_in.index

        X_train_in = X_train_df.loc[idx_train_in]
        y_train_in = y_train_df.loc[idx_train_in].to_numpy()

        X_val_in = X_train_df.loc[idx_val_in]
        y_val_in = y_train_df.loc[idx_val_in].to_numpy()

        return {
            "train_df": (X_train_df, y_train_df.to_numpy()),
            "train_in": (X_train_in, y_train_in),
            "val_in": (X_val_in, y_val_in),
            "test_df": (X_test_df, y_test_df.to_numpy()),
        }

