# =========================
# file1_data_preparation.py
# =========================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


@dataclass
class Split:
    train_df: pd.DataFrame
    train_in: pd.DataFrame
    val_in: pd.DataFrame
    test_df: pd.DataFrame


def hinge_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    cols = [x]
    for k in knots:
        cols.append(np.maximum(0.0, x - float(k)))
    return np.vstack(cols).T


class WindDataManager:
    """
    DATA + FEATURES (single authority)

    Model structure clarity:
      - Linear part: y_t ≈ X_t' β_t
      - Non-linear in wind: hinge(ws) and interactions ws * sin/cos(wd)
      - Seasonality: Fourier (diurnal + annual)
      - Noise is modeled separately in file2 (white or regime-dependent AR noise)

    Key NA principle:
      - We do NOT df.dropna() globally.
      - We drop NAs only in columns strictly required for the model + required lags.
        This avoids removing "previously existing NA" in unrelated columns.
    """

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

        # cache X/y per horizon and split-part
        self._Xy_cache: Dict[Tuple[int, str], Tuple[pd.DataFrame, pd.Series]] = {}

    @staticmethod
    def _add_fourier(df: pd.DataFrame, rad: pd.Series, K: int, prefix: str) -> None:
        for k in range(1, K + 1):
            df[f"{prefix}_sin{k}"] = np.sin(k * rad)
            df[f"{prefix}_cos{k}"] = np.cos(k * rad)

    def load_and_prepare(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, parse_dates=[self.time_col])
        df = df.sort_values(self.time_col).reset_index(drop=True)

        # --- time features (minimal + stable)
        df["hour"] = df[self.time_col].dt.hour.astype(int)
        year = 365.25
        toy_rad = 2 * np.pi * (df[self.toy_col] % year) / year
        hr_rad = 2 * np.pi * df["hour"] / 24.0
        self._add_fourier(df, toy_rad, K=1, prefix="toy")   # seasonal (annual)
        self._add_fourier(df, hr_rad,  K=2, prefix="hr")    # diurnal (1-2 harmonics)
        df["month"] = df[self.time_col].dt.month.astype(int)

        # --- directional trig + centered temperature
        for h in (1, 2, 3):
            rad = np.deg2rad(df[f"Wd{h}"] % 360.0)
            df[f"wd{h}_sin"] = np.sin(rad)
            df[f"wd{h}_cos"] = np.cos(rad)
            df[f"T{h}_c"] = df[f"T{h}"] - df[f"T{h}"].mean()
                # --- wind-change features (forecast-to-forecast changes)
        # Speed changes between horizons: captures ramping / trend in the wind forecast
        df["dWs1"] = df["Ws1"].astype(float) - df["Ws2"].astype(float)   # 1h vs 2h
        df["dWs2"] = df["Ws2"].astype(float) - df["Ws3"].astype(float)   # 2h vs 3h
        df["dWs3"] = df["Ws3"].astype(float) - df["Ws2"].astype(float)   # 3h vs 2h (opposite sign)
        
        # high-signal interaction: ramp effect scales with wind level
        df["Ws1_x_dWs1"] = df["Ws1"].astype(float) * df["dWs1"].astype(float)
        df["Ws2_x_dWs2"] = df["Ws2"].astype(float) * df["dWs2"].astype(float)
        df["Ws3_x_dWs3"] = df["Ws3"].astype(float) * df["dWs3"].astype(float)

        # (optional) magnitude version if you want asymmetry removed
        df["Ws1_x_absdWs1"] = df["Ws1"].astype(float) * df["dWs1"].abs().astype(float)
        df["Ws2_x_absdWs2"] = df["Ws2"].astype(float) * df["dWs2"].abs().astype(float)
        df["Ws3_x_absdWs3"] = df["Ws3"].astype(float) * df["dWs3"].abs().astype(float)


        # Direction changes: use smallest signed angular difference in degrees (-180..180]
        def _angdiff_deg(a: pd.Series, b: pd.Series) -> pd.Series:
            a = a.astype(float)
            b = b.astype(float)
            d = (a - b + 180.0) % 360.0 - 180.0
            return d

        df["dWd1"] = _angdiff_deg(df["Wd1"], df["Wd2"])
        df["dWd2"] = _angdiff_deg(df["Wd2"], df["Wd3"])
        df["dWd3"] = _angdiff_deg(df["Wd3"], df["Wd2"])

        # Optional but often helpful: magnitude-only direction change
        df["abs_dWd1"] = df["dWd1"].abs()
        df["abs_dWd2"] = df["dWd2"].abs()
        df["abs_dWd3"] = df["dWd3"].abs()

        # --- lags (vectorized concat -> avoids fragmentation)
        lag_cols = [f"p_lag{k}" for k in range(1, self.max_lag + 1)]
        missing = [c for c in lag_cols if c not in df.columns]
        if missing:
            lag_df = pd.concat(
                {f"p_lag{k}": df[self.target_col].shift(k) for k in range(1, self.max_lag + 1)},
                axis=1,
            )
            df = pd.concat([df, lag_df], axis=1)

        # --- NA handling: drop only what is strictly required
        essential = (
            [self.target_col]
            + [f"Ws{h}" for h in (1, 2, 3)]
            + [f"Wd{h}" for h in (1, 2, 3)]
            + [f"T{h}"  for h in (1, 2, 3)]
            + ["toy_sin1", "toy_cos1", "hr_sin1", "hr_cos1", "hr_sin2", "hr_cos2"]
        )
        df = df.dropna(subset=essential)
        df = df.dropna(subset=lag_cols).reset_index(drop=True)

        # --- knots from Ws1 quantiles (few hinges)
        knots = df["Ws1"].quantile([0.2, 0.5, 0.8]).to_numpy(dtype=float)
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

    def build_Xy(self, df: pd.DataFrame, h: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Feature builder (single source of truth).
        Nonlinearity: hinge(ws) + ws*dir interactions
        Linearity: y ≈ X'β (β can be adaptive via KF/RLS)
        """
        assert self.knots is not None
        out = df.copy()

        ws = out[f"Ws{h}"].astype(float).to_numpy()
        ws_b = hinge_basis(ws, self.knots)

        cols: Dict[str, np.ndarray] = {"const": np.ones(len(out))}

        # short + daily memory lags (keep small + interpretable)
        for L in (h, h + 1, h + 2, h + 24, h + 48, h + 168):
            if 1 <= L <= self.max_lag:
                cols[f"p_lag{L}"] = out[f"p_lag{L}"].to_numpy()

        # weather (linear in these features, non-linear captured in ws basis)
        cols[f"T{h}_c"] = out[f"T{h}_c"].to_numpy()
        cols[f"wd{h}_sin"] = out[f"wd{h}_sin"].to_numpy()
        cols[f"wd{h}_cos"] = out[f"wd{h}_cos"].to_numpy()
        
        # wind-change features (explicit ramp / veer signals)
        cols[f"dws{h}"] = out[f"dWs{h}"].to_numpy(dtype=float)
        cols[f"dwd{h}"] = out[f"dWd{h}"].to_numpy(dtype=float)
        cols[f"abs_dwd{h}"] = out[f"abs_dWd{h}"].to_numpy(dtype=float)
        cols[f"ws_x_dws{h}"] = out[f"Ws{h}_x_dWs{h}"].to_numpy(dtype=float)
        cols[f"ws_x_absdws{h}"] = out[f"Ws{h}_x_absdWs{h}"].to_numpy(dtype=float)


        # nonlinear wind speed basis
        cols[f"ws{h}"] = ws_b[:, 0]
        for i, k in enumerate(self.knots, start=1):
            cols[f"ws{h}_hinge_{k:g}"] = ws_b[:, i]

        # stable interactions
        cols[f"ws{h}_x_sin"] = cols[f"ws{h}"] * cols[f"wd{h}_sin"]
        cols[f"ws{h}_x_cos"] = cols[f"ws{h}"] * cols[f"wd{h}_cos"]

        # seasonality (diurnal + annual)
        cols["toy_sin1"] = out["toy_sin1"].to_numpy()
        cols["toy_cos1"] = out["toy_cos1"].to_numpy()
        cols["hr_sin1"] = out["hr_sin1"].to_numpy()
        cols["hr_cos1"] = out["hr_cos1"].to_numpy()
        cols["hr_sin2"] = out["hr_sin2"].to_numpy()
        cols["hr_cos2"] = out["hr_cos2"].to_numpy()

        X = pd.DataFrame(cols, index=out.index)
        y = pd.Series(out[self.target_col].to_numpy(dtype=float), index=out.index, name="y")
        return X, y

    def get_cached_Xy(self, h: int, split: Split) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Build X/y ONCE for train_df and test_df.
        Then slice by index for train_in / val_in to ensure strict alignment.
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
