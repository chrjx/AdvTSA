import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from models import lb_pvalue

def residual_diagnostics(out: dict, title: str, lb_lag: int = 48, acf_lags: int = 72):
    """
    Uses standardized innovations if available (Kalman-correct residual object),
    otherwise falls back to plain residuals.
    """
    resid = np.asarray(out["resid"], dtype=float)
    rb = out.get("rb", None)

    if rb is not None and getattr(rb, "std_innov", None) is not None:
        lb_series = np.asarray(rb.std_innov, dtype=float)
        lb_label = "std_innov"
    else:
        lb_series = resid
        lb_label = "resid"

    p = lb_pvalue(lb_series, lag=lb_lag)
    print(f"{title} | LB({lb_lag}) on {lb_label}: p={p:.4g}")
    # Limit plotting for diagnostics
    if len(resid) > 500:
        resid = resid[:500]
        lb_series = lb_series[:500]
    # Time series residual plot
    plt.figure(figsize=(12, 3))
    plt.plot(resid)
    plt.title(f"{title} | Residuals (y - pred)")
    plt.grid(True)
    plt.show()

    # ACF/PACF for residuals (usual)
    plt.figure(figsize=(8, 4))
    plot_acf(resid, lags=acf_lags)
    plt.title(f"{title} | Residual ACF")
    plt.show()

    plt.figure(figsize=(8, 4))
    plot_pacf(resid, lags=min(30, acf_lags))
    plt.title(f"{title} | Residual PACF")
    plt.show()

    # ACF for standardized innovations if present (KF-correct)
    if lb_label == "std_innov":
        plt.figure(figsize=(8, 4))
        plot_acf(lb_series, lags=acf_lags)
        plt.title(f"{title} | Std. innovation ACF")
        plt.show()


def plot_param_traces(beta_trace: np.ndarray, names: list[str], top_k: int = 10, title: str = ""):
    beta_trace = np.asarray(beta_trace, dtype=float)
    dfb = pd.DataFrame(beta_trace, columns=names)
    pick = dfb.std().sort_values(ascending=False).head(top_k).index

    plt.figure(figsize=(12, 5))
    for c in pick:
        plt.plot(dfb[c].values, label=c)
    plt.title(title)
    plt.xlabel("Time index")
    plt.ylabel("Coefficient value")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True)
    plt.show()

# --------------------------------------------------------------------------------------
# Power-curve diagnostics
# --------------------------------------------------------------------------------------

def _binned_mean_curve(x: np.ndarray, y: np.ndarray, n_bins: int = 40):
    """
    Return bin-centers and mean(y) within bins of x.
    Uses quantile bins (more stable with uneven x density).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 10:
        return np.array([]), np.array([])

    # Quantile bin edges (unique to avoid empty/degenerate bins)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if edges.size < 3:
        return np.array([]), np.array([])

    # Digitize to bins
    b = np.digitize(x, edges[1:-1], right=False)  # 0..(nbins-1)
    nb = edges.size - 1

    xb, yb = [], []
    for i in range(nb):
        mask = b == i
        if mask.sum() < 5:
            continue
        xb.append(np.nanmean(x[mask]))
        yb.append(np.nanmean(y[mask]))

    return np.asarray(xb), np.asarray(yb)


def plot_power_curve_diagnostics(
    df: pd.DataFrame,
    h: int = 1,
    p_col: str = "p",
    ws_prefix: str = "Ws",
    wd_prefix: str = "Wd",
    n_ws_bins: int = 40,
    n_dir_bins: int = 8,
    max_scatter_points: int = 20000,
    seed: int = 42,
):
    """
    Power-curve diagnostics for horizon h using SAME-ROW weather forecasts Ws{h}, Wd{h}.

    Produces:
      1) Overall p vs Ws curve with binned mean curve overlay
      2) Same but split by wind-direction sector (n_dir_bins), plotted in a grid

    Assumes df contains columns like:
      - p (power)
      - Ws1..Ws3 (wind speed forecasts aligned to target time)
      - Wd1..Wd3 (wind direction forecasts aligned to target time, degrees 0..360)
    """
    ws_col = f"{ws_prefix}{h}"
    wd_col = f"{wd_prefix}{h}"

    if ws_col not in df.columns:
        raise KeyError(f"Missing column '{ws_col}' in df.")
    if wd_col not in df.columns:
        raise KeyError(f"Missing column '{wd_col}' in df.")
    if p_col not in df.columns:
        raise KeyError(f"Missing column '{p_col}' in df.")

    ws = pd.to_numeric(df[ws_col], errors="coerce").to_numpy()
    wd = pd.to_numeric(df[wd_col], errors="coerce").to_numpy()
    p = pd.to_numeric(df[p_col], errors="coerce").to_numpy()

    m = np.isfinite(ws) & np.isfinite(wd) & np.isfinite(p)
    ws, wd, p = ws[m], wd[m], p[m]
    if ws.size < 50:
        print("Not enough finite data for power-curve diagnostics.")
        return

    # Subsample for scatter readability
    rng = np.random.default_rng(seed)
    if ws.size > max_scatter_points:
        idx = rng.choice(ws.size, size=max_scatter_points, replace=False)
        ws_sc, p_sc = ws[idx], p[idx]
    else:
        ws_sc, p_sc = ws, p

    # ---- Plot 1: overall power curve ----
    xb, yb = _binned_mean_curve(ws, p, n_bins=n_ws_bins)

    plt.figure(figsize=(10, 5))
    plt.scatter(ws_sc, p_sc, s=6, alpha=0.15)
    if xb.size:
        plt.plot(xb, yb, linewidth=2)
    plt.title(f"Power curve diagnostic (h={h}) | {p_col} vs {ws_col}")
    plt.xlabel(ws_col)
    plt.ylabel(p_col)
    plt.grid(True)
    plt.show()

    # ---- Plot 2: power curve by wind-direction sector ----
    # Normalize wd into [0, 360)
    wd = np.mod(wd, 360.0)

    # Sector edges and labels
    edges = np.linspace(0, 360, n_dir_bins + 1)
    # Bin index 0..n_dir_bins-1
    b = np.digitize(wd, edges[1:-1], right=False)

    ncols = 4 if n_dir_bins >= 4 else n_dir_bins
    nrows = int(np.ceil(n_dir_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for k in range(n_dir_bins):
        ax = axes[k // ncols, k % ncols]
        mk = b == k
        if mk.sum() < 50:
            ax.set_title(f"{edges[k]:.0f}–{edges[k+1]:.0f}° (n={mk.sum()})")
            ax.grid(True)
            continue

        ws_k, p_k = ws[mk], p[mk]
        # scatter subsample per sector
        if ws_k.size > max_scatter_points // n_dir_bins:
            idxk = rng.choice(ws_k.size, size=max_scatter_points // n_dir_bins, replace=False)
            ws_k_sc, p_k_sc = ws_k[idxk], p_k[idxk]
        else:
            ws_k_sc, p_k_sc = ws_k, p_k

        xb_k, yb_k = _binned_mean_curve(ws_k, p_k, n_bins=max(12, n_ws_bins // 2))

        ax.scatter(ws_k_sc, p_k_sc, s=6, alpha=0.15)
        if xb_k.size:
            ax.plot(xb_k, yb_k, linewidth=2)
        ax.set_title(f"{edges[k]:.0f}–{edges[k+1]:.0f}° (n={mk.sum()})")
        ax.set_xlabel(ws_col)
        ax.set_ylabel(p_col)
        ax.grid(True)

    # Hide unused axes
    for j in range(n_dir_bins, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle(f"Power curve by wind direction sector (h={h}) | {p_col} vs {ws_col}", y=1.02, fontsize=12)
    fig.tight_layout()
    plt.show()
