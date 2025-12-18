import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
def plot_forecast_vs_actual(y, pred, title="", n_max=1500):
    """
    Plot y_t and \hat{y}_t on the same axes.
    """
    y = np.asarray(y)
    pred = np.asarray(pred)

    if len(y) > n_max:
        y = y[-n_max:]
        pred = pred[-n_max:]

    plt.figure(figsize=(10, 4))
    plt.plot(y, label="Actual", linewidth=1.5)
    plt.plot(pred, label="Forecast", linewidth=1.2)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_residual_diagnostics(resid, title_prefix="", lb_lag=48):
    resid = np.asarray(resid)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    axs[0].plot(resid, linewidth=0.8)
    axs[0].set_title(f"{title_prefix} residuals")

    axs[1].hist(resid, bins=40, density=True)
    axs[1].set_title("Histogram")

    plot_acf(resid, lags=lb_lag, ax=axs[2])
    axs[2].set_title("ACF")

    plt.tight_layout()
    plt.show()
def plot_innovation_diagnostics(rb, title_prefix="", lb_lag=48):
    """
    Uses standardized innovations if available.
    """
    if rb is None or rb.std_innov is None:
        return

    z = np.asarray(rb.std_innov)

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))

    axs[0].plot(z, linewidth=0.8)
    axs[0].set_title(f"{title_prefix} std. innovations")

    plot_acf(z, lags=lb_lag, ax=axs[1])
    axs[1].set_title("ACF (std. innov)")

    plt.tight_layout()
    plt.show()
def plot_beta_traces(beta_trace, names, top_k=5, title=""):
    """
    Plot top-k most variable beta coefficients.
    """
    if beta_trace is None:
        return

    beta_trace = np.asarray(beta_trace)
    if beta_trace.ndim != 2:
        return

    stds = beta_trace.std(axis=0)
    idx = np.argsort(stds)[-top_k:]

    plt.figure(figsize=(10, 4))
    for j in idx:
        plt.plot(beta_trace[:, j], label=names[j])

    plt.title(title or "Beta traces (most dynamic)")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()
def plot_model_result(
    res: dict,
    model_name: str,
    h: int,
    lb_lag: int = 48,
    trace_top_k: int = 5,
):
    y = res["y"]
    pred = res["pred"]
    rb = res.get("rb", None)

    plot_forecast_vs_actual(
        y,
        pred,
        title=f"{model_name} — Horizon h={h}",
    )

    plot_residual_diagnostics(
        resid=res["resid"],
        title_prefix=model_name,
        lb_lag=lb_lag,
    )

    plot_innovation_diagnostics(
        rb=rb,
        title_prefix=model_name,
        lb_lag=lb_lag,
    )

    plot_beta_traces(
        beta_trace=res.get("beta_trace"),
        names=res.get("names", []),
        top_k=trace_top_k,
        title=f"{model_name} — beta dynamics",
    )
