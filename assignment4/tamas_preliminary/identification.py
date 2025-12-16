import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

def regressogram_conditional_stats(x_lag, x_now, n_bins=20):
    """
    Estimate conditional mean and variance using regressograms
    Chapter 3.2 – equations (3.6) and (3.7)
    """
    df = pd.DataFrame({"x_lag": x_lag, "x_now": x_now}).dropna()
    
    df["bin"] = pd.cut(df["x_lag"], bins=n_bins)
    
    grouped = df.groupby("bin")
    
    x_center = grouped["x_lag"].mean()
    lambda_hat = grouped["x_now"].mean()
    gamma_hat = grouped["x_now"].var()
    
    return x_center.values, lambda_hat.values, gamma_hat.values

def cumulative_function(x, y):
    """
    Numerical integral using cumulative trapezoidal rule
    Chapter 3.2 – equations (3.8) and (3.9)
    """
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    
    cumulative = np.cumsum(
        np.diff(x_sorted, prepend=x_sorted[0]) * y_sorted
    )
    return x_sorted, cumulative
def lag_dependence_function(x, max_lag=12, frac=0.3):
    """
    Computes LDF(k) for k = 1,...,max_lag using LOWESS smoothing
    """
    ldf = np.zeros(max_lag + 1)
    ldf[0] = 1.0
    
    for k in range(1, max_lag + 1):
        xt = x[k:]
        xtk = x[:-k]
        
        smooth = lowess(xt, xtk, frac=frac, return_sorted=False)
        
        r2 = r2_score(xt, smooth)
        r2 = max(r2, 0.0)
        
        sign = np.sign(smooth.max() - smooth.min())
        ldf[k] = sign * np.sqrt(r2)
    
    return ldf

def partial_lag_dependence_function(x, max_lag=12):
    """
    Linear PLDF approximation using variance reduction (R²)
    Nonlinear backfitting version can be added later
    """
    pldf = np.zeros(max_lag + 1)
    pldf[0] = 1.0
    
    for k in range(1, max_lag + 1):
        y = x[k:]
        
        X_prev = np.column_stack([x[k-i-1:-i-1] for i in range(k-1)]) if k > 1 else None
        X_full = np.column_stack([x[k-i-1:-i-1] for i in range(k)])
        
        if X_prev is None:
            var_prev = np.var(y)
        else:
            reg_prev = LinearRegression().fit(X_prev, y)
            var_prev = np.var(y - reg_prev.predict(X_prev))
        
        reg_full = LinearRegression().fit(X_full, y)
        var_full = np.var(y - reg_full.predict(X_full))
        
        r2 = max((var_prev - var_full) / var_prev, 0.0)
        pldf[k] = np.sign(reg_full.coef_[-1]) * np.sqrt(r2)
    
    return pldf
