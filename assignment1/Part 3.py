import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# ----- simulate SETAR(2;1;1) (your params from part 2) -----
def simulate_setar(T, c1=3.0, phi1=0.7, sigma1=1.0,
                   c2=-3.0, phi2=-0.3, sigma2=1.0,
                   threshold=0.0, delay=1, burn=200, y0=0.0, seed=7):
    if seed is not None:
        npr.seed(seed)
    N = T + burn
    y = np.zeros(N + 1); y[0] = y0
    s = np.zeros(N + 1, dtype=int)
    for t in range(1, N + 1):
        lag = max(0, t - delay)
        if y[lag] <= threshold:
            s[t] = 1
            y[t] = c1 + phi1*y[t-1] + npr.normal(0, sigma1)
        else:
            s[t] = 2
            y[t] = c2 + phi2*y[t-1] + npr.normal(0, sigma2)
    return y[burn+1:], s[burn+1:]

# true M(x) for overlay (edit if you change params)
def true_M(x, thr=0.0):
    x = np.asarray(x)
    return np.where(x <= thr, 3.0 + 0.7*x, -3.0 - 0.3*x)

# ----- CCM estimators -----
def ccm_running_mean(x, y):
    """Sort by x; return (x_sorted, running mean of y up to each x)."""
    order = np.argsort(x)
    xs = x[order]; ys = y[order]
    running = np.cumsum(ys) / (np.arange(len(ys)) + 1)
    return xs, running

def ccm_rolling_mean(x, y, window=101):
    """Sort by x; return (x_sorted, rolling mean of y over a centered window)."""
    order = np.argsort(x)
    xs = x[order]; ys = y[order]
    w = int(window) | 1  # force odd
    half = w // 2
    out = np.empty_like(ys, dtype=float)
    for i in range(len(ys)):
        lo = max(0, i - half); hi = min(len(ys), i + half + 1)
        out[i] = ys[lo:hi].mean()
    return xs, out

# ----- run example -----
if __name__ == "__main__":
    y, s = simulate_setar(T=1000, burn=300, seed=7)
    x = y[:-1]          # X_t
    y_next = y[1:]      # X_{t+1}

    # CCM curves
    xs_w1, m_w1   = ccm_rolling_mean(x, y_next, 50) 
    xs_w2, m_w2   = ccm_rolling_mean(x, y_next, 201)     # smoother (larger "bandwidth")
    xs_w3, m_w3   = ccm_rolling_mean(x, y_next, 500)
    xs_w4, m_w4   = ccm_rolling_mean(x, y_next, 10)

    # For reference, LOESS with span=0.2 (optional)
    x_grid = np.linspace(x.min(), x.max(), 400)
    
    # Plot
    plt.figure(figsize=(9,6))
    plt.scatter(x, y_next, s=6, alpha=0.2, label="Simulated data")
    plt.plot(x_grid, true_M(x_grid), "--", lw=2, label="True $M(x)$")
    plt.plot(xs_w4, m_w4, lw=2, label="Local CCM using interval size $h_n=10$")
    plt.plot(xs_w1, m_w1, lw=2, label="Local CCM using interval size $h_n=50$")
    plt.plot(xs_w2, m_w2, lw=2, label="Local CCM using interval size $h_n=201$")
    plt.plot(xs_w3, m_w3, lw=2, label="Local CCM using interval size $h_n=500$")
    plt.xlabel("$X_t$"); plt.ylabel("$X_{t+1}$")
    plt.title("Cumulative Conditional Means (CCM) vs true $M(x)$")
    plt.legend(); plt.tight_layout(); 
    plt.savefig("Part 3 CCM.png")

    # CCM curves
    xs_w4, m_w4   = ccm_rolling_mean(x, y_next, 50)

    # For reference, LOESS with span=0.2 (optional)
    x_grid = np.linspace(x.min(), x.max(), 1000)
    
    # Plot
    plt.figure(figsize=(9,6))
    plt.scatter(x, y_next, s=6, alpha=0.2, label="Simulated data")
    plt.plot(x_grid, true_M(x_grid), "--", lw=2, label="True $M(x)$")
    plt.plot(xs_w4, m_w4, lw=2, label="Local CCM using interval size $d_n=10$")
    plt.xlabel("$X_t$"); plt.ylabel("$X_{t+1}$")
    plt.title("Cumulative Conditional Means (CCM) vs true $M(x)$")
    plt.legend(); plt.tight_layout(); 
    #plt.savefig("Part 3 CCM.png")
    plt.show()

