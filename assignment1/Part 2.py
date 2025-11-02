import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

# ---------- Simulator ----------
def simulate_setar(T, c1=3.0, phi1=0.7, sigma1=1.0,
                   c2=-3.0, phi2=-0.3, sigma2=1.0,
                   threshold=0.0, delay=1, burn=50, y0=0.0, seed=7):
    if seed is not None:
        npr.seed(seed)
    N = T + burn
    y = np.zeros(N + 1); y[0] = y0
    s = np.zeros(N + 1, dtype=int)
    for t in range(1, N + 1):
        lag_idx = max(0, t - delay)
        if y[lag_idx] <= threshold:
            s[t] = 1
            y[t] = c1 + phi1 * y[t-1] + npr.normal(0.0, sigma1)
        else:
            s[t] = 2
            y[t] = c2 + phi2 * y[t-1] + npr.normal(0.0, sigma2)
    return y[burn+1:], s[burn+1:]

# ---------- Truth for overlay ----------
def true_M(x):
    x = np.asarray(x)
    return np.where(x <= 0, 3 + 0.7*x, -3 - 0.3*x)

# ---------- Epanechnikov kernel + local linear ----------
def epanechnikov(u):
    out = 0.75 * (1.0 - u*u)
    out[np.abs(u) > 1.0] = 0.0
    return out

def local_linear_epanechnikov(x, y, x_eval, h):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x_eval = np.asarray(x_eval, float)
    mhat = np.empty_like(x_eval, dtype=float)
    for j, x0 in enumerate(x_eval): 
        u = (x - x0) / h
        w = epanechnikov(u)
        ok = w > 0
        if not np.any(ok):
            mhat[j] = y[np.argmin(np.abs(x - x0))]
            continue
        X = np.column_stack([np.ones(np.sum(ok)), (x[ok] - x0)])
        W = w[ok]
        XT_W = X.T * W
        beta, *_ = np.linalg.lstsq(XT_W @ X, XT_W @ y[ok], rcond=None)
        mhat[j] = beta[0]
    return mhat

# ---------- Weights for CV (trim tails) ----------
def pi_weights(x, lower_q=0.05, upper_q=0.95, soft=False):
    lo, hi = np.quantile(x, [lower_q, upper_q])
    w = np.zeros_like(x, dtype=float)
    inside = (x >= lo) & (x <= hi)
    w[inside] = 1.0
    if soft:
        xmin, xmax = x.min(), x.max()
        left = (x < lo); right = (x > hi)
        w[left] = np.clip((x[left] - xmin) / (lo - xmin + 1e-12), 0.0, 1.0)
        w[right] = np.clip((xmax - x[right]) / (xmax - hi + 1e-12), 0.0, 1.0)
    return w

# ---------- K-fold CV over spans (fractions of range) ----------
def loess_kfold_cv(x, y, spans, K=10, lower_q=0.05, upper_q=0.95, shuffle=True, seed=0):
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed); rng.shuffle(idx)
    folds = np.array_split(idx, K)
    scores = {s: 0.0 for s in spans}; counts = {s: 0 for s in spans}
    xr = x.max() - x.min()
    for fold in folds:
        val_idx = fold
        train_idx = np.setdiff1d(idx, val_idx, assume_unique=False)
        x_tr, y_tr = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]
        w_val = pi_weights(x_val, lower_q=lower_q, upper_q=upper_q, soft=False)
        for s in spans:
            h = max(1e-8, s * xr)  # convert span -> absolute bandwidth
            y_hat_val = local_linear_epanechnikov(x_tr, y_tr, x_val, h)
            se = (y_val - y_hat_val)**2
            # divide by N to match MSE2(h) form
            scores[s] += np.sum(se * w_val) / n
            counts[s] += 1
    for s in spans:
        scores[s] /= max(counts[s], 1)
    best_span = min(scores, key=scores.get)
    return scores, best_span 

def run_cv_and_plot(ylag, yt, spans, true_M=None, K=10, lower_q=0.05, upper_q=0.95, seed=0, title_suffix=""):
    scores, best = loess_kfold_cv(ylag, yt, spans, K=K, lower_q=lower_q, upper_q=upper_q, seed=seed)
    # CV curve
    plt.figure(figsize=(6,4))
    xs = np.array(spans); ys = np.array([scores[s] for s in spans])
    plt.plot(xs*xr, ys, marker="o")
    plt.axvline(best*xr, linestyle="--", label=f"best h = {best*xr:.3f}")
    plt.xlabel("Bandwidth (h)"); plt.ylabel("CV MSE")
    plt.title("Cross-validation curve"); plt.legend(); plt.tight_layout()
    plt.savefig(f"Part 2 CV curve.png")
    # Fit with best span
    x = np.asarray(ylag); y = np.asarray(yt); xg = np.linspace(ylag.min(), ylag.max(), 200)
    h_best = max(1e-8, best * (x.max() - x.min()))
    y_hat = local_linear_epanechnikov(x, y, xg, h_best)

    plt.figure(figsize=(8,6))
    plt.scatter(x, y, s=6, alpha=0.25, label="data")
    if true_M is not None:
        plt.plot(xg, true_M(xg), "--", lw=2, label="True M(x)")
    plt.plot(xg, y_hat, lw=2, label=f"best h={best*xr:.3f}")
    plt.xlabel(r"$X_t$"); plt.ylabel(r"$X_{t+1}$")
    plt.title("Final best CV conditional mean estimate for SETAR(2,1,1)"); plt.legend(); plt.tight_layout()
    plt.savefig(f"Part 2 best.png")
    return scores, best

# ---------- Run ----------
y, _ = simulate_setar(T=1000, seed=7)
ylag = y[:-1]; yt = y[1:]

# Quick overlay of a few spans
plt.figure(figsize=(8,6))
plt.scatter(ylag, yt, s=5, alpha=0.3, label="data")
xg = np.linspace(ylag.min(), ylag.max(), 200)
plt.plot(xg, true_M(xg), "k--", lw=2, label="True M(x)")
xr = ylag.max() - ylag.min()
for f in [ 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
    h = f * xr
    plt.plot(xg, local_linear_epanechnikov(ylag, yt, xg, h), lw=2, label=f"h={np.round(f * xr,2)}")
plt.title("Conditional mean using weighted least squares with Epanechnikov kernel for different bandwidths\n SETAR(2,1,1)"); plt.xlabel("$X_t$"); plt.ylabel("$X_{t+1}$")
plt.legend(); plt.tight_layout()
plt.savefig("Part 2 test.png")
# CV choice
spans = [ 0.001,0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
scores, best_span = run_cv_and_plot(ylag, yt, spans, true_M=true_M, K=10, lower_q=0.05, upper_q=0.95, seed=7, title_suffix="(SETAR)")
print("CV scores:", {k: round(v, 5) for k, v in scores.items()})
print("Best span:", best_span)


# ---------- Run ----------
y, _ = simulate_setar(T=1000, seed=7)
ylag = y[:-1]
yt   = y[1:]
xr   = ylag.max() - ylag.min()
xg   = np.linspace(ylag.min(), ylag.max(), 200)

# Candidate spans
spans = [0.001, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
scores, best_span = loess_kfold_cv(
    ylag, yt, spans, K=10, lower_q=0.05, upper_q=0.95, seed=7
)
best_h = best_span * xr

# ----- Create one figure with 3 columns -----
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
ax1, ax2, ax3 = axs

# === (1) Multiple bandwidth fits ===
ax1.scatter(ylag, yt, s=5, alpha=0.3, label="Simulated data")
ax1.plot(xg, true_M(xg), "k--", lw=2, label="True M(x)")
for f in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
    h = f * xr
    ax1.plot(xg, local_linear_epanechnikov(ylag, yt, xg, h),
             lw=1.8, label=f"h={h:.2f}")
ax1.set_title("Conditional mean for different bandwidths")
ax1.set_xlabel(r"$X_t$")
ax1.set_ylabel(r"$X_{t+1}$")
ax1.legend(fontsize=8, frameon=False)

# === (2) Cross-validation curve ===
xs = np.array(spans) * xr
ys = np.array([scores[s] for s in spans])
ax2.plot(xs, ys, marker="o")
ax2.axvline(best_h, linestyle="--", color="r",
            label=f"best h = {best_h:.3f}")
ax2.set_xlabel("Bandwidth (h)")
ax2.set_ylabel("CV MSE")
ax2.set_title("Cross-validation curve")
ax2.legend(fontsize=8, frameon=False)

# === (3) Best CV fit ===
y_hat = local_linear_epanechnikov(ylag, yt, xg, best_h)
ax3.scatter(ylag, yt, s=6, alpha=0.25, label="Simulated data")
ax3.plot(xg, true_M(xg), "--", lw=2, label="True M(x)")
ax3.plot(xg, y_hat, lw=2, color="r", label=f"best h = {best_h:.3f}")
ax3.set_xlabel(r"$X_t$")
ax3.set_ylabel(r"$X_{t+1}$")
ax3.set_title("Best CV conditional mean estimate")
ax3.legend(fontsize=8, frameon=False)

plt.suptitle("Local weighted least squares with Epanechnikov smoothing — SETAR(2,1,1)",
             fontsize=13)
plt.tight_layout(pad = 2.0)
plt.savefig("Part 2 final.png")

