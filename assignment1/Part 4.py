import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

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

# ----------------------------
# 1) Load & prepare the data
# ----------------------------
# Change these to the actual column names in DataPart4.csv:
COL_PHI = "Ph"            # heat load Φ_t
COL_Ti  = "Ti"             # indoor temperature T^i_t
COL_Te  = "Te"             # outdoor temperature T^e_t
COL_W   = "W"              # wind speed W_t

df = pd.read_csv("DataPart4.csv")

# Compute ΔT and Z = Φ / ΔT
df["dT"] = df[COL_Ti] - df[COL_Te]
df["Z"]  = df[COL_PHI] / df["dT"]

# Basic cleaning: remove NA and extreme small |ΔT|
eps = 0.5  # threshold in degrees to avoid exploding noise; tweak as needed
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[COL_W, "dT", "Z"])
df = df.loc[df["dT"].abs() >= eps].copy()

# Optionally winsorize Z lightly to stabilize tails (comment out if not desired)
q1, q99 = df["Z"].quantile([0.01, 0.99])
df["Z"] = df["Z"].clip(q1, q99)

x = df[COL_W].to_numpy()
z = df["Z"].to_numpy()

# ----------------------------
# 2) Cross-validated LOESS span
# ----------------------------
def pi_weights(x, lower_q=0.05, upper_q=0.95):
    lo, hi = np.quantile(x, [lower_q, upper_q])
    return ((x >= lo) & (x <= hi)).astype(float)

def loess_kfold_cv(x, y, spans, K=10, seed=0, lower_q=0.05, upper_q=0.95):
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)
    scores = {s: 0.0 for s in spans}
    xr = x.max() - x.min()
    for fold in folds:
        val = fold
        tr  = np.setdiff1d(idx, val)
        xv, yv = x[val], y[val]
        wv = pi_weights(xv, lower_q, upper_q)
        for s in spans:
            h = max(1e-8, s * xr)
            #yhat = lowess(y[tr], x[tr], frac=s, xvals=xv, return_sorted=False)
            yhat = local_linear_epanechnikov(y[tr], x[tr], xv, h)
            se = (yv - yhat) ** 2
            wt = wv.sum() if wv.sum() > 0 else len(se)
            scores[s] += (se * wv).sum() / wt
    for s in spans:
        scores[s] /= K
    best = min(scores, key=scores.get)
    return scores, best

spans = [0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50]
scores, best_span = loess_kfold_cv(x, z, spans, K=10, seed=7)

print("CV scores:", {k: round(v, 5) for k, v in scores.items()})
print("Best LOESS span =", best_span)

xr = z.max() - z.min()
x_grid = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 300)
u_hat_grid = local_linear_epanechnikov(x, z, x_grid, best_span*xr)

plt.figure(figsize=(9,6))
plt.scatter(x, z, s=10, alpha=0.25, label="Data")
plt.plot(x_grid, u_hat_grid, lw=2, label=f"Optimal fit of $\\hat U_a(w)$ (h={best_span*xr:.2f})")
plt.xlabel("$W_t$")
plt.ylabel(r"$U_a(W_t)$")
plt.title("Optimal fit for $U_a(W_t)$ using CV")
plt.legend()
plt.tight_layout()
plt.savefig("Part 4 best span.png")

plt.figure(figsize=(8,6))
plt.scatter(x, z, s=5, alpha=0.3, lw=2,label="Data")
plt.title("The heat loss coefficient $U_a$ as a function of wind speed $W_t$")
plt.xlabel(r"$W_t$")
plt.ylabel(r"$U_a$")
plt.legend(fontsize=8, frameon=False)
plt.savefig("Part 4 plane.png")

plt.figure(figsize=(8,6))
plt.scatter(x, z, s=5, alpha=0.3,lw=2, label="Data")
for f in [0.01, 0.02, 0.05, 0.1, 0.15]:
    h = f * xr
    plt.plot(x_grid, local_linear_epanechnikov(x, z, x_grid, h),lw=2,
              label=f"h={h:.2f}")
plt.title("The heat loss function $U_a$ as a function of wind speed $W_t$.\nFitted using weighted least squares with Epanechnikov kernel for different bandwidths.")
plt.xlabel(r"$W_t$")
plt.ylabel(r"$U_a$")
plt.legend(fontsize=8, frameon=False)
#plt.savefig("Part 4 bandwidths.png")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# --- Left: raw data ---
# axes[0].scatter(x, z, s=5, alpha=0.3, lw=2, label="Data")
# axes[0].set_title(r"The heat loss coefficient $U_a$ as a function of wind speed $W_t$")
# axes[0].set_xlabel(r"$W_t$")
# axes[0].set_ylabel(r"$U_a$")
# axes[0].legend(fontsize=8, frameon=False)
axes[1].scatter(x, z, s=10, alpha=0.25, label="Data")
axes[1].plot(x_grid, u_hat_grid, lw=2, label=f"Optimal fit of $\\hat U_a(w)$ (h={best_span*xr:.2f})")
axes[1].set_xlabel("$W_t$")
axes[1].set_ylabel(r"$U_a(W_t)$")
axes[1].set_title("Optimal fit for $U_a(W_t)$ using CV")
axes[1].set_xlabel(r"$W_t$")
axes[1].set_ylabel(r"$U_a$")
axes[1].legend(fontsize=8, frameon=False)


# --- Right: fits with different bandwidths ---
axes[0].scatter(x, z, s=5, alpha=0.3, lw=2, label="Data")
for f in [0.01, 0.02, 0.05, 0.1, 0.15]:
    h = f * xr
    axes[0].plot(x_grid,
                 local_linear_epanechnikov(x, z, x_grid, h),
                 lw=2,
                 label=fr"$h={h:.2f}$")

axes[0].set_title(r"Local WLS with Epanechnikov kernel for various bandwidths")
axes[0].set_xlabel(r"$W_t$")
axes[0].set_ylabel(r"$U_a$")
axes[0].legend(fontsize=8, frameon=False)

# --- Save and show ---
plt.savefig("Part 4 combined.png", dpi=300)
plt.show()


