"""
Part 2.1: Linear Rainfall-Runoff Model - COMPLETE & VERIFIED
Implements the exact model from the exercise with proper Kalman filtering
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA
# =============================================================================
data = pd.read_csv("ex1_rainfallrunoff.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['t'] = (data['timestamp'] - data['timestamp'].iloc[0]).dt.total_seconds() / 3600

# Subsample for computational efficiency
subsample = 5
data = data.iloc[::subsample].reset_index(drop=True)
t, y_obs, rain = data['t'].values, data['stormwater'].values, data['rainfall'].values
n_obs = len(t)
print(f"Using {n_obs} observations (subsampled 1:{subsample})")

# =============================================================================
# MODEL CLASS
# =============================================================================
from ctsmr import LinearReservoirKF

def aic_bic(ll, k, n):
    """Compute AIC and BIC."""
    return -2*ll + 2*k, -2*ll + k*np.log(n)


# =============================================================================
# 2.1.1: TWO-STATE MODEL (n=1)
# =============================================================================
print("\n" + "="*70)
print("2.1.1: TWO-STATE MODEL (n=1)")
print("="*70)
print("Model equations:")
print("  dX1 = (A*U - X1/K) dt + sigma*dW1")
print("  dX2 = (X1/K) dt + sigma*dW2")
print("  Y = X2 + noise")

m2 = LinearReservoirKF(t, y_obs, rain, 2)
r2 = m2.fit()
ll2 = -r2.fun

print(f"\nResults:")
print(f"  Log-likelihood: {ll2:.4f}")
print(f"  K = {r2.x[0]:.4f} hours")
print(f"  A = {r2.x[1]:.4f}")
print(f"  sigma = {r2.x[2]:.6f}")
print(f"  S = {r2.x[3]:.6f}")
print(f"  Converged: {r2.success}")

# =============================================================================
# 2.1.2: MODEL SELECTION
# =============================================================================
print("\n" + "="*70)
print("2.1.2: MODEL SELECTION")
print("="*70)

results = []
for m in range(2, 7):
    print(f"  Estimating {m}-state model (n={m-1})...", end=" ", flush=True)
    model = LinearReservoirKF(t, y_obs, rain, m)
    res = model.fit()
    ll = -res.fun
    k = 4 + m  # K, A, sigma, S + m initial states
    a, b = aic_bic(ll, k, n_obs)
    results.append({
        'm': m, 'n': m-1, 'll': ll, 'AIC': a, 'BIC': b,
        'K': res.x[0], 'A': res.x[1], 'sigma': res.x[2], 'S': res.x[3], 'res': res
    })
    print(f"LL={ll:.1f}, AIC={a:.1f}, BIC={b:.1f}")

df = pd.DataFrame(results)

print("\nModel Selection Table:")
print(df[['m', 'n', 'll', 'AIC', 'BIC', 'K', 'A']].to_string(index=False))

best_i_aic = df['AIC'].idxmin()
best_i_bic = df['BIC'].idxmin()
best_i = best_i_bic  # Use BIC as primary criterion

best_m = int(df.loc[best_i, 'm'])
print(f"\nBest model by AIC: {int(df.loc[best_i_aic, 'm'])} states")
print(f"Best model by BIC: {best_m} states (SELECTED)")

# =============================================================================
# 2.1.3: PARAMETER COMPARISON
# =============================================================================
print("\n" + "="*70)
print("2.1.3: PARAMETER COMPARISON - 2-state vs Best model")
print("="*70)

r2_row = df[df['m']==2].iloc[0]
rb = df.loc[best_i]

print(f"\n{'Parameter':<15} {'2-state (n=1)':<18} {f'{best_m}-state (n={best_m-1})':<18}")
print("-" * 51)
print(f"{'K (hours)':<15} {r2_row['K']:<18.4f} {rb['K']:<18.4f}")
print(f"{'A':<15} {r2_row['A']:<18.4f} {rb['A']:<18.4f}")
print(f"{'sigma':<15} {r2_row['sigma']:<18.6f} {rb['sigma']:<18.6f}")
print(f"{'S':<15} {r2_row['S']:<18.6f} {rb['S']:<18.6f}")
print(f"{'Log-likelihood':<15} {r2_row['ll']:<18.2f} {rb['ll']:<18.2f}")

# Drainage rates
rate_2 = 1 / r2_row['K']
rate_best = (best_m - 1) / rb['K']

print(f"\nDrainage rate (n/K):")
print(f"  2-state: 1/{r2_row['K']:.2f} = {rate_2:.4f} per hour")
print(f"  {best_m}-state: {best_m-1}/{rb['K']:.2f} = {rate_best:.4f} per hour")

print(f"\nInterpretation:")
print(f"  • K represents the time scale of the system")
print(f"  • Mean residence time per reservoir: K/n")
print(f"    - 2-state: {r2_row['K']/1:.2f}h per reservoir")
print(f"    - {best_m}-state: {rb['K']/(best_m-1):.2f}h per reservoir")
print(f"  • With more states, the system response becomes smoother and")
print(f"    more delayed (approaches gamma distribution)")
print(f"  • K typically increases with more states to maintain similar")
print(f"    overall system response time")

# =============================================================================
# 2.1.4: CORRELATION MATRIX
# =============================================================================
print("\n" + "="*70)
print("2.1.4: CORRELATION MATRIX")
print("="*70)

best_model = LinearReservoirKF(t, y_obs, rain, int(best_m))
best_res = df.loc[best_i, 'res']
params = best_res.x

# Numerical Hessian (main parameters only)
print("Computing Hessian matrix (this may take a moment)...")
eps = 1e-5
H = np.zeros((4, 4))
for i in range(4):
    for j in range(i, 4):
        p = params.copy()
        p[i] += eps; p[j] += eps; fpp = best_model.negloglik(p)
        p = params.copy()
        p[i] += eps; p[j] -= eps; fpm = best_model.negloglik(p)
        p = params.copy()
        p[i] -= eps; p[j] += eps; fmp = best_model.negloglik(p)
        p = params.copy()
        p[i] -= eps; p[j] -= eps; fmm = best_model.negloglik(p)
        H[i,j] = H[j,i] = (fpp - fpm - fmp + fmm) / (4*eps**2)

try:
    cov = np.linalg.inv(H)
    se = np.sqrt(np.abs(np.diag(cov)))
    corr = cov / np.outer(se, se)
    names = ['K', 'A', 'sigma', 'S']
    
    print("\nCorrelation Matrix:")
    print(f"{'':>10}" + "".join(f"{n:>10}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>10}" + "".join(f"{corr[i,j]:>10.3f}" for j in range(4)))
    
    print(f"\nStandard Errors:")
    for n, s, v in zip(names, se, params[:4]):
        print(f"  {n}: {s:.6f} (estimate: {v:.6f}, CV: {s/v*100:.1f}%)")
    
    print(f"\nInterpretation:")
    print(f"  • Correlation matrix shows interdependence of parameters")
    print(f"  • |r| > 0.7 indicates strong correlation (identifiability issues)")
    print(f"  • |r| > 0.9 suggests parameters are nearly redundant")
    
    high_corr = []
    for i in range(4):
        for j in range(i+1, 4):
            if abs(corr[i,j]) > 0.5:
                high_corr.append((names[i], names[j], corr[i,j]))
    
    if high_corr:
        print(f"\n  High correlations detected:")
        for n1, n2, r in high_corr:
            strength = "VERY STRONG" if abs(r) > 0.9 else "STRONG" if abs(r) > 0.7 else "moderate"
            print(f"    {n1}-{n2}: {r:.3f} ({strength})")
    
    print(f"\n  • K and A often correlate: both affect magnitude and timing")
    print(f"  • High correlation → wide confidence intervals → parameter uncertainty")
    
except np.linalg.LinAlgError:
    print("ERROR: Could not invert Hessian (matrix singular)")
    corr = None

# =============================================================================
# 2.1.5: RESIDUAL ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("2.1.5: RESIDUAL ANALYSIS AND MODEL VALIDATION")
print("="*70)

_, x_filt, std_res = best_model.negloglik(best_res.x, return_full=True)

# Basic statistics
mean_res = np.mean(std_res)
std_res_val = np.std(std_res)
skew_res = stats.skew(std_res)
kurt_res = stats.kurtosis(std_res)

print(f"\nResidual Statistics:")
print(f"  Mean:     {mean_res:>8.4f}  (target: 0, good if |mean| < 0.1)")
print(f"  Std Dev:  {std_res_val:>8.4f}  (target: 1, good if 0.8-1.2)")
print(f"  Skewness: {skew_res:>8.4f}  (target: 0, good if |skew| < 0.5)")
print(f"  Kurtosis: {kurt_res:>8.4f}  (target: 0, good if |kurt| < 1)")

# Normality test
n_test = min(5000, len(std_res))
_, p_shapiro = stats.shapiro(std_res[:n_test])
print(f"  Shapiro-Wilk p-value: {p_shapiro:.4f} (>0.05 suggests normality)")

# Ljung-Box test for autocorrelation
acf = [np.corrcoef(std_res[:-k], std_res[k:])[0,1] for k in range(1, 21)]
Q = len(std_res) * sum(a**2 for a in acf)
p_lb = 1 - stats.chi2.cdf(Q, 20)
print(f"  Ljung-Box Q(20): {Q:.2f}, p-value: {p_lb:.4f} (>0.05 = no autocorr)")

# Validation assessment
print(f"\nValidation Checks:")
checks = [
    ("Mean ≈ 0", abs(mean_res) < 0.1),
    ("Std ≈ 1", 0.8 < std_res_val < 1.2),
    ("Normality", p_shapiro > 0.05),
    ("No autocorrelation", p_lb > 0.05)
]

passed = sum(c[1] for c in checks)
for check, result in checks:
    print(f"  {'✓' if result else '✗'} {check}")

print(f"\nOverall: {passed}/{len(checks)} checks passed")
if passed >= 3:
    print("  → Model appears ADEQUATE")
else:
    print("  → Model may need IMPROVEMENT")

# =============================================================================
# PLOTS
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 1. Model Selection
axes[0,0].plot(df['m'], df['AIC'], 'bo-', lw=2, ms=8, label='AIC')
axes[0,0].plot(df['m'], df['BIC'], 'rs-', lw=2, ms=8, label='BIC')
axes[0,0].axvline(best_m, color='g', ls='--', lw=2, alpha=0.7)
axes[0,0].set_xlabel('Number of States'); axes[0,0].set_ylabel('IC')
axes[0,0].set_title('2.1.2: Model Selection'); axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3); axes[0,0].set_xticks(range(2, 7))

# 2. Correlation Matrix
if corr is not None:
    im = axes[0,1].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0,1].set_xticks(range(4)); axes[0,1].set_yticks(range(4))
    axes[0,1].set_xticklabels(names); axes[0,1].set_yticklabels(names)
    for i in range(4):
        for j in range(4):
            axes[0,1].text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=axes[0,1], fraction=0.046)
axes[0,1].set_title('2.1.4: Correlation Matrix')

# 3. Model Fit
axes[0,2].plot(t, y_obs, 'k.', ms=1, alpha=0.3, label='Observed')
axes[0,2].plot(t[1:], x_filt[:,-1], 'r-', lw=1, label='Filtered')
axes[0,2].set_xlabel('Time (hours)'); axes[0,2].set_ylabel('Stormwater')
axes[0,2].set_title('2.1.5: Model Fit'); axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

# 4. Residuals over time
axes[1,0].plot(t[1:], std_res, 'b-', lw=0.4, alpha=0.7)
axes[1,0].axhline(0, color='k', lw=1)
axes[1,0].axhline(2, color='r', ls='--', lw=1); axes[1,0].axhline(-2, color='r', ls='--', lw=1)
axes[1,0].set_xlabel('Time (hours)'); axes[1,0].set_ylabel('Std. Residuals')
axes[1,0].set_title('Residuals over Time'); axes[1,0].grid(True, alpha=0.3)

# 5. Histogram
axes[1,1].hist(std_res, bins=40, density=True, alpha=0.7, edgecolor='black')
x_norm = np.linspace(-4, 4, 100)
axes[1,1].plot(x_norm, stats.norm.pdf(x_norm), 'r-', lw=2, label='N(0,1)')
axes[1,1].set_xlabel('Std. Residuals'); axes[1,1].set_ylabel('Density')
axes[1,1].set_title('Residual Distribution'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

# 6. ACF
axes[1,2].bar(range(1, 21), acf, alpha=0.7, color='blue')
ci = 1.96/np.sqrt(len(std_res))
axes[1,2].axhline(ci, color='r', ls='--', lw=1, label='95% CI')
axes[1,2].axhline(-ci, color='r', ls='--', lw=1)
axes[1,2].axhline(0, color='k', lw=0.5)
axes[1,2].set_xlabel('Lag'); axes[1,2].set_ylabel('ACF')
axes[1,2].set_title('ACF of Residuals'); axes[1,2].legend(); axes[1,2].grid(True, alpha=0.3)

# --- Save each subplot separately ---
# 1. Model selection
fig_ms, ax = plt.subplots(figsize=(6, 4))
ax.plot(df['m'], df['AIC'], 'bo-', lw=2, ms=8, label='AIC')
ax.plot(df['m'], df['BIC'], 'rs-', lw=2, ms=8, label='BIC')
ax.axvline(best_m, color='g', ls='--', lw=2, alpha=0.7)
ax.set_xlabel('Number of States'); ax.set_ylabel('IC')
ax.set_title('2.1.2: Model Selection'); ax.legend()
ax.grid(True, alpha=0.3); ax.set_xticks(range(2, 7))
fig_ms.tight_layout(); fig_ms.savefig('part21_model_selection.png', dpi=150, bbox_inches='tight'); plt.close(fig_ms)

# 2. Correlation matrix
if corr is not None:
    fig_cm, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(names); ax.set_yticklabels(names)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=9)
    fig_cm.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title('2.1.4: Correlation Matrix')
    fig_cm.tight_layout(); fig_cm.savefig('part21_correlation_matrix.png', dpi=150, bbox_inches='tight'); plt.close(fig_cm)

# 3. Model fit
fig_mf, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, y_obs, 'k.', ms=1, alpha=0.3, label='Observed')
ax.plot(t[1:], x_filt[:, -1], 'r-', lw=1, label='Filtered')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Stormwater')
ax.set_title('2.1.5: Model Fit'); ax.legend(); ax.grid(True, alpha=0.3)
fig_mf.tight_layout(); fig_mf.savefig('part21_model_fit.png', dpi=150, bbox_inches='tight'); plt.close(fig_mf)

# 4. Residuals over time
fig_rt, ax = plt.subplots(figsize=(8, 4))
ax.plot(t[1:], std_res, 'b-', lw=0.4, alpha=0.7)
ax.axhline(0, color='k', lw=1)
ax.axhline(2, color='r', ls='--', lw=1); ax.axhline(-2, color='r', ls='--', lw=1)
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Std. Residuals')
ax.set_title('Residuals over Time'); ax.grid(True, alpha=0.3)
fig_rt.tight_layout(); fig_rt.savefig('part21_residuals_time.png', dpi=150, bbox_inches='tight'); plt.close(fig_rt)

# 5. Histogram of residuals
fig_hist, ax = plt.subplots(figsize=(6, 4))
ax.hist(std_res, bins=40, density=True, alpha=0.7, edgecolor='black')
x_norm = np.linspace(-4, 4, 100)
ax.plot(x_norm, stats.norm.pdf(x_norm), 'r-', lw=2, label='N(0,1)')
ax.set_xlabel('Std. Residuals'); ax.set_ylabel('Density')
ax.set_title('Residual Distribution'); ax.legend(); ax.grid(True, alpha=0.3)
fig_hist.tight_layout(); fig_hist.savefig('part21_residual_hist.png', dpi=150, bbox_inches='tight'); plt.close(fig_hist)

# 6. ACF of residuals
fig_acf, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(1, 21), acf, alpha=0.7, color='blue')
ci = 1.96 / np.sqrt(len(std_res))
ax.axhline(ci, color='r', ls='--', lw=1, label='95% CI')
ax.axhline(-ci, color='r', ls='--', lw=1)
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
ax.set_title('ACF of Residuals'); ax.legend(); ax.grid(True, alpha=0.3)
fig_acf.tight_layout(); fig_acf.savefig('part21_acf_residuals.png', dpi=150, bbox_inches='tight'); plt.close(fig_acf)

plt.tight_layout()
plt.savefig('part21_complete_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY - PART 2.1 COMPLETE")
print("="*70)
print(f"""
✓ 2.1.1 - Two-state model (n=1):
    Log-likelihood = {ll2:.4f}
    K = {r2.x[0]:.4f} hours
    A = {r2.x[1]:.4f}

✓ 2.1.2 - Model selection:
    Best model: {best_m} states (by BIC)
    Improvement: ΔLog-lik = {rb['ll'] - r2_row['ll']:.2f}

✓ 2.1.3 - Parameter comparison:
    K: {r2_row['K']:.2f}h → {rb['K']:.2f}h
    A: {r2_row['A']:.2f} → {rb['A']:.2f}

✓ 2.1.4 - Correlation matrix computed
    {'Strong K-A correlation detected' if corr is not None and abs(corr[0,1]) > 0.7 else 'Moderate parameter correlations'}

✓ 2.1.5 - Validation: {passed}/{len(checks)} checks passed
    Model is {'ADEQUATE' if passed >= 3 else 'needs improvement'}

Ready for Part 2.2: Non-linear overflow model
""")