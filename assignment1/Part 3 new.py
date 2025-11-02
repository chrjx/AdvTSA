import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Simulate SETAR(2;1;1): regime depends on y_{t-1}
# -----------------------------
np.random.seed(0)

n = 1000
r = np.random.normal(size=n)

y = np.empty(n)
y[0] = r[0]
for t in range(1, n):
    if y[t-1] <= 0.0:                     # <-- SETAR threshold on y_{t-1}
        y[t] = 3.0 + 0.7*y[t-1] + r[t]
    else:
        y[t] = -3.0 - 0.3*y[t-1] + r[t]

# conditioning variable is y
x = y

# -----------------------------
# 2) Histogram regression setup
# -----------------------------
n_bin = 10
breaks = np.linspace(-2, 2, num=n_bin + 1)  # equal-width bins
h = np.diff(breaks)[0]                      # bin width

# We'll regress Y_t on bins of X_{t-1}
x_prev = x[:-1]
y_curr = x[1:]

# keep only points that fall inside [breaks[0], breaks[-1])
mask = (x_prev >= breaks[0]) & (x_prev < breaks[-1])
x_prev = x_prev[mask]
y_curr = y_curr[mask]

# bins labeled 0..n_bin-1
bin_idx = np.digitize(x_prev, breaks, right=False) - 1

# Guard: ensure all bins have >= 5 points (as in R)
counts = np.array([np.sum(bin_idx == i) for i in range(n_bin)])
if not np.all(counts >= 5):
    raise ValueError("Stopped: There are less than 5 points in one of the intervals")

# -----------------------------
# 3) Compute lambda, f.hat, gamma (matching R code)
# -----------------------------
lambda_hat = np.zeros(n_bin)  # mean of y in each bin  = regressogram M-hat
f_hat = np.zeros(n_bin)       # same scaling as your R: len / (n_bin*h)
gamma = np.zeros(n_bin)       # within-bin variance around lambda (divide by n_i)

for i in range(n_bin):
    y_bin = y_curr[bin_idx == i]
    lam = y_bin.mean()
    lambda_hat[i] = lam
    f_hat[i] = len(y_bin) / (n_bin * h)     # mirror R scaling
    gamma[i] = np.sum((y_bin - lam) ** 2) / len(y_bin)

# -----------------------------
# 4) Cumulative integral Lambda and H, confidence bands (as in R)
# -----------------------------
c_alpha = 1.273

Lambda = np.cumsum(lambda_hat * h)  # integral sum of lambda over bins

# h.hat[i] = gamma[i] / f.hat[i], H.hat = cumsum(h.hat * h)
h_hat = gamma / f_hat
H_hat = np.cumsum(h_hat * h)
H_hat_b = H_hat[-1]

band_factor = c_alpha * (n_bin ** -0.5) * (H_hat_b ** 0.5)
Lambda_lower = Lambda - band_factor * (1 + H_hat / H_hat_b)
Lambda_upper = Lambda + band_factor * (1 + H_hat / H_hat_b)

# x-positions to plot the cumulative integral at bin upper edges
x_plot = breaks[1:]  # upper edge of each bin

# -----------------------------
# 5) Theoretical M(x) and its integral for this SETAR
# -----------------------------
def M_true(x):
    x = np.asarray(x)
    return np.where(x <= 0.0, 3.0 + 0.7*x, -3.0 - 0.3*x)

def primitive_left(u):  
    return 3.0*u + 0.35*u**2

def primitive_right(u):  
    return -3.0*u - 0.15*u**2

def integrate_M_true(x, a):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    if a < 0.0:
        m_left = x <= 0.0
        out[m_left]  = primitive_left(x[m_left])  - primitive_left(a)
        out[~m_left] = (primitive_left(0.0) - primitive_left(a)) + \
                       (primitive_right(x[~m_left]) - primitive_right(0.0))
    else:
        out = primitive_right(x) - primitive_right(a)
    return out

a0 = breaks[0]                  # integrate from leftmost bin edge
Lambda_true = integrate_M_true(x_plot, a=a0)

# -----------------------------
# 6) Plot the integrated curve + bands + theoretical
# -----------------------------
plt.figure(figsize=(9, 6))
plt.scatter(x[:-1],  y[1:], s=6, alpha=0.2, label="Simulated data")
plt.plot(x_plot, Lambda, lw=2, label=r"Estimated $\Lambda(x)$")
plt.fill_between(x_plot, Lambda_lower, Lambda_upper, alpha=0.2, label="95% band")
plt.plot(x_plot, Lambda_true, "k--", lw=2, label=r"Theoretical $\Lambda(x)$")
plt.xlabel(r"$x$")
plt.ylabel(r"$\Lambda(x)$")
plt.title("Cumulative Conditional Means $\hat\Lambda$ vs true $\Lambda$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("Part 3 CCM.png")

plt.show()
