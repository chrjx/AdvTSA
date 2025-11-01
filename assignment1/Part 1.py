import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import inspect
import matplotlib.colors as mcolors
 
def simulate_setar(
    T,
    c1=4.0,  phi1=0.5,  sigma1=1.0,   # regime 1 params (y_{t-d} <= threshold)
    c2=-4.0, phi2=-0.5, sigma2=1.0,   # regime 2 params (y_{t-d} >  threshold)
    threshold=0.0,
    delay=1,
    burn=200,
    y0=0.0,
    seed=None
):
    if seed is not None:
        npr.seed(seed)

    N = T + burn
    y = np.zeros(N + 1)
    y[0] = y0
    s = np.zeros(N + 1, dtype=int)

    for t in range(1, N + 1):
        # choose regime using self-exciting threshold on y_{t-delay}
        lag_idx = max(0, t - delay)
        if y[lag_idx] <= threshold:
            s[t] = 1
            eps = npr.normal(0.0, sigma1)
            y[t] = c1 + phi1 * y[t-1] + eps
        else:
            s[t] = 2
            eps = npr.normal(0.0, sigma2)
            y[t] = c2 + phi2 * y[t-1] + eps

    return y[burn+1:], s[burn+1:]


def simulate_igar(
    T,
    c=(0.0, 0.8),           # intercepts per regime (r=1,2)
    phi=(0.85, -0.2),       # AR(1) coeffs per regime
    sigma=(1.0, 1.3),       # innovation std per regime
    p=(0.6, 0.4),           # P(S_t = r), independent over t
    burn=200,
    y0=0.0,
    seed=None
):
 
    if seed is not None:
        npr.seed(seed)

    p = np.asarray(p, dtype=float)
    assert np.isclose(p.sum(), 1.0), "p must sum to 1"

    K = 2
    c = np.asarray(c, dtype=float)
    phi = np.asarray(phi, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    N = T + burn
    y = np.zeros(N + 1)
    y[0] = y0
    s = np.zeros(N + 1, dtype=int)

    for t in range(1, N + 1):
        r = 1 + npr.choice(K, p=p)  # regime label 1..K
        s[t] = r
        k = r - 1
        eps = npr.normal(0.0, sigma[k])
        y[t] = c[k] + phi[k] * y[t-1] + eps

    return y[burn+1:], s[burn+1:]

def simulate_mmar(
    T,
    c=(0.0, 0.0), phi=(0.9, 0.2), sigma=(1.0, 1.5),
    P=np.array([[0.95, 0.05],
                [0.08, 0.92]]),  # transition matrix
    s0=1, burn=200, x0=None
):

    P = np.asarray(P)
    assert P.shape == (2, 2)
    assert np.allclose(P.sum(axis=1), 1.0), "Rows of P must sum to 1"
    N = T + burn
    x = np.zeros(N + 1)
    s = np.zeros(N + 1, dtype=int)
    s[0] = s0
    if x0 is not None:
        x[0] = x0

    for t in range(1, N + 1):
        prev = s[t-1] - 1
        r = 1 + npr.choice(2, p=P[prev])
        s[t] = r
        k = r - 1
        eps = npr.normal(0.0, sigma[k])
        x[t] = c[k] + phi[k] * x[t-1] + eps

    return x[burn+1:], s[burn+1:]


def plot_phase(y, s, title="y_t vs y_{t-1} (colored by regime)"):
    ylag = y[:-1]
    yt   = y[1:]
    s1   = s[1:]  # align with yt
    plt.figure(figsize=(6,5))
    for r in sorted(np.unique(s1)):
        mask = (s1 == r)
        plt.step(ylag[mask], yt[mask], s=6, alpha=0.6, label=f"regime {r}")
    plt.xlabel(r"$X_{t-1}$"); plt.ylabel(r"$X_t$")
    plt.title(title); plt.legend()
    plt.tight_layout()

def acf(x, nlags=40):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    out = np.empty(nlags + 1)
    for k in range(nlags + 1):
        out[k] = np.dot(x[:n-k], x[k:]) / denom
    return out

# ---------- helpers that draw into given axes (no new figures) ----------
def phase_scatter(ax, y, s):
    """Scatter plot of y_t vs y_{t-1}, colored by regime."""
    ylag = y[:-1]
    yt   = y[1:]
    s1   = s[1:]
    for r in sorted(np.unique(s1)):
        mask = (s1 == r)
        ax.scatter(ylag[mask], yt[mask], s=6, alpha=0.6, label=f"regime {r}")
    ax.set_ylabel(r"$X_t$")
    ax.legend(loc="best", fontsize=8, frameon=False)

def acf_plot(ax, y, nlags=40):
    """ACF bar-like plot."""
    vals = acf(y, nlags=nlags)
    lags = np.arange(len(vals))
    for k in lags:
        ax.plot([k, k], [0, vals[k]])
    ax.plot(lags, vals, marker="o", linestyle="None")
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel("ACF")

def regime_trace(ax, s):
    """Step plot of regime indicator."""
    ax.step(np.arange(len(s)), s, where="post", lw=1.2)
    regs = sorted(np.unique(s))
    ax.set_yticks(regs)
    ax.set_xlabel("t")
    ax.set_ylabel("Regime")
    ax.set_title("Regime trace")

def colored_timeseries(ax, y, s, palette=None):
    """Plot y_t colored by regime."""
    t = np.arange(len(y))
    ax.plot(t, y)
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylabel(r"$X_t$")
    ax.set_xlabel("t")

def plot_grid_for_model(simulator, row_param_list, row_titles=None, sim_kwargs=None, seed_base=123):
    """
    simulator: e.g. simulate_setar / simulate_igar / simulate_mmar
    row_param_list: list of dicts with row-specific params
    row_titles: optional descriptive labels for each row
    sim_kwargs: shared kwargs passed to simulator
    """
    if sim_kwargs is None:
        sim_kwargs = {}
    if row_titles is None:
        row_titles = [", ".join(f"{k}={v}" for k,v in rp.items()) for rp in row_param_list]

    nrows = len(row_param_list)
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10), constrained_layout=True)

    # ensure axs is 2D array even if nrows=1
    axs = np.atleast_2d(axs)

    for i, rp in enumerate(row_param_list):
        rp_local = dict(rp)
        # Add seed only if simulator accepts it
        if seed_base is not None:
            sig = inspect.signature(simulator)
            if "seed" in sig.parameters:
                rp_local["seed"] = seed_base + i

        # filter only accepted params
        sig = inspect.signature(simulator)
        allowed = set(sig.parameters.keys())
        kwargs = {k: v for k, v in {**sim_kwargs, **rp_local}.items() if k in allowed}
        y, s = simulator(**kwargs)

        # --- Column 1: time series
        ax = axs[i, 0]
        ax.plot(y, lw=1)
        ax.set_title("Time series")
        ax.set_ylabel(r"$X_t$" + "\n" + row_titles[i])
        if i == 2:
            ax.set_xlabel("t")
        fig.supylabel(r"Different AR parameters for each regime: $(a_0^1,a_1^2),(a_0^2,a_1^2)$:", fontsize=12)

        # --- Column 2: phase plot
        ax = axs[i, 1]
        phase_scatter(ax, y, s)
        ax.set_title("Conditional")
        if i == 2:
            ax.set_xlabel(r"$X_{t-1}$")

        # --- Column 3: ACF
        ax = axs[i, 2]
        acf_plot(ax, y, nlags=40)
        ax.set_title("ACF")
        if i == 2:
            ax.set_xlabel("Lag")

        # --- Column 4: Regime trace
        ax = axs[i, 3]
        regime_trace(ax, s)

    return fig, axs


if __name__ == "__main__":

    setar_rows = [
        dict(c1=5.0,  phi1=0.5,  sigma1=1.0,  c2=3.0, phi2=0.5, sigma2=1.0, threshold=6),
        dict(c1=3.0,  phi1=0.5,  sigma1=1.0,  c2=-3.0, phi2=-0.5, sigma2=1.0, threshold=-2.5),
        dict(c1=3.0,  phi1=0.5,  sigma1=1.0,  c2=3.0,  phi2=-0.5,  sigma2=1.0, threshold=4.0),
    ]
    setar_titles = [
        r"(5,0.5), (3,0.5), threshold=6",
        r"(3.0,0.5), (-3.0,-0.5), threshold=-2.5",
        r"(3.0,0.5), (3.0,-0.5), threshold=4.0",
    ]
    _ = plot_grid_for_model(
        simulate_setar,
        row_param_list=setar_rows,
        row_titles=setar_titles,
        sim_kwargs=dict(T=200, threshold=4, delay=1, burn=100, y0=0.0)
    )
    plt.suptitle("SETAR(2,1,1) — rows vary $(a_0^i, a_1^i)$", fontsize=13)
    plt.tight_layout(pad=2)
    plt.savefig("SETAR_grid.png", dpi=300)

    # --- IGAR: vary (c, phi) per regime (independent regime draws), keep p,sigma fixed (or tweak as desired)
    igar_rows = [
        dict(c=(0.2, 0.5),  phi=(0.7, 0.5), p=(0.90, 0.10)),
        dict(c=(0.3, 0.5), phi=(-0.3, -0.5), p=(0.70, 0.30)),
        dict(c=(0.3, 0.5), phi=(0.3,  -0.5), p=(0.50, 0.50)),
    ]
    igar_titles = [
        r"(0.2,0.5), (0.7,0.5), p=(0.9,0.1)",
        r"(0.3,0.5), (-0.3,-0.5), p=(0.7,0.3)",
        r"(0.3,0.5), (0.3,-0.5), p=(0.5,0.5)",
    ]
    _ = plot_grid_for_model(
        simulate_igar,
        row_param_list=igar_rows,
        row_titles=igar_titles,
        sim_kwargs=dict(T=200, sigma=(1.0, 1.0), p=(0.7, 0.3), burn=100, y0=0.0)
    )
    plt.suptitle("IGAR(2,1,1) — rows vary $(a_0^i, a_1^i)$", fontsize=13)
    plt.tight_layout(pad=2)
    plt.savefig("IGAR_grid.png", dpi=300)

    # --- MMAR: vary (c, phi) per regime; keep transition matrix fixed (or change if you like)
    P_fixed = np.array([[0.90, 0.10],
                        [0.10, 0.90]])
    mmar_rows = [
        dict(c=(0.2, 0.5),  phi=(0.7, 0.5), p=(0.85, 0.15)),
        dict(c=(0.3, 0.5), phi=(-0.3, -0.5), p=(0.85, 0.15)),
        dict(c=(0.3, 0.5), phi=(0.3,  -0.5), p=(0.85, 0.15)),
    ]
    mmar_titles = [
        r"(0.2,0.5), (0.7,0.5)",
        r"(0.3,0.5), (-0.3,-0.5)",
        r"(0.3,0.5), (0.3,-0.5)",
    ]
    _ = plot_grid_for_model(
        simulate_mmar,
        row_param_list=mmar_rows,
        row_titles=mmar_titles,
        sim_kwargs=dict(T=200, sigma=(1.0, 1.0), P=P_fixed, s0=1, burn=100, x0=0.0)
    )


    plt.suptitle("MMAR(2,1,1) — rows vary $(a_0^i, a_1^i)$", fontsize=13)
    plt.tight_layout(pad=2.0)
    plt.savefig("MMAR_grid.png", dpi=300)
    plt.show()



