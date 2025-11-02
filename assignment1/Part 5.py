import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf as sm_plot_acf, plot_pacf as sm_plot_pacf
from statsmodels.graphics.gofplots import qqplot  # <— for QQ plot
from LDF import ldf


data = pd.read_csv("DataPart5.csv")
y = data.iloc[:, 0].astype(float).dropna()

# Model orders (increasing complexity)
orders = [(1,1), (2,1), (2,2)]  # (p,q)

models, residuals = [], []
for p, q in orders:
    m = ARIMA(y, order=(p, 0, q)).fit()
    models.append(m)
    residuals.append(m.resid)

fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(17, 10), constrained_layout=True)

for i, ((p, q), resid) in enumerate(zip(orders, residuals)):

    sm_plot_acf(resid, lags=30, ax=axs[i, 0])
    axs[i, 0].set_title(f"ACF — ARMA({p},{q}) residuals")
    axs[2, 0].set_xlabel("lag")

    sm_plot_pacf(resid, lags=30, ax=axs[i, 1], method="ywm")
    axs[i, 1].set_title(f"PACF — ARMA({p},{q}) residuals")
    axs[2, 1].set_xlabel("lag")

    axs[i, 2].plot(resid, lw=1)
    axs[i, 2].axhline(0, ls="--", lw=1, color="gray")
    axs[i, 2].set_title(f"Residuals — ARMA({p},{q})")
    axs[2, 2].set_xlabel("t")
    axs[i, 2].set_ylabel("Residual")

 
    qqplot(resid, line="45", ax=axs[i, 3], color="C1", alpha=0.8)
    axs[i, 3].set_title(f"QQ plot — ARMA({p},{q})")

# Shared labels/titles
fig.suptitle("ARMA model diagnostics by order : ACF, PACF, Residuals, QQ-plot", fontsize=13)
plt.tight_layout(pad=2.0)
plt.savefig("Part 5 ARMA diagnostics.png", dpi=300)


x = residuals[1]  # index 1 corresponds to ARMA(2,1)
lags = [1, 2, 3, 4, 5, 10, 20]

vals, q95 = ldf(x, lags, nBoot=20, plotIt=True, random_state=42)
plt.title("LDF of ARMA(2,1) residuals")
plt.savefig("Part 5 LDF residuals.png", dpi=300)
plt.show()


