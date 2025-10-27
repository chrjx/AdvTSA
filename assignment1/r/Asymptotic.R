## =====================================================
## Asymptotic behaviour of the CCM estimator (clear demo)
## =====================================================
rm(list=ls())
set.seed(1)
library(ggplot2)

## ---- True conditional mean ----
M_true <- function(x) ifelse(x < 0, 4 + 0.5*x, -4 - 0.5*x)

## ---- Numerical integral of M_true ----
Lambda_true_numeric <- function(xgrid){
  dx <- diff(range(xgrid)) / length(xgrid)
  Mvals <- M_true(xgrid)
  cumsum(Mvals) * dx
}

## ---- Simulate SETAR(2,1,1) process ----
simulate_setar <- function(n, sd=2){  # <-- larger noise for visible effect
  y <- numeric(n)
  for(t in 2:n){
    if(y[t-1] < 0) y[t] <- 4 + 0.5*y[t-1] + rnorm(1, 0, sd)
    else            y[t] <- -4 - 0.5*y[t-1] + rnorm(1, 0, sd)
  }
  y
}

## ---- Parameters ----
n_seq  <- c(250, 500, 1000, 2000)
xgrid0 <- seq(-10, 10, length.out=400)
df_all <- data.frame()
rmse_df <- data.frame()

## ---- Main loop ----
for(n in n_seq){
  y <- simulate_setar(n)
  ylag  <- y[-length(y)]
  ylead <- y[-1]
  
  ## Adaptive bandwidth: span ~ n^(-1/5)
  span_val <- 0.8 * n^(-1/5)
  
  fit <- loess(ylead ~ ylag, degree=1, span=span_val)
  Mhat <- predict(fit, newdata=data.frame(ylag=xgrid0))
  valid <- !is.na(Mhat)
  xgrid <- xgrid0[valid]
  Mhat  <- Mhat[valid]
  
  ## Empirical cumulative conditional mean
  dx <- diff(range(xgrid)) / length(xgrid)
  Lhat <- cumsum(Mhat) * dx
  
  ## Theoretical cumulative mean (same grid)
  Ltrue <- Lambda_true_numeric(xgrid)
  
  ## RMSE
  rmse <- sqrt(mean((Lhat - Ltrue)^2, na.rm=TRUE))
  rmse_df <- rbind(rmse_df, data.frame(n=n, RMSE=rmse))
  
  ## Store for plot
  df_all <- rbind(df_all, data.frame(x=xgrid, Lambda=Lhat, n=factor(n)))
}

## ---- Plot CCM curves ----
x_theo <- seq(-10, 10, length.out=400)
Lambda_theo <- Lambda_true_numeric(x_theo)
theo_df <- data.frame(x=x_theo, Lambda=Lambda_theo)

p1 <- ggplot(df_all, aes(x, Lambda, color=n)) +
  geom_line(size=1) +
  geom_line(data=theo_df, aes(x, Lambda),
            color="black", linetype="dashed", size=1.2) +
  labs(title="Asymptotic behaviour of the CCM estimator",
       subtitle="SETAR(2,1,1): Empirical vs theoretical cumulative conditional means",
       x="x", y=expression(Lambda(x)), color="Sample size n") +
  theme_minimal(base_size=14) +
  theme(plot.title=element_text(face="bold"))

print(p1)

## ---- Plot RMSE vs sample size ----
p2 <- ggplot(rmse_df, aes(x=n, y=RMSE)) +
  geom_line(linewidth=1, color="steelblue") +
  geom_point(size=3, color="steelblue") +
  scale_x_continuous(breaks=n_seq) +
  labs(title="Convergence of the CCM estimator",
       subtitle="RMSE between empirical and theoretical Λ(x) decreases with n",
       x="Sample size n", y="RMSE(Λ̂ₙ)") +
  theme_minimal(base_size=14) +
  theme(plot.title=element_text(face="bold"))

print(p2)

## ---- Save figures ----
ggsave("CCM_Asymptotic_Curves.png", p1, width=9, height=5, dpi=300)
ggsave("CCM_Asymptotic_RMSE.png", p2, width=7, height=5, dpi=300)
cat("✅ Saved: CCM_Asymptotic_Curves.png and CCM_Asymptotic_RMSE.png\n")
