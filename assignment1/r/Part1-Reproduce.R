## ==============================================
## Asymptotic behaviour of the CCM estimator
## (automatic numerical integration for Λ_true)
## ==============================================
rm(list=ls())
set.seed(1)

## ---- True conditional mean and cumulative mean ----
M_true <- function(x) ifelse(x < 0, 4 + 0.5*x, -4 - 0.5*x)

## Numerically integrate M_true from min(x) upward
Lambda_true_numeric <- function(xgrid){
  dx <- diff(range(xgrid)) / length(xgrid)
  Mvals <- M_true(xgrid)
  cumsum(Mvals) * dx
}

## ---- Simulate SETAR(2,1,1) process ----
simulate_setar <- function(n){
  y <- numeric(n)
  for(t in 2:n){
    if(y[t-1] < 0) y[t] <- 4 + 0.5*y[t-1] + rnorm(1)
    else            y[t] <- -4 - 0.5*y[t-1] + rnorm(1)
  }
  y
}

## ---- Settings ----
library(ggplot2)
n_seq  <- c(250, 500, 1000, 2000)
xgrid0 <- seq(-10, 10, length.out=400)   # finer grid
df_all <- data.frame()

## ---- Loop over sample sizes ----
for(n in n_seq){
  y <- simulate_setar(n)
  ylag  <- y[-length(y)]
  ylead <- y[-1]
  
  ## Limit xgrid to observed range
  xrange <- range(ylag)
  xgrid  <- xgrid0[xgrid0 >= xrange[1] & xgrid0 <= xrange[2]]
  
  ## Local linear regression (loess)
  fit <- loess(ylead ~ ylag, degree=1, span=0.3)
  Mhat <- predict(fit, newdata=data.frame(ylag=xgrid))
  
  ## Remove possible NA
  valid <- !is.na(Mhat)
  xgrid <- xgrid[valid]
  Mhat  <- Mhat[valid]
  
  ## Empirical cumulative conditional mean
  dx <- diff(range(xgrid)) / length(xgrid)
  Lhat <- cumsum(Mhat) * dx
  
  df_all <- rbind(df_all,
                  data.frame(x=xgrid, Lambda=Lhat, n=factor(n)))
}

## ---- Theoretical cumulative mean (auto integration) ----
x_theo <- seq(-10, 10, length.out=400)
Lambda_theo <- Lambda_true_numeric(x_theo)
theo_df <- data.frame(x=x_theo, Lambda=Lambda_theo)

## ---- Plot ----
p <- ggplot(df_all, aes(x, Lambda, color=n)) +
  geom_line(size=1) +
  geom_line(data=theo_df, aes(x, Lambda),
            color="black", linetype="dashed", size=1.1) +
  labs(title="Asymptotic behaviour of the CCM estimator",
       subtitle="SETAR(2,1,1): empirical vs theoretical cumulative conditional means",
       x="x", y=expression(Lambda(x)), color="Sample size n") +
  theme_minimal(base_size=14) +
  theme(plot.title=element_text(face="bold"))

print(p)

## ---- Save figure ----
ggsave("Asymptotic_CCM_SETAR_final.png", p, width=9, height=5, dpi=300)
cat("✅ Plot saved as 'Asymptotic_CCM_SETAR_final.png' (auto-integrated Λ_true, no offset).\n")
