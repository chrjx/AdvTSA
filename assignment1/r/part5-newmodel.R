# 0) Packages
library(tsDyn)    # SETAR
library(zoo)      # (loaded by tsDyn)
# library(quantmod) # optional; if you load it, the S3 overwrite message appears

# 1) Load & CLEAN data (solve the NA error)
dat_raw <- read.csv("DataPart5.csv")
y_raw   <- as.numeric(dat_raw[[1]])

# remove NA/Inf and keep the time index aligned
ok  <- is.finite(y_raw)
y   <- y_raw[ok]
stopifnot(!anyNA(y), length(y) > 100)  # must be NA-free
y_ts <- ts(y, frequency = 1)

# Quick sanity check
anyNA(y_ts)      # should be FALSE
summary(y_ts)

# 2) Fit a small SETAR first (robust choice)
#    Start simple to avoid unit-root issues; ensure enough obs per regime with trim
fit_setar <- setar(y_ts, nthres = 1, thDelay = 1,
                   mL = 1, mH = 1, trim = 0.1, include = "const")
summary(fit_setar)

# If that works, try a tiny grid and pick BIC
cand <- expand.grid(mL = 1:2, mH = 1:2, d = 1:2)
fits <- lapply(seq_len(nrow(cand)), function(i){
  with(cand[i,], try(setar(y_ts, nthres=1, thDelay=d, mL=mL, mH=mH,
                           trim=0.1, include="const"), silent=TRUE))
})
bic  <- sapply(fits, function(f) if(inherits(f,"try-error")) Inf else BIC(f))
best_i <- which.min(bic)
best   <- fits[[best_i]]
cat("Best by BIC:", paste(cand[best_i,], collapse=", "), "\n")
summary(best)

# 3) Diagnostics
par(mfrow=c(2,2))
acf(residuals(best), 40, main="ACF: SETAR residuals")
pacf(residuals(best), 40, main="PACF: SETAR residuals")
plot(residuals(best), type="l", main="SETAR residuals"); abline(h=0, lty=3)

# 4) LDF on SETAR residuals (uses your teacher’s functions)
source("leaveOneOut.R")  # the function you pasted
source("ldf.R")          # the LDF wrapper that calls leaveOneOut()
ldf(residuals(best), lags=1:10, nBoot=200, plotIt=TRUE, plotFits=FALSE)

