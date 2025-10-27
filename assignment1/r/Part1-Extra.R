## =========================================================
## Part 1 - 补充可视化（不改原图）
##   1) regime/状态着色的时间序列
##   2) 相图 y[t] vs y[t-1]
##   3) 不同参数组合的对比
## =========================================================

## ---------- 兼容：如果上文没定义保存选项，则设默认 ----------
if(!exists("SAVE_PLOTS")) SAVE_PLOTS <- TRUE
if(!exists("OUTDIR"))      OUTDIR      <- "part1_extra_out"
if(!exists("FORMAT"))      FORMAT      <- "png"
if(!exists("DPI"))         DPI         <- 150
if(!exists("W_THIN"))      W_THIN      <- 1400
if(!exists("H_THIN"))      H_THIN      <- 500
if(!exists("W_TALL"))      W_TALL      <- 1400
if(!exists("H_TALL"))      H_TALL      <- 700

if(!exists("open_dev")){
  open_dev <- function(filename, w, h, fmt=FORMAT, dpi=DPI, outdir=OUTDIR){
    if(!SAVE_PLOTS) return(invisible(FALSE))
    if(!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
    path <- file.path(outdir, filename)
    switch(fmt,
           "png" = png(path, width=w, height=h, res=dpi),
           "pdf" = pdf(path,  width=w/96, height=h/96),
           "svg" = svg(path,  width=w/96, height=h/96),
           stop("Unsupported FORMAT (use 'png'/'pdf'/'svg').")
    )
    invisible(TRUE)
  }
  close_dev <- function(opened){ if(isTRUE(opened)) dev.off() }
}

## ---------- 依赖包（仅补充部分用到） ----------
pkgs <- c("ggplot2", "patchwork")
for(p in pkgs){
  if(!requireNamespace(p, quietly = TRUE)) install.packages(p)
}
library(ggplot2)
library(patchwork)

## ---------- 若上文未定义这些随机种子，则给默认 ----------
if(!exists("SEED_SETAR")) SEED_SETAR <- 123
if(!exists("SEED_IGAR"))  SEED_IGAR  <- 456
if(!exists("SEED_MMAR"))  SEED_MMAR  <- 789
if(!exists("n"))          n          <- 1000

## =========================================================
## 一、不同视角：regime/状态着色 + 相图
## （SETAR/IGAR 重新模拟以拿到regime；MMAR之前已返回state）
## =========================================================

## 1) SETAR：与报告一致的参数 + 返回regime
simulate_SETAR_report_with_reg <- function(n, a1=0.2, b1=1.0, a2=15, b2=0.95,
                                           thr=50, sd=1, y0=0, seed=NULL){
  if(!is.null(seed)) set.seed(seed)
  y <- numeric(n); y[1] <- y0
  reg <- integer(n); reg[1] <- ifelse(y[1] <= thr, 1, 2)
  for(t in 2:n){
    if(y[t-1] <= thr){
      y[t] <- a1 + b1*y[t-1] + rnorm(1, 0, sd); reg[t] <- 1
    } else {
      y[t] <- a2 + b2*y[t-1] + rnorm(1, 0, sd); reg[t] <- 2
    }
  }
  df_ts    <- data.frame(t=1:n, y=y, reg=factor(reg))
  df_phase <- data.frame(y_lag=y[-n], y=y[-1], reg=factor(reg[-1]))
  list(df_ts=df_ts, df_phase=df_phase, thr=thr)
}

setar_sup <- simulate_SETAR_report_with_reg(n, seed=SEED_SETAR)
## 时间序列（着色）
p_setar_ts <- ggplot(setar_sup$df_ts, aes(t, y, color=reg))+
  geom_line(linewidth=0.6)+
  scale_color_manual(values=c("#1f77b4","#d62728"),
                     labels=c("Regime 1 (y[t-1] ≤ 50)","Regime 2 (y[t-1] > 50)"),
                     name="Regime")+
  labs(title="SETAR(2,1,1)：按regime着色的时间序列", x="t", y="y[t]")+
  theme_minimal(base_size = 13)+
  theme(legend.position="bottom")

## 相图
p_setar_phase <- ggplot(setar_sup$df_phase, aes(y_lag, y, color=reg))+
  geom_point(alpha=0.6, size=1)+
  geom_vline(xintercept = setar_sup$thr, linetype=2)+
  scale_color_manual(values=c("#1f77b4","#d62728"), name="Regime")+
  labs(title="SETAR(2,1,1)：相图 y[t] vs y[t-1]", x="y[t-1]", y="y[t]")+
  theme_minimal(base_size = 13)+
  theme(legend.position="bottom")

opened <- open_dev("Figure1_supp_SETAR_TS_and_Phase.png", 1400, 900)
print(p_setar_ts / p_setar_phase)
close_dev(opened)

## 2) IGAR：与报告一致的参数 + 返回regime（由 j_t < p 决定）
simulate_IGAR_report_with_reg <- function(n, p=0.9,
                                          a1=0.1, b1=1.0, a2=-5, b2=0.9,
                                          sd=1, y0=0, seed=NULL){
  if(!is.null(seed)) set.seed(seed)
  y <- numeric(n); y[1] <- y0
  j <- runif(n)
  reg <- integer(n); reg[1] <- ifelse(j[1] < p, 1, 2)
  for(t in 2:n){
    if(j[t] < p){
      y[t] <- a1 + b1*y[t-1] + rnorm(1,0,sd); reg[t] <- 1
    } else {
      y[t] <- a2 + b2*y[t-1] + rnorm(1,0,sd); reg[t] <- 2
    }
  }
  df_ts    <- data.frame(t=1:n, y=y, reg=factor(reg))
  df_phase <- data.frame(y_lag=y[-n], y=y[-1], reg=factor(reg[-1]))
  list(df_ts=df_ts, df_phase=df_phase)
}

igar_sup <- simulate_IGAR_report_with_reg(n, seed=SEED_IGAR)

p_igar_ts <- ggplot(igar_sup$df_ts, aes(t, y, color=reg))+
  geom_line(linewidth=0.6)+
  scale_color_manual(values=c("#2ca02c","#ff7f0e"),
                     labels=c("Regime 1 (j[t]<0.9)","Regime 2 (j[t]≥0.9)"),
                     name="Regime")+
  labs(title="IGAR(2,1)：按regime着色的时间序列", x="t", y="y[t]")+
  theme_minimal(base_size = 13)+
  theme(legend.position="bottom")

p_igar_phase <- ggplot(igar_sup$df_phase, aes(y_lag, y, color=reg))+
  geom_point(alpha=0.6, size=1)+
  scale_color_manual(values=c("#2ca02c","#ff7f0e"), name="Regime")+
  labs(title="IGAR(2,1)：相图 y[t] vs y[t-1]", x="y[t-1]", y="y[t]")+
  theme_minimal(base_size = 13)+
  theme(legend.position="bottom")

opened <- open_dev("Figure2_supp_IGAR_TS_and_Phase.png", 1400, 900)
print(p_igar_ts / p_igar_phase)
close_dev(opened)

## 3) MMAR：时间序列按状态着色 + 相图（用已有 mmar 结果）
## 若 mmar 不存在，则重算一次（与报告参数一致）
if(!exists("mmar")){
  simulate_MMAR_report <- function(n,
                                   a=c(0.1, -12), b=c(0.9, 0.6),
                                   P=matrix(c(0.95,0.05, 0.10,0.90), 2, 2, byrow=TRUE),
                                   sd=1, y0=0, s0=1, seed=NULL){
    if(!is.null(seed)) set.seed(seed)
    y <- numeric(n); y[1] <- y0
    s <- integer(n); s[1] <- s0
    U <- runif(n)
    for(t in 2:n){
      prev <- s[t-1]
      s[t] <- if(U[t] <= P[prev,1]) 1 else 2
      y[t] <- a[s[t]] + b[s[t]]*y[t-1] + rnorm(1,0,sd)
    }
    list(df_ts = data.frame(t=1:n, y=y),
         df_state = data.frame(t=1:n, state=s))
  }
  mmar <- simulate_MMAR_report(n, seed=SEED_MMAR)
}

df_mmar_ts    <- transform(mmar$df_ts, state=factor(mmar$df_state$state))
df_mmar_phase <- data.frame(y_lag=df_mmar_ts$y[-n], y=df_mmar_ts$y[-1],
                            state=df_mmar_ts$state[-1])

p_mmar_ts <- ggplot(df_mmar_ts, aes(t, y, color=state))+
  geom_line(linewidth=0.6)+
  scale_color_manual(values=c("#9467bd","#8c564b"), name="State",
                     labels=c("State 1","State 2"))+
  labs(title="MMAR(2,1)：按状态着色的时间序列", x="t", y="y[t]")+
  theme_minimal(base_size = 13)+
  theme(legend.position="bottom")

p_mmar_phase <- ggplot(df_mmar_phase, aes(y_lag, y, color=state))+
  geom_point(alpha=0.6, size=1)+
  scale_color_manual(values=c("#9467bd","#8c564b"), name="State")+
  labs(title="MMAR(2,1)：相图 y[t] vs y[t-1]", x="y[t-1]", y="y[t]")+
  theme_minimal(base_size = 13)+
  theme(legend.position="bottom")

opened <- open_dev("Figure3_supp_MMAR_TS_and_Phase.png", 1400, 900)
print(p_mmar_ts / p_mmar_phase)
close_dev(opened)

## =========================================================
## 二、不同参数组合（Parameter Sensitivity）
##   每个模型给 3 组参数；仅生成补充图，不影响原图
## =========================================================

## -- SETAR 参数网格 --
setar_grid <- list(
  list(name="A baseline",
       a1=0.2, b1=1.00, a2=15, b2=0.95, thr=50, sd=1, y0=0),
  list(name="B lower threshold",
       a1=0.2, b1=0.95, a2=12, b2=0.90, thr=30, sd=1, y0=0),
  list(name="C stronger stabilization",
       a1=0.2, b1=0.90, a2=10, b2=0.80, thr=50, sd=1, y0=0)
)

setar_panels <- lapply(seq_along(setar_grid), function(i){
  g <- setar_grid[[i]]
  sim <- simulate_SETAR_report_with_reg(
    n, a1=g$a1, b1=g$b1, a2=g$a2, b2=g$b2, thr=g$thr, sd=g$sd, y0=g$y0,
    seed = SEED_SETAR + i
  )
  p1 <- ggplot(sim$df_ts, aes(t, y, color=reg))+
    geom_line(linewidth=0.5)+
    scale_color_manual(values=c("#1f77b4","#d62728"), guide="none")+
    labs(title=paste0("SETAR ", g$name), x=NULL, y="y[t]")+
    theme_minimal(base_size=11)
  p2 <- ggplot(sim$df_phase, aes(y_lag, y, color=reg))+
    geom_point(alpha=0.5, size=0.8)+
    scale_color_manual(values=c("#1f77b4","#d62728"), guide="none")+
    labs(x="y[t-1]", y="y[t]")+
    theme_minimal(base_size=11)
  p1 / p2
})
opened <- open_dev("Figure1_supp_SETAR_param_sensitivity.png", 1400, 1800)
print(wrap_plots(setar_panels, ncol=1))
close_dev(opened)

## -- IGAR 参数网格 --
igar_grid <- list(
  list(name="A p=0.9", p=0.9, a1=0.1, b1=1.0, a2=-5,  b2=0.9, sd=1, y0=0),
  list(name="B p=0.7", p=0.7, a1=0.4, b1=0.9, a2=-6,  b2=0.8, sd=1, y0=0),
  list(name="C p=0.5", p=0.5, a1=0.8, b1=0.85,a2=-8,  b2=0.7, sd=1, y0=0)
)

igar_panels <- lapply(seq_along(igar_grid), function(i){
  g <- igar_grid[[i]]
  sim <- simulate_IGAR_report_with_reg(
    n, p=g$p, a1=g$a1, b1=g$b1, a2=g$a2, b2=g$b2, sd=g$sd, y0=g$y0,
    seed = SEED_IGAR + i
  )
  p1 <- ggplot(sim$df_ts, aes(t, y, color=reg))+
    geom_line(linewidth=0.5)+
    scale_color_manual(values=c("#2ca02c","#ff7f0e"), guide="none")+
    labs(title=paste0("IGAR ", g$name), x=NULL, y="y[t]")+
    theme_minimal(base_size=11)
  p2 <- ggplot(sim$df_phase, aes(y_lag, y, color=reg))+
    geom_point(alpha=0.5, size=0.8)+
    scale_color_manual(values=c("#2ca02c","#ff7f0e"), guide="none")+
    labs(x="y[t-1]", y="y[t]")+
    theme_minimal(base_size=11)
  p1 / p2
})
opened <- open_dev("Figure2_supp_IGAR_param_sensitivity.png", 1400, 1800)
print(wrap_plots(igar_panels, ncol=1))
close_dev(opened)

## -- MMAR 参数网格（持久性与AR强度） --
simulate_MMAR_report <- function(n,
                                 a=c(0.1, -12), b=c(0.9, 0.6),
                                 P=matrix(c(0.95,0.05, 0.10,0.90), 2, 2, byrow=TRUE),
                                 sd=1, y0=0, s0=1, seed=NULL){
  if(!is.null(seed)) set.seed(seed)
  y <- numeric(n); y[1] <- y0
  s <- integer(n); s[1] <- s0
  U <- runif(n)
  for(t in 2:n){
    prev <- s[t-1]
    s[t] <- if(U[t] <= P[prev,1]) 1 else 2
    y[t] <- a[s[t]] + b[s[t]]*y[t-1] + rnorm(1,0,sd)
  }
  list(df_ts = data.frame(t=1:n, y=y, state=factor(s)))
}

mmar_grid <- list(
  list(name="A baseline",
       a=c(0.1,-12), b=c(0.9,0.6),
       P=matrix(c(0.95,0.05, 0.10,0.90),2,2,byrow=TRUE)),
  list(name="B high persistence",
       a=c(0.1,-10), b=c(0.88,0.65),
       P=matrix(c(0.98,0.02, 0.05,0.95),2,2,byrow=TRUE)),
  list(name="C low persistence",
       a=c(0.3,-14), b=c(0.92,0.55),
       P=matrix(c(0.90,0.10, 0.20,0.80),2,2,byrow=TRUE))
)

mmar_panels <- lapply(seq_along(mmar_grid), function(i){
  g <- mmar_grid[[i]]
  sim <- simulate_MMAR_report(n, a=g$a, b=g$b, P=g$P, seed=SEED_MMAR + i)
  df_phase <- data.frame(y_lag=sim$df_ts$y[-n], y=sim$df_ts$y[-1],
                         state=sim$df_ts$state[-1])
  p1 <- ggplot(sim$df_ts, aes(t, y, color=state))+
    geom_line(linewidth=0.5)+
    scale_color_manual(values=c("#9467bd","#8c564b"), guide="none")+
    labs(title=paste0("MMAR ", g$name), x=NULL, y="y[t]")+
    theme_minimal(base_size=11)
  p2 <- ggplot(df_phase, aes(y_lag, y, color=state))+
    geom_point(alpha=0.5, size=0.8)+
    scale_color_manual(values=c("#9467bd","#8c564b"), guide="none")+
    labs(x="y[t-1]", y="y[t]")+
    theme_minimal(base_size=11)
  p1 / p2
})
opened <- open_dev("Figure3_supp_MMAR_param_sensitivity.png", 1400, 1800)
print(wrap_plots(mmar_panels, ncol=1))
close_dev(opened)

cat("✅ 补充图已保存到：", normalizePath(OUTDIR), "\n", sep="")
