## ==============================================
## Part 1 完整代码：非线性模型仿真与可视化（SETAR / IGAR / MMAR）
## 生成内容：
##  - 单组参数：时间序列 + 相图 + 状态轨迹（MMAR）
##  - 多组参数：对比图（facets）
## 输出目录：part1_outputs/
## ==============================================

rm(list = ls())
set.seed(42)

## -------- 依赖包 --------
pkgs <- c("ggplot2", "patchwork")
for(p in pkgs){
  if(!requireNamespace(p, quietly = TRUE)){
    install.packages(p)
  }
}
library(ggplot2)
library(patchwork)

## -------- 工具&主题 --------
outdir <- "part1_outputs"
if(!dir.exists(outdir)) dir.create(outdir)

theme_clean <- function(){
  theme_minimal(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(face="bold"),
      legend.position = "bottom"
    )
}

save_plot <- function(p, fname, w=8, h=4.5){
  ggsave(file.path(outdir, fname), p, width = w, height = h, dpi = 200)
}

lagged_df <- function(y){
  ## 生成 (y_{t-1}, y_t) 相图数据
  data.frame(
    t = 2:length(y),
    y_lag = y[-length(y)],
    y = y[-1]
  )
}

## =========================================================
## 1) SETAR(2;1;1) 模拟与作图
##    y_t = { a1 + b1*y_{t-1} + e_t, if y_{t-1} <= thr
##          { a2 + b2*y_{t-1} + e_t, if y_{t-1} >  thr
## =========================================================
simulate_SETAR <- function(n, a1, b1, a2, b2, thr, sd=1, y0=0){
  y <- numeric(n); y[1] <- y0
  reg <- integer(n); reg[1] <- ifelse(y[1] <= thr, 1, 2)
  for(t in 2:n){
    if(y[t-1] <= thr){
      y[t] <- a1 + b1*y[t-1] + rnorm(1, 0, sd)
      reg[t] <- 1
    } else {
      y[t] <- a2 + b2*y[t-1] + rnorm(1, 0, sd)
      reg[t] <- 2
    }
  }
  list(y=y, reg=reg, thr=thr)
}

plot_setar <- function(sim, title_prefix="SETAR(2;1;1)"){
  y <- sim$y; reg <- factor(sim$reg); thr <- sim$thr
  df_ts <- data.frame(t=1:length(y), y=y, reg=reg)
  df_phase <- lagged_df(y)
  df_phase$reg <- factor(ifelse(df_phase$y_lag <= thr, 1, 2))
  
  p_ts <- ggplot(df_ts, aes(t, y, color=reg))+
    geom_line(linewidth=0.6)+
    scale_color_manual(values=c("#1f77b4","#d62728"), name="Regime",
                       labels=c(paste0("y[t-1] ≤ ", thr), paste0("y[t-1] > ", thr)))+
    labs(title=paste0(title_prefix, "：时间序列"),
         x="t", y="y[t]")+
    theme_clean()
  
  p_phase <- ggplot(df_phase, aes(y_lag, y, color=reg))+
    geom_point(alpha=0.5, size=1)+
    geom_vline(xintercept = thr, linetype=2, linewidth=0.6)+
    scale_color_manual(values=c("#1f77b4","#d62728"), name="Regime")+
    labs(title=paste0(title_prefix, "：相图 y[t] vs y[t-1]"),
         x="y[t-1]", y="y[t]")+
    theme_clean()
  
  p_ts / p_phase
}

## 示例参数（与你报告风格一致，且稳定）：可自行调整
n <- 1000
setar_params_default <- list(a1=0.2, b1=0.9, a2=15, b2=0.95, thr=50, sd=1, y0=0)

setar_sim <- do.call(simulate_SETAR, c(list(n=n), setar_params_default))
p_setar <- plot_setar(setar_sim)
save_plot(p_setar, "SETAR_default_timeseries_phase.png", w=10, h=8)

## 多组参数对比（老师建议：展示不同参数的行为差异）
setar_grid <- list(
  list(a1=0.2,b1=0.9,a2=15,b2=0.6, thr=50, sd=1, y0=0),    # 稳定 + 高阈值
  list(a1=0.2,b1=0.8,a2=10,b2=0.7, thr=30, sd=1, y0=0),    # 更稳 + 低阈值（更频繁切换）
  list(a1=1.0,b1=0.95,a2=-12,b2=0.6, thr=0, sd=1, y0=0)    # 正负分段（阈值在0）
)

plots <- list()
for(i in seq_along(setar_grid)){
  sim_i <- do.call(simulate_SETAR, c(list(n=n), setar_grid[[i]]))
  plots[[i]] <- plot_setar(sim_i, title_prefix=paste0("SETAR 方案 ", i))
}
save_plot(wrap_plots(plots, ncol=1), "SETAR_param_sensitivity.png", w=10, h=20)

## =========================================================
## 2) IGAR(2;1) 模拟与作图
##    随机开关： j_t ~ Uniform(0,1)
##    if j_t < p: y_t = a1 + b1*y_{t-1} + e_t
##    else:        y_t = a2 + b2*y_{t-1} + e_t
## =========================================================
simulate_IGAR <- function(n, p, a1, b1, a2, b2, sd=1, y0=0){
  y <- numeric(n); y[1] <- y0
  reg <- integer(n)
  jt <- runif(n)
  reg[1] <- ifelse(jt[1] < p, 1, 2)
  for(t in 2:n){
    if(jt[t] < p){
      y[t] <- a1 + b1*y[t-1] + rnorm(1, 0, sd)
      reg[t] <- 1
    } else {
      y[t] <- a2 + b2*y[t-1] + rnorm(1, 0, sd)
      reg[t] <- 2
    }
  }
  list(y=y, reg=reg, jt=jt, p=p)
}

plot_igar <- function(sim, title_prefix="IGAR(2;1)"){
  y <- sim$y; reg <- factor(sim$reg); p <- sim$p
  df_ts <- data.frame(t=1:length(y), y=y, reg=reg)
  df_phase <- lagged_df(y)
  df_phase$reg <- reg[-1]
  
  p_ts <- ggplot(df_ts, aes(t, y, color=reg))+
    geom_line(linewidth=0.6)+
    scale_color_manual(values=c("#2ca02c","#ff7f0e"), name="Regime",
                       labels=c(paste0("j[t]<",p), paste0("j[t]≥",p)))+
    labs(title=paste0(title_prefix, "：时间序列（随机切换）"),
         x="t", y="y[t]")+
    theme_clean()
  
  p_phase <- ggplot(df_phase, aes(y_lag, y, color=reg))+
    geom_point(alpha=0.5, size=1)+
    scale_color_manual(values=c("#2ca02c","#ff7f0e"), name="Regime")+
    labs(title=paste0(title_prefix, "：相图 y[t] vs y[t-1]"),
         x="y[t-1]", y="y[t]")+
    theme_clean()
  
  p_ts / p_phase
}

igar_params_default <- list(p=0.9, a1=0.1, b1=0.9, a2=-5, b2=0.9, sd=1, y0=0)
igar_sim <- do.call(simulate_IGAR, c(list(n=n), igar_params_default))
p_igar <- plot_igar(igar_sim)
save_plot(p_igar, "IGAR_default_timeseries_phase.png", w=10, h=8)

## 多组参数对比（切换概率 & 系数变化）
igar_grid <- list(
  list(p=0.9, a1=0.1,  b1=0.9, a2=-5,  b2=0.9, sd=1, y0=0),
  list(p=0.7, a1=0.5,  b1=0.85,a2=-7,  b2=0.7, sd=1, y0=0),
  list(p=0.5, a1=1.0,  b1=0.8, a2=-10, b2=0.6, sd=1, y0=0)
)
plots <- list()
for(i in seq_along(igar_grid)){
  sim_i <- do.call(simulate_IGAR, c(list(n=n), igar_grid[[i]]))
  plots[[i]] <- plot_igar(sim_i, title_prefix=paste0("IGAR 方案 ", i))
}
save_plot(wrap_plots(plots, ncol=1), "IGAR_param_sensitivity.png", w=10, h=20)

## =========================================================
## 3) MMAR(2;1) 模拟与作图
##    隐马尔可夫状态 s_t ∈ {1,2}，转移矩阵 P
##    y_t = a[s_t] + b[s_t]*y_{t-1} + e_t
## =========================================================
simulate_MMAR <- function(n, a, b, P, sd=1, y0=0, s0=1){
  stopifnot(length(a)==2, length(b)==2, all(dim(P)==c(2,2)))
  y <- numeric(n); y[1] <- y0
  s <- integer(n); s[1] <- s0
  
  ## 预先抽取均匀数以确保可复现
  U <- runif(n)
  
  for(t in 2:n){
    ## 先转移状态
    prev <- s[t-1]
    ## 按行 prev 从 P[prev,] 采样下一个状态
    s[t] <- if(U[t] <= P[prev,1]) 1 else 2
    ## 再生成 y_t
    y[t] <- a[s[t]] + b[s[t]]*y[t-1] + rnorm(1,0,sd)
  }
  list(y=y, s=s, P=P)
}

plot_mmar <- function(sim, title_prefix="MMAR(2;1)"){
  y <- sim$y; s <- factor(sim$s)
  df_ts <- data.frame(t=1:length(y), y=y, s=s)
  df_phase <- lagged_df(y)
  df_phase$s <- s[-1]
  df_state <- data.frame(t=1:length(y), s=as.integer(s))
  
  p_ts <- ggplot(df_ts, aes(t, y, color=s))+
    geom_line(linewidth=0.6)+
    scale_color_manual(values=c("#9467bd","#8c564b"), name="State",
                       labels=c("State 1","State 2"))+
    labs(title=paste0(title_prefix, "：时间序列（隐状态着色）"),
         x="t", y="y[t]")+
    theme_clean()
  
  p_phase <- ggplot(df_phase, aes(y_lag, y, color=s))+
    geom_point(alpha=0.5, size=1)+
    scale_color_manual(values=c("#9467bd","#8c564b"), name="State")+
    labs(title=paste0(title_prefix, "：相图 y[t] vs y[t-1]"),
         x="y[t-1]", y="y[t]")+
    theme_clean()
  
  p_state <- ggplot(df_state, aes(t, s))+
    geom_step(linewidth=0.6)+
    scale_y_continuous(breaks=c(1,2), labels=c("State 1","State 2"))+
    labs(title=paste0(title_prefix, "：状态轨迹（step）"),
         x="t", y="state")+
    theme_clean()
  
  (p_ts / p_phase) / p_state
}

mmar_params_default <- list(
  a=c(0.1, -12), b=c(0.9, 0.6),
  P=matrix(c(0.95,0.05, 0.10,0.90), nrow=2, byrow=TRUE),
  sd=1, y0=0, s0=1
)

mmar_sim <- do.call(simulate_MMAR, c(list(n=n), mmar_params_default))
p_mmar <- plot_mmar(mmar_sim)
save_plot(p_mmar, "MMAR_default_ts_phase_state.png", w=10, h=12)

## 多组参数对比（持久性 & 自回归强度）
mmar_grid <- list(
  list(a=c(0.1,-12), b=c(0.9,0.6),
       P=matrix(c(0.95,0.05, 0.10,0.90),2,2,byrow=TRUE), sd=1, y0=0, s0=1),
  list(a=c(0.0,-8),  b=c(0.85,0.7),
       P=matrix(c(0.98,0.02, 0.05,0.95),2,2,byrow=TRUE), sd=1, y0=0, s0=1),
  list(a=c(0.5,-15), b=c(0.95,0.5),
       P=matrix(c(0.90,0.10, 0.20,0.80),2,2,byrow=TRUE), sd=1, y0=0, s0=1)
)

plots <- list()
for(i in seq_along(mmar_grid)){
  sim_i <- do.call(simulate_MMAR, c(list(n=n), mmar_grid[[i]]))
  plots[[i]] <- plot_mmar(sim_i, title_prefix=paste0("MMAR 方案 ", i))
}
save_plot(wrap_plots(plots, ncol=1), "MMAR_param_sensitivity.png", w=10, h=30)

## ------- 结束 -------
message("✅ Part 1 全部图已输出到: ", normalizePath(outdir))

