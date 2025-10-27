# ----------------------------------------------------------------
# A. 修正后的数据加载
# ----------------------------------------------------------------

# 1. 设置 header=TRUE 以将第一行识别为列名并跳过。
# 2. [, 1] 选取第一列。
# 3. as.numeric() 显式确保数据是数值类型。
data <- as.numeric(read.csv("DataPart5.csv", header=TRUE)[, 1]) 

# ----------------------------------------------------------------
# B. 拟合 ARMA(2,1) 模型
# ----------------------------------------------------------------

# 使用 arima() 函数拟合 ARMA(2,1) 模型
model_arma21 <- arima(data, order = c(2, 0, 1))

print("--- ARMA(2,1) Model Fit Summary ---")
print(model_arma21)

# 提取模型残差
residuals_arma21 <- residuals(model_arma21)

# Ljung-Box 检验 (可选但推荐)
Box.test(residuals_arma21, lag = 20, type = "Ljung-Box")


# ----------------------------------------------------------------
# C. 对残差运行 LDF
# ----------------------------------------------------------------

# x: 模型的残差 (e_t)
# lags: 想要检验的滞后数 (例如，检验前 10 个滞后)
# nBoot: 用于计算置信带的 Bootstrap 次数 (建议 100 或更高)

print("--- Running LDF on ARMA(2,1) Residuals ---")

ldf_results <- ldf(
  x = residuals_arma21, 
  lags = 10, 
  nBoot = 100, 
  plotIt = TRUE, # 绘制 LDF 图
  plotFits = FALSE
)

# ----------------------------------------------------------------
# D. 分析与改进模型
# ----------------------------------------------------------------

# 1. 查看生成的 LDF 图。如果 LDF 曲线在任何滞后 n 处
#    显著超出置信带，则表示残差中存在显著的非线性。
# 2. 找到显著的滞后 n。
# 3. 绘制诊断图 e_t vs. e_{t-n}：
#    plot(residuals_arma21[-c(1:n)], residuals_arma21[-c((n+1):length(residuals_arma21))],
#         xlab=paste0("e_{t-", n, "}"), ylab="e_t", main=paste0("Residuals: e_t vs. e_{t-", n, "}"))
# 4. 根据诊断图的形状（例如，SETAR 的 V 形、ARCH 的喇叭形），
#    提出一个更合适的非线性模型结构（如 SETAR, ARCH/GARCH 等）。