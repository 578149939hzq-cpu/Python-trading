import os

# ==========================================
# 1. 基础设施配置 (Infrastructure)
# ==========================================
# 项目根目录 (自动获取当前文件所在目录)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据存储路径
# C 语言习惯：不要把路径写死成 "C:\\Users\\..."，用相对路径
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# 默认交易对文件
DEFAULT_SYMBOL = "BTCUSDT"
DATA_FILE_NAME = "Binance_BTCUSDT_1h.csv"
DATA_PATH = os.path.join(DATA_RAW_DIR, DATA_FILE_NAME)

# ==========================================
# 2. Alpha 核心参数 (Brain Parameters)
# ==========================================
# 这里的参数决定了 Jarvis 怎么看待市场

# Robert Carver 的 EWMA 周期组合
# 这里的数字代表 "Span" (指数衰减跨度)，不是简单的 Window
# 8=极快, 64=长期趋势
STRATEGY_PARAMS = {
        'fast_span': [8, 16, 32, 64],
        'slow_span': [32, 64, 128, 256],
        # 对应上面四组的缩放系数
        'scalars': [5.6, 3.8, 2.6, 1.9] 
    }
# 4. 权重 (Chapter 9 - Ensemble)
# 我们认为这4个策略一样好，所以权重平均
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
# 波动率计算周期 (用于归一化)
VOLATILITY_SPAN = 36

# 预测值放大系数 (Scalar)
# 原始预测值通常在 -2 到 +2 之间，乘以这个系数方便观察
# 在 Phase 3 网格搜索中，这个值会被动态替换
DEFAULT_SCALAR = 10.0

# ==========================================
# 3. 策略执行参数 (Execution Parameters)
# ==========================================
# 这里的参数决定了 Jarvis 怎么下单

# 缓冲区 (Buffer / Hysteresis)
# 只有当 (目标仓位 - 当前仓位) 的绝对值大于此值时，才调仓
# 防止在震荡市被手续费磨损
POSITION_BUFFER = 0.10  # 10%
# ==========================================
# 4. 回测环境配置 (Simulation)
# ==========================================
# 初始资金 (美元)
INITIAL_CAPITAL = 10000.0
TARGET_VOLATILITY = 1.00  # 年化波动率目标 (20% is Carver's standard)
IDM = 1.0                  # 暂时设为 1.0 (单一资产)
# 最大杠杆限制 (硬顶)
# 防止波动率极低时算出一个 100倍杠杆把账户爆了
MAX_LEVERAGE=4.0
# 交易手续费率
# Binance 现货通常是 0.1% (0.001)，BNB 抵扣是 0.075% (0.00075)
# 量化通常设得高一点作为滑点保护，比如 万五 (0.0005)
FEE_RATE = 0.0005

# 回测时间段 (样本内/样本外切分点)
# 格式: 'YYYY-MM-DD'
START_DATE = '2020-01-01'
SPLIT_DATE = '2023-01-01' # 之前是 IS，之后是 OOS
END_DATE = '2026-01-01'
# [V2.0 Update] Sigma 熔断阈值
# 如果单小时涨跌幅超过 3倍标准差，视为流动性黑洞，强制清零
SIGMA_THRESHOLD = 4.0