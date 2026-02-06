import os

class Config:
    """
    Jarvis-Code 全局配置中心 (V3.3 Survival Edition)
    Vol-Targeting 主导仓位 + 6.0x ATR 灾难阻断器。
    """
    
    # ==========================================
    # 1. 基础设施 (Infrastructure)
    # ==========================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 自动探测数据路径
    _path_v1 = os.path.join(BASE_DIR, "data_raw", "Binance_BTCUSDT_1h.csv")
    _path_v2 = os.path.join(BASE_DIR, "data", "raw", "Binance_BTCUSDT_1h.csv")
    
    if os.path.exists(_path_v1):
        DATA_PATH = _path_v1
    elif os.path.exists(_path_v2):
        DATA_PATH = _path_v2
    else:
        DATA_PATH = _path_v1

    # ==========================================
    # 2. Alpha 策略参数 (Brain Parameters)
    # ==========================================
    STRATEGY_PARAMS = {
        'fast_span': [8, 16, 32, 64],
        'slow_span': [32, 64, 128, 256],
        'scalars': [5.6, 3.8, 2.6, 1.9] 
    }
    
    WEIGHTS = [0.25, 0.25, 0.25, 0.25]
    
    # 长期波动率周期 (480小时)
    # 用于平滑的仓位调整，这是策略的主引擎
    VOL_LOOKBACK = 240
    # [B] 均值回归模块 (Mean Reversion - RSI) [V4.0 New]
    RSI_PERIOD = 14
    # RSI 归一化系数: (50 - RSI) * Scalar
    # 例如 RSI=70 (超买), Diff=-20, Scalar=1.0 -> Forecast=-20 (强空)
    RSI_SCALAR = 1.0 
    
    # [C] 混合权重分配 (Signal Weights) [V4.0 New]
    # 趋势 70% (进攻), RSI 30% (防守/反转)
    TREND_WEIGHT = 0.9
    RSI_WEIGHT = 0.1
    # ==========================================
    # 3. 风险管理 (Risk Management)
    # ==========================================
    # A. 波动率目标制 (Vol Scaling) - 负责日常风控
    TARGET_VOLATILITY = 1.2
    MAX_LEVERAGE = 3
    
    # ------------------------------------------------
    # [B] 环境过滤器 (Regime Filter) [V4.0 New]
    # MA200 (日线) = 4800 (小时)
    REGIME_MA_WINDOW = 2400
    # 熊市/猴市下的杠杆上限 (当 Price < MA200)
    # 开启保命模式，强制降杠杆
    BEAR_MODE_MAX_LEVERAGE = 1.0
    # 正常波动 (1x-3x ATR) 由波动率调仓自动处理，不触发硬止损。
    
    SURVIVAL_ATR_WINDOW = 24       # 快速感知崩盘 (1天)
    SURVIVAL_ATR_MULTIPLIER = 6  # 极宽阈值 (约等于单小时暴跌 6%~10%)
    
    # 波动率地板 (防止死鱼行情误触，至少要跌够这个幅度才算崩盘)
    MIN_HOURLY_VOL = 0.005 
    
    # ==========================================
    # 4. 回测仿真 (Simulation)
    # ==========================================
    POSITION_BUFFER = 0.25 
    FEE_RATE = 0.0005
    INITIAL_CAPITAL = 10000.0