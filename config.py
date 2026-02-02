import os

class Config:
    """
    Jarvis-Code 全局配置中心 (V3.1 Alpha Lab Edition)
    集成 MAD 去噪、ATR 动态风控与长期稳健定仓。
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
    
    # [Core Upgrade C] 稳健波动率定仓 (Stable Sizing)
    # 从 36 调整为 480 (约20天)。
    # 目的: 让仓位锚定长期波动率趋势，忽略短期噪音，避免仓位剧烈抖动。
    VOL_LOOKBACK = 480

    # ==========================================
    # 3. 预处理降噪参数 (Denoising Layer)
    # ==========================================
    # [Core Upgrade A] MAD 去噪配置
    MAD_WINDOW = 24       # 滚动窗口 24小时
    MAD_THRESHOLD = 5.0   # 偏离中位数 5倍 MAD 视为插针

    # ==========================================
    # 4. 风险引擎 V3.1 (ATR Dynamic Risk)
    # ==========================================
    # [Core Upgrade B] ATR 动态止损
    RISK_METRIC = 'ATR'   # 强制使用 ATR
    ATR_WINDOW = 24       # 保持对波动率变化的快速响应
    
    # 3倍 ATR 约等于 99.7% 置信区间 (正态分布下 3sigma)
    # 在高波动率时自动放宽止损，低波动率时自动收紧
    ATR_MULTIPLIER = 3.0 
    
    STOP_LOSS_MULTIPLIER = 5.0 # (备用，兼容旧逻辑)
    MELTDOWN_DIRECTION = 'down'

    # 基础风控
    TARGET_VOLATILITY = 0.20
    MAX_LEVERAGE = 2.0
    
    # ==========================================
    # 5. 回测仿真 (Simulation)
    # ==========================================
    POSITION_BUFFER = 0.10 
    FEE_RATE = 0.0005
    INITIAL_CAPITAL = 10000.0