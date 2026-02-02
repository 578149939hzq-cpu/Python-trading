import os

class Config:
    """
    Jarvis-Code 全局配置中心 (V3.2 Simplified Edition)
    回归原始信号，宽幅 ATR 动态风控。
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
    
    # 保持 480 (长期稳健定仓)
    VOL_LOOKBACK = 480

    # ==========================================
    # 3. 风险引擎 V3.2 (Relaxed ATR Risk)
    # ==========================================
    RISK_METRIC = 'ATR'
    ATR_WINDOW = 24
    
    # [Config Change] 放宽至 6.0
    # 6倍 ATR 是一个极宽的防线，意味着只有发生巨大的结构性破坏时才离场
    # 避免被常规噪音止损
    ATR_MULTIPLIER = 6.0 
    
    MELTDOWN_DIRECTION = 'down'

    # 基础风控
    TARGET_VOLATILITY = 0.20
    MAX_LEVERAGE = 2.0
    
    # ==========================================
    # 4. 回测仿真 (Simulation)
    # ==========================================
    POSITION_BUFFER = 0.10 
    FEE_RATE = 0.0005
    INITIAL_CAPITAL = 10000.0