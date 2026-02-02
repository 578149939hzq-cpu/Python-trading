import os

class Config:
    # ==========================================
    # 1. 基础设施 (Infrastructure)
    # ==========================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 自动识别：优先找 data_raw (你的实际文件夹)，找不到再找 data/raw
    if os.path.exists(os.path.join(BASE_DIR, "data_raw")):
        DATA_PATH = os.path.join(BASE_DIR, "data_raw", "Binance_BTCUSDT_1h.csv")
    else:
        DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Binance_BTCUSDT_1h.csv")

    # ==========================================
    # 2. 策略参数 (Alpha Params)
    # ==========================================
    STRATEGY_PARAMS = {
        'fast_span': [8, 16, 32, 64],
        'slow_span': [32, 64, 128, 256],
        'scalars': [5.6, 3.8, 2.6, 1.9]
    }
    WEIGHTS = [0.25, 0.25, 0.25, 0.25]
    VOL_LOOKBACK = 36  # 1周 (更稳健的波动率基准)
    
    # ==========================================
    # 3. 执行参数 (Execution)
    # ==========================================
    POSITION_BUFFER = 0.10
    FEE_RATE = 0.0005
    
    # ==========================================
    # 5. 风险引擎 V3.0 (Robust Risk Engine)
    # ==========================================
    # 核心指标选择: 'MAD' (推荐), 'ATR', 'STD', 'QUANTILE'
    RISK_METRIC = 'MAD' 
    
    # 稳健观察窗口 (1周 = 168小时)
    # 相比 V2.1 的 168，这里专门用于计算中位数/分位数，样本量需足够
    MAD_WINDOW = 168  
    
    # 止损乘数 (配合 MAD 使用)
    # MAD * 5.0 ≈ 3.3 Sigma (正态分布下 1 Sigma ≈ 1.4826 MAD)
    # 5.0 是一个非常稳健的防插针阈值
    STOP_LOSS_MULTIPLIER = 5.0
    
    # ATR 乘数 (如果使用 ATR 模式)
    ATR_MULTIPLIER = 3.0
    ATR_WINDOW = 24