import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 🆕 引入配置文件
from config import Config

def load_price_data(csv_path: str) -> pd.DataFrame:
    # ... (保持你原有的加载代码不变，非常完美) ...
    # 1. 初次尝试读取
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return pd.DataFrame()

    if len(df) > 0 and ("http" in str(df.columns[0]) or "www" in str(df.columns[0])):
        df = pd.read_csv(csv_path, skiprows=1, low_memory=False)

    df.columns = [c.strip().lower() for c in df.columns]
    
    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"])
    elif "unix" in df.columns:
        df["unix"] = pd.to_numeric(df["unix"], errors='coerce')
        max_ts = df["unix"].max()
        if pd.isna(max_ts) or max_ts == 0:
            return pd.DataFrame()
        if max_ts > 1e14: unit = 'us'
        elif max_ts > 1e11: unit = 'ms'
        else: unit = 's'
        df["time"] = pd.to_datetime(df["unix"], unit=unit)
    elif "date" in df.columns:
         df["time"] = pd.to_datetime(df["date"])
    else:
        return pd.DataFrame() 

    df = df.set_index("time").sort_index()
    df = df[df.index > pd.to_datetime("2010-01-01")]
    
    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"文件 {csv_path} 缺少列: {col}")
    if "volume" not in df.columns and "vol" in df.columns:
        df["volume"] = df["vol"]

    df["ret"] = df["close"].pct_change().fillna(0)
    return df

def calculate_scaled_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    🔥 核心升级: 基于 Config 参数的 EWMAC (均线交叉) 策略
    
    逻辑链条:
    1. 计算波动率 (分母)
    2. 计算 4 组均线交叉 (分子)
    3. 乘以对应的 Scalar (缩放)
    4. 加权平均 (集成)
    """
    data = df.copy()
    
    # ==========================================
    # 🧠 步骤 A: 计算波动率 (Standard Deviation)
    # ==========================================
    # 使用 Config 中的窗口 (通常是 36)
    # 物理意义：风险标尺。
    vol_span = Config.VOL_LOOKBACK
    data['volatility'] = data['close'].ewm(span=vol_span).std()
    
    # 防除零保护 (加上一个极小值)
    data['volatility'] = data['volatility'].replace(0, np.nan).fillna(method='ffill') + 1e-8
    
    # ==========================================
    # ⚡ 步骤 B: 循环计算 4 个子策略
    # ==========================================
    fast_spans = Config.STRATEGY_PARAMS['fast_span']
    slow_spans = Config.STRATEGY_PARAMS['slow_span']
    scalars = Config.STRATEGY_PARAMS['scalars']
    weights = Config.WEIGHTS
    
    # 用于存储各子策略的"标准化 Forecast"
    forecast_cols = []
    
    print(f"🔄 正在计算 {len(fast_spans)} 组 EWMAC 策略...")
    
    for i in range(len(fast_spans)):
        fast = fast_spans[i]
        slow = slow_spans[i]
        scalar = scalars[i]
        
        # 1. 计算快慢均线
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        
        # 2. 原始交叉值 (Raw Cross) = 快线 - 慢线
        raw_cross = ema_fast - ema_slow
        
        # 3. 标准化预测 (Scaled Forecast)
        # 公式: (快 - 慢) * Scalar / 波动率
        # 含义: 当前的均线差值，相当于多少倍的日波动率？
        col_name = f'fc_{fast}_{slow}'
        data[col_name] = (raw_cross * scalar) / data['volatility']
        
        forecast_cols.append(col_name)
        # print(f"   ✅ 策略 {fast}/{slow}: Scalar={scalar}")

    # ==========================================
    # ⚖️ 步骤 C: 集成 (Ensemble)
    # ==========================================
    # 加权平均
    # 这里的 weights 都在 Config 里 (0.25, 0.25, 0.25, 0.25)
    combined_forecast = data[forecast_cols].mul(weights).sum(axis=1)
    
    # ==========================================
    # 🛡️ 步骤 D: 封顶 (Capping)
    # ==========================================
    # Carver 建议单个策略通常限制在 +/- 20 之间
    data['forecast'] = combined_forecast.clip(lower=-20.0, upper=20.0).fillna(0)
    
    # 记录一些调试信息
    data['ema_slow_base'] = data['close'].ewm(span=slow_spans[-1]).mean() # 画图用最慢的线
    
    return data

# ... (run_vectorized_backtest 和 calculate_position_target 保持不变) ...
def run_vectorized_backtest(df:pd.DataFrame,fee_rate=0.0005)->pd.DataFrame:

    # 保持原样
    data=df.copy()
    data["market_log_ret"]=np.log(data['close']).diff().fillna(0)
    data['strategy_log_ret']=data['position']*data['market_log_ret']
    position_change=data['position'].diff().abs().fillna(0)
    data['cost']=position_change*fee_rate
    data['net_log_ret']=data['strategy_log_ret']-data['cost']
    data['equity'] = np.exp(data['net_log_ret'].cumsum())
    data['buy_hold_equity']=np.exp(data['market_log_ret'].cumsum())
    return data

def calculate_position_target(df: pd.DataFrame, forecast_col='forecast', buffer=0.1) -> pd.DataFrame:
    """
    [Risk Engine V2.0]
    1. 统一波动率计算 (Return Volatility)
    2. 动态杠杆 (Vol-Targeting)
    3. Sigma 熔断 (Safety Airbag)
    """
    # 局部引入 Config，确保能读取到 main.py 中注入的最新参数
    from config import Config

    data = df.copy()

    # ==========================================
    # 1. 统一波动率计算 (The Right Way)
    # ==========================================
    # 直接计算"收益率"的波动率，而非价格的标准差
    hourly_ret = data['close'].pct_change().fillna(0)

    # 使用 config 中的长周期 (默认168=一周) 计算稳健波动率
    # 注意：这里得到的是"小时级标准差" (Hourly Sigma)
    hourly_sigma = hourly_ret.ewm(span=Config.VOL_LOOKBACK).std().fillna(0)

    # 转化为年化波动率 (用于计算杠杆)
    # Annual Vol = Hourly Sigma * sqrt(8760)
    data['ann_vol_pct'] = hourly_sigma * np.sqrt(365 * 24)

    # ==========================================
    # 2. 计算动态杠杆 (Dynamic Leverage)
    # ==========================================
    # 避免除以零
    safe_vol = data['ann_vol_pct'].replace(0, 1e-6)

    # 公式：目标波动率 / 当前波动率
    # 如果目标是20%，当前波动率是10%，则上2倍杠杆
    leverage_ratio = Config.TARGET_VOLATILITY / safe_vol

    # 封顶：不超过最大允许杠杆 (例如 2.0x)
    leverage_ratio = leverage_ratio.clip(upper=Config.MAX_LEVERAGE)
    data['leverage_ratio'] = leverage_ratio

    # ==========================================
    # 3. 基础目标仓位
    # ==========================================
    # 归一化预测值 (-1 ~ 1)
    raw_position = data[forecast_col] / 20.0

    # 叠加杠杆
    ideal_position = raw_position * leverage_ratio

    # ==========================================
    # 4. Sigma 熔断机制 (Safety Airbag) !!!
    # ==========================================
    # 计算当前的容忍上限：N倍标准差
    # 如果当前这一小时跌幅超过了 3倍的历史平均波动，说明市场流动性枯竭
    sigma_limit = hourly_sigma * Config.SIGMA_THRESHOLD

    # 标记熔断时刻
    # abs(hourly_ret) 代表无论暴涨还是暴跌，只要剧烈波动就熔断
    meltdown_mask = abs(hourly_ret) > sigma_limit

    # 记录熔断事件 (供诊断绘图用)
    data['sigma_event'] = meltdown_mask

    # ⚡️ 强制清仓
    # 在熔断时刻，将目标仓位强行设为 0
    ideal_position = np.where(meltdown_mask, 0.0, ideal_position)

    # 再次截断最终仓位 (防止逻辑漏洞)
    ideal_position = np.clip(ideal_position, -Config.MAX_LEVERAGE, Config.MAX_LEVERAGE)

    # ==========================================
    # 5. 缓冲器 (Buffer)
    # ==========================================
    ideal_values = ideal_position
    n = len(ideal_values)
    buffered_position = np.zeros(n)
    current_pos = 0.0

    for i in range(n):
        ideal = ideal_values[i]
        # 只有当新目标和当前持仓的差距超过 buffer 时，才调仓
        if abs(ideal - current_pos) > buffer:
            current_pos = ideal
        buffered_position[i] = current_pos

    data['raw_target'] = ideal_position
    data['buffered_pos'] = buffered_position
    # Shift(1) 代表“下根K线执行”，防止未来函数
    data['position'] = data['buffered_pos'].shift(1).fillna(0)

    return data