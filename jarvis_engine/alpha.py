import pandas as pd
import numpy as np
from config import Config

def load_price_data(csv_path: str) -> pd.DataFrame:
    """
    加载并清洗数据
    """
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
        unit = 'ms'
        if df["unix"].max() > 1e14: unit = 'us'
        elif df["unix"].max() < 1e11: unit = 's'
        df["time"] = pd.to_datetime(df["unix"], unit=unit)
    elif "date" in df.columns:
         df["time"] = pd.to_datetime(df["date"])
    else:
        return pd.DataFrame() 

    df = df.set_index("time").sort_index()
    for c in ['open', 'high', 'low', 'close']:
        if c not in df.columns: df[c] = df['close']
            
    return df

def calculate_scaled_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    [V4.3 The Silence Protocol] 深度平滑混合信号
    --------------------------------------------
    1. Trend: 保持不变
    2. RSI: 实施 "Soft Deadzone" + "Deep Smoothing"，消除跳变。
    """
    data = df.copy()
    
    # --- 1. 基础数据准备 ---
    vol_span = getattr(Config, 'VOL_LOOKBACK', 480) 
    data['volatility'] = data['close'].ewm(span=vol_span).std().replace(0, np.nan).fillna(method='ffill') + 1e-8
    
    # --- 2. 计算趋势信号 (Trend Component) ---
    fast_spans = Config.STRATEGY_PARAMS['fast_span']
    slow_spans = Config.STRATEGY_PARAMS['slow_span']
    scalars = Config.STRATEGY_PARAMS['scalars']
    weights = getattr(Config, 'TREND_INTERNAL_WEIGHTS', [0.25, 0.25, 0.25, 0.25])
    
    forecast_cols = []
    for i in range(len(fast_spans)):
        fast, slow, scalar = fast_spans[i], slow_spans[i], scalars[i]
        raw = data['close'].ewm(span=fast).mean() - data['close'].ewm(span=slow).mean()
        col = f'fc_{fast}_{slow}'
        data[col] = (raw * scalar) / data['volatility']
        forecast_cols.append(col)

    trend_forecast = data[forecast_cols].mul(weights).sum(axis=1)
    
    # --- 3. [V4.3] 深度平滑版 RSI 反转信号 ---
    rsi_period = getattr(Config, 'RSI_PERIOD', 14)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsi_period, adjust=False).mean()
    rs = gain / loss
    raw_rsi = 100 - (100 / (1 + rs))
    
    # [V4.3 优化 A] 加强输入平滑 (3 -> 12小时)
    # 彻底过滤掉短期的 RSI 噪点
    smooth_rsi = raw_rsi.rolling(window=12).mean().fillna(50)
    
    rsi_diff = 50 - smooth_rsi
    rsi_scalar = getattr(Config, 'RSI_SCALAR', 1.0)
    
    # [V4.3 优化 B] 软死区逻辑 (Soft Deadzone)
    # 替换之前的 np.where 硬截断。
    # 逻辑: 当 abs(diff) <= 10 时，输出 0。
    #       当 abs(diff) > 10 时，输出 (abs(diff) - 10) * sign。
    # 效果: 信号从 0 线性爬升，不再突变，解决 Buffer 频繁触发问题。
    # 修复: np.maximum 和 np.sign 返回的是 Series，不会报 ndarray 错误。
    rsi_forecast = np.sign(rsi_diff) * np.maximum(0, rsi_diff.abs() - 10) * rsi_scalar
    
    # [V4.3 优化 C] 输出信号再次平滑
    # 对最终生成的 Forecast 再做一次 EMA，确保曲线像丝绸一样顺滑
    rsi_forecast = rsi_forecast.ewm(span=24).mean()
    
    # --- 4. 信号融合 ---
    # 使用配置的权重，默认倾向于趋势 (0.9/0.1 or 0.8/0.2)
    w_trend = getattr(Config, 'TREND_WEIGHT', 0.9)
    w_rsi = getattr(Config, 'RSI_WEIGHT', 0.1)
    
    data['trend_forecast'] = trend_forecast.clip(-20, 20).fillna(0)
    data['rsi_forecast'] = rsi_forecast.clip(-20, 20).fillna(0)
    
    # 混合输出
    data['forecast'] = (data['trend_forecast'] * w_trend + data['rsi_forecast'] * w_rsi)
    
    return data

def calculate_position_target(df: pd.DataFrame, forecast_col='forecast', buffer=0.1) -> pd.DataFrame:
    """
    [Risk Engine V4.0] 环境感知型风控 (保持不变)
    Regime Filter + Vol Scaling + Survival Stop
    """
    data = df.copy()
    
    # --- 1. 环境过滤器 (Regime Filter) ---
    ma_window = getattr(Config, 'REGIME_MA_WINDOW', 4800)
    regime_ma = data['close'].rolling(window=ma_window).mean()
    is_bull_regime = data['close'] > regime_ma
    
    # 动态杠杆上限
    normal_cap = getattr(Config, 'MAX_LEVERAGE', 2.5)      
    bear_cap = getattr(Config, 'BEAR_MODE_MAX_LEVERAGE', 1.0) 
    
    dynamic_max_cap = np.where(is_bull_regime, normal_cap, bear_cap)
    data['regime_ma'] = regime_ma 
    data['dynamic_max_cap'] = dynamic_max_cap
    
    # --- 2. 波动率目标管理 (Vol Scaling) ---
    hourly_ret = data['close'].pct_change().fillna(0)
    long_term_vol = hourly_ret.ewm(span=Config.VOL_LOOKBACK).std().fillna(0)
    ann_vol_pct = long_term_vol * np.sqrt(365 * 24)
    data['ann_vol_pct'] = ann_vol_pct
    
    safe_vol = ann_vol_pct.replace(0, 1e-6)
    target_vol = getattr(Config, 'TARGET_VOLATILITY', 0.8)
    raw_leverage_ratio = (target_vol / safe_vol)
    
    # --- 3. 仓位计算 ---
    # [V3.7] 信号增强: / 10.0
    ideal_position = (data[forecast_col] / 2.0) * raw_leverage_ratio
    
    # [V4.0] 应用动态环境上限
    ideal_position = ideal_position.clip(-dynamic_max_cap, dynamic_max_cap)
    data['leverage_ratio'] = ideal_position.abs()
    
    # --- 4. 灾难阻断器 (Survival Hard Stop) [V3.3] ---
    h, l, c = data['high'], data['low'], data['close']
    pc = c.shift(1).fillna(c)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    
    atr_window = getattr(Config, 'SURVIVAL_ATR_WINDOW', 24)
    atr = tr.ewm(span=atr_window).mean().fillna(0)
    
    multiplier = getattr(Config, 'SURVIVAL_ATR_MULTIPLIER', 4.5)
    min_vol = getattr(Config, 'MIN_HOURLY_VOL', 0.005) 
    
    raw_threshold = (atr * multiplier) / c
    crash_threshold = np.maximum(raw_threshold, min_vol * multiplier)
    
    is_crash = hourly_ret < -crash_threshold
    
    # 熔断归零
    ideal_position = np.where(is_crash, 0.0, ideal_position)
    
    data['sl_threshold'] = crash_threshold
    data['sigma_event'] = is_crash
    data['is_meltdown'] = is_crash
    
    # --- 5. 缓冲器 (Buffer) ---
    ideal_values = ideal_position
    n = len(ideal_values)
    buffered_position = np.zeros(n)
    current_pos = 0.0
    for i in range(n):
        if abs(ideal_values[i] - current_pos) > buffer:
            current_pos = ideal_values[i]
        buffered_position[i] = current_pos
        
    data['raw_target'] = ideal_position
    data['buffered_pos'] = buffered_position
    data['position'] = data['buffered_pos'].shift(1).fillna(0)
    
    return data

def run_vectorized_backtest(df: pd.DataFrame, fee_rate=0.0005, funding_rate=0.00001):
    data = df.copy()
    
    # [V4.8] 放弃 Log Returns，改用 Simple Returns
    # 计算基础的市场百分比变化 (Simple Return)
    data['market_ret'] = data['close'].pct_change().fillna(0)
    
    # 灾难止损修正 (使用你之前的劣后成交逻辑，但改为简单收益率)
    risk_mask = data.get('sigma_event', False)
    adjusted_ret = data['market_ret'].copy()
    if risk_mask.any():
        sl_values = data.loc[risk_mask, 'sl_threshold']
        open_price = data.loc[risk_mask, 'open']
        close_price = data.loc[risk_mask, 'close']
        prev_close = data['close'].shift(1).loc[risk_mask]
        
        # 劣后成交：取 理论止损价 和 实际收盘价 的最小值，并扣除 0.5% 极端滑点
        execution_price = np.minimum(open_price * (1.0 - sl_values), close_price) * 0.995
        adjusted_ret.loc[risk_mask] = (execution_price / prev_close) - 1.0

    # 策略毛收益 = 仓位 * 调整后的市场收益
    data['strat_ret_raw'] = data['position'] * adjusted_ret
    
    # 费用计算
    pos_change = data['position'].diff().abs().fillna(0)
    transaction_cost = pos_change * fee_rate
    funding_cost = data['position'].abs() * funding_rate
    
    # 净收益率
    data['net_ret'] = data['strat_ret_raw'] - transaction_cost - funding_cost
    
    # [V4.8 核心修改] 模拟真实资金曲线 (逐行相乘，而非对数累加)
    # 这样能更准确地反映大杠杆下的损耗
    initial_cap = Config.INITIAL_CAPITAL
    data['equity'] = initial_cap * (1 + data['net_ret']).cumprod()
    data['buy_hold_equity'] = initial_cap * (1 + data['market_ret']).cumprod()
    
    # 为了兼容以前的统计函数，保留 net_log_ret
    data['net_log_ret'] = np.log(1 + data['net_ret'])
    data['market_log_ret'] = np.log(1 + data['market_ret'])
    
    return data
   