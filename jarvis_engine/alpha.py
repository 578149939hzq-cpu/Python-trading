import pandas as pd
import numpy as np
from config import Config

def load_price_data(csv_path: str) -> pd.DataFrame:
    """
    加载并清洗数据 (Standard Loading)
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
    # 补全 OHLC
    for c in ['open', 'high', 'low', 'close']:
        if c not in df.columns: df[c] = df['close']
            
    return df

# ==========================================
# [Core Upgrade A] 预处理降噪层
# ==========================================
def denoise_price_data(df: pd.DataFrame, window=24, threshold=5.0) -> pd.DataFrame:
    """
    Alpha Lab 降噪算法 (MAD Outlier Filter)
    -------------------------------------------------------
    金融逻辑: 
    Crypto 市场常出现 "Flash Crash" (瞬间插针) 或 "Scam Wick"。
    这些非理性的价格突变会破坏均线系统，导致频繁的虚假信号。
    本函数使用稳健统计学方法，识别并压平这些噪音。
    
    算法:
    1. 计算价格的中位数趋势 (Rolling Median)。
    2. 计算绝对偏差 (Absolute Deviation)。
    3. 识别偏离中位数超过 N 倍 MAD 的点。
    4. Winsorization: 将离群点替换为边界值，而非删除。
    """
    df_clean = df.copy()
    close = df_clean['close']
    
    # 1. 稳健趋势 (Robust Trend)
    roll_median = close.rolling(window=window).median()
    
    # 2. 绝对偏差 (Robust Dispersion)
    abs_diff = abs(close - roll_median)
    
    # 3. MAD (Median Absolute Deviation)
    mad = abs_diff.rolling(window=window).median()
    
    # 4. 动态边界 (Dynamic Bounds)
    # 允许价格在 [Median - K*MAD, Median + K*MAD] 范围内波动
    upper_bound = roll_median + threshold * mad
    lower_bound = roll_median - threshold * mad
    
    # 5. 去噪动作 (Winsorization)
    # 逻辑: 如果价格向上插针，强制压回上轨；向下插针，强制托回下轨。
    df_clean['close'] = np.where(close > upper_bound, upper_bound, 
                                 np.where(close < lower_bound, lower_bound, close))
    
    # 同步修正 High/Low，防止出现 High < Close 的逻辑错误
    if 'high' in df_clean.columns:
        df_clean['high'] = np.where(df_clean['high'] > upper_bound, upper_bound, df_clean['high'])
    if 'low' in df_clean.columns:
        df_clean['low'] = np.where(df_clean['low'] < lower_bound, lower_bound, df_clean['low'])
        
    return df_clean

def calculate_scaled_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Refactored] 集成去噪层的信号生成器
    """
    # 1. 调用降噪层 (Preprocessing)
    # 使用清洗后的数据来计算均线，信号更纯净
    clean_df = denoise_price_data(df, window=Config.MAD_WINDOW, threshold=Config.MAD_THRESHOLD)
    
    # 我们将在 clean_df 上计算指标，最后将 forecast 赋值回原 df
    data = df.copy()
    
    # [Core Upgrade C] 长期稳健波动率
    vol_span = getattr(Config, 'VOL_LOOKBACK', 480) 
    clean_df['volatility'] = clean_df['close'].ewm(span=vol_span).std().replace(0, np.nan).fillna(method='ffill') + 1e-8
    
    fast_spans = Config.STRATEGY_PARAMS['fast_span']
    slow_spans = Config.STRATEGY_PARAMS['slow_span']
    scalars = Config.STRATEGY_PARAMS['scalars']
    weights = Config.WEIGHTS
    
    forecast_cols = []
    for i in range(len(fast_spans)):
        fast, slow, scalar = fast_spans[i], slow_spans[i], scalars[i]
        # 使用去噪后的价格计算均线交叉
        raw = clean_df['close'].ewm(span=fast).mean() - clean_df['close'].ewm(span=slow).mean()
        col = f'fc_{fast}_{slow}'
        # 信号归一化
        clean_df[col] = (raw * scalar) / clean_df['volatility']
        forecast_cols.append(col)

    combined = clean_df[forecast_cols].mul(weights).sum(axis=1)
    
    # 将计算好的纯净信号注入回原始数据流
    data['forecast'] = combined.clip(-20, 20).fillna(0)
    # 同时也保存去噪后的波动率，供后续参考
    data['volatility'] = clean_df['volatility']
    
    return data

def calculate_position_target(df: pd.DataFrame, forecast_col='forecast', buffer=0.1) -> pd.DataFrame:
    """
    [Core Upgrade B] 集成 ATR 动态止损的持仓计算
    """
    data = df.copy()
    
    # ==========================================
    # 1. 稳健杠杆计算 (Stable Sizing)
    # ==========================================
    # 使用 Config.VOL_LOOKBACK (480) 计算长期波动率
    hourly_ret = data['close'].pct_change().fillna(0)
    long_term_vol = hourly_ret.ewm(span=Config.VOL_LOOKBACK).std().fillna(0)
    
    # 年化
    ann_vol_pct = long_term_vol * np.sqrt(365 * 24)
    data['ann_vol_pct'] = ann_vol_pct # For diagnostics
    
    safe_vol = ann_vol_pct.replace(0, 1e-6)
    leverage_ratio = (Config.TARGET_VOLATILITY / safe_vol).clip(upper=Config.MAX_LEVERAGE)
    data['leverage_ratio'] = leverage_ratio
    
    # 基础目标仓位
    ideal_position = (data[forecast_col] / 20.0) * leverage_ratio
    
    # ==========================================
    # 2. ATR 动态风控 (Dynamic Risk)
    # ==========================================
    # 金融逻辑: ATR (平均真实波幅) 代表了市场当前的"呼吸节奏"。
    # 止损阈值不应是固定的 %，而应是 ATR 的倍数。
    # 波动大时，止损放宽，避免被正常波动洗出；波动小时，止损收紧，快速截断亏损。
    
    # A. 计算 True Range
    high = data['high']
    low = data['low']
    close = data['close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    # 逐元素取最大值
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # B. 计算 ATR (Window=24)
    atr = tr.ewm(span=Config.ATR_WINDOW).mean()
    
    # C. 计算动态止损阈值 (百分比)
    # Stop Threshold % = (ATR * Multiplier) / Price
    atr_threshold_pct = (atr * Config.ATR_MULTIPLIER) / close
    
    # 更新系统阈值 (这将直接影响回测中的 Gap 修正)
    data['sl_threshold'] = atr_threshold_pct
    data['vol_raw'] = atr / close # 记录原始波动率单位(%)
    
    # ==========================================
    # 3. 执行风控 (Intraday Stop & Meltdown)
    # ==========================================
    # 使用 ATR 阈值进行判断
    
    # A. 瞬时止损 (Low Price Check)
    if 'low' in data.columns and 'open' in data.columns:
        intraday_drop = (data['low'] - data['open']) / data['open']
        stop_loss_mask = intraday_drop < -atr_threshold_pct
    else:
        stop_loss_mask = pd.Series(False, index=data.index)
        
    # B. 单向熔断 (Close Price Check)
    meltdown_dir = getattr(Config, 'MELTDOWN_DIRECTION', 'down')
    if meltdown_dir == 'down':
        meltdown_mask = hourly_ret < -atr_threshold_pct
    else:
        meltdown_mask = abs(hourly_ret) > atr_threshold_pct
    
    # 触发风控
    risk_event = meltdown_mask | stop_loss_mask
    
    data['sigma_event'] = risk_event
    data['is_meltdown'] = meltdown_mask
    data['is_stop_loss'] = stop_loss_mask
    
    # 仓位清零
    ideal_position = np.where(risk_event, 0.0, ideal_position)
    ideal_position = ideal_position.clip(-Config.MAX_LEVERAGE, Config.MAX_LEVERAGE)
    
    # ==========================================
    # 4. 缓冲器 (Buffer)
    # ==========================================
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

def run_vectorized_backtest(df: pd.DataFrame, fee_rate=0.0005) -> pd.DataFrame:
    """
    [Backtest Engine] ATR 适配版
    """
    data = df.copy()
    data["market_log_ret"] = np.log(data['close']).diff().fillna(0)
    
    # 1. 复制市场回报
    adjusted_market_ret = data['market_log_ret'].copy()
    
    # 2. 修正风控时刻的回报 (ATR Gap Correction)
    risk_mask = data.get('sigma_event', False)
    
    if risk_mask.any():
        # 获取 ATR 动态阈值
        sl_values = data.loc[risk_mask, 'sl_threshold']
        
        prev_close = data.loc[risk_mask, 'close'].shift(1)
        open_price = data.loc[risk_mask, 'open']
        
        # 止损价 = 开盘价 * (1 - ATR_Threshold%)
        stop_price = open_price * (1.0 - sl_values)
        
        correction = np.log(stop_price / prev_close)
        correction = correction.fillna(adjusted_market_ret.loc[risk_mask])
        adjusted_market_ret.loc[risk_mask] = correction
        
    # 3. 计算策略回报
    data['strategy_log_ret'] = data['position'] * adjusted_market_ret
    position_change = data['position'].diff().abs().fillna(0)
    
    data['cost'] = position_change * fee_rate
    data['net_log_ret'] = data['strategy_log_ret'] - data['cost']
    
    # 4. 资金映射
    initial_cap = getattr(Config, 'INITIAL_CAPITAL', 10000.0)
    norm_equity = np.exp(data['net_log_ret'].cumsum())
    norm_bh_equity = np.exp(data['market_log_ret'].cumsum())
    
    data['equity'] = norm_equity * initial_cap
    data['buy_hold_equity'] = norm_bh_equity * initial_cap
    
    return data