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
    [Rollback] 原始信号生成器
    移除 MAD 去噪层，直接使用 Raw Close 计算均线。
    """
    data = df.copy()
    
    # 1. 长期波动率 (用于信号归一化)
    vol_span = getattr(Config, 'VOL_LOOKBACK', 480) 
    data['volatility'] = data['close'].ewm(span=vol_span).std().replace(0, np.nan).fillna(method='ffill') + 1e-8
    
    fast_spans = Config.STRATEGY_PARAMS['fast_span']
    slow_spans = Config.STRATEGY_PARAMS['slow_span']
    scalars = Config.STRATEGY_PARAMS['scalars']
    weights = Config.WEIGHTS
    
    forecast_cols = []
    for i in range(len(fast_spans)):
        fast, slow, scalar = fast_spans[i], slow_spans[i], scalars[i]
        # 直接使用原始 Close
        raw = data['close'].ewm(span=fast).mean() - data['close'].ewm(span=slow).mean()
        col = f'fc_{fast}_{slow}'
        # 信号归一化
        data[col] = (raw * scalar) / data['volatility']
        forecast_cols.append(col)

    combined = data[forecast_cols].mul(weights).sum(axis=1)
    
    data['forecast'] = combined.clip(-20, 20).fillna(0)
    
    return data

def calculate_position_target(df: pd.DataFrame, forecast_col='forecast', buffer=0.1) -> pd.DataFrame:
    """
    [Simplified] 宽幅 ATR 风控 (Closing Basis)
    移除 Intraday Stop，仅使用收盘价确认风险。
    """
    data = df.copy()
    
    # ==========================================
    # 1. 稳健杠杆计算
    # ==========================================
    hourly_ret = data['close'].pct_change().fillna(0)
    long_term_vol = hourly_ret.ewm(span=Config.VOL_LOOKBACK).std().fillna(0)
    
    ann_vol_pct = long_term_vol * np.sqrt(365 * 24)
    data['ann_vol_pct'] = ann_vol_pct
    
    safe_vol = ann_vol_pct.replace(0, 1e-6)
    leverage_ratio = (Config.TARGET_VOLATILITY / safe_vol).clip(upper=Config.MAX_LEVERAGE)
    data['leverage_ratio'] = leverage_ratio
    
    ideal_position = (data[forecast_col] / 20.0) * leverage_ratio
    
    # ==========================================
    # 2. 宽幅 ATR 动态阈值 (Wide Dynamic Threshold)
    # ==========================================
    high = data['high']
    low = data['low']
    close = data['close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR Window = 24 (保持灵敏度)
    atr = tr.ewm(span=getattr(Config, 'ATR_WINDOW', 24)).mean()
    
    # Threshold = (ATR * 6.0) / Price
    # 6倍 ATR 意味着极度宽容，允许市场大幅震荡
    atr_threshold_pct = (atr * Config.ATR_MULTIPLIER) / close
    
    data['sl_threshold'] = atr_threshold_pct
    data['vol_raw'] = atr / close 
    
    # ==========================================
    # 3. 收盘确认风控 (Closing Basis Check Only)
    # ==========================================
    # [Rollback] 移除了对 Low 价格的瞬时检测
    # 只有当小时收盘价跌幅超过阈值时，才触发熔断
    
    meltdown_dir = getattr(Config, 'MELTDOWN_DIRECTION', 'down')
    if meltdown_dir == 'down':
        # 只看收盘跌幅
        risk_event = hourly_ret < -atr_threshold_pct
    else:
        risk_event = abs(hourly_ret) > atr_threshold_pct
    
    data['sigma_event'] = risk_event
    data['is_meltdown'] = risk_event # 简化：现在的 sigma_event 就是 meltdown
    data['is_stop_loss'] = pd.Series(False, index=data.index) # 瞬时止损已禁用
    
    # 触发熔断 -> 归零
    ideal_position = np.where(risk_event, 0.0, ideal_position)
    ideal_position = ideal_position.clip(-Config.MAX_LEVERAGE, Config.MAX_LEVERAGE)
    
    # ==========================================
    # 4. 缓冲器
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
    [Backtest Engine] 保持 ATR Gap 修正逻辑
    """
    data = df.copy()
    data["market_log_ret"] = np.log(data['close']).diff().fillna(0)
    
    adjusted_market_ret = data['market_log_ret'].copy()
    
    # 修正风控时刻的回报
    # 虽然是收盘确认，但我们依然假设在触发阈值的那个点位附近成交了（或者是收盘价）
    # 为保守起见，依然保留 Gap Correction 逻辑
    risk_mask = data.get('sigma_event', False)
    
    if risk_mask.any():
        sl_values = data.loc[risk_mask, 'sl_threshold']
        prev_close = data.loc[risk_mask, 'close'].shift(1)
        open_price = data.loc[risk_mask, 'open']
        
        # Stop Price
        stop_price = open_price * (1.0 - sl_values)
        
        correction = np.log(stop_price / prev_close)
        correction = correction.fillna(adjusted_market_ret.loc[risk_mask])
        adjusted_market_ret.loc[risk_mask] = correction
        
    data['strategy_log_ret'] = data['position'] * adjusted_market_ret
    position_change = data['position'].diff().abs().fillna(0)
    
    data['cost'] = position_change * fee_rate
    data['net_log_ret'] = data['strategy_log_ret'] - data['cost']
    
    initial_cap = getattr(Config, 'INITIAL_CAPITAL', 10000.0)
    norm_equity = np.exp(data['net_log_ret'].cumsum())
    norm_bh_equity = np.exp(data['market_log_ret'].cumsum())
    
    data['equity'] = norm_equity * initial_cap
    data['buy_hold_equity'] = norm_bh_equity * initial_cap
    
    return data