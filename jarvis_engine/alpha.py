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
    原始信号生成器
    """
    data = df.copy()
    
    # 长期波动率 (用于信号归一化)
    vol_span = getattr(Config, 'VOL_LOOKBACK', 480) 
    data['volatility'] = data['close'].ewm(span=vol_span).std().replace(0, np.nan).fillna(method='ffill') + 1e-8
    
    fast_spans = Config.STRATEGY_PARAMS['fast_span']
    slow_spans = Config.STRATEGY_PARAMS['slow_span']
    scalars = Config.STRATEGY_PARAMS['scalars']
    weights = Config.WEIGHTS
    
    forecast_cols = []
    for i in range(len(fast_spans)):
        fast, slow, scalar = fast_spans[i], slow_spans[i], scalars[i]
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
    [Risk Engine V3.7] 信号增强版
    """
    data = df.copy()
    
    # 1. 波动率目标管理
    hourly_ret = data['close'].pct_change().fillna(0)
    long_term_vol = hourly_ret.ewm(span=Config.VOL_LOOKBACK).std().fillna(0)
    ann_vol_pct = long_term_vol * np.sqrt(365 * 24)
    data['ann_vol_pct'] = ann_vol_pct
    
    safe_vol = ann_vol_pct.replace(0, 1e-6)
    leverage_ratio = (Config.TARGET_VOLATILITY / safe_vol).clip(upper=Config.MAX_LEVERAGE)
    data['leverage_ratio'] = leverage_ratio
    
    # ==========================================
    # 2. [V3.7 Update] 信号增强 (Signal Boosting)
    # ==========================================
    # 旧逻辑: / 20.0 (过于保守，导致平均杠杆极低)
    # 新逻辑: / 10.0 (将 Forecast=10 视为满确信度)
    # 效果: 在同等预测值下，基础仓位翻倍，显著提高资金利用率
    ideal_position = (data[forecast_col] / 5.0) * leverage_ratio
    
    # 硬性杠杆限制
    ideal_position = ideal_position.clip(-Config.MAX_LEVERAGE, Config.MAX_LEVERAGE)
    
    # 3. 灾难阻断器 (Survival Hard Stop)
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
    
    # 触发熔断
    ideal_position = np.where(is_crash, 0.0, ideal_position)
    
    data['sl_threshold'] = crash_threshold
    data['sigma_event'] = is_crash
    data['is_meltdown'] = is_crash
    
    # 4. 缓冲器
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

def run_vectorized_backtest(df: pd.DataFrame, fee_rate=0.0005, funding_rate=0.00001) -> pd.DataFrame:
    """
    [Backtest Engine V3.7] 引入资金费率 (Funding Cost)
    ----------------------------------------------------
    funding_rate: 默认 0.00005 (万分之五/小时), 约等于年化 40% 的持仓成本。
    这模拟了在牛市中做多所需的昂贵资金成本。
    """
    data = df.copy()
    data["market_log_ret"] = np.log(data['close']).diff().fillna(0)
    
    # 1. 灾难阻断修正 (Gap Correction)
    adjusted_market_ret = data['market_log_ret'].copy()
    risk_mask = data.get('sigma_event', False)
    
    if risk_mask.any():
        sl_values = data.loc[risk_mask, 'sl_threshold']
        prev_close = data.loc[risk_mask, 'close'].shift(1)
        open_price = data.loc[risk_mask, 'open']
        
        stop_price = open_price * (1.0 - sl_values)
        correction = np.log(stop_price / prev_close)
        correction = correction.fillna(adjusted_market_ret.loc[risk_mask])
        adjusted_market_ret.loc[risk_mask] = correction
        
    # 2. 策略毛回报
    data['strategy_log_ret'] = data['position'] * adjusted_market_ret
    
    # 3. 交易手续费 (Transaction Cost)
    position_change = data['position'].diff().abs().fillna(0)
    data['cost'] = position_change * fee_rate
    
    # ==========================================
    # 4. [V3.7 Update] 资金费率 (Funding Cost)
    # ==========================================
    # 逻辑: 只要持有仓位(Position != 0)，每小时都需要支付资金费。
    # 这里采用绝对值计算，假设无论多空都需要支付费率（保守回测模型）
    # 如果是做空收费，回测会更乐观，但为了健壮性我们假设是支付。
    position_size = data['position'].abs()
    data['funding_cost'] = position_size * funding_rate
    
    # 5. 净回报 (Net Return)
    # Net = Strategy - Transaction - Funding
    data['net_log_ret'] = data['strategy_log_ret'] - data['cost'] - data['funding_cost']
    
    # 6. 资金曲线
    initial_cap = getattr(Config, 'INITIAL_CAPITAL', 10000.0)
    norm_equity = np.exp(data['net_log_ret'].cumsum())
    norm_bh_equity = np.exp(data['market_log_ret'].cumsum())
    
    data['equity'] = norm_equity * initial_cap
    data['buy_hold_equity'] = norm_bh_equity * initial_cap
    
    return data
   