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
    Day 23.5 Upgrade: 引入动态波动率缩放 (Vol-Scaling)
    逻辑：
    1. 计算当前价格的年化波动率。
    2. 计算杠杆系数 = 目标波动率 / 当前波动率。
    3. 动态调整仓位。
    """
    data = df.copy()
    
    # -------------------------------------------------------
    # 1. 计算年化波动率百分比 (Annualized Volatility %)
    # -------------------------------------------------------
    # data['volatility'] 是小时级别的绝对价格波动 (std)
    # 我们需要把它转化为：相对于当前价格的、年化的百分比
    # 公式：(Vol / Price) * sqrt(24小时 * 365天)
    
    # 安全检查：防止除以0
    close_price = data['close'].replace(0, np.nan)
    hourly_vol_pct = data['volatility'] / close_price
    
    # 年化因子 (根号下 8760 小时)
    sqrt_time = np.sqrt(24 * 365)
    
    data['ann_vol_pct'] = hourly_vol_pct * sqrt_time
    # 填补空值 (防止开局数据不足导致的空值)
    data['ann_vol_pct'] = data['ann_vol_pct'].fillna(method='ffill').fillna(0)

    # -------------------------------------------------------
    # 2. 计算动态杠杆 (Leverage Ratio)
    # -------------------------------------------------------
    # 目标：保持组合的年化波动率恒定在 Config.TARGET_VOLATILITY (比如 20%)
    # 如果当前波动率是 10%，我们就上 2.0x 杠杆
    # 如果当前波动率是 40%，我们就降到 0.5x 仓位
    
    # 防除零处理 (加一个极小值 1e-6)
    vol_safe = data['ann_vol_pct'].replace(0, 1e-6)
    
    leverage_ratio = Config.TARGET_VOLATILITY / vol_safe
    
    # -------------------------------------------------------
    # 3. 计算目标仓位 (Ideal Position)
    # -------------------------------------------------------
    # 原始信号强度 (-20 ~ +20) 归一化到 (-1 ~ +1)
    raw_signal_strength = data[forecast_col] / 20.0
    
    # 叠加波动率杠杆
    ideal_position = raw_signal_strength * leverage_ratio
    
    # -------------------------------------------------------
    # 4. 风控熔断 (Risk Cap)
    # -------------------------------------------------------
    # 无论模型多有信心，绝对不能超过最大允许杠杆 (比如 4倍)
    limit = Config.MAX_LEVERAGE
    ideal_position = ideal_position.clip(-limit, limit)
    
    # -------------------------------------------------------
    # 5. 缓冲器逻辑 (Buffer) - 保持 Day 20 的逻辑不变
    # -------------------------------------------------------
    ideal_values = ideal_position.values
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
  