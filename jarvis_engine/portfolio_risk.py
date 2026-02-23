"""
风控与执行层：仓位目标计算（Regime + Vol Scaling + Survival + Buffer）与向量化回测。
仓位缓冲器为状态机逻辑，保留原有实现；其余均为 Pandas/NumPy 向量化。
"""
import pandas as pd
import numpy as np
from config import Config


def calculate_position_target(
    df: pd.DataFrame,
    forecast_col: str = "forecast",
    buffer: float = 0.1,
) -> pd.DataFrame:
    """
    [Risk Engine V4.0] 环境感知型风控
    Regime Filter + Vol Scaling + Survival Stop + Buffer（状态机，保留原有实现）。

    Args:
        df: 须含 close, high, low 及 forecast_col（如 forecast）。
        forecast_col: 信号列名。
        buffer: 调仓缓冲阈值，来自 Config.POSITION_BUFFER。

    Returns:
        同索引 DataFrame，新增 regime_ma, dynamic_max_cap, ann_vol_pct, position 等列。
    """
    data = df.copy()

    # --- 1. 环境过滤器 (Regime Filter) ---
    ma_window = getattr(Config, "REGIME_MA_WINDOW", 4800)
    regime_ma = data["close"].rolling(window=ma_window).mean()
    is_bull_regime = data["close"] > regime_ma

    normal_cap = getattr(Config, "MAX_LEVERAGE", 2.5)
    bear_cap = getattr(Config, "BEAR_MODE_MAX_LEVERAGE", 1.0)
    dynamic_max_cap = np.where(is_bull_regime, normal_cap, bear_cap)
    data["regime_ma"] = regime_ma
    data["dynamic_max_cap"] = dynamic_max_cap

    # --- 2. 波动率目标管理 (Vol Scaling) ---
    hourly_ret = data["close"].pct_change().fillna(0)
    long_term_vol = hourly_ret.ewm(span=Config.VOL_LOOKBACK).std().fillna(0)
    ann_vol_pct = long_term_vol * np.sqrt(365 * 24)
    data["ann_vol_pct"] = ann_vol_pct

    safe_vol = ann_vol_pct.replace(0, 1e-6)
    target_vol = getattr(Config, "TARGET_VOLATILITY", 0.8)
    raw_leverage_ratio = target_vol / safe_vol

    # --- 3. 仓位计算 ---
    ideal_position = (data[forecast_col] / 2.0) * raw_leverage_ratio
    ideal_position = ideal_position.clip(-dynamic_max_cap, dynamic_max_cap)
    data["leverage_ratio"] = ideal_position.abs()

    # --- 4. 灾难阻断器 (Survival Hard Stop) [V3.3] ---
    h, l, c = data["high"], data["low"], data["close"]
    pc = c.shift(1).fillna(c)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))

    atr_window = getattr(Config, "SURVIVAL_ATR_WINDOW", 24)
    atr = tr.ewm(span=atr_window).mean().fillna(0)

    multiplier = getattr(Config, "SURVIVAL_ATR_MULTIPLIER", 4.5)
    min_vol = getattr(Config, "MIN_HOURLY_VOL", 0.005)

    raw_threshold = (atr * multiplier) / c
    crash_threshold = np.maximum(raw_threshold, min_vol * multiplier)
    is_crash = hourly_ret < -crash_threshold

    ideal_position = np.where(is_crash, 0.0, ideal_position)

    data["sl_threshold"] = crash_threshold
    data["sigma_event"] = is_crash
    data["is_meltdown"] = is_crash

    # --- 5. 缓冲器 (Buffer)：状态机，保留原有实现 ---
    ideal_values = ideal_position
    n = len(ideal_values)
    buffered_position = np.zeros(n)
    current_pos = 0.0
    for i in range(n):
        if abs(ideal_values[i] - current_pos) > buffer:
            current_pos = ideal_values[i]
        buffered_position[i] = current_pos

    data["raw_target"] = ideal_position
    data["buffered_pos"] = buffered_position
    data["position"] = data["buffered_pos"].shift(1).fillna(0)

    return data


def run_vectorized_backtest(
    df: pd.DataFrame,
    fee_rate: float = 0.0005,
    funding_rate: float = 0.00001,
) -> pd.DataFrame:
    """
    全向量化回测：根据 position 与行情计算净收益与资金曲线（简单收益累乘）。
    灾难日采用劣后成交逻辑；费用含手续费与资金费。

    Args:
        df: 须含 close, open, position, sigma_event, sl_threshold（由 calculate_position_target 产出）。
        fee_rate: 单边手续费率，默认从 Config.FEE_RATE 读取时由调用方传入。
        funding_rate: 资金费率。

    Returns:
        同索引 DataFrame，新增 market_ret, net_ret, equity, buy_hold_equity, net_log_ret, market_log_ret 等。
    """
    data = df.copy()

    data["market_ret"] = data["close"].pct_change().fillna(0)

    risk_mask = data.get("sigma_event", False)
    adjusted_ret = data["market_ret"].copy()
    if risk_mask.any():
        sl_values = data.loc[risk_mask, "sl_threshold"]
        open_price = data.loc[risk_mask, "open"]
        close_price = data.loc[risk_mask, "close"]
        prev_close = data["close"].shift(1).loc[risk_mask]
        execution_price = (
            np.minimum(open_price * (1.0 - sl_values), close_price) * 0.995
        )
        adjusted_ret.loc[risk_mask] = (execution_price / prev_close) - 1.0

    data["strat_ret_raw"] = data["position"] * adjusted_ret

    pos_change = data["position"].diff().abs().fillna(0)
    transaction_cost = pos_change * fee_rate
    funding_cost = data["position"].abs() * funding_rate
    data["net_ret"] = data["strat_ret_raw"] - transaction_cost - funding_cost

    initial_cap = Config.INITIAL_CAPITAL
    data["equity"] = initial_cap * (1 + data["net_ret"]).cumprod()
    data["buy_hold_equity"] = initial_cap * (1 + data["market_ret"]).cumprod()

    data["net_log_ret"] = np.log(1 + data["net_ret"])
    data["market_log_ret"] = np.log(1 + data["market_ret"])

    return data
