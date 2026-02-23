"""
策略层：信号生成。基于 Config 计算趋势 + RSI 混合预测（特征与 forecast）。
全部为 Pandas 向量化运算，无行级 for 循环。
"""
import pandas as pd
import numpy as np
from config import Config


def calculate_scaled_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    [V4.3 The Silence Protocol] 深度平滑混合信号
    1. Trend: 多周期 EWMA 差 / 波动率，加权合成
    2. RSI: Soft Deadzone + Deep Smoothing

    Args:
        df: 必须包含 close，时间索引。可由 data_handler.load_price_data 产出。

    Returns:
        同索引 DataFrame，新增 volatility, trend_forecast, rsi_forecast, forecast 等列。
    """
    data = df.copy()

    # --- 1. 基础数据准备 (Vol Scaling 用) ---
    vol_span = getattr(Config, "VOL_LOOKBACK", 480)
    data["volatility"] = (
        data["close"].ewm(span=vol_span).std().replace(0, np.nan).ffill() + 1e-8
    )

    # --- 2. 趋势信号 (Trend Component) ---
    fast_spans = Config.STRATEGY_PARAMS["fast_span"]
    slow_spans = Config.STRATEGY_PARAMS["slow_span"]
    scalars = Config.STRATEGY_PARAMS["scalars"]
    weights = getattr(Config, "TREND_INTERNAL_WEIGHTS", [0.25, 0.25, 0.25, 0.25])

    forecast_cols: list[str] = []
    for i in range(len(fast_spans)):
        fast, slow, scalar = fast_spans[i], slow_spans[i], scalars[i]
        raw = data["close"].ewm(span=fast).mean() - data["close"].ewm(span=slow).mean()
        col = f"fc_{fast}_{slow}"
        data[col] = (raw * scalar) / data["volatility"]
        forecast_cols.append(col)

    trend_forecast = data[forecast_cols].mul(weights).sum(axis=1)

    # --- 3. [V4.3] 深度平滑 RSI 反转信号 ---
    rsi_period = getattr(Config, "RSI_PERIOD", 14)
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = gain / loss
    raw_rsi = 100 - (100 / (1 + rs))

    smooth_rsi = raw_rsi.rolling(window=12).mean().fillna(50)
    rsi_diff = 50 - smooth_rsi
    rsi_scalar = getattr(Config, "RSI_SCALAR", 1.0)

    rsi_forecast = (
        np.sign(rsi_diff) * np.maximum(0, rsi_diff.abs() - 10) * rsi_scalar
    )
    rsi_forecast = rsi_forecast.ewm(span=24).mean()

    # --- 4. 信号融合 ---
    w_trend = getattr(Config, "TREND_WEIGHT", 0.9)
    w_rsi = getattr(Config, "RSI_WEIGHT", 0.1)

    data["trend_forecast"] = trend_forecast.clip(-20, 20).fillna(0)
    data["rsi_forecast"] = rsi_forecast.clip(-20, 20).fillna(0)
    data["forecast"] = data["trend_forecast"] * w_trend + data["rsi_forecast"] * w_rsi

    return data
