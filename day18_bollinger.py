import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据加载 (直接复用 Day 17 的完美版)
# ==========================================
def load_price_data(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return pd.DataFrame()

    if len(df) > 0 and ("http" in str(df.columns[0]) or "www" in str(df.columns[0])):
        df = pd.read_csv(csv_path, skiprows=1, low_memory=False)

    df.columns = [c.strip().lower() for c in df.columns]
    
    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"])
    elif "unix" in df.columns:
        df["unix"] = pd.to_numeric(df["unix"], errors='coerce')
        max_ts = df["unix"].max()
        if max_ts > 1e14: unit = 'us'
        elif max_ts > 1e11: unit = 'ms'
        else: unit = 's'
        df["time"] = pd.to_datetime(df["unix"], unit=unit)
    elif "date" in df.columns:
         df["time"] = pd.to_datetime(df["date"])
    else:
        return pd.DataFrame() 

    df = df.set_index("time").sort_index()
    df = df[df.index > pd.to_datetime("2010-01-01")] # 过滤脏数据
    
    if "volume" not in df.columns and "vol" in df.columns:
        df["volume"] = df["vol"]

    return df

# ==========================================
# 2. 🔥 新策略：布林带均值回归 (Day 18 核心)
# ==========================================
def calc_bollinger_signal(df, window=20, num_std=2.0):
    """
    计算布林带策略信号
    window: 均线周期 (默认20)
    num_std: 标准差倍数 (默认2.0，越大越难触发，越稳)
    """
    data = df.copy()
    
    # 1. 计算布林带
    # 中轨 = 移动平均线
    data["ma"] = data["close"].rolling(window).mean()
    # 标准差 (波动率)
    data["std"] = data["close"].rolling(window).std()
    
    # 上轨 = 中轨 + N倍标准差
    data["upper"] = data["ma"] + (num_std * data["std"])
    # 下轨 = 中轨 - N倍标准差
    data["lower"] = data["ma"] - (num_std * data["std"])
    
    # 2. 生成信号
    data["signal"] = 0
    
    # 买入逻辑：价格 < 下轨 (跌得太深了，抄底!)
    # 卖出逻辑：价格 > 上轨 (涨得太猛了，卖出!)
    
    # 这里我们用 loc 来标记
    # 信号 1: 买入
    data.loc[data["close"] < data["lower"], "signal"] = 1
    
    # 信号 -1: 卖出
    data.loc[data["close"] > data["upper"], "signal"] = -1
    
    return data

# ==========================================
# 3. 回测引擎 (复用)
# ==========================================
def run_simple_backtest(df_signals, initial_capital=10000, fee_rate=0.0005):
    """
    简化版回测引擎 (不带止损，纯跑策略逻辑)
    """
    balance = initial_capital
    position = 0 
    equity_curve = []
    
    for i in range(len(df_signals)):
        price = df_signals["close"].iloc[i]
        signal = df_signals["signal"].iloc[i]
        
        # 信号逻辑：
        # 1 = 即使有仓位也保持，没仓位就买
        # -1 = 清仓
        # 0 = 保持现状 (Hold)
        
        if signal == 1 and position == 0:
            # 买入 (All in)
            cost = balance * (1 - fee_rate)
            position = cost / price
            balance = 0
            
        elif signal == -1 and position > 0:
            # 卖出 (Close)
            balance = position * price * (1 - fee_rate)
            position = 0
            
        # 计算当前净值
        current_equity = balance + (position * price)
        equity_curve.append(current_equity)
        
    return pd.Series(equity_curve, index=df_signals.index)

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # 配置
    # ⚠️ 确保你用的是昨天下载好的干净数据！
    file_path = "Binance_BTCUSDT_1h.csv" 
    
    print(f"🚀 Jarvis Day 18: 布林带均值回归启动...")
    
    # 1. 加载数据
    df = load_price_data(file_path)
    if df.empty:
        print("❌ 数据加载失败，请检查文件名")
        exit()
        
    # 2. 运行策略 (使用默认参数: 20, 2.0)
    # 你可以试着改 num_std，比如 2.5 或 3.0
    df_res = calc_bollinger_signal(df, window=20, num_std=2.0)
    
    # 3. 回测
    curve = run_simple_backtest(df_res)
    
    # 4. 计算 Buy & Hold 作为对比
    buy_hold = df["close"] / df["close"].iloc[0] * 10000
    
    # 5. 打印最终结果
    final_equity = curve.iloc[-1]
    bh_equity = buy_hold.iloc[-1]
    print(f"\n💰 最终资金: ${final_equity:,.0f}")
    print(f"📉 囤币资金: ${bh_equity:,.0f}")
    print(f"📊 收益率: {(final_equity/10000 - 1):.2%}")

    # 6. 画图 (带布林带通道)
    plt.figure(figsize=(12, 8))
    
    # 子图1: 资金曲线
    plt.subplot(2, 1, 1)
    plt.plot(curve, label="Bollinger Strategy", color='purple')
    plt.plot(buy_hold, label="Buy & Hold", color='grey', linestyle='--', alpha=0.5)
    plt.title("Equity Curve: Mean Reversion vs HODL")
    plt.legend()
    plt.grid()
    
    # 子图2: 价格与布林带 (只画最后500根K线，不然看不清)
    plt.subplot(2, 1, 2)
    last_500 = df_res.iloc[-500:]
    plt.plot(last_500.index, last_500["close"], label="Price", color='black', alpha=0.6)
    plt.plot(last_500.index, last_500["upper"], label="Upper Band", color='green', linestyle='--')
    plt.plot(last_500.index, last_500["lower"], label="Lower Band", color='red', linestyle='--')
    
    # 标出买卖点
    buys = last_500[last_500["signal"] == 1]
    sells = last_500[last_500["signal"] == -1]
    plt.scatter(buys.index, buys["close"], marker='^', color='red', s=100, label="Buy")
    plt.scatter(sells.index, sells["close"], marker='v', color='green', s=100, label="Sell")
    
    plt.title("Bollinger Bands Trade Signals (Last 500 Hours)")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig("Day18_Bollinger.png")
    print("📸 结果已保存为 Day18_Bollinger.png")