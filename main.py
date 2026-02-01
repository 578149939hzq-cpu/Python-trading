import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config

# ==========================================
# 🛠️ 适配层 (保持之前的热修复逻辑)
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 自动寻找 data_raw
DATA_PATH_GUESS = os.path.join(PROJECT_ROOT, "data_raw", "Binance_BTCUSDT_1h.csv")
if os.path.exists(DATA_PATH_GUESS):
    config.DATA_PATH = DATA_PATH_GUESS

class ConfigAdapter:
    VOL_LOOKBACK = getattr(config, 'VOLATILITY_SPAN', 36)
    STRATEGY_PARAMS = config.STRATEGY_PARAMS
    WEIGHTS = config.WEIGHTS
    DATA_PATH = config.DATA_PATH
    # 新增风控参数透传
    TARGET_VOLATILITY = getattr(config, 'TARGET_VOLATILITY', 0.20)
    MAX_LEVERAGE = getattr(config, 'MAX_LEVERAGE', 4.0)

config.Config = ConfigAdapter

from jarvis_engine.alpha import load_price_data, calculate_scaled_forecast
from jarvis_engine.alpha import calculate_position_target, run_vectorized_backtest

# ==========================================
# 📊 新增：风险诊断绘图引擎
# ==========================================
def plot_leverage_diagnostic(df_res):
    print("🏥 正在生成风险诊断报告 (Leverage Diagnostic)...")
    
    # 准备画布：3行1列
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # 子图 1: 价格走势
    ax1 = axes[0]
    ax1.plot(df_res.index, df_res['close'], color='black', alpha=0.6)
    ax1.set_title(f"BTC Price Action", fontweight='bold')
    ax1.set_ylabel("Price ($)")
    ax1.grid(True, alpha=0.2)
    
    # 子图 2: 波动率 (Vol) vs 目标 (Target)
    ax2 = axes[1]
    # 绘制实际波动率
    ax2.plot(df_res.index, df_res['ann_vol_pct'], color='blue', linewidth=1.5, label='Actual Vol (Ann.)')
    # 绘制目标波动率红线
    target_vol = ConfigAdapter.TARGET_VOLATILITY
    ax2.axhline(target_vol, color='red', linestyle='--', linewidth=2, label=f'Target Vol ({target_vol})')
    
    ax2.set_title("Market Volatility vs Target", fontweight='bold')
    ax2.set_ylabel("Annualized Volatility")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.2)
    
    # 子图 3: 动态杠杆 (Leverage)
    ax3 = axes[2]
    ax3.plot(df_res.index, df_res['leverage_ratio'], color='green', linewidth=1.5, label='Dynamic Leverage')
    
    # 标记被强制封顶 (Clipped) 的区域
    max_lev = ConfigAdapter.MAX_LEVERAGE
    ax3.axhline(max_lev, color='red', linestyle=':', label=f'Max Cap ({max_lev}x)')
    
    # 填充因为波动率过低而触顶的区域
    ax3.fill_between(df_res.index, df_res['leverage_ratio'], max_lev, 
                     where=(df_res['leverage_ratio'] >= max_lev), 
                     color='red', alpha=0.3, label='Clipped Region')

    ax3.set_title("System Leverage Ratio", fontweight='bold')
    ax3.set_ylabel("Leverage (x)")
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    # 保存
    results_dir = os.path.join(PROJECT_ROOT, "data_results")
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    save_path = os.path.join(results_dir, "Leverage_Diagnostic.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"✅ 诊断报告已保存: {save_path}")
    plt.show()

# ==========================================
# 🚀 主任务流程
# ==========================================
def mission_start():
    print("🚀 Jarvis System Initializing...")
    df = load_price_data(config.DATA_PATH)
    
    if df.empty: return

    print("🧠 Calculating Alpha...")
    df = calculate_scaled_forecast(df)
    
    print(f"🛡️ Risk Engine: Vol-Targeting (Target={ConfigAdapter.TARGET_VOLATILITY}, Max={ConfigAdapter.MAX_LEVERAGE}x)...")
    df = calculate_position_target(df, buffer=config.POSITION_BUFFER)
    
    print("⚡ Backtesting...")
    df_res = run_vectorized_backtest(df, fee_rate=config.FEE_RATE)
    
    # 打印简报
    final_equity = df_res['equity'].iloc[-1]
    sharpe = (df_res['net_log_ret'].mean() / df_res['net_log_ret'].std()) * np.sqrt(365*24)
    print(f"🏆 最终净值: {final_equity:.4f} | 夏普比率: {sharpe:.2f}")
    
    # 🔥 调用诊断函数
    plot_leverage_diagnostic(df_res)

if __name__ == "__main__":
    mission_start()