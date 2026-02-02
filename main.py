import config
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config

# ==========================================
# 🛠️ 适配层 (Config Adapter)
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_GUESS = os.path.join(PROJECT_ROOT, "data_raw", "Binance_BTCUSDT_1h.csv")
if os.path.exists(DATA_GUESS):
    config.DATA_PATH = DATA_GUESS

class ConfigAdapter:
    # 基础参数
    VOL_LOOKBACK = getattr(config, 'VOLATILITY_SPAN', 168)
    STRATEGY_PARAMS = config.STRATEGY_PARAMS
    WEIGHTS = config.WEIGHTS
    DATA_PATH = config.DATA_PATH
    
    # 风控 V2.0 参数
    TARGET_VOLATILITY = getattr(config, 'TARGET_VOLATILITY', 0.20)
    MAX_LEVERAGE = getattr(config, 'MAX_LEVERAGE', 2.0)
    SIGMA_THRESHOLD = getattr(config, 'SIGMA_THRESHOLD', 3.0)
    
    # [V2.1 New] 风控 V2.1 参数
    STOP_LOSS_SIGMA = getattr(config, 'STOP_LOSS_SIGMA', 2.0)
    MELTDOWN_DIRECTION = getattr(config, 'MELTDOWN_DIRECTION', 'down')

config.Config = ConfigAdapter

from jarvis_engine.alpha import load_price_data, calculate_scaled_forecast
from jarvis_engine.alpha import calculate_position_target, run_vectorized_backtest

# ==========================================
# 📊 诊断绘图引擎 V2.1
# ==========================================
def plot_leverage_diagnostic(df_res):
    print("🏥 Generating Risk Engine V2.1 Diagnostic Report...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # --- 图1: 净值对比 ---
    ax0 = axes[0]
    ax0.plot(df_res.index, df_res['equity'], color='#FF9900', linewidth=2.5, label='Jarvis Strategy')
    ax0.plot(df_res.index, df_res['buy_hold_equity'], color='gray', linestyle='--', alpha=0.6, label='Buy & Hold')
    ax0.set_title("🏆 Equity Curve", fontweight='bold')
    ax0.legend(loc='upper left')
    ax0.grid(True, alpha=0.2)
    
    # --- 图2: 价格与风险事件 ---
    ax1 = axes[1]
    ax1.plot(df_res.index, df_res['close'], color='black', alpha=0.6, label='Price')
    
    # 🔴 标记熔断 (Meltdown > 3σ)
    meltdowns = df_res[df_res.get('is_meltdown', False) == True]
    if not meltdowns.empty:
        ax1.scatter(meltdowns.index, meltdowns['close'], color='red', s=40, marker='v', zorder=5, label='Meltdown (>3σ)')

    # 🟣 标记瞬时止损 (Stop Loss > 2σ)
    stoplosses = df_res[df_res.get('is_stop_loss', False) == True]
    if not stoplosses.empty:
        ax1.scatter(stoplosses.index, stoplosses['close'], color='purple', s=20, marker='x', zorder=4, label='Intraday Stop (>6σ)')
        
    ax1.set_title("BTC Price & Risk Events (Red=Crash, Purple=Intraday)", fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    
    # --- 图3: 波动率 ---
    ax2 = axes[2]
    ax2.plot(df_res.index, df_res['ann_vol_pct'], color='blue', label='Actual Vol')
    ax2.axhline(ConfigAdapter.TARGET_VOLATILITY, color='green', linestyle='--', label='Target Vol')
    ax2.set_title("Volatility Monitor", fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.2)
    
    # --- 图4: 杠杆与强制平仓 ---
    ax3 = axes[3]
    ax3.plot(df_res.index, df_res['leverage_ratio'], color='gray', alpha=0.5, label='Raw Leverage')
    ax3.plot(df_res.index, df_res['position'].abs(), color='#FF9900', label='Actual Position')
    
    # 标记强制归零点
    crashes = df_res[(df_res.get('sigma_event', False) == True) & (df_res['position'] == 0)]
    if not crashes.empty:
        ax3.scatter(crashes.index, [0]*len(crashes), color='red', marker='x', s=50, label='Forced Exit')
        
    ax3.set_title("System Leverage", fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.2)
    
    plt.tight_layout()
    results_dir = os.path.join(PROJECT_ROOT, "data_results")
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    save_path = os.path.join(results_dir, "Risk_Engine_V2_Diagnostic.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved: {save_path}")
    plt.show()

# ==========================================
# 🚀 主任务
# ==========================================
def mission_start():
    print("🚀 Jarvis System Initializing (Risk Engine V2.1)...")
    
    # 1. 强制刷新配置 (防止 IDE 缓存旧参数)
    import importlib
    importlib.reload(config)
    # --- 🔍 侦探代码 (新增) ---
    print("-" * 30)
    print(f"📂 正在尝试读取文件: {config.DATA_PATH}")
    if os.path.exists(config.DATA_PATH):
        print("✅ 文件存在！")
    else:
        print("❌ 文件不存在！请检查路径或文件名！")
        # 打印当前目录下有什么，方便排查
        print(f"👀 当前目录下有: {os.listdir(PROJECT_ROOT)}")
    print("-" * 30)
    # -------------------------
    df = load_price_data(config.DATA_PATH)
    if df.empty: return

    print("🧠 Calculating Alpha...")
    df = calculate_scaled_forecast(df)
    
    print(f"🛡️ Applying Risk Control V2.1 (StopLoss={ConfigAdapter.STOP_LOSS_SIGMA}σ, Direction={ConfigAdapter.MELTDOWN_DIRECTION})...")
    df = calculate_position_target(df, buffer=config.POSITION_BUFFER)
    
    print("⚡ Backtesting (with Cost Correction)...")
    df_res = run_vectorized_backtest(df, fee_rate=config.FEE_RATE)
    
    final = df_res['equity'].iloc[-1]
    sharpe = (df_res['net_log_ret'].mean() / df_res['net_log_ret'].std()) * np.sqrt(365*24)
    print(f"🏆 Final Equity: {final:.4f} | Sharpe: {sharpe:.2f}")
    
    plot_leverage_diagnostic(df_res)

if __name__ == "__main__":
    mission_start()