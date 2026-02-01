import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config

# ==========================================
# 🛠️ 适配层 (Config Adapter) - 严禁修改
# ==========================================
# 保持原有的路径修复和参数注入逻辑不变
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 尝试自动修正数据路径
DATA_GUESS = os.path.join(PROJECT_ROOT, "data_raw", "Binance_BTCUSDT_1h.csv")
if os.path.exists(DATA_GUESS):
    config.DATA_PATH = DATA_GUESS

class ConfigAdapter:
    # 基础参数
    VOL_LOOKBACK = getattr(config, 'VOLATILITY_SPAN', 36)
    STRATEGY_PARAMS = config.STRATEGY_PARAMS
    WEIGHTS = config.WEIGHTS
    DATA_PATH = config.DATA_PATH
    
    # 风控参数 (V2.0)
    TARGET_VOLATILITY = getattr(config, 'TARGET_VOLATILITY', 0.80)
    MAX_LEVERAGE = getattr(config, 'MAX_LEVERAGE', 4.0)
    SIGMA_THRESHOLD = getattr(config, 'SIGMA_THRESHOLD', 3.0)

config.Config = ConfigAdapter

from jarvis_engine.alpha import load_price_data, calculate_scaled_forecast
from jarvis_engine.alpha import calculate_position_target, run_vectorized_backtest

# ==========================================
# 📊 诊断绘图引擎 (Diagnostic Engine V2.1)
# ==========================================
def plot_leverage_diagnostic(df_res):
    print("🏥 Generating Risk Engine V2.1 Diagnostic Report...")
    
    # 🆕 改动点：从 3行 变为 4行，高度增加到 16
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # --------------------------------------------------------
    # 子图 1 (新增): 净值曲线对比 (Equity Comparison)
    # --------------------------------------------------------
    ax0 = axes[0]
    
    # 绘制 Jarvis 策略净值 (橙色粗线)
    ax0.plot(df_res.index, df_res['equity'], 
             color='#FF9900', linewidth=2.5, label='Jarvis Strategy')
    
    # 绘制 Buy & Hold 净值 (灰色虚线)
    ax0.plot(df_res.index, df_res['buy_hold_equity'], 
             color='gray', linestyle='--', alpha=0.6, label='Buy & Hold (BTC)')
    
    # 填充超额收益区域 (绿色=跑赢, 红色=跑输)
    ax0.fill_between(df_res.index, df_res['equity'], df_res['buy_hold_equity'],
                     where=(df_res['equity'] >= df_res['buy_hold_equity']),
                     color='green', alpha=0.1, interpolate=True)
    ax0.fill_between(df_res.index, df_res['equity'], df_res['buy_hold_equity'],
                     where=(df_res['equity'] < df_res['buy_hold_equity']),
                     color='red', alpha=0.1, interpolate=True)
    
    ax0.set_title("🏆 Equity Curve: Jarvis vs Buy & Hold", fontweight='bold')
    ax0.set_ylabel("Normalized Equity ($)")
    ax0.legend(loc='upper left')
    ax0.grid(True, alpha=0.2)
    
    # --------------------------------------------------------
    # 子图 2: 价格与熔断点 (原 ax1)
    # --------------------------------------------------------
    ax1 = axes[1]
    ax1.plot(df_res.index, df_res['close'], color='black', alpha=0.6, label='Price')
    
    # 标记熔断点
    meltdowns = df_res[df_res.get('sigma_event', False) == True]
    if not meltdowns.empty:
        ax1.scatter(meltdowns.index, meltdowns['close'], color='red', s=25, zorder=5, label='Sigma Meltdown')
        
    ax1.set_title(f"BTC Price & Meltdown Events (Sigma > {ConfigAdapter.SIGMA_THRESHOLD})", fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    
    # --------------------------------------------------------
    # 子图 3: 波动率监测 (原 ax2)
    # --------------------------------------------------------
    ax2 = axes[2]
    ax2.plot(df_res.index, df_res['ann_vol_pct'], color='blue', linewidth=1.5, label='Actual Vol (Ann.)')
    ax2.axhline(ConfigAdapter.TARGET_VOLATILITY, color='green', linestyle='--', label=f'Target ({ConfigAdapter.TARGET_VOLATILITY})')
    
    ax2.set_title("Annualized Volatility Monitor", fontweight='bold')
    ax2.set_ylabel("Vol %")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.2)
    
    # --------------------------------------------------------
    # 子图 4: 杠杆率 (原 ax3)
    # --------------------------------------------------------
    ax3 = axes[3]
    ax3.plot(df_res.index, df_res['leverage_ratio'], color='gray', alpha=0.5, label='Raw Leverage')
    
    # 绘制实际仓位 (绝对值)
    real_lev = df_res['position'].abs()
    ax3.plot(df_res.index, real_lev, color='#FF9900', linewidth=1.5, label='Actual Position (Abs)')
    
    # 标记强制清零点
    crashes = df_res[(df_res.get('sigma_event', False) == True) & (df_res['position'] == 0)]
    if not crashes.empty:
        ax3.scatter(crashes.index, [0]*len(crashes), color='red', marker='x', s=50, label='Forced Liquidation')

    ax3.set_title("System Leverage & Circuit Breakers", fontweight='bold')
    ax3.set_ylabel("Leverage (x)")
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    # 保存图片
    results_dir = os.path.join(PROJECT_ROOT, "data_results")
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    save_path = os.path.join(results_dir, "Risk_Engine_V2_Diagnostic.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"✅ 全能诊断报告已保存: {save_path}")
    plt.show()

# ==========================================
# 🚀 主任务 (保持逻辑不变)
# ==========================================
def mission_start():
    print("🚀 Jarvis System Initializing (Risk Engine V2.1)...")
    # --- 🔍 调试代码开始 ---
    print(f"DEBUG: Vol Span = {config.Config.VOL_LOOKBACK}")
    print(f"DEBUG: Target Vol = {config.Config.TARGET_VOLATILITY}")
    print(f"DEBUG: Max Leverage = {config.Config.MAX_LEVERAGE}")
    df = load_price_data(config.DATA_PATH)
    if df.empty: return

    # 1. 大脑计算
    print("🧠 Calculating Alpha...")
    df = calculate_scaled_forecast(df)
    
    # 2. 风控介入 (V2.0 逻辑)
    print(f"🛡️ Applying Risk Control (TargetVol={ConfigAdapter.TARGET_VOLATILITY}, Sigma={ConfigAdapter.SIGMA_THRESHOLD})...")
    df = calculate_position_target(df, buffer=config.POSITION_BUFFER)
    
    # 3. 回测
    print("⚡ Backtesting...")
    df_res = run_vectorized_backtest(df, fee_rate=config.FEE_RATE)
    
    # 4. 打印结果
    final = df_res['equity'].iloc[-1]
    sharpe = (df_res['net_log_ret'].mean() / df_res['net_log_ret'].std()) * np.sqrt(365*24)
    print(f"🏆 最终净值: {final:.4f} | 夏普: {sharpe:.2f}")
    
    # 5. 画图
    plot_leverage_diagnostic(df_res)

if __name__ == "__main__":
    mission_start()