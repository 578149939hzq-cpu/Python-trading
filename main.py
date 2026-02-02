import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 引入 Config
from config import Config
from jarvis_engine.alpha import load_price_data, calculate_scaled_forecast
from jarvis_engine.alpha import calculate_position_target, run_vectorized_backtest

# ==========================================
# 📊 1. 全景战报 (Full History Report)
# ==========================================
def plot_full_report(df_res):
    print("🎨 Generating Institutional Static Report (Matplotlib)...")
    
    plt.style.use('bmh') 
    
    # 4行1列
    fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
    
    # --- 子图 1: 净值曲线 ---
    ax0 = axes[0]
    ax0.plot(df_res.index, df_res['equity'], color='#FF9900', linewidth=2, label='Jarvis Strategy')
    ax0.plot(df_res.index, df_res['buy_hold_equity'], color='gray', linestyle='--', alpha=0.6, label='Buy & Hold')
    ax0.set_title("🏆 Equity Curve (Net of Fees)", fontweight='bold', fontsize=14)
    ax0.set_ylabel("Account Value ($)")
    ax0.legend(loc='upper left')
    
    # --- 子图 2: 价格与风控事件 ---
    ax1 = axes[1]
    ax1.plot(df_res.index, df_res['close'], color='black', alpha=0.6, linewidth=1, label='Price')
    
    # 标记熔断 (红色倒三角) - V3.2 只有 ATR Closing Stop
    meltdowns = df_res[df_res.get('is_meltdown', False) == True]
    if not meltdowns.empty:
        ax1.scatter(meltdowns.index, meltdowns['close'], color='red', marker='v', s=40, zorder=5, label=f'ATR Stop (> {Config.ATR_MULTIPLIER}x)')
        
    # V3.2 已移除瞬时止损，此处不再绘制紫色X

    ax1.set_title(f"📉 Price Action & Wide ATR Risk Control ({Config.ATR_MULTIPLIER}x)", fontweight='bold', fontsize=14)
    ax1.legend(loc='upper left')

    # --- 子图 3: 波动率监测 ---
    ax2 = axes[2]
    # 显示长期波动率
    ax2.plot(df_res.index, df_res['ann_vol_pct'], color='blue', linewidth=1.5, label=f'Long-Term Vol (Span={Config.VOL_LOOKBACK})')
    ax2.axhline(Config.TARGET_VOLATILITY, color='green', linestyle='--', linewidth=2, label=f'Target ({Config.TARGET_VOLATILITY})')
    ax2.set_title("🌊 Volatility Regime (Stable Sizing)", fontweight='bold', fontsize=14)
    ax2.set_ylabel("Annualized Vol %")
    ax2.legend(loc='upper left')

    # --- 子图 4: 杠杆管理 ---
    ax3 = axes[3]
    ax3.plot(df_res.index, df_res['leverage_ratio'], color='gray', alpha=0.5, label='Max Allowed Leverage')
    ax3.plot(df_res.index, df_res['position'].abs(), color='#FF9900', linewidth=1.5, label='Actual Position (Abs)')
    
    # 标记强制平仓点
    crashes = df_res[(df_res.get('sigma_event', False) == True) & (df_res['position'] == 0)]
    if not crashes.empty:
        ax3.scatter(crashes.index, [0]*len(crashes), color='red', marker='x', s=50, label='Forced Exit')

    ax3.set_title("⚙️ Leverage System", fontweight='bold', fontsize=14)
    ax3.set_ylabel("Leverage (x)")
    ax3.legend(loc='upper left')

    plt.tight_layout()
    
    results_dir = os.path.join(Config.BASE_DIR, "data_results")
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    save_path = os.path.join(results_dir, "Jarvis_Full_Report.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"✅ 全景报告已保存: {save_path}")

# ==========================================
# 📸 2. 智能特写快照 (Smart Snapshots)
# ==========================================
def plot_crash_snapshots(df_res, top_n=3):
    """
    自动寻找风险事件时刻，生成局部特写图
    """
    print(f"📸 Generating Top {top_n} Crash Snapshots...")
    
    risk_events = df_res[df_res.get('sigma_event', False) == True].copy()
    
    if risk_events.empty:
        print("🎉 Good News: No risk events triggered (System is extremely robust).")
        return

    # 按波动率排序
    risk_events = risk_events.sort_values('ann_vol_pct', ascending=False)
    
    risk_events['date'] = risk_events.index.date
    top_days = risk_events.drop_duplicates(subset=['date']).head(top_n)
    
    results_dir = os.path.join(Config.BASE_DIR, "data_results")

    for idx, (timestamp, row) in enumerate(top_days.iterrows()):
        start_t = timestamp - pd.Timedelta(days=5) # 稍微拉长一点观察周期，看趋势
        end_t = timestamp + pd.Timedelta(days=5)
        subset = df_res.loc[start_t:end_t]
        
        if subset.empty: continue

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        date_str = timestamp.strftime('%Y-%m-%d')
        fig.suptitle(f"🚨 Risk Event: {date_str} (Vol: {row['ann_vol_pct']:.1%})", fontsize=16, fontweight='bold', color='darkred')
        
        # 图1
        ax0 = axes[0]
        ax0.plot(subset.index, subset['close'], color='black', label='Price')
        
        local_melt = subset[subset.get('is_meltdown', False) == True]
        ax0.scatter(local_melt.index, local_melt['close'], color='red', marker='v', s=100, label='ATR Stop')
        
        ax0.set_title("Price Action", fontsize=10)
        ax0.legend()
        ax0.grid(True, alpha=0.3)
        
        # 图2
        ax1 = axes[1]
        ax1.plot(subset.index, subset['position'].abs(), color='#FF9900', linewidth=2, label='Position (Abs)')
        ax1.fill_between(subset.index, subset['position'].abs(), color='#FF9900', alpha=0.1)
        ax1.set_title("Position Deleveraging", fontsize=10)
        ax1.set_ylabel("Position Size")
        ax1.grid(True, alpha=0.3)
        
        # 图3
        ax2 = axes[2]
        ax2.plot(subset.index, subset['ann_vol_pct'], color='blue', label='Long-Term Vol')
        ax2.axhline(Config.TARGET_VOLATILITY, color='green', linestyle='--', label='Target')
        ax2.set_title("Volatility", fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=45)
        
        fname = f"Snapshot_{idx+1}_{date_str}.png"
        save_path = os.path.join(results_dir, fname)
        plt.savefig(save_path, dpi=200)
        plt.close()
        
        print(f"📸 快照已生成: {fname}")

# ==========================================
# 🚀 主任务
# ==========================================
def mission_start():
    print("🚀 Jarvis System Initializing (Risk Engine V3.2 Simplified)...")
    
    # 强制重载配置
    import importlib
    import config
    importlib.reload(config)

    print(f"📂 Data Path: {Config.DATA_PATH}")
    
    df = load_price_data(Config.DATA_PATH)
    if df.empty: 
        print("❌ Data not found.")
        return

    print("🧠 Calculating Alpha (Raw Signal)...")
    df = calculate_scaled_forecast(df)
    
    print(f"🛡️ Risk Engine V3.2 (Metric=ATR, Multiplier={Config.ATR_MULTIPLIER}x, VolSpan={Config.VOL_LOOKBACK})...")
    
    df = calculate_position_target(df, buffer=Config.POSITION_BUFFER)
    
    print("⚡ Backtesting (Closing Basis)...")
    df_res = run_vectorized_backtest(df, fee_rate=Config.FEE_RATE)
    
    final = df_res['equity'].iloc[-1]
    sharpe = (df_res['net_log_ret'].mean() / df_res['net_log_ret'].std()) * np.sqrt(365*24)
    print("-" * 40)
    print(f"🏆 Final Equity: ${final:,.2f} (Initial: ${Config.INITIAL_CAPITAL})")
    print(f"📊 Sharpe Ratio: {sharpe:.2f}")
    print("-" * 40)
    
    plot_full_report(df_res)
    plot_crash_snapshots(df_res, top_n=3)

if __name__ == "__main__":
    mission_start()