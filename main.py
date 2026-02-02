import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 引入正规化后的 Config
from config import Config
from jarvis_engine.alpha import load_price_data, calculate_scaled_forecast
from jarvis_engine.alpha import calculate_position_target, run_vectorized_backtest

# ==========================================
# 📊 1. 全景战报 (Full History Report)
# ==========================================
def plot_full_report(df_res):
    print("🎨 Generating Institutional Static Report (Matplotlib)...")
    
    # 设置风格：专业、硬朗
    plt.style.use('bmh') 
    
    # 4行1列
    fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
    
    # --- 子图 1: 净值曲线 ---
    ax0 = axes[0]
    ax0.plot(df_res.index, df_res['equity'], color='#FF9900', linewidth=2, label='Jarvis Strategy')
    ax0.plot(df_res.index, df_res['buy_hold_equity'], color='gray', linestyle='--', alpha=0.6, label='Buy & Hold')
    ax0.set_title("🏆 Equity Curve (Net of Fees)", fontweight='bold', fontsize=14)
    ax0.set_ylabel("Normalized Equity ($)")
    ax0.legend(loc='upper left')
    
    # --- 子图 2: 价格与风控事件 ---
    ax1 = axes[1]
    ax1.plot(df_res.index, df_res['close'], color='black', alpha=0.6, linewidth=1, label='Price')
    
    # 标记熔断 (红色倒三角)
    meltdowns = df_res[df_res.get('is_meltdown', False) == True]
    if not meltdowns.empty:
        ax1.scatter(meltdowns.index, meltdowns['close'], color='red', marker='v', s=30, zorder=5, label='Meltdown (>3σ)')
        
    # 标记瞬时止损 (紫色X)
    stops = df_res[df_res.get('is_stop_loss', False) == True]
    if not stops.empty:
        ax1.scatter(stops.index, stops['close'], color='purple', marker='x', s=20, zorder=4, label='Intraday Stop (>2σ)')

    ax1.set_title("📉 Price Action & Risk Events", fontweight='bold', fontsize=14)
    ax1.legend(loc='upper left')

    # --- 子图 3: 波动率监测 ---
    ax2 = axes[2]
    ax2.plot(df_res.index, df_res['ann_vol_pct'], color='blue', linewidth=1.5, label='Realized Vol')
    ax2.axhline(Config.TARGET_VOLATILITY, color='green', linestyle='--', linewidth=2, label=f'Target ({Config.TARGET_VOLATILITY})')
    ax2.set_title("🌊 Volatility Regime", fontweight='bold', fontsize=14)
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
    
    # 保存高清大图
    results_dir = os.path.join(Config.BASE_DIR, "data_results")
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    save_path = os.path.join(results_dir, "Jarvis_Full_Report.png")
    
    plt.savefig(save_path, dpi=300) # 300 DPI 打印级清晰度
    print(f"✅ 全景报告已保存: {save_path}")
    # plt.show() # 如果你想在窗口看，可以取消注释

# ==========================================
# 📸 2. 智能特写快照 (Smart Snapshots)
# ==========================================
def plot_crash_snapshots(df_res, top_n=3):
    """
    自动寻找波动率最大的前 N 个风险时刻，生成局部特写图
    """
    print(f"📸 Generating Top {top_n} Crash Snapshots...")
    
    # 筛选出发生过风控事件的时刻
    risk_events = df_res[df_res.get('sigma_event', False) == True].copy()
    
    if risk_events.empty:
        print("🎉 Good News: No risk events triggered. No snapshots needed.")
        return

    # 按“波动率”从大到小排序，找到最剧烈的时刻
    # 我们不仅看熔断，也看那一刻的波动率有多高
    risk_events = risk_events.sort_values('ann_vol_pct', ascending=False)
    
    # 为了避免重复拍同一天的图（比如连续3小时熔断），我们简单去重
    # 取每天波动最大的那个小时作为代表
    risk_events['date'] = risk_events.index.date
    top_days = risk_events.drop_duplicates(subset=['date']).head(top_n)
    
    results_dir = os.path.join(Config.BASE_DIR, "data_results")

    for idx, (timestamp, row) in enumerate(top_days.iterrows()):
        # 截取前后 3 天的数据
        start_t = timestamp - pd.Timedelta(days=3)
        end_t = timestamp + pd.Timedelta(days=3)
        subset = df_res.loc[start_t:end_t]
        
        if subset.empty: continue

        # --- 绘图 ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # 标题
        date_str = timestamp.strftime('%Y-%m-%d')
        fig.suptitle(f"🚨 Crash Forensics: {date_str} (Vol: {row['ann_vol_pct']:.1%})", fontsize=16, fontweight='bold', color='darkred')
        
        # 图1: 价格与事件
        ax0 = axes[0]
        ax0.plot(subset.index, subset['close'], color='black', label='Price')
        
        # 标记具体的熔断点
        local_melt = subset[subset.get('is_meltdown', False) == True]
        ax0.scatter(local_melt.index, local_melt['close'], color='red', marker='v', s=100, label='Meltdown')
        
        local_stop = subset[subset.get('is_stop_loss', False) == True]
        ax0.scatter(local_stop.index, local_stop['close'], color='purple', marker='x', s=80, label='Intraday Stop')
        
        ax0.set_title("Price Action", fontsize=10)
        ax0.legend()
        ax0.grid(True, alpha=0.3)
        
        # 图2: 杠杆归零过程
        ax1 = axes[1]
        ax1.plot(subset.index, subset['position'].abs(), color='#FF9900', linewidth=2, label='Position (Abs)')
        ax1.fill_between(subset.index, subset['position'].abs(), color='#FF9900', alpha=0.1)
        ax1.set_title("Position Deleveraging", fontsize=10)
        ax1.set_ylabel("Position Size")
        ax1.grid(True, alpha=0.3)
        
        # 图3: 波动率飙升
        ax2 = axes[2]
        ax2.plot(subset.index, subset['ann_vol_pct'], color='blue', label='Realized Vol')
        ax2.axhline(Config.TARGET_VOLATILITY, color='green', linestyle='--', label='Target')
        ax2.set_title("Volatility Spike", fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 格式化日期显示
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
        plt.xticks(rotation=45)
        
        # 保存
        fname = f"Snapshot_{idx+1}_{date_str}.png"
        save_path = os.path.join(results_dir, fname)
        plt.savefig(save_path, dpi=200)
        plt.close() # 关闭画布释放内存
        
        print(f"📸 快照已生成: {fname}")

# ==========================================
# 🚀 主任务
# ==========================================
def mission_start():
    print("🚀 Jarvis System Initializing (Institutional Static Mode)...")
    
    # 强制重载配置
    import importlib
    import config
    importlib.reload(config)

    print(f"📂 Data Path: {Config.DATA_PATH}")
    
    # 1. 加载
    df = load_price_data(Config.DATA_PATH)
    if df.empty: 
        print("❌ Data not found.")
        return

    # 2. 计算
    print("🧠 Calculating Alpha...")
    df = calculate_scaled_forecast(df)
    
    print(f"🛡️ Risk Engine V2.1 (StopLoss={Config.STOP_LOSS_SIGMA}σ)...")
    df = calculate_position_target(df, buffer=Config.POSITION_BUFFER)
    
    print("⚡ Backtesting...")
    df_res = run_vectorized_backtest(df, fee_rate=Config.FEE_RATE)
    
    # 3. 业绩
    final = df_res['equity'].iloc[-1]
    sharpe = (df_res['net_log_ret'].mean() / df_res['net_log_ret'].std()) * np.sqrt(365*24)
    print("-" * 40)
    print(f"🏆 Final Equity: {final:.4f}")
    print(f"📊 Sharpe Ratio: {sharpe:.2f}")
    print("-" * 40)
    
    # 4. 生成全景图 (Matplotlib)
    plot_full_report(df_res)
    
    # 5. 生成特写快照 (New Feature!)
    plot_crash_snapshots(df_res, top_n=3)

if __name__ == "__main__":
    mission_start()