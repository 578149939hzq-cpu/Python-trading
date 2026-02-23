import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 引入 Config 与三层架构：数据 / 策略 / 风控执行
from config import Config
from jarvis_engine.data_handler import CSVDataLoader
from jarvis_engine.signal_generator import calculate_scaled_forecast
from jarvis_engine.portfolio_risk import calculate_position_target, run_vectorized_backtest

# ==========================================
# 📊 全景战报 (Full History Report)
# ==========================================
def plot_full_report(df_res):
    print("🎨 Generating Institutional Static Report (Matplotlib)...")
    
    plt.style.use('bmh') 
    
    # 改为 5 行子图，新增 "Survival Monitor"
    fig, axes = plt.subplots(5, 1, figsize=(14, 24), sharex=True)
    
    # --- 子图 1: 净值曲线 ---
    ax0 = axes[0]
    ax0.plot(df_res.index, df_res['equity'], color='#FF9900', linewidth=2, label='Jarvis Strategy')
    ax0.plot(df_res.index, df_res['buy_hold_equity'], color='gray', linestyle='--', alpha=0.6, label='Buy & Hold')
    ax0.set_title("🏆 Equity Curve (Survival Mode)", fontweight='bold', fontsize=12)
    ax0.set_ylabel("Account Value ($)")
    ax0.legend(loc='upper left')
    
    # --- 子图 2: 价格与熔断点 ---
    ax1 = axes[1]
    ax1.plot(df_res.index, df_res['close'], color='black', alpha=0.6, linewidth=1, label='Price')
    
    # 标记熔断点 (Red Triangle)
    meltdowns = df_res[df_res.get('sigma_event', False) == True]
    if not meltdowns.empty:
        ax1.scatter(meltdowns.index, meltdowns['close'], color='red', marker='v', s=80, zorder=5, label=f'Survival Stop Triggered')
        
    ax1.set_title(f"📉 Price Action", fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left')

    # --- [NEW] 子图 3: 生存监控 (Survival Monitor) ---
    # 核心特征分析：为什么归零？
    ax2 = axes[2]
    
    # 1. 计算每小时涨跌幅
    hourly_ret = df_res['close'].pct_change().fillna(0)
    
    # 2. 绘制涨跌幅 (灰色区域)
    ax2.fill_between(df_res.index, hourly_ret, 0, color='gray', alpha=0.3, label='Hourly Return')
    
    # 3. 绘制灾难阈值 (红线, 负值)
    # sl_threshold 是正数 (e.g. 0.06)，我们需要画成 -0.06
    if 'sl_threshold' in df_res.columns:
        threshold_line = -1 * df_res['sl_threshold']
        ax2.plot(df_res.index, threshold_line, color='red', linewidth=1.5, linestyle='--', label=f'Crash Threshold ({Config.SURVIVAL_ATR_MULTIPLIER}x ATR)')
        
        # 4. 标记刺穿时刻 (特征)
        crashes = df_res[hourly_ret < threshold_line]
        if not crashes.empty:
            ax2.scatter(crashes.index, hourly_ret.loc[crashes.index], color='red', marker='x', s=50, label='Breach Point')

    ax2.set_title("☠️ Survival Monitor (Return vs Threshold)", fontweight='bold', fontsize=12)
    ax2.set_ylabel("Return %")
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    # --- 子图 4: 波动率 ---
    ax3 = axes[3]
    ax3.plot(df_res.index, df_res['ann_vol_pct'], color='blue', linewidth=1.5, label=f'Long-Term Vol (Span={Config.VOL_LOOKBACK})')
    ax3.axhline(Config.TARGET_VOLATILITY, color='green', linestyle='--', linewidth=2, label=f'Target ({Config.TARGET_VOLATILITY})')
    ax3.set_title("🌊 Volatility Regime", fontweight='bold', fontsize=12)
    ax3.set_ylabel("Ann Vol %")
    ax3.legend(loc='upper left')

    # --- 子图 5: 仓位/杠杆 ---
    ax4 = axes[4]
    ax4.plot(df_res.index, df_res['leverage_ratio'], color='gray', alpha=0.5, label='Max Leverage Cap')
    # 绘制实际仓位 (填充橙色)
    ax4.fill_between(df_res.index, df_res['position'].abs(), 0, color='#FF9900', alpha=0.5, label='Actual Position')
    
    # 再次强调归零点
    if not meltdowns.empty:
        # 在归零的地方画红竖线
        for date in meltdowns.index:
            ax4.axvline(date, color='red', alpha=0.3, linestyle=':')

    ax4.set_title("⚙️ Leverage System (Zero = Meltdown)", fontweight='bold', fontsize=12)
    ax4.set_ylabel("Leverage")
    ax4.legend(loc='upper left')

    plt.tight_layout()
    
    results_dir = os.path.join(Config.BASE_DIR, "data_results")
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    save_path = os.path.join(results_dir, "Jarvis_Full_Report.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"✅ 全景报告已保存: {save_path}")

# ==========================================
# 📸 2. 智能特写快照 (增强版)
# ==========================================
def plot_crash_snapshots(df_res, top_n=3):
    print(f"📸 Generating Top {top_n} Crash Snapshots...")
    
    risk_events = df_res[df_res.get('sigma_event', False) == True].copy()
    
    if risk_events.empty:
        print("🎉 Good News: No DISASTER events triggered.")
        return

    risk_events = risk_events.sort_values('ann_vol_pct', ascending=False)
    risk_events['date'] = risk_events.index.date
    top_days = risk_events.drop_duplicates(subset=['date']).head(top_n)
    
    results_dir = os.path.join(Config.BASE_DIR, "data_results")

    for idx, (timestamp, row) in enumerate(top_days.iterrows()):
        # 缩短观察窗口，放大细节 (前后 2 天)
        start_t = timestamp - pd.Timedelta(days=2) 
        end_t = timestamp + pd.Timedelta(days=2)
        subset = df_res.loc[start_t:end_t]
        
        if subset.empty: continue

        # 4行特写
        fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
        date_str = timestamp.strftime('%Y-%m-%d')
        fig.suptitle(f"🚨 DISASTER FORENSICS: {date_str}", fontsize=16, fontweight='bold', color='darkred')
        
        # 图1: 价格
        ax0 = axes[0]
        ax0.plot(subset.index, subset['close'], color='black', label='Price')
        local_melt = subset[subset.get('sigma_event', False) == True]
        ax0.scatter(local_melt.index, local_melt['close'], color='red', marker='v', s=100, label='Survival Trigger')
        ax0.legend()
        ax0.grid(True, alpha=0.3)
        
        # [NEW] 图2: 刺穿特征 (Return vs Threshold)
        ax1 = axes[1]
        hourly_ret = subset['close'].pct_change().fillna(0)
        ax1.bar(subset.index, hourly_ret, color='gray', alpha=0.5, label='Hourly Ret', width=0.04) # bar chart
        
        if 'sl_threshold' in subset.columns:
            thresh = -1 * subset['sl_threshold']
            ax1.plot(subset.index, thresh, color='red', linestyle='--', label='Crash Threshold')
            
            # 标记刺穿
            breach = subset[hourly_ret < thresh]
            ax1.scatter(breach.index, hourly_ret.loc[breach.index], color='red', marker='x', s=100, zorder=5)
            
        ax1.set_title("Why Zero? (Return pierced Threshold)", fontsize=10, fontweight='bold')
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)
        
        # 图3: 仓位归零
        ax2 = axes[2]
        ax2.fill_between(subset.index, subset['position'].abs(), 0, color='#FF9900', alpha=0.6, label='Position')
        ax2.set_ylabel("Position")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 图4: 波动率
        ax3 = axes[3]
        ax3.plot(subset.index, subset['ann_vol_pct'], color='blue', label='Vol')
        ax3.axhline(Config.TARGET_VOLATILITY, color='green', linestyle='--', label='Target')
        ax3.grid(True, alpha=0.3)
        
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %Hh'))
        plt.xticks(rotation=45)
        
        fname = f"Snapshot_{idx+1}_{date_str}.png"
        save_path = os.path.join(results_dir, fname)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"📸 快照已生成: {fname}")
# ==========================================
# 📊 [V3.5 New] Sortino Metric
# ==========================================
def calculate_sortino(series, target_return=0, periods=24*365):

    """
    辅助函数：计算年化 Sortino Ratio
    只考虑下行偏差 (Downside Deviation)，不惩罚上涨波动。
    """
    # 1. 计算年化收益
    mean_ret = series.mean() * periods
    
    # 2. 计算下行偏差 (只取负收益部分)
    downside_returns = series[series < target_return]
    
    if len(downside_returns) == 0:
        return np.nan
        
    # 下行标准差
    downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(periods)
    
    if downside_std == 0:
        return np.nan
        
    return mean_ret / downside_std
# ==========================================
# 🚀 主任务
# ==========================================
def calculate_drawdown_metrics(equity_series):

    """
    计算最大回撤 (MDD) 和 Calmar 比率
    """
    # 1. 计算累计最大值 (Running Max)
    roll_max = equity_series.cummax()
    
    # 2. 计算回撤序列 (Drawdown Series)
    drawdown = (equity_series / roll_max) - 1.0
    
    # 3. 提取最大回撤 (是一个负数，例如 -0.40)
    max_drawdown = drawdown.min()
    
    # 4. 计算 Calmar Ratio (年化收益 / |最大回撤|)
    # 假设数据是 1小时频率，总长度 N
    total_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1.0
    n_years = len(equity_series) / (365 * 24)
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    
    # 防止分母为 0
    if max_drawdown == 0:
        calmar = np.nan
    else:
        calmar = ann_ret / abs(max_drawdown)
        
    return max_drawdown, calmar
def calculate_trade_metrics(df_res: pd.DataFrame) -> None:

    """
    [V3.7 Analytics] 交易维度统计
    将连续的持仓序列拆解为独立的 'Round-Trip' 交易进行统计。
    使用纯 Pandas 向量化 groupby/agg 消除显式 Python 循环。
    """
    df = df_res.copy()

    # 1. 定义交易分组 (Trade Grouping)
    # 逻辑: 只要仓位符号(多/空)发生变化，就算作新的一笔交易
    # 0 (空仓) 也会被分一组，后面会过滤掉
    # 精度过滤: 忽略 < 0.01 的微小仓位(可能是浮点误差)
    df["pos_sign"] = np.sign(df["position"])
    df.loc[df["position"].abs() < 0.01, "pos_sign"] = 0

    # 当符号变化时，累加 group_id
    df["trade_id"] = (df["pos_sign"] != df["pos_sign"].shift(1)).cumsum()

    # 获取时间索引 (假设索引是 datetime，如果不是请先转换)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 2. 仅保留非空仓段，并向量化聚合
    df_active = df[df["pos_sign"] != 0].copy()
    if df_active.empty:
        print("⚠️ No trades executed.")
        return

    df_active = df_active.reset_index()
    # 索引列名可能为 "index" 或 None 等，统一改为 timestamp 供 agg 使用
    ts_col = df_active.columns[0]
    df_active = df_active.rename(columns={ts_col: "timestamp"})

    grouped = df_active.groupby("trade_id")
    agg = grouped.agg(
        start_time=("timestamp", "first"),
        end_time=("timestamp", "last"),
        pos_sign_first=("pos_sign", "first"),
        trade_return=("net_log_ret", "sum"),
    )

    # 计算每笔交易的持续时间 (小时)
    duration_hours = (
        (agg["end_time"] - agg["start_time"]).dt.total_seconds() / 3600.0
    )

    # 方向与结果 DataFrame，与原实现字段保持一致
    df_trades = agg.copy()
    df_trades["duration"] = duration_hours
    df_trades["direction"] = np.where(
        df_trades["pos_sign_first"] > 0, "Long", "Short"
    )
    df_trades = df_trades.reset_index()[
        ["trade_id", "direction", "duration", "trade_return"]
    ].rename(columns={"trade_return": "return"})

    # 3. 计算核心指标
    total_trades = len(df_trades)
    win_trades = len(df_trades[df_trades["return"] > 0])
    loss_trades = len(df_trades[df_trades["return"] <= 0])

    win_rate = win_trades / total_trades if total_trades > 0 else 0.0

    # 盈亏比 (Profit Factor): 总盈利 / |总亏损|
    gross_win = df_trades[df_trades["return"] > 0]["return"].sum()
    gross_loss = abs(df_trades[df_trades["return"] <= 0]["return"].sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf

    # 平均持仓 (小时)
    avg_duration = df_trades["duration"].mean()

    # 平均单笔收益 (已扣费)
    avg_pnl = df_trades["return"].mean()

    # 4. 打印战报
    print("\n📊 --- Trade Statistics (Round-Trip) ---")
    print(f"🔹 Total Trades    : {total_trades}")
    print(f"🔹 Win Rate        : {win_rate:.2%} ({win_trades} W / {loss_trades} L)")
    print(f"🔹 Profit Factor   : {profit_factor:.2f}")
    print(f"🔹 Avg PnL / Trade : {avg_pnl:.2%}")
    print(f"🔹 Avg Duration    : {avg_duration:.1f} Hours ({avg_duration/24:.1f} Days)")

    if win_rate < 0.4 and profit_factor > 1.2:
        print("✅ 风格: 典型的趋势策略 (低胜率，高盈亏比)。抓大放小。")
    elif win_rate > 0.5:
        print("✅ 风格: 胜率较高，稳健型。")
    else:
        print("⚠️ 风格: 胜率与赔率需进一步平衡。")
    print("------------------------------------------\n")
def calculate_performance_summary(equity_series, periods_per_year=24*365):
    """
    [V4.6] 计算年化收益与复合增长率
    """
    # 1. 计算总收益率
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1.0
    
    # 2. 计算回测跨越的年数
    # 数据点总数 / (每年的小时数)
    n_days = len(equity_series) / 24
    n_years = len(equity_series) / periods_per_year
    
    # 3. 计算年化收益率 (CAGR - 复合年均增长率)
    if n_years > 0:
        ann_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        ann_return = np.nan
        
    return total_return, ann_return, n_days
def mission_start():
    print("🚀 Jarvis System Initializing (V3.3 Visualization Upgrade + Leverage Stats)...")
    
    import importlib
    import config
    importlib.reload(config)

    print(f"📂 Data Path: {Config.DATA_PATH}")

    data_loader = CSVDataLoader(Config.DATA_PATH)
    df = data_loader.fetch_data()
    if df.empty: 
        print("❌ Data not found.")
        return

    print("🧠 Calculating Alpha...")
    df = calculate_scaled_forecast(df)
    
    print(f"🛡️ Risk Engine V3.3 (Survival Threshold = {Config.SURVIVAL_ATR_MULTIPLIER}x ATR)...")
    df = calculate_position_target(df, buffer=Config.POSITION_BUFFER)
    
    print("⚡ Backtesting...")
    df_res = run_vectorized_backtest(df, fee_rate=Config.FEE_RATE)
    
    # ------------------------------------------------------
    # [新增] 杠杆率统计 (Leverage Statistics)
    # ------------------------------------------------------
    # 计算平均持仓杠杆 (绝对值)
    avg_leverage = df_res['position'].abs().mean()
    # 计算最大使用杠杆
    max_leverage_used = df_res['position'].abs().max()
    # ------------------------------------------------------

    final = df_res['equity'].iloc[-1]
    sharpe = (df_res['net_log_ret'].mean() / df_res['net_log_ret'].std()) * np.sqrt(365*24)
    total_ret_strat, ann_ret_strat, n_days = calculate_performance_summary(df_res['equity'])
    total_ret_bh, ann_ret_bh, _ = calculate_performance_summary(df_res['buy_hold_equity'])
    
    mdd_strat, calmar_strat = calculate_drawdown_metrics(df_res['equity'])
    # [V3.5 New] Sortino Analysis
    print("\n📊 --- Performance Analytics ---")
    strat_sortino = calculate_sortino(df_res['net_log_ret'])
    btc_sortino = calculate_sortino(df_res['market_log_ret'])
    print(f"🔹 Backtest Period : {n_days:.1f} Days ({n_days/365:.2f} Years)")
    print(f"🔹 Total Return    : {total_ret_strat:.2%} (B&H: {total_ret_bh:.2%})")
    print(f"🚀 Annualized Ret  : {ann_ret_strat:.2%} (B&H: {ann_ret_bh:.2%})")
    print(f"🏆 Final Equity: ${final:,.2f} (Initial: ${Config.INITIAL_CAPITAL})")
    print(f"📈 Sharpe Ratio : {sharpe:.2f}")
    
    # [新增] 打印杠杆率数据
    print(f"⚖️ Avg Leverage  : {avg_leverage:.2f}x (Target: ~1.0x)")
    print(f"🚀 Max Leverage  : {max_leverage_used:.2f}x (Cap: {Config.MAX_LEVERAGE}x)")
    
    print(f"🔹 Strategy Sortino: {strat_sortino:.4f}")
    print(f"🔸 Bitcoin Sortino : {btc_sortino:.4f}")
    calculate_trade_metrics(df_res)
    print("\n📉 --- Risk Analysis (Drawdown) ---")
    
    # 1. 策略回撤
    mdd_strat, calmar_strat = calculate_drawdown_metrics(df_res['equity'])
    
    # 2. 只有买入持有的回撤
    mdd_bh, calmar_bh = calculate_drawdown_metrics(df_res['buy_hold_equity'])
    
    print(f"🔹 Strategy MDD   : {mdd_strat:.2%} (Calmar: {calmar_strat:.2f})")
    print(f"🔸 Buy & Hold MDD : {mdd_bh:.2%} (Calmar: {calmar_bh:.2f})")
    
    if abs(mdd_strat) < abs(mdd_bh):
        print("✅ 结论: 策略显著降低了极端风险。")
    else:
        print("⚠️ 结论: 策略风险控制未跑赢大盘，请检查杠杆率。")
    print("------------------------------------------\n")
    
    if strat_sortino > btc_sortino:
        print("✅ 结论: 策略在承担单位下行风险时，回报优于囤币。")
    else:
        print("⚠️ 结论: 策略下行风险控制仍需优化。")
    print("-" * 40)
    
    plot_full_report(df_res)
    plot_crash_snapshots(df_res, top_n=3)

if __name__ == "__main__":
    mission_start()