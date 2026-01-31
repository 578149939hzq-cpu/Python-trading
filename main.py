import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 第一步：先导入 config 模块
import config

# ==========================================
# 🚑 紧急热修复 (Hotfix) - 适配层 (保持不变)
# ==========================================
# ⚠️ 注意：这段代码必须在 "from jarvis_engine.alpha" 之前执行！

# --- 修复 A: 强行纠正路径错误 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 修正读取路径
REAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data_raw", "Binance_BTCUSDT_1h.csv")
config.DATA_PATH = REAL_DATA_PATH 

# --- 修复 B: 伪造 Config 类 ---
class ConfigAdapter:
    VOL_LOOKBACK = config.VOLATILITY_SPAN 
    STRATEGY_PARAMS = config.STRATEGY_PARAMS
    WEIGHTS = config.WEIGHTS

config.Config = ConfigAdapter

# ==========================================
# 🛑 补丁打完后，再导入 alpha 模块
# ==========================================
from jarvis_engine.alpha import load_price_data
from jarvis_engine.alpha import calculate_scaled_forecast
from jarvis_engine.alpha import calculate_position_target
from jarvis_engine.alpha import run_vectorized_backtest

def mission_start():
    print("🚀 Jarvis System Initializing...")
    
    # 1. 加载数据
    print(f"📂 Loading data from: {config.DATA_PATH}")
    df = load_price_data(config.DATA_PATH)
    
    if df.empty:
        print("❌ 数据加载失败，请检查 data_raw 文件夹。")
        return

    # 2. 计算 Alpha (大脑)
    print("🧠 Calculating Alpha (EWMAC)...")
    df = calculate_scaled_forecast(df)
    
    # 3. 计算 仓位 (手脚)
    print(f"🛡️ Adjusting Positions (Buffer={config.POSITION_BUFFER})...")
    df = calculate_position_target(df, buffer=config.POSITION_BUFFER)
    
    # 4. 回测 (模拟场)
    print("⚡ Running Vectorized Backtest...")
    df_result = run_vectorized_backtest(df, fee_rate=config.FEE_RATE)
    
    # 5. 战报展示
    if 'equity' not in df_result.columns:
        print("❌ 回测未能生成净值曲线。")
        return

    final_equity = df_result['equity'].iloc[-1]
    total_return = (final_equity - 1) * 100
    
    net_ret = df_result['net_log_ret']
    std = net_ret.std()
    sharpe = (net_ret.mean() / std) * np.sqrt(365 * 24) if std != 0 else 0

    print("-" * 40)
    print(f"🏆 最终战报 (Final Report)")
    print(f"💰 最终净值: {final_equity:.4f}")
    print(f"📈 总回报率: {total_return:.2f}%")
    print(f"📊 夏普比率: {sharpe:.2f}")
    print("-" * 40)
    
    # ==========================================
    # 📸 6. 画图并保存 (升级部分)
    # ==========================================
    
    # A. 准备文件夹
    # 在项目根目录下，找一个叫 data_results 的文件夹
    results_dir = os.path.join(PROJECT_ROOT, "data_results")
    
    # 如果文件夹不存在，就自动创建一个 (os.makedirs 会帮你搞定)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"📁 已自动创建结果文件夹: {results_dir}")
        
    # B. 设置图片文件名
    # 我们可以加上夏普比率在文件名里，方便以后对比
    file_name = f"Backtest_Result_Sharpe_{sharpe:.2f}.png"
    save_path = os.path.join(results_dir, file_name)

    # C. 开始画图
    plt.figure(figsize=(12, 6))
    
    # 画 Buy & Hold (基准)
    plt.plot(df_result.index, df_result['buy_hold_equity'], 
             label='Buy & Hold (BTC)', color='gray', linestyle='--', alpha=0.5)
    
    # 画 Jarvis 策略
    plt.plot(df_result.index, df_result['equity'], 
             label='Jarvis Strategy', color='#FF9900', linewidth=2)
    
    plt.title(f'Jarvis Strategy Equity Curve (Sharpe: {sharpe:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # D. 保存图片 (关键一步!)
    # dpi=300 代表高清大图
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图片已保存至: {save_path}")
    
    # E. 最后再弹窗显示
    plt.show()

if __name__ == "__main__":
    mission_start()