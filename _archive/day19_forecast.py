import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
def calculate_scaled_forecast(df: pd.DataFrame, spans=[8, 16, 32, 64]) -> pd.DataFrame:
   """
    Day 22 Upgrade: 基于 Carver 逻辑的 EWMA 集成预测系统
    
    参数:
        df: 必须包含 'close' 列
        spans: EWMA 的跨度列表 (Carver 风格：快/中/慢/极慢)
    """
   data=df.copy()

   # 🧠 步骤 A: 计算 EWMA 波动率 (Vol)
   # 物理意义：昨天的波动率比上个月的波动率更重要。
   volatility=data['close'].ewm(span=36).std()

   #防止波动率除0
   volatility=volatility+1e-8
   # ==========================================
   # ⚡ 步骤 B: 向量化子策略 (Sub-Forecasts)
   # ==========================================
   # 我们遍历 span 列表，生成多个维度的“原始预测”
   # 公式：(价格 - EWMA) / 波动率
   forecast_list=[]

   for span in spans:
      #计算该周期的指数均线
      ema=data['close'].ewm(span=span).mean()
      #标准化差异
      raw_forecast=(data['close']-ema)/volatility
      forecast_list.append(raw_forecast)
# ==========================================
# ⚖️ 步骤 C: 集成 (Ensemble)
# ==========================================
# 将列表转为 DataFrame (列 = 不同的 span)
   forecast_df=pd.concat(forecast_list,axis=1)
   # 等权平均：听取所有周期的意见
   combined_forecast=forecast_df.mean(axis=1)
# ==========================================
# 🛡️ 步骤 D: 后处理 (Post-Processing)
# ==========================================
# 1. 放大: 乘以 10，映射到 -20 ~ +20
# 2. 截断: 超过 20 的极值强制拉回，防止系统爆炸
   final_forecast=(combined_forecast*10.0) .clip(lower=-20.0,upper=20.0)
   #填充清洗
   final_forecast=final_forecast.fillna(0)

   data['forecast']=final_forecast
   # 为了画图方便，我们把 64 周期的均线也存下来
   data['ema_64'] = data['close'].ewm(span=64).mean()
    
   return data
# ==========================================
# 📊 验证与可视化
# ==========================================
if __name__ == "__main__":
    try:
        print("🚀 [Day 22] 正在智能搜索数据文件...")

        # ====================================================
        # 🗺️ 自动寻路逻辑 (方法二的核心)
        # ====================================================
        
        # 1. 找到当前脚本(代码)在哪里
        # 结果可能是: .../Python-trading/jarvis_engine
        current_script_folder = os.path.dirname(os.path.abspath(__file__))

        # 2. 找到项目根目录 (往上跳一级)
        # 结果可能是: .../Python-trading
        project_root = os.path.dirname(current_script_folder)

        # 3. 拼接数据的绝对路径
        # 意思是在根目录下，找 data_raw 文件夹，再找那个 csv
        csv_path = os.path.join(project_root, "data_raw", "Binance_BTCUSDT_1h.csv")

        print(f"📂 锁定文件路径: {csv_path}")
        
        # 4. 读取 (这时候就不会报错了)
        df = pd.read_csv(csv_path)
        
        # ====================================================
        # 下面接着写你原来的代码...
        # ====================================================
        
        df.columns = [c.strip().lower() for c in df.columns]
        # ... (后续代码保持不变)
        if 'timestamp' in df.columns: df['time'] = pd.to_datetime(df['timestamp'])
        elif 'unix' in df.columns: df['time'] = pd.to_datetime(df['unix'], unit='ms')
        df = df.set_index('time').sort_index()
        # 取最近 1000 个小时的数据，方便看细节
        df = df[df.index > '2023-01-01'].tail(1000)
        
        print("🧠 正在计算 EWMA Forecast...")
        df_res = calculate_scaled_forecast(df, spans=[8, 16, 32, 64])
        
        # 画图
        plt.figure(figsize=(12, 10))
        
        # 图1: 价格与长期趋势线
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(df_res.index, df_res['close'], color='black', label='Price', alpha=0.6)
        ax1.plot(df_res.index, df_res['ema_64'], color='#FF9900', label='EWMA (64)', linewidth=2)
        ax1.set_title('BTC Price vs EWMA(64) Trend')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: 最终预测信号
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(df_res.index, df_res['forecast'], color='blue', label='Final Forecast')
        
        # 画出红绿区域
        ax2.fill_between(df_res.index, df_res['forecast'], 0, 
                         where=(df_res['forecast']>0), color='red', alpha=0.3)
        ax2.fill_between(df_res.index, df_res['forecast'], 0, 
                         where=(df_res['forecast']<0), color='green', alpha=0.3)
        
        # 阈值线
        ax2.axhline(0, color='black', linewidth=1)
        ax2.axhline(10, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(-10, color='green', linestyle='--', alpha=0.5)
        
        ax2.set_title('Carver EWMA Forecast (-20 to +20)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("Day22_EWMA_Forecast.png")
        print("\n📸 升级完毕！图表已保存: Day22_EWMA_Forecast.png")
        print("👉 观察图1中的橙色线：你会发现 EWMA 比之前的 SMA 更加平滑且贴合价格。")

    except Exception as e:
        print(f"❌ 出错: {e}")