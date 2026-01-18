import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def calculate_scaled_forecast(df:pd.DataFrame,windows:list[int]=[8,16,32,64])->pd.DataFrame:
    """
    Day 19: 连续型趋势预测系统 (Carver Logic)
    
    这是 Jarvis 的核心逻辑升级。我们将策略从简单的“开关(0/1)”升级为“油门(0-20)”。
    
    参数:
        df: 必须包含 'close' 的数据表
        windows: 我们观察市场的四个窗口 (短期->长期)，默认 [8, 16, 32, 64]
    """
    # 1. 创建副本：好的习惯，防止改坏原始数据
    data=df.copy()
    # ==========================================
    # 🧠 步骤 A: 感知市场体温 (波动率 Volatility)
    # ==========================================
    # 物理意义：
    # 如果市场波动很大 (体温高)，价格涨 1000 点可能只是噪音。
    # 如果市场波动很小 (体温低)，价格涨 1000 点可能就是巨变。
    # 我们用 20 天的滚动标准差来衡量。
    volatility=data["close"].rolling(window=20).std()
    # 🔧 工程细节：加一个极小数 (epsilon)，防止波动率为 0 时除法报错
    volatility=volatility+1e-8
    # ==========================================
    # ⚡ 步骤 B: 并行计算 (Vectorization)
    # ==========================================
    # 这里我们不用 for 循环一行行算，太慢了。
    # 我们用“列表推导式”，一次性生成 4 个维度的预测。
    # 公式：(当前价格 - 均线) / 波动率
    forecast_list=[
        (data["close"]-data['close'].rolling(window=w).mean())/volatility
        for w in windows
    ]
    # ==========================================
    # ⚖️ 步骤 C: 委员会投票 (Aggregation)
    # ==========================================
    # 我们把 4 个维度的结果拼成一张表，然后横向取平均。
    # 物理意义：短期(8)可能看涨，长期(64)可能看跌，我们听取所有人的意见，取折中值。
    forecast_df=pd.concat(forecast_list,axis=1)
    combined_forecast=forecast_df.mean(axis=1)

    # ==========================================
    # 🛡️ 步骤 D: 风控与映射 (Post-processing)
    # ==========================================
    # 1. clip(-2, 2): 安全阀。
    #    不管市场多么疯狂，我们认为偏离度超过 2 倍标准差就是极限了。
    #    防止因为黑天鹅事件导致信号爆表，系统梭哈。
    # 2. * 10: 放大。
    #    把小数 (-2.0 ~ 2.0) 变成直观的整数 (-20 ~ +20)。
    #    +20 = 极强多头，-20 = 极强空头，0 = 震荡/无方向。
    final_forecast=(
        combined_forecast
        .clip(lower=-2.0,upper=2.0)
        .mul(10)
        .fillna(0)# 填补最开始计算不出来的空值
    )
    # 把计算好的大脑信号写入数据表
    data['forecast']=final_forecast
    return data
# ==========================================
# 🎨 验证环节 (让数据说话)
# ==========================================
if __name__ == "__main__":
    print("🚀 Jarvis Day 19: 正在启动大脑皮层...")
    # 1. 尝试读取数据
    try:
        df=pd.read_csv("Binance_BTCUSDT_1h.csv")
        df_columns=[c.strip().lower() for c in df.columns]
        #智能识别时间
        if 'timestamp' in df.columns:
            df['time']=pd.to_datetime(df['timestamp'])
        elif 'unix' in df.columns:
            df['time']=pd.to_datetime(df['unix'],unit='ms')
        if 'time' in df.columns:
            df=df.set_index('time').sort_index()
        #过滤掉过远的数据 只看最近几年
        df=df[df.index>'2020-01-01']
    except Exception as e:
        print(f"⚠️ 没找到数据，我先造点假数据演示给你看逻辑: {e}")
        dates=pd.date_range(start='2023-01-01',period=1000,frea='1h')
        # 造一个先涨后跌的假价格
        price=10000+np.cumsum(np.random.randn(1000))*100
        df=pd.DataFrame({'close':price},index=dates)
    # 2. 核心计算 (调用上面的函数)
    df_result=calculate_scaled_forecast(df)
    print("✅ 计算完成！来看看 Jarvis 现在的脑电波：")
    print(df_result[['close','forecast']].tail(10))

    # 3. 画图
    plt.figure(figsize=(12, 8))
    # 上半部分：币价
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df_result.index, df_result['close'], color='black', label='Price', linewidth=1)
    ax1.set_title('BTC Price Action')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 下半部分：Forecast (预测值)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1) # 共享 X 轴，方便对比
    # 画出我们的信号线 (蓝色)
    ax2.plot(df_result.index, df_result['forecast'], color='#0066CC', label='Jarvis Forecast', linewidth=1.5)
    # 涂色：由红变绿，一目了然
    # 红色区域 = 做多信号 (Forecast > 0)
    ax2.fill_between(df_result.index,df_result['forecast'],0,
                    where=(df_result['forecast']>0),color='red',alpha=0.3)
    # 绿色区域 = 做空信号 (Forecast < 0)
    ax2.fill_between(df_result.index, df_result['forecast'], 0, 
                     where=(df_result['forecast'] < 0), color='green', alpha=0.3)
    # 画几条参考线，方便你看
    ax2.axhline(0, color='black', linewidth=1) # 零轴
    ax2.axhline(10, color='red', linestyle='--', alpha=0.5) # 强多头线 (+10)
    ax2.axhline(-10, color='green', linestyle='--', alpha=0.5) # 强空头线 (-10)

    ax2.set_title('Jarvis Forecast Signal (-20 to +20)')
    ax2.set_ylim(-22, 22) # 固定 Y 轴范围，看起来更整齐
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("Day19_Forecast_Chart.png")
    print("\n📸 战报已生成: Day19_Forecast_Chart.png")
    print("👉 快去打开这张图，看看红绿波浪是不是比之前的“死叉”更灵敏？")