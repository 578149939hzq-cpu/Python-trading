import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from day19_forecast import calculate_scaled_forecast
import plotly.graph_objects as go
def calculate_position_target(df:pd.DataFrame,forecast_col='forecast',buffer=0.1)->pd.DataFrame:
    """
    Day 20 核心逻辑：将预测信号转换为实际持仓，并应用“阻尼器”。
    
    参数:
        buffer: 缓冲区大小 (默认 0.1，即 10%)。
                只有当 (目标仓位 - 当前仓位) 的差值 > 10% 时，才真正调仓。
    """
    data=df.copy()
    # ==========================================
    # 1. 归一化 (Scaling)
    # ==========================================
    # 我们的 forecast 是 -20 到 +20
    # 我们需要的仓位是 -1.0 (满仓空) 到 +1.0 (满仓多)
    # 所以除以 10，然后掐头去尾
    ideal_position=data[forecast_col]/10.0
    ideal_position=ideal_position.clip(lower=-1.0,upper=1.0)

    # ==========================================
    # 2. 缓冲循环 (The Hysteresis Loop)
    # ==========================================
    # ⚠️ 难点：这是一个"有记忆"的过程。今天的持仓，取决于昨天手里拿着什么。
    # 这种逻辑很难用 Pandas 的 apply 并行化，所以我们回归 C 语言风格的循环。
    ideal_values=ideal_position.values
    n=len(ideal_values)

    #创建一个全0数组用来储存结果
    buffered_position=np.zeros(n)

    #初始建仓设置为0、
    current_pos=0.0

    #开始循环
    for i in range(n):
        ideal=ideal_values[i]
        if abs(ideal-current_pos)>buffer:
            current_pos=ideal
            # 否则 current_pos 保持不变 (也就是 else: current_pos = current_pos)
        else: current_pos = current_pos
        # 记录当天的最终决定
        buffered_position[i]=current_pos
    # 把算好的数组放回 DataFrame
    data['raw_target']=ideal_position #理想结果
    data['buffered_pos']=buffered_position #现实结果

    # ==========================================
    # 3. 防未来函数 (Lagging)
    # ==========================================
    # 今天的收盘价算出来的信号，只能在"明天开盘"执行。
    # 所以实际持仓必须向后移一位。
    data['position']=data['buffered_pos'].shift(1)

    # 填补因为 shift 产生的第一个空洞
    data['position']=data['position'].fillna(0)
    return data
if __name__ =="__main__":
    try:
        print("🚀 正在加载真实 BTC 数据...")
        df = pd.read_csv("Binance_BTCUSDT_1h.csv")
        # 2. 数据清洗
        df.columns = [c.strip().lower() for c in df.columns]
        if 'timestamp' in df.columns:
            df['time'] = pd.to_datetime(df['timestamp'])
        elif 'unix' in df.columns:
            df['time'] = pd.to_datetime(df['unix'], unit='ms')
        df = df.set_index('time').sort_index()
        df = df[df.index > '2020-01-01'] # 过滤掉太早的数据
        print("🧠 正在调用 Day 19 大脑计算 Forecast...")
        df = calculate_scaled_forecast(df)
        # 检查一下 Day 19 算没算出来
        if 'forecast' not in df.columns:
            raise ValueError("Day 19 函数没返回 forecast 列，请检查代码！")
    except ImportError:
        print("❌ 找不到 'day19_forecast.py' 文件！")
        print("💡 解决方法：请把 Day 19 的代码保存为 'day19_forecast.py' 并放在旁边。")
        exit()
    except Exception as e:
        print(f"❌ 数据处理出错: {e}")
        exit()
    # 2. 运行 Day 20 的逻辑
    print("🛡️ 正在应用阻尼器 (Buffer = 0.1)...")
    df_res=calculate_position_target(df,buffer=0.1)
    # # 3. 画图对比
    # print("🎨 正在生成交互式图表...")
    # fig=go.Figure()
    # # 第一条线：理想仓位 (灰色虚线)
    # fig.add_trace(go.Scatter(
    #     x=df_res.index,
    #     y=df_res['raw_target'],
    #     mode='lines',
    #     name='Raw Target(理想)',
    #     line=dict(color='gray',width=1,dash='dash'),
    #     opacity=0.5
    # ))
    # # 第二条线：实际持仓 (橙色实线 - 你的阶梯！)
    # fig.add_trace(go.Scatter(
    #         x=df_res.index, 
    #         y=df_res['buffered_pos'],
    #         mode='lines',
    #         name='Buffered Position (实际)',
    #         line=dict(color="#B433FF", width=3)
    #     ))
    # # 0轴参考线
    # fig.add_hline(y=0, line_color="white", opacity=0.2)
    # fig.update_layout(
    #         title='<b>Jarvis Day 20: 阻尼器效果分析</b> (请用鼠标滚轮缩放)',
    #         yaxis_title='仓位 (-1.0 到 1.0)',
    #         template='plotly_dark', # 深色背景
    #         hovermode='x unified'   # 鼠标放上去显示数值
    #     )
    # # 保存为 HTML
    # output_file = "Day20_Dampener_Interactive.html"
    # fig.write_html(output_file)
    # print(f"\n✅ 成功！请打开这个文件查看细节: {output_file}")
    # print("👉 双击 HTML 文件，在浏览器里 缩放(Zoom) 看看那些漂亮的阶梯吧！")
    # ==========================================
    # 3. 画图对比 (修改版：只看最后 500 小时)
    # ==========================================
    # 为了看清细节，我们只截取最后 500 行数据
    subset = df_res.tail(500)
    
    plt.figure(figsize=(12, 6))
    
    # 1. 画出原始的、躁动的理想仓位 (虚线)
    plt.plot(subset.index, subset['raw_target'], 
             label='Raw Target (Ideal)', color='gray', linestyle='--', alpha=0.5)
    
    # 2. 画出加了阻尼器后的、稳健的实际持仓 (实线)
    plt.plot(subset.index, subset['buffered_pos'], 
             label='Buffered Position (Actual)', color='#FF5733', linewidth=2)
    
    plt.title("The Dampener Effect (Last 500 Hours)")
    plt.ylabel("Position Size (-1.0 to 1.0)")
    plt.axhline(0, color='black', alpha=0.3)
    
    # 加上网格，更容易看清台阶
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("Day20_Dampener_Zoomed.png")
    print("\n📸 结果已保存: Day20_Dampener_Zoomed.png")
    print("👉 现在去看看新图片，你应该能看到明显的‘台阶’了！")

  