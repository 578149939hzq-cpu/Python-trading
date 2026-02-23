import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
# ==========================================
# 🚀 核心组件：光速回测引擎 (Vectorized)
# ==========================================
def run_vectorized_backtest(df:pd.DataFrame,fee_rate=0.0005)->pd.DataFrame:
    """
    Day 21 任务: 全向量化回测引擎
    
    输入:
        df: 必须包含 'close' 和 'position' (已 shift) 的 DataFrame
        fee_rate: 手续费率 (默认万五)
    输出:
        df: 包含 'equity' (资金曲线) 的 DataFrame
    """
    data=df.copy()

    #计算对数收益率
    data["market_log_ret"]=np.log(data['close']).diff().fillna(0)

    #计算策略回报
    # 矩阵点乘：持仓 * 市场回报
    # 如果做多 (1.0)，就赚市场的钱；如果做空 (-1.0)，就赚市场反向的钱；空仓 (0) 不赚不赔
    data['strategy_log_ret']=data['position']*data['market_log_ret']
    

    # Step C: 计算交易成本 (Transaction Costs)
    # .diff().abs() 计算仓位变化的绝对值 (比如从 1.0 变到 0.5，变化量是 0.5)
    # 这里的成本是相对于资本的比例拖累
    position_change=data['position'].diff().abs().fillna(0)
    data['cost']=position_change*fee_rate

    # ==========================================
    # Step D: 净回报与资金曲线 (Net Return & Equity)
    # ==========================================
    # 净回报 = 策略回报 - 交易成本
    data['net_log_ret']=data['strategy_log_ret']-data['cost']

    

    # 资金曲线 = exp(累计的对数回报)
    # 初始资金设为 1.0 (归一化)
    data['equity'] = np.exp(data['net_log_ret'].cumsum())
    data['buy_hold_equity']=np.exp(data['market_log_ret'].cumsum())

    return data
def calculate_metrics(df):
    """辅助函数：计算核心指标"""
    total_ret=df['equity'].iloc[-1]-1

    #计算Sharpe
    mean_ret=df['net_log_ret'].mean()
    std_ret=df['net_log_ret'].std()

    if std_ret==0:
        sharpe=0
    else:
        sharpe=(mean_ret/std_ret)*np.sqrt(365*24)
    return total_ret,sharpe
# ==========================================
# ⏱️ 性能测试场 (Benchmark Arena)
# ==========================================
if __name__ =="__main__":
    # 1. 制造海量假数据 (50,000 行，约等于 6 年的小时数据)
    print("🛠️ 正在制造 50,000 行测试数据...")
    np.random.seed(42)
    n_rows = 50000
    dates = pd.date_range(start='2018-01-01', periods=n_rows, freq='1h')

    #随机漫步价格
    price=10000*np.exp(np.cumsum(np.random.randn(n_rows)*0.001))

    positions=np.round(np.random.uniform(-1,1,n_rows),1)
    #uniform 均匀分布函数 随机生成-1到1之间的整数 判断仓位强弱
    positions_array = np.round(np.random.uniform(-1, 1, n_rows), 1)

    df_test=pd.DataFrame({
        'close':price,
        'position':positions_array
    },index=dates)
    df_test['position'] = df_test['position'].shift(1).fillna(0)
    print(f"✅ 数据就绪: {df_test.shape}")

    # 2. 启动计时器
    print("\n🏁 开始基准测试 (Benchmark)...")
    start_time = time.time()
    #调用函数
    df_result=run_vectorized_backtest(df_test,fee_rate=0.0005)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000 # 转换为毫秒
    # 3. 输出报告
    tot_ret, sharpe = calculate_metrics(df_result)
    
    print("-" * 40)
    print(f"🚀 执行耗时: {elapsed_time:.2f} ms") # 毫秒
    print("-" * 40)
    if elapsed_time < 50:
        print("🏆 评级: S级 (极速)")
        print("💬 评价: 这种速度足够你一晚上跑完几百万次参数组合。")
    elif elapsed_time < 100:
        print("🥈 评级: A级 (合格)")
    else:
        print("🐢 评级: C级 (太慢了，代码需要优化)")
        
    print("-" * 40)
    print(f"📈 策略总回报: {tot_ret*100:.2f}%")
    print(f"📊 夏普比率:   {sharpe:.2f}")

    # 4. 画图验证 (只画最后 1000 小时)
    plt.figure(figsize=(12, 6))
    subset = df_result.iloc[-1000:]
    plt.plot(subset.index, subset['buy_hold_equity'], label='Buy & Hold', color='gray', linestyle='--')
    plt.plot(subset.index, subset['equity'], label='Jarvis Strategy', color='orange', linewidth=2)
    plt.title(f"Performance Check (Last 1000 Hours) - Time: {elapsed_time:.2f}ms")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("Day21_Backtest_Performance.png")
    print("\n📸 回测图已生成: Day21_Backtest_Performance.png")


    

