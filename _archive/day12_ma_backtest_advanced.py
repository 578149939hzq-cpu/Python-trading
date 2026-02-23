import pandas as pd
import matplotlib.pyplot as plt
# equity_tools.py
def calc_equity_curve(initial_capital, trade_returns):
    equity = initial_capital
    equity_curve = [equity]
    for r in trade_returns:
        equity = equity * (1 + r)
        equity_curve.append(equity)
    return equity_curve


def calc_max_drawdown(equity_curve):
    max_peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > max_peak:
            max_peak = eq
        dd = eq / max_peak - 1
        if dd < max_dd:
            max_dd = dd
    return max_dd

def load_price_data(csv_path: str) -> pd.DataFrame:
    # 1. 跳过第一行说明文字读取
    df = pd.read_csv(csv_path, skiprows=1)
    
    # 2. 清洗列名
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ==== 🕵️‍♂️ 侦探代码：先看看最大值到底是多少 ====
    # (这一步是为了让你自己在终端看到罪魁祸首，运行一次后可注释掉)
    print(f"最大时间戳: {df['unix'].max()}") 
    print(f"样本数据: {df['unix'].head().tolist()}")

    # ==== ✅ 修复核心 ====
    # 1) 强制转成数字，如果有非数字的乱码，变成 NaN (errors='coerce')
    df["unix"] = pd.to_numeric(df["unix"], errors='coerce')
    
    # 2) 智能判断：如果是微秒(16位)，就除以1000变成毫秒
    # 1e14 是一个分界线：毫秒通常是 1.7e12，微秒是 1.7e15
    mask_micro = df["unix"] > 1e14 
    df.loc[mask_micro, "unix"] = df.loc[mask_micro, "unix"] / 1000

    # 3) 现在统一都是毫秒了，安全转换
    df["time"] = pd.to_datetime(df["unix"], unit="ms")

    # 4) 处理完设为索引
    df = df.set_index("time")
    df = df.sort_index()

    return df
def add_indicators(df:pd.DataFrame,short:int,long: int)->pd.DataFrame:
    """在 df 上添加收盘收益率和均线列。"""
    df=df.copy()

    df["ret"]=df["close"].pct_change()
    df[f"ma_{short}"]=df["close"].rolling(short).mean()
    df[f"ma_{long}"]=df["close"].rolling(long).mean()

    return df
# ========= 2. 核心：MA 策略回测函数 =========
def backtest_ma(
        df:pd.DataFrame,
        short:int=3,
        long:int=5,
        fee_rate:float=0.0005, #单边手续费
)->tuple[dict,pd.DataFrame]:
     """
    在给定 DataFrame 上跑只做多的 MA 策略回测。
    返回 (结果字典, 带回测列的 df)
    """
     df=add_indicators(df,short,long)
     #1) 生成信号:ma_short>ma_long->做多
     df["signal"]=0
     df.loc[df[f"ma_{short}"]>df[f"ma_{long}"],"signal"]=1
     #2)用前一根k的signal来参与当前收益，避免未来函数
     df["signal_shift"]=df["signal"].shift(1).fillna(0)
     #保持空仓或者无动作

     #3) 策略毛收益(未知手续费)
     df["strategy_ret_gross"]=df["signal_shift"]*df["ret"]
    #4）仓位变化&&手续费
    #position_change: 0=无变化，1=开仓或者平仓
     df["position_change"]=df["signal_shift"].diff().fillna(0).abs()

     #每次仓位变化都扣一次手续费，简化为fee_rate*资金
     #因为这里ret是收益率，所以手续费也用一个负的"收益率"近似处理
     df["fee_ret"]=-fee_rate*df["position_change"]
     
     #5)净收益=毛收益+手续费收益(负数)
     df["strategy_ret_net"]=df["strategy_ret_gross"]+df["fee_ret"]

     #6)累计收益曲线(从1开始的归一化净值)
     df["buy_hold_cum"]=(1+df["ret"]).cumprod()
     df["strategy_cum"]=(1+df["strategy_ret_net"]).cumprod()

     #7)关键指标计算
     buy_hold_final=df["buy_hold_cum"].iloc[-1]
     strategy_final=df["strategy_cum"].iloc[-1]

     buy_hold_total_return=buy_hold_final-1
     strategy_total_return=strategy_final-1

     #最大回撤
     strategy_equity=df["strategy_cum"].fillna(1).tolist()
     max_dd=calc_max_drawdown(strategy_equity)

     #简单sharpe(按日频率，年化因子252)
     #注意dropna避免NaN
     strat_ret_series=df["strategy_ret_net"].dropna()

     if len(strat_ret_series)>1 and strat_ret_series.std()!=0:
         sharpe=(strat_ret_series.mean()/strat_ret_series.std())*(252**0.5)
     else:
         sharpe=float("nan")

     result={
         "short":short,
         "long":long,
         "fee_rate":fee_rate,
         "buy_hold_total_return":buy_hold_total_return,
         "strategy_total_return":strategy_total_return,
         "max_drawdown": max_dd,
         "sharpe":sharpe,
     }
     return result,df
# ========= 3. 主程序：实际跑一下 =========
if __name__=="__main__":
    csv_path="Binance_BTCUSDT_1h.csv"
    df_raw=load_price_data(csv_path)
    #可以随便改参数
    short=105
    long=200
    fee_rate=0.0005

    result,df_bt=backtest_ma(df_raw,short=short,long=long,fee_rate=fee_rate)
    trade_count=df_bt["position_change"].sum()

    print("回测结果：")
    print("-" * 40)
    print(f"总交易次数:{int(trade_count)}")
    print(f"参数:MA{result['short']}/{result['long']}, 手续费: {result['fee_rate']:.4f}")
    print(f"买入持有总收益率:{result['buy_hold_total_return']:.2%}")
    print(f"策略总收益率    : {result['strategy_total_return']:.2%}")
    print(f"最大回撤        : {result['max_drawdown']:.2%}")
    print(f"夏普比率        : {result['sharpe']:.2f}")

    # 画曲线对比一下
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(df_bt.index, df_bt["buy_hold_cum"], label="Buy & Hold")
    plt.plot(df_bt.index, df_bt["strategy_cum"], label=f"MA{short}/{long} Strategy")

    plt.title("MA Strategy vs Buy & Hold (with fee)")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

