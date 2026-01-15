import pandas as pd
import matplotlib.pyplot as plt
#读csv
df=pd.read_csv("btc_demo.csv")

#处理时间
df["time"]=pd.to_datetime(df["time"])
df=df.set_index("time")

#计算均线
short=3
long=5

df[f"ma_{short}"]=df["close"].rolling(short).mean()
df[f"ma_{long}"]=df["close"].rolling(long).mean()

#收盘价收益率
df["ret"]=df["close"].pct_change()

#丢弃前5行数据 因为它们包含NaN
df.dropna(inplace=True)
print(f"Data Shape:{df}")
#策略信号 1=做多 0=持仓
df["signal"]=0
df.loc[df[f"ma_{short}"]>df[f"ma_{long}"],"signal"]=1
#使用前一日的signal来计算策略收益，避免未来函数
df["signal_shift"]=df["signal"].shift(1)
#防止未来函数 整体下移一格 1月1日的信号--要在一月2日买入
#策略每日收益=前一日的持仓*当日标的收益率
df["strategy_ret"]=df["signal_shift"]*df["ret"]

#计算累计收益(策略vs买入持有)

df["buy_hold_cum"]=(1+df["ret"]).cumprod() #买入持有
df["strategy_cum"]=(1+df["strategy_ret"]).cumprod() #策略

#取最后一个非NaN的值作为最终的收益倍数
buy_hold_final=df["buy_hold_cum"].iloc[-1]
strategy_final=df["strategy_cum"].iloc[-1]


buy_hold_total_return=buy_hold_final-1
strategy_total_return=strategy_final-1

print(f"买入持有的总收益率:{buy_hold_total_return:.2%}")
print(f"策略的收益率率:{strategy_total_return:.2%}")

#从 CSV → 均线 → 信号 → 策略收益曲线 → 和买入持有比较

#画收益率曲线图
plt.figure(figsize=(10,5))
plt.plot(df.index,df["buy_hold_cum"],label="Buy & Hold")
plt.plot(df.index,df["strategy_cum"],label=f"MA{short}/{long}stragety")

plt.title("Stragety vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
