import pandas as pd

#读csv
df=pd.read_csv("btc_demo.csv")
print("原始数据:")
print(df.head())

#2.把time转换成日期类型，并且设置为索引
df["time"]=pd.to_datetime(df["time"])
df=df.set_index("time")

print("\n设置time为索引后的数据:")
print("df,head()")

print("\n数据概括:")
print(df.info())

short=3
long=5
df[f"ma_{short}"]=df["close"].rolling(short).mean()
df[f"ma_{long}"]=df["close"].rolling(long).mean()

print("\n加入均线后的数据:")
print(df[["close",f"ma_{short}",f"ma_{long}"]].head(long+2))
df["ret"]=df["close"].pct_change()
df["cum_ret"]=(1+df["ret"]).cumprod()
final_cum=df["cum_ret"].iloc[-1]
total_return=final_cum-1
print("\n买入持有从第一天到最后一天的总收益率:",f"{total_return:.2%}")
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))

plt.plot(df.index,df["close"],label="Close")#收盘价
plt.plot(df.index,df[f"ma_{short}"],label=f"MA{short}")#短均线
plt.plot(df.index,df[f"ma_{long}"],label=f"MA{long}")#长均线

plt.title("BTC Demo Price Moving Averges")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



