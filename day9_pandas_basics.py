import pandas as pd
data={
    "time":[
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
        "2025-01-04",
        "2025-01-05",
    ],
    "open":[100,102,101,103,104],
    "high": [103,103,104,105,106],
    "low": [99,100,100,102,103],
    "close":[102,101,103,104,105],
    "volume":[1000,1200,900,1100,1300],
}

# ==== 侦探代码：检查每一列的长度 ====
print("正在检查数据长度...")
for key, value in data.items():
    print(f"{key} 的长度是: {len(value)}")
# ================================

df=pd.DataFrame(data)

print("原始 DataFrame:")
print(df)

print("\n基本信息:")
print(df.info())

print("\n统计描述:")
print(df.describe())
# ==== 把 time 列转换为真正的日期类型，并设为索引 ====
df["time"]=pd.to_datetime(df["time"])
df=df.set_index("time")

print("\n把time设为索引后的DataFrame:")
print(df)
# ==== 计算每日收益率：ret = close_t / close_{t-1} - 1 ====
df["ret"]=df["close"].pct_change()

print("\n加入日收益率ret列:")
print(df)

#=== 计算从第一天持有到现在的累计收益===
df["cum_ret"]=(1+df["ret"]).cumprod()
print("\n加入累计收益 cum_ret 列")
print(df)

#DataFrame里面算简单均线
# ==== 简单移动平均线（MA） ====
short = 2   # 短周期
long = 3    # 长周期

df[f"ma_{short}"]=df["close"].rolling(short).mean()
df[f"ma_{long}"]=df["close"].rolling(short).mean()

print("\n加入短期 / 长期均线后的 DataFrame:")
print(df[["close",f"ma_{short}",f"ma_{long}","ret","cum_ret"]])
print(df.tail(3))  # 看最后三行
