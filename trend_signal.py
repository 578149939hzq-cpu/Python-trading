#根据当前价格 短均线 长均线 给出简单的趋势判断
price=float(input("当前价格:"))
ma_short=float(input("短周期均线(例如20日):"))
ma_long=float(input("长周期均线(例如60日):"))

print("-"*40)
print(f"price={price},ma_short={ma_short},ma_long={ma_long}")
if price>ma_short and ma_short>ma_long:
     print("📈 多头趋势：价格在短均线之上，短均线在长均线上方，考虑做多或持多。")
elif price<ma_short and ma_short<ma_long:
     print("📉 空头趋势：价格在短均线之下，短均线在长均线下方，考虑做空或持空。")
else:
     print("🔁 震荡/过渡区：多空结构不清晰，暂时观望或轻仓。")
