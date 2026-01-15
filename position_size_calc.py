#根据账户资金，允许风险百分比，入场价格和止损价计算最大仓位

#使用input() 从键盘获取数据
capital=float(input("请输入账户的总资金(USD):"))
risk_pct=float(input("请输入单笔最大亏损占比(例如1表示1%)"))
entry_price=float(input("请输入入场价格"))
stop_price=float(input("请输入止损价格"))
profit_price=float(input("输入止盈价格"))
# 计算这笔交易的允许亏损
risk_amount=capital*risk_pct/100

#计算这笔交易允许亏多少钱
risk_amount=capital*risk_pct/100

#每一单位亏多少钱
per_unit_risk=abs(entry_price-stop_price)

#最大可以下单的数量
position_size=risk_amount/per_unit_risk

#profit
profit_amount=abs(entry_price-stop_price)*position_size
print("-"*40)
print(f"账户资金    :{capital:.2f}USD")
print(f"单笔风险比例    {risk_pct:.2f}%")
print(f"允许最大亏损金额    :{risk_amount:.2f}USD")
print(f"本次最多可下仓位    :{position_size:.4F}单位(例如BTC/ETH)")
print(f"本次盈利    :{profit_amount:.2f}USD")
print("-"*40)