#计算一串价格的单期收益率和总收益率
# prices=[100,102,101,105,99,100]
# print("价格序列:",prices)
# returns=[] #存每一步的收益率
# for i in range(1,len(prices)):
#     p_prev=prices[i-1]
#     p_now=prices[i]
#     r=(p_now-p_prev)/p_prev
#     returns.append(r)
#     print(f"从{p_prev}->{p_now}的收益率:{r:.4f}")

# #计算持有到最后的总体收益率
# total_return=(prices[-1]/prices[0])-1
# #price[-1]指向price数组的最后一个元素
# print("-"*40)
# print("所有单期收益率:",returns)
# print(f"总体收益率:{total_return:.4f}")

# equity_curve_sim.py
#用一串""交易结束"模拟账户的资金曲线
initial_capital=1000.0

#假设这是N笔交易的收益率
trade_returns=[0.02,-0.01,0.03,-0.02,0.01,0.04,-0.03]

equity=initial_capital
equity_curve=[equity]

for r in trade_returns:
    equity=equity*(1+r)
    equity_curve.append(equity)
print("初始资金:.",initial_capital)
print("每笔交易收益率:",trade_returns)
##
print("资金曲线:",[f"{x:.3f}"for x in equity_curve])
print(f"最终资金:{equity:.2f},总收益率:{equity/initial_capital}")