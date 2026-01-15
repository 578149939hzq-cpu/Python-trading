import numpy as np
# 为了每次运行结果一致，设一个随机种子（可以理解为固定随机数序列）
np.random.seed(42)

initial_capital=1000.0 #初始资金
n_days=100 #模拟100天
mu_daily=0.001 #日均收益率
sigma_daily=0.02 #日波动率

daily_returns=np.random.normal(mu_daily,sigma_daily,size=n_days)

#资金曲线
equity_curve=initial_capital*(1+daily_returns).cumprod()

#看看前几天的表现
print("前10天的收益率:",np.round(daily_returns[:10],4))
print("前10天的资金曲线:",np.round(equity_curve[:10],2))

final_equity=equity_curve[-1]
total_return=final_equity/initial_capital-1

print("-"*40)
print(f"最终资金:{final_equity:.2f}USD")
print(f"总收益率:{total_return:.2%}")
print(f"日收益率均值:{daily_returns.mean():.2%}")
print(f"日收益率标准差:{daily_returns.std():.2%}")
