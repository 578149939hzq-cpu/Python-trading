#模拟做N笔同样收益率的交易后，账户的资金变化
initial_capital=10000  # 初始资金
r=0.01                 # 每笔交易收益率2%
n_trades=10            #交易笔数
n=100
final_capital=initial_capital*(1+r)**n_trades
print(f"初始资金:{initial_capital}USD")
print(f"每笔收益率:{r*100:.2f}%")
print(f"{n_trades}笔交易后账户的资金约为{final_capital:.2f}USD")
print(f"做{n}笔交易后我们就能发财了!!!!")