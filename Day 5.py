#trade_list_stats 用列表统计多笔交易表现
# trade_pnl=[50,-30,80,-20,100,-10,40]
# print("每笔交易盈亏:",trade_pnl)

# total_pnl=0
# win_count=0

# for pnl in trade_pnl:
#     total_pnl+=pnl
#     if pnl>0:
#         win_count+=1
# num_trades=len(trade_pnl)
# avg_pnl=total_pnl/num_trades
# win_rate=win_count/num_trades

# print(f"总盈亏:{total_pnl:.2f}USD")
# print(f"平均每笔盈亏:{avg_pnl:.2f}USD")
# print(f"总共{num_trades}笔交易，其中盈利{win_count}笔")
# print(f"胜率:{win_rate:.2%}")

#trade_log_dicts 用「字典 + 列表」保存每笔详情
trade1={
    "symbol":"BTCUSDT",
    "direction":"long", #long
    "entry":27000,
    "exit":27350,
    "size":0.05 #position
}
trade2={
    "symbol":"ETHUSDT",
    "direction":"short",
    "entry":2000,
    "exit":1900,
    "size":0.5
}
trades=[trade1,trade2]
for t in trades:
    if t["direction"]=="long":
        pnl=(t["exit"]-t["entry"])*t["size"]
    else:#short
        pnl=(t["entry"]-t["exit"])*t["size"]
    print("-"*40)
    print(f"品种:{t['symbol']}")
    print(f"方向:{t['direction']}")
    print(f"入场:{t['entry']},平仓:{t['exit']}，仓位:{t['size']}")
    print(f"本次盈亏:{pnl:.2f}USD")
