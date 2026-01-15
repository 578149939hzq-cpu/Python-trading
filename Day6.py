#equity_tools.py
def calc_equity_curve(initial_capital,trade_returns):
    """
    根据初始资金和每笔交易收益率列表，返回资金曲线列表。
    trade_returns 例如：[0.02, -0.01, 0.03]
    """
    equity=initial_capital
    equity_curve=[equity]
    for r in trade_returns:
        equity=equity*(1+r)
        equity_curve.append(equity)
    return equity_curve
def calc_max_drawdown(equity_curve):
    """
    计算最大回撤
    回撤=当前资金/历史最高资金
    返回值为一个负数 比如-15表示-15%
    """
    max_peak=equity_curve[0]
    max_dd=0.0 #为负数
    for eq in equity_curve:
        if eq>max_peak:
            max_peak=eq;
        dd=eq/max_peak-1
        if dd<max_dd:
            max_dd=dd
    return max_dd
if __name__=="__main__":
    initial_capital=1000.0
    trade_returns=[0.02,-0.01,0.03,-0.02,0.01]
    curve=calc_equity_curve(initial_capital,trade_returns)
    print("资金曲线:",[f"{x:.2f}"for x in curve])
    final_equity=curve[-1]
    total_return =final_equity/initial_capital-1

    max_dd = calc_max_drawdown(curve)
    print(f"最大回撤：{max_dd:.2%}")
    print(f"最终资金:{final_equity:.2f}USD")
    print(f"总收益率:{total_return:.2%}")


