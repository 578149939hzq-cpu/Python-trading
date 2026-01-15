import pandas as pd
import matplotlib.pyplot as plt # 加上画图库

# ==== 0. 配置参数 ====
PARAMS = {
    "short_window": 5,
    "long_window": 20,
    "fee_rate": 0.0005,
    "initial_capital": 10000
}

# ==== 1. 数据加载模块 ====
def load_price_data(csv_path: str) -> pd.DataFrame:
    # 修正1: skiprows (复数)
    df = pd.read_csv(csv_path, skiprows=1)
    
    # 修正2: lower() 是函数，要加括号
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # 时间戳清洗逻辑 (保持你之前的修正)
    df["unix"] = pd.to_numeric(df["unix"], errors='coerce')
    mask_micro = df["unix"] > 1e14 
    df.loc[mask_micro, "unix"] = df.loc[mask_micro, "unix"] / 1000
    df["time"] = pd.to_datetime(df["unix"], unit="ms")
    
    df = df.set_index("time").sort_index()
    #预计算收益率 因为ret的值与ma策略无关
    df["ret"]=df["close"].pct_change()
    return df

# ==== 2. 指标与信号模块 (向量化) ====
def calc_ma_signal(df: pd.DataFrame, short: int, long: int,atr_window:int=20,atr_threshold:float=0.5) -> pd.DataFrame:
    #df = df.copy()##可以选择传入拷贝值
    ##或者我们提取我们只需要的列数即可
    """
    df:原始数据
    short/long:均线参数
    atr_window:计算ATR的窗口(默认20)
    atr_threshold:NATR阈值(默认0.5小于10.5的时候不交易)
    """
    data=df[["close","ret","high","low"]].copy()

    #向量化计算均线
    data["ma_short"] = data["close"].rolling(short).mean()
    data["ma_long"] = data["close"].rolling(long).mean()
    
    #计算ATR
    #TR1=H-L
    #TR2=|H-Prevclose|
    #TR3=|L-Prevclose|
    #TR=max(TR1,TR2,TR3)

    prev_close=data["close"].shift(1)
    tr1=data["high"]-data["low"]
    tr2=(data["high"]-prev_close).abs()
    tr3=(data["low"]-prev_close).abs()
    # 向量化计算最大值
    data["tr"]=pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)

    #计算ATR
    data["atr"]=data["tr"].rolling(atr_window).mean()

    #计算NATR(波动率百分比)->方便我们设定统一的阈值
    data["natr"]=(data["atr"]/data["close"])*100

    #part C:生成信号(加入风控逻辑)
    condition_trend=data["ma_short"]>data["ma_long"]

    # 风控逻辑: 只有当波动率足够大 (市场活跃) 时才允许交易
    #& (data["natr"] < 5.0) 防止极端暴跌接飞刀#

    condition_volatility = (data["natr"] > atr_threshold)&(data["natr"]<5.0)
    data["signal"]=0
    # 只有 "趋势来了" 且 "车速够快(活跃)" 才上车
    data.loc[condition_trend&condition_volatility,"signal"]=1
    return data
# ==== 3. 回测引擎 (核心重构：向量化) ====
# 修正3: 拼写 DataFrame
def run_backtest(df: pd.DataFrame, fee_rate: float, initial_capital: float)->pd.Series:
    
    # 1. 信号防未来函数 (Shift 1)
    # 昨天的信号，决定今天的持仓
    df["position"] = df["signal"].shift(1).fillna(0)
    
    # 2. 计算扣费前的策略收益
    df["strategy_ret_gross"] =(df["position"] * df["ret"]).fillna(0)
    
    # 3. 计算手续费
    # 当 position 发生变化时 (0->1 或 1->0)，产生手续费
    # diff() != 0 代表仓位变了
    df["trade_count"] = df["position"].diff().fillna(0).abs()
    df["fee"] = df["trade_count"] * fee_rate
    
    # 4. 计算净收益 (收益 - 手续费)
    df["strategy_ret_net"] = df["strategy_ret_gross"] - df["fee"]
    
    # 5. 计算资金曲线 (累计乘积)
    df["equity_curve"] = initial_capital * (1 + df["strategy_ret_net"]).cumprod()
    strat_ret_series=df["strategy_ret_net"].dropna()

   
    # 返回整列数据，方便后续分析
    return df["equity_curve"]

def run_backtest_with_stoploss(df:pd.DataFrame,fee_rate:float,initial_capital:float,stop_loss_pct:float=0.05)->pd.Series:
    """
    事件驱动回测：支持固定比例止损+冷却机制
    stop_loss_pct: 止损比例 (例如 0.05 代表亏 5% 止损)
    """
    #准备数据容器
    capital=initial_capital
    position=0.0 #当前持仓数量(币)
    entry_price=0.0 #入场价格
    equity_curve=[] #记录每天的资金
    #将dataFrame转换为命名元组列表(急速遍历)
    #我们需要time.open.high,low,close,signal
    #注意:这里假设df索引是时间。且列名包含signal close low

    #True：代表这波上涨已经出局，不要再买回来
    stop_triggered=False
    # --- 0. 状态重置逻辑 (关键!) ---
        # 如果信号变成了 0 (死叉/空仓信号)，说明上一波趋势结束了
        # 我们就可以解除 "冷却状态"，准备迎接下一次金叉
    
    for row in df.itertuples():
        # --- A. 每日结算前，先记录当前的资产净值 ---
        # 如果持仓，市值 = 币数 * 当前收盘价
        # 如果空仓，市值 = 现金
        if row.signal==0:
            stop_triggered=False
        if position>0:
            current_equity=position*row.close
        else:
            current_equity=capital
        equity_curve.append(current_equity)
        # 情况1：持有仓位，检查是否要卖
    #情况1:持有仓位，检查是否要卖
        if position>0:
        #1. 🛑 止损检查 (优先级最高！)
        # 为什么用 Low？因为只要这一小时内最低价跌破了，就会触发止损单
            stop_price=entry_price*(1-stop_loss_pct)

            if row.low<=stop_price:
                #触发止损 卖出
                #实际成交价格通常就是止损价格
                sell_price=stop_price

                #卖出逻辑
                revenue=position*sell_price
                fee=revenue*fee_rate
                capital=revenue-fee #变现回现金
                position=0.0 #仓位归零
                entry_price=0.0
                # 🆕 标记：这波我不玩了！
                stop_triggered=True
                continue #遍历完一行去到下一行
        #2.正常离场检查(死叉)
        #如果没有触发止损，但策略要卖(signal==0)
            elif row.signal==0:
                sell_price=row.close
                revenue=position*sell_price
                fee=revenue*fee_rate
                capital=revenue-fee
                position=0.0
                entry_price=0.0
                continue
        #情况2:空仓，检查是否要购买
        elif position==0:
            # ✅ 买入条件升级：
            # 1. 信号必须是 1
            # 2. 必须没有处于 "冷却状态" (not stop_triggered)
            if row.signal==1 and not stop_triggered:
                buy_price=row.close
                #全仓买入(扣除手续费)
                cost=capital*(1-fee_rate)
                position=cost/buy_price
                capital=0.0#现金变成币
                entry_price=buy_price#记录成本价格 关键
    return pd.Series(equity_curve,index=df.index)    

# ==== 4. 结果分析模块（(升级版：加入 Calmar)） ====
import numpy as np
def calculate_metrics(equity_curve:pd.Series)->dict:
    """
    计算核心指标:总回报、最大回撤、sharpe
    """
    #1基础数据
    final_equity=equity_curve[-1]
    initial_capital=equity_curve.iloc[0]
    ##或者我们可以函数传参
    total_return=final_equity/initial_capital-1
    # ---- 🆕 新增：计算年化收益率 (用于 Calmar) ----
    # 数据大约 8 年 (2018-2026)
    years=8.0
    #年化公式:(1+总收益)^(1/年数)-1
    cagr=(final_equity/initial_capital)**(1/years)-1
    #最大回撤 向量版
    running_max=equity_curve.cummax()
    drawdown=(running_max-equity_curve)/running_max
    max_dd=drawdown.max()
    #==== 计算sharpe =====
    #先反推出每根k线的收益率序列
    ret_series=equity_curve.pct_change().dropna()

    #防止0错误(如果策略从头到尾都没有开单，std是0)
    if len(ret_series)>1 and ret_series.std()>0:
        # 核心公式：(均值 / 标准差) * sqrt(年化周期数)
        # 你的数据是 1小时级别，一年有 365 * 24 = 8760 小时
        annual_factor=8760**0.5
        sharpe=(ret_series.mean()/ret_series.std())*annual_factor
    else:
        sharpe=0.0
    # ---- 🆕 新增：卡玛比率 (Calmar Ratio) ----
    # 核心公式：年化收益 / 最大回撤
    if max_dd>0:
        calmar=cagr/max_dd
    else:
        # 如果没有回撤（神仙策略），给个极大的数字
        calmar=999.0
    return{
        "Final Equity":final_equity,
        "Total Return":total_return,
        "Max Drawdown":max_dd,
        "Sharpe":sharpe,
        "Calmar":calmar,
    }

# ==========================================
# 5. 优化层 (Optimizer Layer) - 网格搜索 2.0
# ==========================================
import time
# from itertools import product
# def grid_search(df_raw:pd.DataFrame,short_range:list,long_range:list,stop_loss_range:list,fee:float,capital:float):
#     """
#     三维参数扫描: Short x Long x StopLoss
#     """
#     results=[]
#     start_time=time.time()
#     # 使用 product 生成笛卡尔积，比写两层 for 循环更优雅
#     #笛卡尔积 两两配对 不用写多层for循环
#     # e.g., [(5, 20), (5, 50), (10, 20)...]
#     # 🆕 使用 product 生成三维笛卡尔积
#     # 例如: [(5, 20, 0.05), (5, 20, 0.10)...]
#     param_combinations=list(product(short_range,long_range,stop_loss_range))
#     print(f"🕵️‍♂️ Jarvis 正在扫描 {len(param_combinations)} 组参数组合...")
#     for s,l,sl in param_combinations:## s=short, l=long, sl=stop_loss
#         #逻辑防呆
#         if s>=l:
#             continue
#         # A. 生成信号 (依然是用 ATR 计算函数，虽然我们暂时不用 ATR 阈值)
#         # 这里 atr_threshold 我们先给个极小值 0.001，相当于暂时关闭 ATR 过滤，只测止损
#         df_sig = calc_ma_signal(df_raw, short=s, long=l, atr_window=20, atr_threshold=0.001)

#         #B.跑回测
#         curve=run_backtest_with_stoploss(df_sig,fee,capital,stop_loss_pct=sl)

#         #强制5%止损
#         #curve=run_backtest_with_stoploss(df_sig,fee,capital,stop_loss_pct=0.05)
#         #C.算指标
#         metrcis=calculate_metrics(curve)

#         #D.存结果
#         results.append({
#             "Short":s,
#             "Long":l,
#             "Stop_Loss":sl,#记录改组数据实验用多少止损跑的
#             "Return":metrcis["Total Return"],
#             "Max_DD":metrcis["Max Drawdown"],
#             "Equity":metrcis["Final Equity"],
#             "Sharpe":metrcis["Sharpe"],
#             "Calmar":metrcis["Calmar"]

#         })
#         print(f"✅ 扫描完成! 耗时: {time.time() - start_time:.2f} 秒")

#         #转成DataFrame并且排序
#         df_res=pd.DataFrame(results)
#     return df_res.sort_values(by="Calmar",ascending=False)
def get_best_params(df_train,short_params,long_params,stop_loss_params,fee,capital):
    """
    安静版的网格搜索，只返回 best_params 字典
    """
    results=[]
    from itertools import product
    combinations=list(product(short_params,long_params,stop_loss_params))
    if len(df_train) < 300: 
            print(f"   ⚠️ 数据不足 ({len(df_train)}行), 跳过此训练集")
            return None # 返回空，让主程序跳过
    for s,l,sl in combinations:
        if s>=l:continue
        #算信号
        df_sig=calc_ma_signal(df_train,int(s),int(l),atr_threshold=0.001)
        #跑回测
        curve=run_backtest_with_stoploss(df_sig,fee,capital,stop_loss_pct=sl)
        # 🆕 新增：如果这一年的数据太少（少于最长均线），直接放弃，别浪费时间算
        #算指标
        
        #只算Sharpe和Calmar
        if len(curve)>0:
            # A. 算年化收益 (CAGR)
            total_ret = curve.iloc[-1] / curve.iloc[0] - 1
            # B. 算最大回撤 (MaxDD)
            cummax = curve.cummax()
            dd = (cummax - curve) / cummax
            max_dd = dd.max()
            # C. 核心修改：用 "卡玛比率" 作为评分标准！
            # 如果回撤太小(比如0)，给个极大值防止除以0
            if max_dd > 0.01:
                score = total_ret / max_dd
            else:
                score = 0.0 # 没回撤通常意味着没交易，给0分
            # D. 额外惩罚：如果最大回撤超过 30%，直接判死刑 (Score = 0)
            # 这一句是强行让 Jarvis 选保守参数！
            if max_dd > 0.30:
                score = 0     
        else:
            score = 0
        results.append({"s":s,"l":l,"sl":sl,"score":score})
    if not results:
        return None
    best=sorted(results,key=lambda x:x["score"],reverse=True)[0]
    return best
def run_walk_forward(df_raw,short_params,long_params,stop_loss_params,fee,initial_capital):
    """
    滚动回测主引擎
    """
    # 1. 按年份切分数据
    # df.index 必须是 datetime 类型
    years=df_raw.index.year.unique().sort_values()
    print(f"📅 数据涵盖年份: {years.tolist()}")
    #2.初始化
    final_equity_curve=pd.Series(dtype="float64")
    current_capital=initial_capital# 这一年的本金是上一年的余额
    history_params=[]#记录每一年使用的参数

    # 3. 开始滚动 (从第2年开始，因为第1年只能用来做训练)
    # Train: Year i
    # Test: Year i+1
    for i in range(len(years)-1):
        train_year=years[i]
        test_year=years[i+1]
        print(f"\n🔄 正在进行滚动: 训练 {train_year} -> 实战 {test_year}")
        # 切分数据
        df_train=df_raw[df_raw.index.year==train_year].copy()
        df_test=df_raw[df_raw.index.year==test_year].copy() 
        # A. 在训练集上找最佳参数 (Optimization)
        print(f" Searching best params in{train_year}...")
        best=get_best_params(df_train,short_params,long_params,stop_loss_params,fee,current_capital)
        if best is None:
            print("   ❌ 这一年数据不足或无法交易，跳过")
            continue
        print(f"   ✅ 冠军参数: MA {best['s']}/{best['l']} | SL {best['sl']:.1%}")
        history_params.append({"year":test_year,"params": best})
        # B. 在测试集上跑实盘 (Validation)
        # 注意：这里用的是刚刚算出来的 best 参数！
        print(f"   🏃 Running trade in {test_year}...")
        df_test_sig=calc_ma_signal(df_test,best['s'],best['l'],atr_threshold=0.001)
        #跑回测,初始资金是current_capital(复利滚动)
        curve_test=run_backtest_with_stoploss(df_test_sig,fee,current_capital,stop_loss_pct=best['sl'])

        #C.拼接资金曲线
        if final_equity_curve.empty:
            final_equity_curve=curve_test
        else:
            #拼接到后面
            final_equity_curve=pd.concat([final_equity_curve,curve_test])
        #D.更新本金，为明年做准备
        current_capital=final_equity_curve.iloc[-1]
        print(f"   💰 {test_year} 年底资产: {current_capital:,.0f}")
    return final_equity_curve,history_params

# ==========================================
# 🚀 主程序入口
# ==========================================
if __name__ == "__main__":
    # 配置区
    CSV_PATH = "Binance_BTCUSDT_1h.csv"
    INITIAL_CAPITAL = 10000
    FEE_RATE = 0.0005  # 万五手续费
    
    # 1. 准备数据
    df = load_price_data(CSV_PATH)
    
    # 2. 定义你要测试的参数池
    # range(5, 55, 5) 意味着: 5, 10, 15 ... 50
    short_params = [10, 20,30,50] 
    long_params = [100,150,200.300]
    # 🆕 止损参数池: 测 3% 到 15%
    # 太窄(0.03)容易被打脸，太宽(0.15)扛单太久，看看哪个最好
    stop_loss_params=[0.05,0.08,0.10,0.15]
    # # 3. 启动网格搜索
    # leaderboard = grid_search(df, short_params, long_params,stop_loss_params, FEE_RATE, INITIAL_CAPITAL)
    # 2. 启动滚动回测
    start_t=time.time()
    wfa_curve,wfa_history=run_walk_forward(df,short_params,long_params,stop_loss_params,FEE_RATE,INITIAL_CAPITAL)
    print(f"\n⏱️ 滚动回测总耗时: {time.time() - start_t:.2f} 秒")
    # # 4. 展示前 15 名
    # print("\n🏆 策略排行榜 (Top 15 By Calmar) 🏆")
    # print(leaderboard.head(15).to_string(formatters={
    #     'Return': '{:,.2%}'.format,
    #     'Max_DD': '{:,.2%}'.format,
    #     'Equity': '{:,.0f}'.format,
    #     'Sharpe': '{:,.2f}'.format,
    #     'Calmar': '{:,.2f}'.format
    # }))
    # # 5. 画出第一名的曲线
    # if not leaderboard.empty:
    #     best_row = leaderboard.iloc[0]
    #     best_s, best_l = int(best_row["Short"]), int(best_row["Long"])
    #     best_sl=float(best_row["Stop_Loss"])  
    #     print(f"\n📈 正在绘制最佳策略: MA {best_s} / {best_l}|Stoploss{best_sl:.1%}")
    #     # 复现最佳结果（传入best_sl）
    #     df_best_sig = calc_ma_signal(df, best_s, best_l,atr_threshold=0.001)
    #     curve_best = run_backtest_with_stoploss(df_best_sig, FEE_RATE, INITIAL_CAPITAL,stop_loss_pct=best_sl)
    #     # B. 🆕 计算 "买入持有 (Buy & Hold)" 基准曲线
    #     # 逻辑：资金随价格比例波动。今天的钱 = 初始钱 * (今天价 / 初始价)
    #     bh_curve=df["close"]/df["close"].iloc[0]*INITIAL_CAPITAL
    #     bh_return=bh_curve.iloc[-1]/INITIAL_CAPITAL-1        
    #     plt.figure(figsize=(12, 6))
    #     # 🆕 画囤币曲线 (灰色虚线，作为背景参考)
    #     plt.plot(bh_curve, label=f"Buy & Hold Benchmark (Return: {bh_return:.2%})", color='grey', linestyle='--', alpha=0.7)
    #     plt.plot(curve_best, label=f"Strategy (MA {best_s}/{best_l}, SL {best_sl:.0%})", color='blue', linewidth=1.5)
    #     plt.title(f"Jarvis vs Benchmark | Calmar: {best_row['Calmar']:.2f} | Max DD: {best_row['Max_DD']:.2%}|Sharpe:{best_row['Sharpe']:.2f}")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()
    # else:
    #     print("所有策略都亏光了？或者参数设置有误？")
    
    #3.结果分析
    if not wfa_curve.empty:
        #计算最终指标
        metrics=calculate_metrics(wfa_curve)
        print("\n🎓 滚动回测 (WFA) 最终成绩单 🎓")
        print(f"最终资金: {metrics['Final Equity']:,.0f}")
        print(f"总收益率: {metrics['Total Return']:,.2%}")
        print(f"最大回撤: {metrics['Max Drawdown']:,.2%}")
        print(f"卡玛比率: {metrics['Calmar']:.2f}")
        # 4. 画图
        # 算个基准 (从回测开始的那一天算起)
        start_date=wfa_curve.index[0]
        base_price=df.loc[start_date,"close"]
        bh_curve=df.loc[start_date:,"close"]/base_price*INITIAL_CAPITAL
        plt.figure(figsize=(12, 6))
        plt.plot(wfa_curve, label="Walk-Forward Strategy (Dynamic Params)", color='blue')
        plt.plot(bh_curve, label="Buy & Hold", color='grey', linestyle='--', alpha=0.5)
        plt.title(f"Walk-Forward Analysis (2019-2026) | End Capital: {metrics['Final Equity']:,.0f}")
        plt.legend()
        plt.grid(True)
        plt.show()
        # 5. 打印每年的参数变化 (这很有趣!)
        print("\n📜 每年最佳参数进化史:")
        params_df=pd.DataFrame([
            {"Year":h['year'],"Short":h['params']['s'],"Long":h['params']['l'],"SL":h['params']['sl']}
            for h in wfa_history
        ])
        print(params_df)
    else:
        print("❌ 回测生成失败")