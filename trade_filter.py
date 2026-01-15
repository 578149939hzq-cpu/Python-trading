#根据胜率和盈亏比，简单评价这套系统是否有优势

win_rate=float(input("请输入策略的历史胜率(0~100):"))
rr_ratio=float(input("请输入平均盈亏比R(例如2表示1:2):"))

print("-"*40)
print(f"胜率:{win_rate:.1f}%")
print(f"盈亏比{rr_ratio:.2f}%")

if win_rate>=40 and rr_ratio>=2:
    print("有优势的系统")
elif win_rate<30 and rr_ratio<1.5:
    print("弱势系统")
else:
    print("中间地带的系统")