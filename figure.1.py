import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# --- 1. 录入数据 (保持你原来的录入顺序：100% -> 0%) ---
soc_points = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0])

v_05c = np.array([3.32, 3.31, 3.30, 3.29, 3.28, 3.27, 3.26, 3.25, 3.23, 3.00, 2.50])
v_1c  = np.array([3.30, 3.28, 3.25, 3.23, 3.20, 3.18, 3.16, 3.13, 3.10, 2.80, 2.50])
v_2c  = np.array([3.25, 3.20, 3.15, 3.10, 3.05, 3.00, 2.95, 2.90, 2.85, 2.65, 2.50])

# --- 2. 关键修正：将数据翻转为“从小到大”以满足函数要求 ---
# 解释：插值函数要求 X 轴必须递增，所以我们在这里临时把数据倒过来算
soc_points_sorted = soc_points[::-1]  # 变成 [0, 10, ... 100]
v_05c_sorted = v_05c[::-1]            # 对应的电压也倒过来
v_1c_sorted = v_1c[::-1]
v_2c_sorted = v_2c[::-1]

# --- 3. 定义平滑函数 ---
def smooth_curve(x, y):
    x_new = np.linspace(x.min(), x.max(), 300) 
    # k=3 代表三次样条插值
    spl = make_interp_spline(x, y, k=3) 
    y_smooth = spl(x_new)
    # 修正：插值可能会导致电压微微低于2.5或高于3.32，强制截断一下
    y_smooth = np.clip(y_smooth, 2.5, 3.4)
    return x_new, y_smooth

# --- 4. 绘图设置 ---
# 如果还是报字体错误，请删除下面这行，或者改成 'sans-serif'
plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['font.size'] = 12
plt.figure(figsize=(7, 5), dpi=300)

# 生成平滑数据 (使用翻转后的数据计算)
x_smooth, y_05c_smooth = smooth_curve(soc_points_sorted, v_05c_sorted)
_, y_1c_smooth = smooth_curve(soc_points_sorted, v_1c_sorted)
_, y_2c_smooth = smooth_curve(soc_points_sorted, v_2c_sorted)

# 绘制曲线 (颜色：蓝/橙/红)
plt.plot(x_smooth, y_05c_smooth, label='0.5 C', color='#1f77b4', linewidth=2)
plt.plot(x_smooth, y_1c_smooth, label='1.0 C', color='#ff7f0e', linewidth=2)
plt.plot(x_smooth, y_2c_smooth, label='2.0 C', color='#d62728', linewidth=2)

# --- 5. 坐标轴调整 ---
# 【关键】因为数据被我们翻转计算了，为了显示符合习惯，这里再次反转 X 轴
plt.xlim(100, 0) 
plt.ylim(2.4, 3.4)

plt.xlabel('State of Charge (SOC) [%]', fontsize=13)
plt.ylabel('Voltage [V]', fontsize=13)
plt.title('Discharge Curves of LiFePO$_4$ Battery', fontsize=14, pad=10)

plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(frameon=False, loc='upper right')

# 添加文字说明
plt.text(50, 2.6, 'Nominal Voltage Plateau ~3.2V\nVoltage drops as C-rate increases', 
         fontsize=10, ha='center', color='gray', style='italic',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.tight_layout()
plt.savefig('LFP_Discharge_Curve_Fixed.png')
plt.show()