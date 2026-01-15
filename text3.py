import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- 1. 数据准备 ---
materials = ['NCM (Ternary)', 'LiFePO$_4$ (LFP)']
t_onset = [150, 230] 
t_runaway = [240, 450]

# --- 2. 绘图设置 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.figure(figsize=(6.5, 5), dpi=300)
plt.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.12)

bar_width = 0.35
index = np.arange(len(materials))

# --- 3. 绘制柱状图 (关键优化：加纹理) ---
# 给起始温度 (Onset) 加上斜线纹理 (hatch='///')，这样黑白图例就有意义了
# NCM (Red)
plt.bar(index[0], t_onset[0], bar_width, color='#ffb3b3', edgecolor='black', hatch='///', alpha=0.9)
plt.bar(index[0] + bar_width, t_runaway[0], bar_width, color='#d62728', edgecolor='black', alpha=0.9)

# LFP (Blue)
plt.bar(index[1], t_onset[1], bar_width, color='#aec7e8', edgecolor='black', hatch='///', alpha=0.9)
plt.bar(index[1] + bar_width, t_runaway[1], bar_width, color='#1f77b4', edgecolor='black', alpha=0.9)

# --- 4. 添加数值标签 ---
# 手动添加，位置更准
# NCM
plt.text(index[0], t_onset[0] + 5, f'{t_onset[0]}°C', ha='center', va='bottom', fontweight='bold')
plt.text(index[0] + bar_width, t_runaway[0] + 5, f'{t_runaway[0]}°C', ha='center', va='bottom', fontweight='bold')
# LFP
plt.text(index[1], t_onset[1] + 5, f'{t_onset[1]}°C', ha='center', va='bottom', fontweight='bold')
plt.text(index[1] + bar_width, t_runaway[1] + 5, f'{t_runaway[1]}°C', ha='center', va='bottom', fontweight='bold')

# --- 5. 装饰图表 ---
plt.ylabel('Temperature (°C)', fontsize=14, weight='bold')
plt.title('Thermal Stability Comparison: NCM vs LFP', fontsize=16, pad=20)
# 让 X 轴标签居中对齐两根柱子
plt.xticks(index + bar_width / 2, materials, fontsize=14)
plt.ylim(0, 580)

# --- 6. 修复图例 (现在能看懂了) ---
# 解释：带斜线的 = Onset，实心的 = Runaway
legend_patches = [
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Self-heating Onset'),
    mpatches.Patch(facecolor='gray', edgecolor='black', label='Thermal Runaway Trigger')
]
leg = plt.legend(handles=legend_patches, frameon=True, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=11)
leg.get_frame().set_alpha(0.9) # 半透明背景防止遮挡网格

# --- 7. 修复文字重叠 (移到两组柱子中间) ---
# 计算中间位置 (Gap Center)
# NCM组右边缘 ≈ 0.525, LFP组左边缘 ≈ 0.825, 中间 ≈ 0.675
arrow_x = 0.675 

# 绿色安全区
plt.axhspan(0, 60, color='#2ca02c', alpha=0.1)
plt.text(1.5, 25, 'Normal Operation Range', color='#2ca02c', fontsize=11, style='italic', ha='center')

# 绘制箭头：连接 NCM Runaway (240) 和 LFP Runaway (450) 的高度
plt.annotate('', xy=(arrow_x, 450), xytext=(arrow_x, 240),
             arrowprops=dict(arrowstyle='<|-|>', color='black', lw=1.5, shrinkA=0, shrinkB=0))

# 文字放在箭头旁边
plt.text(arrow_x, (240+450)/2, 'Higher Safety\nMargin (~210°C)', 
         color='#1f77b4', fontsize=12, fontweight='bold', va='center', ha='center', 
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2)) # 加个白底更清晰

plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.savefig('Safety_Comparison_Fixed.png', dpi=300, facecolor='white', bbox_inches='tight')
plt.show()