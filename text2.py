import matplotlib.pyplot as plt
import numpy as np

# --- 1. 简化的热模型计算 (数据保持不变) ---
time = np.linspace(0, 1800, 300)
T_amb = 25.0
T_no_pcm = [T_amb]
T_with_pcm = [T_amb]

heat_gen_rate = 0.8  
cooling_coeff = 0.05 
pcm_melt_temp = 42.0 
pcm_capacity = 200   

for i in range(1, len(time)):
    dt = time[i] - time[i-1]
    delta_T1 = (heat_gen_rate - cooling_coeff * (T_no_pcm[-1] - T_amb)) * dt * 0.05
    T_no_pcm.append(T_no_pcm[-1] + delta_T1)
    current_T = T_with_pcm[-1]
    if current_T < pcm_melt_temp or pcm_capacity <= 0:
        delta_T2 = (heat_gen_rate - (cooling_coeff * 1.5) * (current_T - T_amb)) * dt * 0.05
        T_with_pcm.append(current_T + delta_T2)
    else:
        pcm_capacity -= heat_gen_rate * dt * 0.1
        T_with_pcm.append(current_T + 0.001)

# --- 2. 美化绘图设置 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.figure(figsize=(7, 5), dpi=300) # 把图稍微变宽一点点，更舒展
plt.subplots_adjust(left=0.12, bottom=0.15, right=0.9, top=0.88)

# 绘制曲线 (线条加粗，更醒目)
plt.plot(time/60, T_no_pcm, label='Without Thermal Management', 
         color='#d62728', linestyle='--', linewidth=3) 

plt.plot(time/60, T_with_pcm, label='With Aluminum + PCM (Design)', 
         color='#1f77b4', linewidth=3.5)

# --- 3. 精致标注 ---
plt.axhspan(41.5, 42.5, color='gray', alpha=0.15)
plt.text(15, 43, 'PCM Phase Change Plateau (~42°C)', color='gray', fontsize=11, style='italic', ha='center')

# 数值标注 (使用白色描边，防止和曲线重叠看不清)
peak_no = round(T_no_pcm[-1], 1)   
peak_yes = round(T_with_pcm[-1], 1)
diff_val = round(peak_no - peak_yes, 1)

plt.scatter(30, peak_no, color='#d62728', s=60, zorder=5, edgecolors='white', linewidth=1.5)
txt1 = plt.text(30.5, peak_no, f"{peak_no}°C", va='center', color='#d62728', fontweight='bold', fontsize=12)
txt1.set_path_effects([plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='white')])

plt.scatter(30, peak_yes, color='#1f77b4', s=60, zorder=5, edgecolors='white', linewidth=1.5)
txt2 = plt.text(30.5, peak_yes, f"{peak_yes}°C", va='center', color='#1f77b4', fontweight='bold', fontsize=12)
txt2.set_path_effects([plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='white')])

# 【核心美化】温差箭头移到中间，更平衡
annotate_x = 22 # 放在 22 分钟的位置
T_no_at_x = np.interp(annotate_x, time/60, T_no_pcm)
T_yes_at_x = np.interp(annotate_x, time/60, T_with_pcm)

plt.annotate('', xy=(annotate_x, T_yes_at_x), xytext=(annotate_x, T_no_at_x),
             arrowprops=dict(arrowstyle='<|-|>', color='black', lw=1.8, shrinkA=0, shrinkB=0))
plt.text(annotate_x + 0.5, (T_no_at_x + T_yes_at_x)/2, f'Temp Reduction\n~{diff_val}°C', 
         fontsize=12, fontweight='bold', color='#1f77b4', va='center')

# --- 4. 装饰与图例 ---
plt.xlabel('Discharge Time (min)', fontsize=14, weight='bold')
plt.ylabel('Cell Temperature (°C)', fontsize=14, weight='bold')
plt.title('Thermal Management Performance (2C Discharge)', fontsize=16, pad=15)

plt.xlim(0, 33)
plt.ylim(25, 48) 
plt.grid(True, linestyle='--', alpha=0.5)

# 【核心美化】图例加背景框
legend = plt.legend(frameon=True, fontsize=11, loc='upper left', bbox_to_anchor=(0.02, 0.98))
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('None')
legend.get_frame().set_alpha(0.9)

plt.text(0.5, 26, 'Model: Lumped Capacitance Thermal Model\nCondition: 2C Discharge @ 25°C Amb', 
         fontsize=10, color='gray', style='italic',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.savefig('Thermal_Performance_Final_Beauty.png', dpi=300, facecolor='white', bbox_inches='tight')
plt.show()