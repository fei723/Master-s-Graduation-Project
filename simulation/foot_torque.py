import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -------------------- Mac/Win 中文乱码修复 --------------------
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------- 1. 机构参数 --------------------
L1, L2, L3, L4, L5 = 0.15, 0.24, 0.24, 0.15, 0.085

# 为了找到完整的物理安全区，生成背景云图的采样范围 (主要探索下方半圆)
bg_t1_range = np.deg2rad(np.linspace(0, 180, 150))
bg_t4_range = np.deg2rad(np.linspace(0, 180, 150))

# -------------------- 2. 核心数学与运动学 (新坐标系) --------------------
# 新坐标系: O(0,0)在中心, X正向向左, Z正向向下

def check_active_link_interference(xB, zB, xD, zD):
    """
    判断左右主动杆 (A->B 和 E->D) 是否发生交叉碰撞
    A = (L5/2, 0), E = (-L5/2, 0)
    """
    def ccw(Ax, Ay, Bx, By, Cx, Cy):
        return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)
    
    xA, zA = L5/2, 0
    xE, zE = -L5/2, 0
    cond1 = ccw(xA, zA, xE, zE, xD, zD) != ccw(xB, zB, xE, zE, xD, zD)
    cond2 = ccw(xA, zA, xB, zB, xE, zE) != ccw(xA, zA, xB, zB, xD, zD)
    return cond1 and cond2

def five_bar_fk(theta1, theta4):
    # 基座坐标
    xA, zA = L5/2, 0.0
    xE, zE = -L5/2, 0.0

    # 左膝盖 B 点 (相对于 +X 轴)
    xB = xA + L1 * np.cos(theta1)
    zB = zA + L1 * np.sin(theta1)
    
    # 右膝盖 D 点 (相对于 -X 轴, 所以 X 方向用减号)
    xD = xE - L4 * np.cos(theta4)
    zD = zE + L4 * np.sin(theta4)

    # 求解末端 C 点
    dx = xD - xB
    dz = zD - zB
    d = np.hypot(dx, dz)

    if d > L2+L3 or d < abs(L2-L3):
        return None, None, None, None, xB, zB, xD, zD

    cos_a = np.clip((L2**2 + d**2 - L3**2) / (2*L2*d), -1, 1)
    alpha = np.arccos(cos_a)
    phi = np.arctan2(dz, dx)

    # 采用 Knees-Out (膝盖外翻) 装配模式，使用 phi - alpha
    theta2 = phi - alpha
    xC = xB + L2 * np.cos(theta2)
    zC = zB + L2 * np.sin(theta2)
    
    # 计算右从动杆绝对角度 theta3 (相对于 -X 轴)
    theta3 = np.arctan2(zC - zD, xD - xC)

    return theta2, theta3, xC, zC, xB, zB, xD, zD

def get_singularity_type(theta1, theta4, theta2, theta3, tol=0.03):
    # Type I: 串联奇异 (主动杆与从动杆共线，左右侧条件数学上一致)
    type1 = abs(np.sin(theta2 - theta1)) < tol or abs(np.sin(theta3 - theta4)) < tol
    
    # Type II: 并联奇异 (两从动杆共线)。
    # 【注意】：由于 theta3 是相对 -X 定义的，这里条件变成了 sin(theta2 + theta3)
    type2 = abs(np.sin(theta2 + theta3)) < tol
    
    if type1 and type2: return "Type I & II 同时奇异"
    elif type1: return "Type I (串联奇异: A-B-C 或 E-D-C 共线)"
    elif type2: return "Type II (并联奇异: B-C-D 共线)"
    return "正常"

# -------------------- 3. 预计算物理约束云图 --------------------
print("正在计算新坐标系下的安全工作空间...")
x_reach, z_reach, x_sing, z_sing = [], [], [], []
for t1 in bg_t1_range:
    for t4 in bg_t4_range:
        t2, t3, x, z, xB, zB, xD, zD = five_bar_fk(t1, t4)
        
        if t2 is None: continue                   # 运动学不可达
        if z <= 0: continue                       # 物理约束1：Z正向朝下，足端必须 Z > 0
        if check_active_link_interference(xB, zB, xD, zD): continue # 物理约束2：主动杆不干涉
        
        x_reach.append(x)
        z_reach.append(z)
        
        if get_singularity_type(t1, t4, t2, t3, 0.04) != "正常":
            x_sing.append(x)
            z_sing.append(z)

# -------------------- 4. 初始化绘图与 UI --------------------
fig, ax = plt.subplots(figsize=(9, 8))
plt.subplots_adjust(bottom=0.25)

# 画背景云图
ax.scatter(x_reach, z_reach, s=1.5, c='#4488ff', alpha=0.3, label='安全可达空间 (Z>0)')
ax.scatter(x_sing, z_sing, s=8, c='red', alpha=0.8, label='奇异边界')

# 初始化连杆线条
line_left, = ax.plot([], [], 'o-', lw=4.5, color='#2c3e50', markersize=9)  # A-B-C
line_right, = ax.plot([], [], 'o-', lw=4.5, color='#e74c3c', markersize=9) # E-D-C
line_base, = ax.plot([L5/2, -L5/2], [0, 0], 's-', lw=6, color='black', markersize=11) # 基座 A-E

ax.set_aspect('equal')
ax.set_xlim(0.4, -0.4)   # 【关键操作】: 翻转 X 轴显示，让正方向朝左，匹配你的图纸
ax.set_ylim(0.45, -0.1)  # 【关键操作】: 翻转 Z 轴显示，让正方向朝下，匹配你的图纸
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlabel('X (m) [正向朝左]')
ax.set_ylabel('Z (m) [正向朝下]')
ax.legend(loc='upper right')
title_text = ax.set_title('对称坐标系运动学仿真', fontsize=14)

# -------------------- 5. 添加交互滑块 --------------------
ax_t1 = plt.axes([0.15, 0.12, 0.7, 0.03])
ax_t4 = plt.axes([0.15, 0.06, 0.7, 0.03])

# 初始角度：直立状态（此时 theta1 = theta4）
slider_t1 = Slider(ax_t1, 'θ1 (左角)', -90, 270, valinit=90, valstep=1)
slider_t4 = Slider(ax_t4, 'θ4 (右角)', -90, 270, valinit=90, valstep=1)

def update(val):
    t1 = np.deg2rad(slider_t1.val)
    t4 = np.deg2rad(slider_t4.val)
    
    t2, t3, x, z, xB, zB, xD, zD = five_bar_fk(t1, t4)
    
    if t2 is None:
        # 连杆断开
        line_left.set_data([L5/2, L5/2 + L1*np.cos(t1)], [0, L1*np.sin(t1)])
        line_right.set_data([-L5/2, -L5/2 - L4*np.cos(t4)], [0, L4*np.sin(t4)])
        title_text.set_text(f'状态: 机构无法闭合')
        title_text.set_color('purple')
    else:
        # 连杆闭合，更新坐标
        line_left.set_data([L5/2, xB, x], [0, zB, z])
        line_right.set_data([-L5/2, xD, x], [0, zD, z])
        
        # 物理约束排查
        if check_active_link_interference(xB, zB, xD, zD):
            title_text.set_text(f'状态: 严重警告 - 左右主动杆交叉！')
            title_text.set_color('red')
        elif z <= 0:
            title_text.set_text(f'状态: 警告 - 足端进入上半区 (Z={z:.3f} <= 0)')
            title_text.set_color('orange')
        else:
            status = get_singularity_type(t1, t4, t2, t3)
            if status != "正常":
                title_text.set_text(f'状态: 奇异位形 - {status}')
                title_text.set_color('red')
            else:
                # 显示完美的对称性
                title_text.set_text(f'安全 | 坐标 (X:{x:.3f}, Z:{z:.3f}) | θ2:{np.rad2deg(t2):.1f}°, θ3:{np.rad2deg(t3):.1f}°')
                title_text.set_color('green')
            
    fig.canvas.draw_idle()

slider_t1.on_changed(update)
slider_t4.on_changed(update)
update(None)

plt.show()