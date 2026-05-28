import numpy as np
import matplotlib.pyplot as plt

# ====================== Mac 防止中文乱码（核心修复） ======================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 1. 机器人五连杆尺寸参数 ======================
L1 = 0.15   # 大腿 1
L2 = 0.24   # 小腿 1
L3 = 0.24   # 小腿 2
L4 = 0.15   # 大腿 2
L5 = 0.085  # 电机基座间距

# 主动关节角度范围（合理范围，不干涉）
theta1_range = np.deg2rad(np.linspace(-70, 70, 300))
theta4_range = np.deg2rad(np.linspace(110, 250, 300))

# ====================== 2. 五连杆闭环几何正解（修正版） ======================
def five_bar_forward(theta1, theta4):
    xA, zA = 0.0, 0.0
    xB, zB = L5, 0.0

    xP = xA + L1 * np.cos(theta1)
    zP = zA + L1 * np.sin(theta1)
    xQ = xB + L4 * np.cos(theta4)
    zQ = zB + L4 * np.sin(theta4)

    dx = xQ - xP
    dz = zQ - zP
    d = np.hypot(dx, dz)

    if d > L2 + L3 or d < abs(L2 - L3):
        return None, None, None, None

    cos_a = (L2**2 + d**2 - L3**2) / (2 * L2 * d)
    cos_a = np.clip(cos_a, -1, 1)
    alpha = np.arccos(cos_a)
    phi = np.arctan2(dz, dx)

    theta2 = phi + alpha
    theta3 = np.arctan2(zP + L2*np.sin(theta2) - zB, xP + L2*np.cos(theta2) - xB)

    x = xP + L2 * np.cos(theta2)
    z = zP + L2 * np.sin(theta2)
    return theta2, theta3, x, z

# ====================== 3. 雅可比矩阵 & 奇异位形判定（修正版） ======================
def is_singular(theta1, theta4, theta2, theta3):
    J11 = -L1 * np.sin(theta1)
    J12 = -L4 * np.sin(theta4)
    J21 =  L1 * np.cos(theta1)
    J22 =  L4 * np.cos(theta4)

    J = np.array([[J11, J12], [J21, J22]])
    det = np.linalg.det(J)
    return abs(det) < 1e-5

# ====================== 4. 蒙特卡洛采样 ======================
x_reach = []
z_reach = []
x_sing = []
z_sing = []

for t1 in theta1_range:
    for t4 in theta4_range:
        t2, t3, x, z = five_bar_forward(t1, t4)
        if t2 is None:
            continue
        x_reach.append(x)
        z_reach.append(z)
        if is_singular(t1, t4, t2, t3):
            x_sing.append(x)
            z_sing.append(z)

# ====================== 5. 绘图（无乱码） ======================
plt.figure(figsize=(8, 8))
plt.scatter(x_reach, z_reach, s=1.5, c="#4488ff", alpha=0.4, label="可达空间")
plt.scatter(x_sing, z_sing, s=12, c="red", label="奇异位形")
plt.grid(True, linestyle="--", alpha=0.3)
plt.axis("equal")
plt.xlabel("X 方向 (m)")
plt.ylabel("Z 方向 (m)")
plt.title("五连杆单腿 可达工作空间与奇异位形")
plt.legend()
plt.tight_layout()
plt.show()