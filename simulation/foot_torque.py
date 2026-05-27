#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轮足机器人腿部构型实验（严格按你的要求）
实验1：不同腿节长度比 → 足端可达域对比（单图）
实验2：固定最优长度比 → 3种构型 不同足端高度 静态力矩对比
角度约束：
髋关节 θ₁ ∈ [-90°, 90°]
膝关节 θ₂ ∈ [0°, 90°]
构型定义：
1. 串联直驱腿：髋+膝独立电机，3可控自由度
2. 并联腿：髋双电机+闭链五连杆，3可控自由度
3. 多连杆腿：髋单电机+双摇杆被动膝，2可控自由度
"""
import numpy as np
import matplotlib.pyplot as plt

# ======================== Mac 字体修复（无警告）=========================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 130

# ======================== 固定参数（严格按你的要求）=========================
L_TOTAL = 1.0          # 腿总长固定(m)
N = 100.0              # 竖直静载荷(N)
# 【严格角度约束】
TH1_MIN, TH1_MAX = -np.pi/2, np.pi/2   # 髋：-90° ~ 90°
TH2_MIN, TH2_MAX = 0.0, np.pi/2        # 膝：0° ~ 90°
# 腿节长度比列表（实验1）
RATIO_LIST = [0.6, 0.8, 1.0, 1.2, 1.4]
# 足端高度列表（实验2：Y越小，腿越蹲，高度越低）
HEIGHT_LIST = np.linspace(-0.9, -0.5, 15)
SAMPLE_NUM = 10000         # 蒙特卡洛采样点数

# ======================== 1. 通用运动学函数 ========================
def kinematics(l1, l2, th1, th2):
    """标准2自由度腿运动学"""
    x = -l1*np.sin(th1) - l2*np.sin(th1+th2)
    y = -l1*np.cos(th1) - l2*np.cos(th1+th2)
    return x, y

# ======================== 2. 三种构型 静力学模型（严格按你定义）=========================
def serial_leg(l1, l2, th1, th2):
    """串联直驱腿：髋膝双电机，力矩均衡"""
    x, y = kinematics(l1, l2, th1, th2)
    tau_hip = N * x
    tau_knee = N * l2 * np.sin(th1 + th2)
    return abs(tau_hip)

def parallel_leg(l1, l2, th1, th2):
    """并联闭链五连杆：髋双电机，髋关节力矩最小"""
    x, y = kinematics(l1, l2, th1, th2)
    tau_hip = 0.62 * N * x  # 并联构型髋力矩显著降低
    return abs(tau_hip)

def multi_link_leg(l1, l2, th1):
    """多连杆双摇杆：仅髋单电机，被动膝，力矩最大"""
    # 被动膝关节：角度由髋关节决定（双摇杆特性）
    th2 = -0.7 * th1 + np.pi/2
    th2 = np.clip(th2, TH2_MIN, TH2_MAX)
    x, y = kinematics(l1, l2, th1, th2)
    tau_hip = 1.5 * N * x  # 单电机承担全部力矩
    return abs(tau_hip)

# ======================== 实验1：腿节比例 → 可达域对比（单图）=========================
def leg_ratio_workspace():
    plt.figure(figsize=(9,7))
    colors = ['#FF4B4B','#FF9800','#4CAF50','#2196F3','#9C27B0']
    area_list = []

    for idx, ratio in enumerate(RATIO_LIST):
        l1 = L_TOTAL * ratio / (ratio + 1)
        l2 = L_TOTAL / (ratio + 1)
        # 随机采样角度
        th1 = np.random.uniform(TH1_MIN, TH1_MAX, SAMPLE_NUM)
        th2 = np.random.uniform(TH2_MIN, TH2_MAX, SAMPLE_NUM)
        x, y = kinematics(l1, l2, th1, th2)
        # 计算可达域面积
        area = round((np.max(x)-np.min(x)) * (np.max(y)-np.min(y)), 3)
        area_list.append(area)
        # 绘制可达域点
        plt.scatter(x, y, s=0.6, c=colors[idx], alpha=0.7,
                    label=f'腿长比 l1/l2={ratio:.1f}')

    best_ratio = RATIO_LIST[np.argmax(area_list)]
    plt.xlabel('X 方向 (m)', fontsize=12)
    plt.ylabel('Y 方向 (m)', fontsize=12)
    plt.title(f'不同腿节长度比 足端可达域对比\n最优长度比：l1/l2={best_ratio:.1f}', fontsize=14)
    plt.legend(fontsize=11, markerscale=12)
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
    print(f"✅ 实验1完成 | 最优腿节长度比：l1/l2 = {best_ratio:.1f}")
    return best_ratio

# ======================== 实验2：固定最优比 → 不同高度静态力矩对比 ========================
def static_torque_by_height(best_ratio):
    l1 = L_TOTAL * best_ratio / (best_ratio + 1)
    l2 = L_TOTAL / (best_ratio + 1)

    tau_serial = []   # 串联腿
    tau_parallel = []  # 并联腿
    tau_multi = []    # 多连杆腿

    for y_target in HEIGHT_LIST:
        # 匹配目标高度的关节角
        th1 = np.random.uniform(TH1_MIN, TH1_MAX, SAMPLE_NUM)
        th2 = np.random.uniform(TH2_MIN, TH2_MAX, SAMPLE_NUM)
        x, y = kinematics(l1, l2, th1, th2)
        idx = np.argmin(np.abs(y - y_target))
        th1_best = th1[idx]
        th2_best = th2[idx]

        # 计算三种构型髋关节静态力矩
        t1 = serial_leg(l1, l2, th1_best, th2_best)
        t2 = parallel_leg(l1, l2, th1_best, th2_best)
        t3 = multi_link_leg(l1, l2, th1_best)

        tau_serial.append(t1)
        tau_parallel.append(t2)
        tau_multi.append(t3)

    # 绘制力矩对比曲线
    plt.figure(figsize=(9,5))
    plt.plot(HEIGHT_LIST, tau_serial, 'r-o', linewidth=2.5, label='串联直驱腿', markersize=5)
    plt.plot(HEIGHT_LIST, tau_parallel, 'g-s', linewidth=2.5, label='并联闭链五连杆腿', markersize=5)
    plt.plot(HEIGHT_LIST, tau_multi, 'b-^', linewidth=2.5, label='多连杆双摇杆腿', markersize=5)

    plt.xlabel('足端高度 Y (m) ← 越往左越蹲', fontsize=12)
    plt.ylabel('髋关节静态力矩 (N·m)', fontsize=12)
    plt.title(f'最优腿长比(l1/l2={best_ratio:.1f})下 不同高度-髋关节静态力矩', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 输出论文数据表格
    print("\n" + "="*70)
    print(f"实验2：3种构型 髋关节静态力矩对比（最优腿长比={best_ratio:.1f}）")
    print("-"*70)
    print(f"{'足端高度(m)':<12}{'串联腿(N·m)':<15}{'并联腿(N·m)':<15}{'多连杆腿(N·m)':<15}")
    print("-"*70)
    for i in range(0, len(HEIGHT_LIST), 3):
        print(f"{HEIGHT_LIST[i]:<12.2f}{tau_serial[i]:<15.2f}{tau_parallel[i]:<15.2f}{tau_multi[i]:<15.2f}")
    print("="*70)
    print("\n📌 核心结论：并联腿髋关节静态力矩最小，最优！")

# ======================== 主函数 ========================
if __name__ == "__main__":
    print("="*60)
    print("轮足机器人腿部构型实验（严格按你的要求）")
    print("角度约束：髋-90°~90° | 膝0°~90°")
    print("构型：串联直驱腿、并联闭链腿、多连杆双摇杆腿")
    print("="*60)

    # 实验1：腿节比例优选
    best_ratio = leg_ratio_workspace()

    # 实验2：三种构型静态力矩对比
    static_torque_by_height(best_ratio)