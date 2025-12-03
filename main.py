from sys import path
import os

# 检查路径是否存在，避免报错
if os.path.exists("./FDTD_func/"):
    path.append("./FDTD_func/")
from torchTEM3D import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from discretize import TensorMesh
import time

# =============================================================================
# 一、构建计算网格（TensorMesh）
# =============================================================================
dh = 5  # 网格核心区域的基本单元宽度（单位：m）
XI = 300  # x方向网格数
YJ = 300  # y方向网格数
ZK = 300  # z方向网格数
pad_num = 30  # 每个方向上两侧的“扩展层”数量

# 核心区域的实际计算网格数
core_num_x = XI - 2 * pad_num
core_num_y = YJ - 2 * pad_num
core_num_z = ZK - 2 * pad_num

padd_value = 1.2  # 外层网格的扩展比例

# 定义每个方向的网格步长序列
hx = [(dh, pad_num, -padd_value), (dh, core_num_x), (dh, pad_num, padd_value)]
hy = [(dh, pad_num, -padd_value), (dh, core_num_y), (dh, pad_num, padd_value)]
hz = [(dh, pad_num, -padd_value), (dh, core_num_z), (dh, pad_num, padd_value)]

# 创建三维张量网格
mesh = TensorMesh([hx, hy, hz])

# 提取网格尺寸
x = mesh.h[0]
y = mesh.h[1]
z = mesh.h[2]

# =============================================================================
# 二、计算设备设置
# =============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# 三、定义电导率模型（model_EC）
# =============================================================================
model_EC = torch.ones((XI, YJ, ZK))

# ---------- 模型设置 ----------
# 注意：这里将所有区域都设置为 1e-2 (0.01 S/m)，相当于均匀全空间
# 如果需要半空间，请将空气层设置为极小值 (如 1e-8)
model_EC[:, :, 0:7] = model_EC[:, :, 0:7] * 1e-2
model_EC[:, :, 7:-1] = model_EC[:, :, 7:-1] * 1e-2

# 记录背景电导率用于解析解计算 (对应上面的 1e-2)
bg_sigma = 1e-2

model_EC = torch.tensor(model_EC, device=device, requires_grad=False)

# =============================================================================
# 四、发射接收系统参数设置
# =============================================================================
L_loop = 5  # 发射回线边长（m）
n_subloop = 64  # 子线圈数量
Alpha = 0.2  # 稳定性因子
Rx_st = 150  # 接收点位置
Rx_end = 150
i0, j0, k0 = 150, 150, 150  # 发射中心
iter_n_max = 2000  # 迭代步数

# =============================================================================
# 五、执行正演计算 (任意角度叠加)
# =============================================================================
strat_time = time.time()

# 定义角度
theta_deg = 45
phi_deg = 30
theta = np.radians(theta_deg)
phi = np.radians(phi_deg)

# 计算方向余弦
nx = np.sin(theta) * np.cos(phi)
ny = np.sin(theta) * np.sin(phi)
nz = np.cos(theta)

print(f"Simulating for normal vector: ({nx:.2f}, {ny:.2f}, {nz:.2f})")

dBz_total = 0

# 1. Z 方向分量
if abs(nz) > 1e-5:
    print(f"--- Simulating Z-component source (weight={nz:.3f}) ---")
    t_H, dBz_z = forward_3DTEM(L_loop, n_subloop, i0, j0, k0, XI, YJ, ZK, x, y, z, Alpha, model_EC, Rx_st, Rx_end,
                               iter_n_max, device=device, source_dir=2)
    dBz_total += nz * dBz_z

# 2. X 方向分量
if abs(nx) > 1e-5:
    print(f"--- Simulating X-component source (weight={nx:.3f}) ---")
    t_H, dBz_x = forward_3DTEM(L_loop, n_subloop, i0, j0, k0, XI, YJ, ZK, x, y, z, Alpha, model_EC, Rx_st, Rx_end,
                               iter_n_max, device=device, source_dir=0)
    dBz_total += nx * dBz_x

# 3. Y 方向分量
if abs(ny) > 1e-5:
    print(f"--- Simulating Y-component source (weight={ny:.3f}) ---")
    t_H, dBz_y = forward_3DTEM(L_loop, n_subloop, i0, j0, k0, XI, YJ, ZK, x, y, z, Alpha, model_EC, Rx_st, Rx_end,
                               iter_n_max, device=device, source_dir=1)
    dBz_total += ny * dBz_y

print("Total response calculated.")
end_time = time.time()
print(f"Total calculation time: {end_time - strat_time:.2f} s")

# =============================================================================
# 六、绘图对比
# =============================================================================
# 将数据从 GPU 移至 CPU 并转为 numpy
# dBz_total 形状可能是 (iter_n_max, 1)，需要展平
if isinstance(t_H, torch.Tensor):
    t_plot = t_H.cpu().detach().numpy()
else:
    t_plot = t_H

if isinstance(dBz_total, torch.Tensor):
    emf_plot = np.abs(dBz_total.cpu().detach().numpy())
else:
    emf_plot = np.abs(dBz_total)

# 如果是二维数组 (Time, Rx)，取第一个接收点
if emf_plot.ndim > 1:
    emf_plot = emf_plot[:, 0]

plt.figure(figsize=(10, 7))

# 1. 绘制 FDTD 模拟结果
plt.loglog(t_plot, emf_plot, 'r.', markersize=4, label='FDTD: Tilted Loop (45 deg)')
plt.loglog(t_plot, emf_plot, 'r-', linewidth=0.5, alpha=0.5)


# 2. 绘制均匀全空间解析解 (作为参考基准)
# 注意：你的模型设置中 model_EC 全部乘了 1e-2，所以相当于均匀全空间 (Whole Space)
# 下面的公式是中心回线在均匀介质中的解析解形式 (Impulse Response)
def calc_analytical_ref(time_array):
    mu_0 = 4 * np.pi * 1e-7
    # 等效半径
    r0 = L_loop / np.sqrt(np.pi)

    # 扩散参数 theta (或 u)
    u = (r0 / 2.0) * np.sqrt(mu_0 * bg_sigma / time_array)

    term1 = 1.0 / r0
    term2 = mu_0 / (np.sqrt(np.pi) * time_array)
    term3 = u ** 3 * np.exp(-u ** 2)

    # 垂直分量解析解 * 投影系数 (假设 Rx 在中心，水平分量贡献为0，只有 Mz 贡献)
    # 对于中心回线，水平磁偶极子(Mx, My)在中心产生的 Bz 为 0。
    # 所以理论解 = 垂直磁偶极子产生的 Bz * cos(theta)
    vert_bz_dt = -1.0 * term1 * term2 * term3

    return np.abs(vert_bz_dt * np.cos(theta))  # 使用之前定义的 theta


# 计算解析解
# 避免时间为0导致除零错误，从第2个点开始计算
t_ref = t_plot[1:]
emf_ref = calc_analytical_ref(t_ref)

plt.loglog(t_ref, emf_ref, 'k--', linewidth=2, label=f'Ref: Analytical ($\sigma$={bg_sigma} S/m)')

# 3. 装饰
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude |dB/dt| (V/m^2)', fontsize=14)
plt.title(f'3D FDTD vs Analytical\nSource Tilt: {theta_deg} deg, Azimuth: {phi_deg} deg', fontsize=16)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()