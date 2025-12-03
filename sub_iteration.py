import torch


def sub_iteration_torch(
        i0, j0, k0, XI, YJ, ZK, dx, dy, dz, Rx_st, Rx_end,
        iter_n_max, Alpha, model_EC_2, EX, EY, EZ, HX, HY, HZ, t1_E, t1_H,
        device="cuda"
):
    # ============================= Parameter Preprocessing =============================
    # 将 Matlab 风格的 1-based 索引转换为 PyTorch 0-based 索引
    i0 -= 1
    j0 -= 1
    k0 -= 1
    Rx_st -= 1
    Rx_end -= 1

    # 接收器数量（沿测线的接收点个数）
    Rx_size = Rx_end - Rx_st + 1

    # 转为张量并放到 device 上（dx,dy,dz 为 1D 网格尺寸向量）
    dx = torch.as_tensor(dx, dtype=torch.float32, device=device)
    dy = torch.as_tensor(dy, dtype=torch.float32, device=device)
    dz = torch.as_tensor(dz, dtype=torch.float32, device=device)

    # 将场张量从形状 [x, y, z, time] 变为 [time, x, y, z]，便于按时间步更新
    EX = EX.permute(3, 0, 1, 2)
    EY = EY.permute(3, 0, 1, 2)
    EZ = EZ.permute(3, 0, 1, 2)
    HX = HX.permute(3, 0, 1, 2)
    HY = HY.permute(3, 0, 1, 2)
    HZ = HZ.permute(3, 0, 1, 2)

    # ============================= Conductivity Node Calculation =============================
    # 在 Yee 网格上为 E 分量位置计算等效导电率（面积/体积加权平均）
    EC_x = torch.zeros(XI, YJ, ZK, device=device)
    EC_y = torch.zeros(XI, YJ, ZK, device=device)
    EC_z = torch.zeros(XI, YJ, ZK, device=device)

    # EC_x ：位于 x 方向边的等效导电率（加权：相邻四个体素在 y-z 面的面积权重）
    i, j, k = torch.meshgrid(
        torch.arange(XI, device=device),
        torch.arange(1, YJ, device=device),  # y 方向中心差分，跳过边界
        torch.arange(1, ZK, device=device),  # z 方向中心差分，跳过边界
        indexing='ij'
    )

    numerator = (
            dy[j - 1] * dz[k - 1] * model_EC_2[i, j - 1, k - 1] +
            dy[j] * dz[k - 1] * model_EC_2[i, j, k - 1] +
            dy[j - 1] * dz[k] * model_EC_2[i, j - 1, k] +
            dy[j] * dz[k] * model_EC_2[i, j, k]
    )
    denominator = (dy[j - 1] + dy[j]) * (dz[k - 1] + dz[k])
    EC_x[i, j, k] = numerator / denominator

    # EC_y ：位于 y 方向边的等效导电率（加权：相邻四个体素在 x-z 面的面积权重）
    i, j, k = torch.meshgrid(
        torch.arange(1, XI, device=device),  # x 方向中心差分
        torch.arange(YJ, device=device),
        torch.arange(1, ZK, device=device),  # z 方向中心差分
        indexing='ij'
    )
    numerator = (
            dx[i - 1] * dz[k - 1] * model_EC_2[i - 1, j, k - 1] +
            dx[i] * dz[k - 1] * model_EC_2[i, j, k - 1] +
            dx[i - 1] * dz[k] * model_EC_2[i - 1, j, k] +
            dx[i] * dz[k] * model_EC_2[i, j, k]
    )
    denominator = (dx[i - 1] + dx[i]) * (dz[k - 1] + dz[k])
    EC_y[i, j, k] = numerator / denominator

    # EC_z ：位于 z 方向边的等效导电率（加权：相邻四个体素在 x-y 面的面积权重）
    i, j, k = torch.meshgrid(
        torch.arange(1, XI, device=device),  # x 方向中心差分
        torch.arange(1, YJ, device=device),  # y 方向中心差分
        torch.arange(ZK, device=device),
        indexing='ij'
    )
    numerator = (
            dx[i - 1] * dy[j - 1] * model_EC_2[i - 1, j - 1, k] +
            dx[i] * dy[j - 1] * model_EC_2[i, j - 1, k] +
            dx[i - 1] * dy[j] * model_EC_2[i - 1, j, k] +
            dx[i] * dy[j] * model_EC_2[i, j, k]
    )
    denominator = (dx[i - 1] + dx[i]) * (dy[j - 1] + dy[j])
    EC_z[i, j, k] = numerator / denominator

    # ============================= Time Iteration Initialization =============================
    permeability_vac = 4 * torch.pi * 1e-7  # 真空磁导率 μ0
    t_iteration_E = torch.zeros(iter_n_max, device=device)  # 每步的电场时间
    t_iteration_H = torch.zeros(iter_n_max, device=device)  # 每步的磁场时间
    DBZ_Rx = torch.zeros(iter_n_max, Rx_size, device=device)  # 接收点 dBz 记录

    # 时步初始化（取发射中心处的导电率 model_t）
    model_t = model_EC_2[i0, j0, k0].detach()
    t_iteration_E[0] = t1_E
    t_iteration_H[0] = t1_H

    # 初始 dt（与 Alpha、网格尺寸 dz 及介质参数有关）
    dt0 = Alpha * dz[k0] * torch.sqrt(permeability_vac * model_t * t1_E / 6)

    # 初始化时刻的接收数据（沿 y 向的测线：Rx_st..Rx_end）
    rx_slice = slice(Rx_st, Rx_end + 1)  # y 索引区间
    rx_slice_plus_1 = slice(Rx_st + 1, Rx_end + 2)

    # dBz = dEx/dy - dEy/dx（Faraday 旋度关系）
    dEx_dy = (EX[0, i0, rx_slice_plus_1, k0 + 1] - EX[0, i0, rx_slice, k0 + 1]) / dy[rx_slice]
    dEy_dx = (EY[0, i0 + 1, rx_slice, k0 + 1] - EY[0, i0, rx_slice, k0 + 1]) / dx[i0]
    DBZ_Rx[0] = dEx_dy - dEy_dx

    # ============================= Main Iteration Loop =============================
    iter_n = 1
    while iter_n < iter_n_max:

        # 时间推进（电场时间与磁场时间交错）
        t_iteration_E[iter_n] = t_iteration_E[iter_n - 1] + dt0
        dt1 = Alpha * dz[k0] * torch.sqrt(permeability_vac * model_t * t_iteration_E[iter_n] / 6)
        t_iteration_H[iter_n] = t_iteration_H[iter_n - 1] + (dt0 + dt1) / 2
        gamma = (4 / permeability_vac) * (dt0 / dz[k0]) ** 2  # 便利系数（合并常数项）

        # --------------------------- Electric Field Update ---------------------------
        # Ex 更新：使用中心差分近似 curl(H) 的 y、z 分量
        i, j, k = torch.meshgrid(
            torch.arange(XI, device=device),
            torch.arange(1, YJ, device=device),
            torch.arange(1, ZK, device=device),
            indexing='ij'
        )

        term1 = (2 * gamma - EC_x[i, j, k] * dt0) / (2 * gamma + EC_x[i, j, k] * dt0)
        term2 = 4 * dt0 / (2 * gamma + EC_x[i, j, k] * dt0)

        # curl H 的 x 分量：∂Hz/∂y - ∂Hy/∂z（非均匀网格用相邻两单元尺寸和）
        dHZ_dy = (HZ[0, i, j, k] - HZ[0, i, j - 1, k]) / (dy[j - 1] + dy[j])
        dHY_dz = (HY[0, i, j, k] - HY[0, i, j, k - 1]) / (dz[k - 1] + dz[k])
        EX[1, i, j, k] = term1 * EX[0, i, j, k] + term2 * (dHZ_dy - dHY_dz)

        # Ey 更新：curl H 的 y 分量：∂Hx/∂z - ∂Hz/∂x
        i, j, k = torch.meshgrid(
            torch.arange(1, XI, device=device),
            torch.arange(YJ, device=device),
            torch.arange(1, ZK, device=device),
            indexing='ij'
        )

        term1 = (2 * gamma - EC_y[i, j, k] * dt0) / (2 * gamma + EC_y[i, j, k] * dt0)
        term2 = 4 * dt0 / (2 * gamma + EC_y[i, j, k] * dt0)
        dHX_dz = (HX[0, i, j, k] - HX[0, i, j, k - 1]) / (dz[k - 1] + dz[k])
        dHZ_dx = (HZ[0, i, j, k] - HZ[0, i - 1, j, k]) / (dx[i - 1] + dx[i])
        EY[1, i, j, k] = term1 * EY[0, i, j, k] + term2 * (dHX_dz - dHZ_dx)

        # Ez 更新：curl H 的 z 分量：∂Hy/∂x - ∂Hx/∂y
        i, j, k = torch.meshgrid(
            torch.arange(1, XI, device=device),
            torch.arange(1, YJ, device=device),
            torch.arange(ZK, device=device),
            indexing='ij'
        )
        term1 = (2 * gamma - EC_z[i, j, k] * dt0) / (2 * gamma + EC_z[i, j, k] * dt0)
        term2 = 4 * dt0 / (2 * gamma + EC_z[i, j, k] * dt0)
        dHY_dx = (HY[0, i, j, k] - HY[0, i - 1, j, k]) / (dx[i - 1] + dx[i])
        dHX_dy = (HX[0, i, j, k] - HX[0, i, j - 1, k]) / (dy[j - 1] + dy[j])
        EZ[1, i, j, k] = term1 * EZ[0, i, j, k] + term2 * (dHY_dx - dHX_dy)

        # ------------------------ Dirichlet Boundary Conditions ------------------------
        # 对边界施加零值（Dirichlet），避免越界访问/反射
        # EX 边界
        EX[1, :, [0, -1], :] = 0
        EX[1, :, :, [0, -1]] = 0

        # EY 边界
        EY[1, [0, -1], :, :] = 0
        EY[1, :, :, [0, -1]] = 0

        # EZ 边界
        EZ[1, [0, -1], :, :] = 0
        EZ[1, :, [0, -1], :] = 0

        # --------------------------- Magnetic Field Update ---------------------------
        # Hx 更新：curl E 的 x 分量：∂Ey/∂z - ∂Ez/∂y
        i, j, k = torch.meshgrid(
            torch.arange(XI + 1, device=device),
            torch.arange(YJ, device=device),
            torch.arange(ZK, device=device),
            indexing='ij'
        )
        dEY_dz = (EY[1, i, j, k + 1] - EY[1, i, j, k]) / dz[k]
        dEZ_dy = (EZ[1, i, j + 1, k] - EZ[1, i, j, k]) / dy[j]
        HX[1, i, j, k] = HX[0, i, j, k] + (dt0 + dt1) / (2 * permeability_vac) * (dEY_dz - dEZ_dy)

        # Hy 更新：curl E 的 y 分量：∂Ez/∂x - ∂Ex/∂z
        i, j, k = torch.meshgrid(
            torch.arange(XI, device=device),
            torch.arange(YJ + 1, device=device),
            torch.arange(ZK, device=device),
            indexing='ij'
        )
        dEZ_dx = (EZ[1, i + 1, j, k] - EZ[1, i, j, k]) / dx[i]
        dEX_dz = (EX[1, i, j, k + 1] - EX[1, i, j, k]) / dz[k]
        HY[1, i, j, k] = HY[0, i, j, k] + (dt0 + dt1) / (2 * permeability_vac) * (dEZ_dx - dEX_dz)

        # Hz 更新：这里使用 z 方向的积分推进（自底向上 / 自顶向下）并设置顶部/底部边界
        HZ[:, :, 0, 0] = 0
        HZ[:, :, -1, 0] = 0

        # 自底向上（k: ZK -> k0）
        i, j = torch.meshgrid(
            torch.arange(XI, device=device),
            torch.arange(YJ, device=device),
            indexing='ij'
        )
        for k in range(ZK, k0, -1):
            dHX_dx = (HX[1, i + 1, j, k - 1] - HX[1, i, j, k - 1]) / dx[i]
            dHY_dy = (HY[1, i, j + 1, k - 1] - HY[1, i, j, k - 1]) / dy[j]
            HZ[1, i, j, k - 1] = HZ[1, i, j, k] + dz[k - 1] * (dHX_dx + dHY_dy)

        # 自顶向下（k: 1 -> k0-1），注意使用 detach 打断环依赖
        m, n = torch.meshgrid(
            torch.arange(YJ, device=device),
            torch.arange(XI, device=device),
            indexing='ij'
        )
        for k in range(1, k0):
            dHX_dx = (HX[1, n + 1, m, k - 1] - HX[1, n, m, k - 1]) / dx[n]
            dHY_dy = (HY[1, n, m + 1, k - 1] - HY[1, n, m, k - 1]) / dy[m]
            HZ[1, n, m, k] = HZ[1, n, m, k - 1].detach() - dz[k - 1] * (dHX_dx + dHY_dy)

        # ------------------------ Receiver Data Recording ------------------------
        # 记录接收点的 dBz（沿测线 y 方向）
        rx_slice = slice(Rx_st, Rx_end + 1)
        rx_slice_plus_1 = slice(Rx_st + 1, Rx_end + 2)

        dEx_dy = (EX[1, i0, rx_slice_plus_1, k0 + 1] - EX[1, i0, rx_slice, k0 + 1]) / dy[rx_slice]
        dEy_dx = (EY[1, i0 + 1, rx_slice, k0 + 1] - EY[1, i0, rx_slice, k0 + 1]) / dx[i0]
        DBZ_Rx[iter_n] = dEx_dy - dEy_dx

        # [3.4] Roll time steps ---------------------------------------------
        # 将当前时间步（索引 1）的场拷贝到上一时间步（索引 0），进入下一次迭代
        EX[0], EY[0], EZ[0] = EX[1], EY[1], EZ[1]
        HX[0], HY[0], HZ[0] = HX[1], HY[1], HZ[1]
        dt0 = dt1  # 更新时间步
        iter_n += 1

    # ============================= Output Results =============================
    print(f'Computation finished.')
    return t_iteration_H.detach(), DBZ_Rx
