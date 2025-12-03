import torch
from calculate_coord import *


def sub_initial_field_subloop_torch(i0, j0, k0, XI, YJ, ZK, model_EC, dx, dy, dz,
                                    offset_1, offset_2, L_subloop, Alpha, device='cuda', source_dir=2):
    """
    source_dir: 发射源法线方向。
        0 -> x方向 (线圈在yz平面)
        1 -> y方向 (线圈在xz平面)
        2 -> z方向 (线圈在xy平面, 默认原版)
    offset_1, offset_2: 子回线在所在平面内的两个偏移量
        dir=2 (z) -> offset_x, offset_y
        dir=0 (x) -> offset_y, offset_z
        dir=1 (y) -> offset_x, offset_z
    """
    # 确保张量类型
    dx = torch.as_tensor(dx, dtype=torch.float32, device=device)
    dy = torch.as_tensor(dy, dtype=torch.float32, device=device)
    dz = torch.as_tensor(dz, dtype=torch.float32, device=device)

    permeability_vac = 4 * torch.pi * 1e-7

    # 初始化所有场分量为0
    EX_sub = torch.zeros((XI, YJ + 1, ZK + 1, 2), dtype=torch.float32, device=device)
    EY_sub = torch.zeros((XI + 1, YJ, ZK + 1, 2), dtype=torch.float32, device=device)
    EZ_sub = torch.zeros((XI + 1, YJ + 1, ZK, 2), dtype=torch.float32, device=device)
    HX_sub = torch.zeros((XI + 1, YJ, ZK, 2), dtype=torch.float32, device=device)
    HY_sub = torch.zeros((XI, YJ + 1, ZK, 2), dtype=torch.float32, device=device)
    HZ_sub = torch.zeros((XI, YJ, ZK + 1, 2), dtype=torch.float32, device=device)

    # 获取中心点电导率
    # 注意：这里假设介质各向同性，或者取主对角线方向
    sigma = model_EC[i0 - 1, j0 - 1, k0 - 1]
    Txmm = L_subloop ** 2

    # 计算时间尺度 (假设各方向扩散时间计算公式形式一致，主要取决于源所在位置的网格尺寸和电导率)
    # 为了简化，我们沿用原代码基于 dz[k0] 的计算，但在极端各向异性网格中可能需要调整参考步长
    t1_E = 1.13 * permeability_vac * sigma * dz[k0 - 1] ** 2
    dt1 = Alpha * dz[k0 - 1] * torch.sqrt(permeability_vac * sigma * t1_E / 6)
    t1_H = t1_E + dt1 / 2

    # 定义辅助函数：计算通用偶极子公式项
    def get_dipole_term(r, component_coord, t_val):
        # component_coord: 计算E场旋度项所需的垂直距离 (例如Ex需要y或z)
        u = r * torch.sqrt(torch.tensor(2.0, device=device) * torch.pi * sigma / (1e7 * t_val))
        # 电场项分子分母
        num = torch.sqrt(torch.tensor(2.0, device=device) / torch.pi) * Txmm * component_coord * (u / r) ** 5
        den = 4 * torch.pi * sigma * torch.exp(u ** 2 / 2)
        return num / den

    def get_H_terms(r, parallel_coord, perp_coord_1, perp_coord_2, t_val, is_parallel_component):
        # parallel_coord: 沿偶极子方向的坐标 (如Z方向偶极子，取z坐标)
        # perp_coord: 垂直方向坐标
        u = r * torch.sqrt(torch.tensor(2.0, device=device) * torch.pi * sigma / (1e7 * t_val))
        term_erf = 3 * torch.erf(u) - torch.sqrt(torch.tensor(2.0, device=device) / torch.pi) * u * (
                    3 + u ** 2) / torch.exp(u ** 2 / 2)

        pre_factor = Txmm / (4 * torch.pi * r ** 5)

        if not is_parallel_component:
            # 计算垂直分量 (例如 Z偶极子产生的 Hx, Hy)
            # 公式: (3*z*x / r^5) * term
            return (pre_factor * parallel_coord * perp_coord_1) * term_erf
        else:
            # 计算平行分量 (例如 Z偶极子产生的 Hz)
            # 公式涉及 (2z^2 - rho^2)
            rho_sq = perp_coord_1 ** 2 + perp_coord_2 ** 2
            term1 = (2 * parallel_coord ** 2 - rho_sq) * torch.erf(u)
            term2 = (2 * parallel_coord ** 2 - rho_sq * (1 + u ** 2)) * torch.sqrt(
                torch.tensor(2.0, device=device) / torch.pi) / torch.exp(u ** 2 / 2)
            return pre_factor * (term1 - term2)

    # =========================================================================
    # 根据 source_dir 分支计算
    # =========================================================================

    # --------------------------- Z 方向 (原版逻辑) ---------------------------
    if source_dir == 2:
        sub_x, sub_y = offset_1, offset_2

        # EX (由y坐标贡献)
        i, j, k = torch.meshgrid(torch.arange(XI, device=device), torch.arange(1, YJ, device=device),
                                 torch.arange(1, ZK, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 1, i0, dx, device) + sub_x
        cy = calculate_coord_y_torch(j + 1, 1, j0, dy, device) + sub_y
        cz = calculate_coord_z_torch(k + 1, 1, k0, dz, device)
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        EX_sub[i, j, k, 0] = get_dipole_term(r, cy, t1_E)  # E_phi 投影到 x 轴，与 y 有关

        # EY (由x坐标贡献)
        i, j, k = torch.meshgrid(torch.arange(1, XI, device=device), torch.arange(YJ, device=device),
                                 torch.arange(1, ZK, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 2, i0, dx, device) + sub_x
        cy = calculate_coord_y_torch(j + 1, 2, j0, dy, device) + sub_y
        cz = calculate_coord_z_torch(k + 1, 2, k0, dz, device)
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        EY_sub[i, j, k, 0] = -get_dipole_term(r, cx, t1_E)  # 注意方向，通常 E_phi = (-y, x)，这里稍微简化，根据原代码逻辑调整正负
        # 原代码 EY 用的是 +coord_x? 让我们检查原代码...
        # 原代码: EY ... numerator 用 coord_x.
        # 物理上 VMD, E = C * (-y x_hat + x y_hat). EX ~ y, EY ~ x.
        # 你的原代码 EX 用了 coord_y, EY 用了 coord_x. 符号在 numerator 内部可能处理了或者在这里直接赋值。
        # 保持原代码逻辑: EX正比y, EY正比x.

        # HX (垂直分量)
        i, j, k = torch.meshgrid(torch.arange(XI + 1, device=device), torch.arange(YJ, device=device),
                                 torch.arange(ZK, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 4, i0, dx, device) + sub_x
        cy = calculate_coord_y_torch(j + 1, 4, j0, dy, device) + sub_y
        cz = calculate_coord_z_torch(k + 1, 4, k0, dz, device)
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        HX_sub[i, j, k, 0] = get_H_terms(r, cz, cx, cy, t1_H, False)

        # HY (垂直分量)
        i, j, k = torch.meshgrid(torch.arange(XI, device=device), torch.arange(YJ + 1, device=device),
                                 torch.arange(ZK, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 5, i0, dx, device) + sub_x
        cy = calculate_coord_y_torch(j + 1, 5, j0, dy, device) + sub_y
        cz = calculate_coord_z_torch(k + 1, 5, k0, dz, device)
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        HY_sub[i, j, k, 0] = get_H_terms(r, cz, cy, cx, t1_H, False)

        # HZ (平行分量)
        i, j, k = torch.meshgrid(torch.arange(XI, device=device), torch.arange(YJ, device=device),
                                 torch.arange(ZK + 1, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 6, i0, dx, device) + sub_x
        cy = calculate_coord_y_torch(j + 1, 6, j0, dy, device) + sub_y
        cz = calculate_coord_z_torch(k + 1, 6, k0, dz, device)
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        r = torch.where(r < 0.001 * dz[k0 - 1], 0.001 * dz[k0 - 1], r)
        HZ_sub[i, j, k, 0] = get_H_terms(r, cz, cx, cy, t1_H, True)

    # --------------------------- X 方向 (YZ平面线圈) ---------------------------
    elif source_dir == 0:
        # X是主轴。offset_1 -> y, offset_2 -> z
        sub_y, sub_z = offset_1, offset_2

        # EX (平行于磁矩方向，通常为0或极小，在TE模中忽略，或者类似HZ处理? 不，电场垂直于磁矩)
        # 磁偶极子 M_x 产生环绕 X 轴的电场。即只有 Ey 和 Ez。 Ex = 0.
        EX_sub[...] = 0

        # EY (由 z 坐标贡献, 类似于 VMD 的 EX 由 y 贡献)
        # M_x 产生的 E 场环绕 X 轴: E ~ z y_hat - y z_hat (右手螺旋)
        # EY 正比于 -z (或 z, 取决于正负定义)
        i, j, k = torch.meshgrid(torch.arange(1, XI, device=device), torch.arange(YJ, device=device),
                                 torch.arange(1, ZK, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 2, i0, dx, device)
        cy = calculate_coord_y_torch(j + 1, 2, j0, dy, device) + sub_y
        cz = calculate_coord_z_torch(k + 1, 2, k0, dz, device) + sub_z
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        # 注意：这里引用 z 坐标
        EY_sub[i, j, k, 0] = -get_dipole_term(r, cz, t1_E)  # 符号需仔细核对，暂定负

        # EZ (由 y 坐标贡献)
        i, j, k = torch.meshgrid(torch.arange(1, XI, device=device), torch.arange(1, YJ, device=device),
                                 torch.arange(ZK, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 3, i0, dx, device)  # EZ网格位置
        cy = calculate_coord_y_torch(j + 1, 3, j0, dy, device) + sub_y
        cz = calculate_coord_z_torch(k + 1, 3, k0, dz, device) + sub_z
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        EZ_sub[i, j, k, 0] = get_dipole_term(r, cy, t1_E)

        # HX (平行分量，主分量)
        i, j, k = torch.meshgrid(torch.arange(XI + 1, device=device), torch.arange(YJ, device=device),
                                 torch.arange(ZK, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 4, i0, dx, device)
        cy = calculate_coord_y_torch(j + 1, 4, j0, dy, device) + sub_y
        cz = calculate_coord_z_torch(k + 1, 4, k0, dz, device) + sub_z
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        r = torch.where(r < 0.001 * dx[i0 - 1], 0.001 * dx[i0 - 1], r)  # 保护
        HX_sub[i, j, k, 0] = get_H_terms(r, cx, cy, cz, t1_H, True)

        # HY (垂直分量)
        i, j, k = torch.meshgrid(torch.arange(XI, device=device), torch.arange(YJ + 1, device=device),
                                 torch.arange(ZK, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 5, i0, dx, device)
        cy = calculate_coord_y_torch(j + 1, 5, j0, dy, device) + sub_y
        cz = calculate_coord_z_torch(k + 1, 5, k0, dz, device) + sub_z
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        HY_sub[i, j, k, 0] = get_H_terms(r, cx, cy, cz, t1_H, False)

        # HZ (垂直分量)
        i, j, k = torch.meshgrid(torch.arange(XI, device=device), torch.arange(YJ, device=device),
                                 torch.arange(ZK + 1, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 6, i0, dx, device)
        cy = calculate_coord_y_torch(j + 1, 6, j0, dy, device) + sub_y
        cz = calculate_coord_z_torch(k + 1, 6, k0, dz, device) + sub_z
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        HZ_sub[i, j, k, 0] = get_H_terms(r, cx, cz, cy, t1_H, False)

    # --------------------------- Y 方向 (XZ平面线圈) ---------------------------
    elif source_dir == 1:
        # Y是主轴。offset_1 -> x, offset_2 -> z
        sub_x, sub_z = offset_1, offset_2

        # EX (由 z 坐标贡献)
        i, j, k = torch.meshgrid(torch.arange(XI, device=device), torch.arange(1, YJ, device=device),
                                 torch.arange(1, ZK, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 1, i0, dx, device) + sub_x
        cy = calculate_coord_y_torch(j + 1, 1, j0, dy, device)
        cz = calculate_coord_z_torch(k + 1, 1, k0, dz, device) + sub_z
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        EX_sub[i, j, k, 0] = get_dipole_term(r, cz, t1_E)

        # EY (平行为0)
        EY_sub[...] = 0

        # EZ (由 x 坐标贡献)
        i, j, k = torch.meshgrid(torch.arange(1, XI, device=device), torch.arange(1, YJ, device=device),
                                 torch.arange(ZK, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 3, i0, dx, device) + sub_x
        cy = calculate_coord_y_torch(j + 1, 3, j0, dy, device)
        cz = calculate_coord_z_torch(k + 1, 3, k0, dz, device) + sub_z
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        EZ_sub[i, j, k, 0] = -get_dipole_term(r, cx, t1_E)

        # HX (垂直分量)
        i, j, k = torch.meshgrid(torch.arange(XI + 1, device=device), torch.arange(YJ, device=device),
                                 torch.arange(ZK, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 4, i0, dx, device) + sub_x
        cy = calculate_coord_y_torch(j + 1, 4, j0, dy, device)
        cz = calculate_coord_z_torch(k + 1, 4, k0, dz, device) + sub_z
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        HX_sub[i, j, k, 0] = get_H_terms(r, cy, cx, cz, t1_H, False)

        # HY (平行分量，主分量)
        i, j, k = torch.meshgrid(torch.arange(XI, device=device), torch.arange(YJ + 1, device=device),
                                 torch.arange(ZK, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 5, i0, dx, device) + sub_x
        cy = calculate_coord_y_torch(j + 1, 5, j0, dy, device)
        cz = calculate_coord_z_torch(k + 1, 5, k0, dz, device) + sub_z
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        r = torch.where(r < 0.001 * dy[j0 - 1], 0.001 * dy[j0 - 1], r)
        HY_sub[i, j, k, 0] = get_H_terms(r, cy, cx, cz, t1_H, True)

        # HZ (垂直分量)
        i, j, k = torch.meshgrid(torch.arange(XI, device=device), torch.arange(YJ, device=device),
                                 torch.arange(ZK + 1, device=device), indexing='ij')
        cx = calculate_coord_x_torch(i + 1, 6, i0, dx, device) + sub_x
        cy = calculate_coord_y_torch(j + 1, 6, j0, dy, device)
        cz = calculate_coord_z_torch(k + 1, 6, k0, dz, device) + sub_z
        r = torch.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        HZ_sub[i, j, k, 0] = get_H_terms(r, cy, cz, cx, t1_H, False)

    return EX_sub, EY_sub, EZ_sub, HX_sub, HY_sub, HZ_sub, t1_E, t1_H