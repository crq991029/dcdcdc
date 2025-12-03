import torch

def calculate_coord_x_torch(x_matlab_tensor, m, i0, dx_tensor, device):
    """Calculate x-coordinate
    计算 x 方向坐标（基于 Matlab 风格的索引定义：索引从 1 开始）
    参数：
        x_matlab_tensor: 位置索引（Matlab 风格，1-based）
        m: 网格/变量类型标识（控制不同的坐标定义分支）
        i0: 参考中心索引（Matlab 风格，1-based）
        dx_tensor: 每个 x 单元的宽度数组
        device: 运行设备（'cpu' 或 'cuda'）
    返回：
        coord_x_tensor: 与 x_matlab_tensor 同形状的物理坐标（以 i0 为中心的相对坐标）
    """
    # 确保输入是 PyTorch 张量（且在指定 device 上）
    if not isinstance(x_matlab_tensor, torch.Tensor):
        x_matlab_tensor = torch.tensor(x_matlab_tensor, dtype=torch.float32, device=device)
    if not isinstance(dx_tensor, torch.Tensor):
        dx_tensor = torch.tensor(dx_tensor, dtype=torch.float32, device=device)

    # 初始化输出坐标张量（形状与 x_matlab_tensor 一致）
    coord_x_tensor = torch.zeros_like(x_matlab_tensor, dtype=torch.float32, device=device)
    # 累积和：cumsum_dx[k] = dx[0] + dx[1] + ... + dx[k]
    cumsum_dx = torch.cumsum(dx_tensor, dim=0)

    # 处理 m 在 [2, 3] 的情形（注：此处与 m=4 的分支不同，具体差异体现在右侧的索引与半单元处理）
    if m in [2, 3]:
        mask = x_matlab_tensor <= i0  # 左侧（含中心）与右侧的布尔掩码
        # 小于等于 i0 的索引（Matlab 1-based -> PyTorch 0-based，要 -1）
        indices_less_i0 = (x_matlab_tensor[mask] - 1).to(torch.long)
        # 左侧距离：从 i0-1 到目标索引的区间长度和（使用前缀和差法）
        sum_dx_less_i0 = cumsum_dx[i0 - 1] - cumsum_dx[indices_less_i0]
        # 大于 i0 的索引（同样 1-based -> 0-based）
        indices_greater_i0 = (x_matlab_tensor[~mask] - 1).to(torch.long)
        # 右侧距离：从 i0 到目标索引的区间长度和
        sum_dx_greater_i0 = cumsum_dx[indices_greater_i0] - cumsum_dx[i0 - 1]

        # 左侧坐标：负号表示相对 i0 向左
        coord_x_tensor[mask] = -(sum_dx_less_i0 - 0.5 * dx_tensor[i0 - 1] + dx_tensor[indices_less_i0])
        # 右侧坐标：正向
        coord_x_tensor[~mask] = sum_dx_greater_i0 + 0.5 * dx_tensor[i0 - 1] - dx_tensor[indices_greater_i0]

    # 单独处理 m=4 的情形（右侧索引与半单元处理与上面略有不同）
    elif m in [4]:
        mask = x_matlab_tensor <= i0
        indices_less_i0 = (x_matlab_tensor[mask] - 1).to(torch.long)
        sum_dx_less_i0 = cumsum_dx[i0 - 1] - cumsum_dx[indices_less_i0]

        # 注意这里是 -2（意味着右侧区间的起点向前多移一单元，与上分支差异所在）
        indices_greater_i0 = (x_matlab_tensor[~mask] - 2).to(torch.long)
        sum_dx_greater_i0 = cumsum_dx[indices_greater_i0] - cumsum_dx[i0 - 1]

        coord_x_tensor[mask] = -(sum_dx_less_i0 - 0.5 * dx_tensor[i0 - 1] + dx_tensor[indices_less_i0])
        coord_x_tensor[~mask] = sum_dx_greater_i0 + 0.5 * dx_tensor[i0 - 1]

    # 其他 m 的情形（对左右两侧均采用以相邻单元平均为中心的 0.5*dx 处理）
    else:
        mask = x_matlab_tensor <= i0
        indices_less_i0 = (x_matlab_tensor[mask] - 1).to(torch.long)
        sum_dx_less_i0 = cumsum_dx[i0 - 1] - cumsum_dx[indices_less_i0]

        indices_greater_i0 = (x_matlab_tensor[~mask] - 1).to(torch.long)
        sum_dx_greater_i0 = cumsum_dx[indices_greater_i0] - cumsum_dx[i0 - 1]

        # 左侧：考虑左侧单元与中心单元的“半单元平均”校正
        coord_x_tensor[mask] = -(sum_dx_less_i0 - 0.5 * (dx_tensor[indices_less_i0] + dx_tensor[i0 - 1]) + dx_tensor[indices_less_i0])
        # 右侧：同理采用半单元平均
        coord_x_tensor[~mask] = sum_dx_greater_i0 - 0.5 * dx_tensor[indices_greater_i0] + 0.5 * dx_tensor[i0 - 1]

    return coord_x_tensor


def calculate_coord_y_torch(y_matlab_tensor, m, j0, dy, device):
    """Calculate y-coordinate
    计算 y 方向坐标（同 x 的策略，按 m 的不同分支处理）
    参数：
        y_matlab_tensor: Matlab 风格 y 索引（1-based）
        m: 类型标识
        j0: y 方向中心索引（1-based）
        dy: 每个 y 单元宽度
        device: 设备
    返回：
        coord_y_tensor: 相对 j0 的物理坐标
    """
    # 确保输入为张量（并在指定 device 上）
    if not isinstance(y_matlab_tensor, torch.Tensor):
        y_matlab_tensor = torch.tensor(y_matlab_tensor, dtype=torch.float32, device=device)
    if not isinstance(dy, torch.Tensor):
        dy = torch.tensor(dy, dtype=torch.float32, device=device)

    coord_y_tensor = torch.zeros_like(y_matlab_tensor, dtype=torch.float32, device=device)
    cumsum_dy = torch.cumsum(dy, dim=0)

    # m 在 [1, 3]：一种半单元处理方案
    if m in [1, 3]:
        mask = y_matlab_tensor <= j0
        indices_less_j0 = (y_matlab_tensor[mask] - 1).to(torch.long)
        sum_dy_less_j0 = cumsum_dy[j0 - 1] - cumsum_dy[indices_less_j0]

        indices_greater_j0 = (y_matlab_tensor[~mask] - 1).to(torch.long)
        sum_dy_greater_j0 = cumsum_dy[indices_greater_j0] - cumsum_dy[j0 - 1]

        coord_y_tensor[mask] = -(sum_dy_less_j0 - 0.5 * dy[j0 - 1] + dy[indices_less_j0])
        coord_y_tensor[~mask] = sum_dy_greater_j0 + 0.5 * dy[j0 - 1] - dy[indices_greater_j0]

    # m 在 [5]：与上面不同的右侧处理（-2 的位移）
    elif m in [5]:
        mask = y_matlab_tensor <= j0
        indices_less_j0 = (y_matlab_tensor[mask] - 1).to(torch.long)
        sum_dy_less_j0 = cumsum_dy[j0 - 1] - cumsum_dy[indices_less_j0]

        indices_greater_j0 = (y_matlab_tensor[~mask] - 2).to(torch.long)
        sum_dy_greater_j0 = cumsum_dy[indices_greater_j0] - cumsum_dy[j0 - 1]

        coord_y_tensor[mask] = -(sum_dy_less_j0 - 0.5 * dy[j0 - 1] + dy[indices_less_j0])
        coord_y_tensor[~mask] = sum_dy_greater_j0 + 0.5 * dy[j0 - 1]

    # 其他 m：采用对称的半单元平均校正
    else:
        mask = y_matlab_tensor <= j0
        indices_less_j0 = (y_matlab_tensor[mask] - 1).to(torch.long)
        sum_dy_less_j0 = cumsum_dy[j0 - 1] - cumsum_dy[indices_less_j0]

        indices_greater_j0 = (y_matlab_tensor[~mask] - 1).to(torch.long)
        sum_dy_greater_j0 = cumsum_dy[indices_greater_j0] - cumsum_dy[j0 - 1]

        coord_y_tensor[mask] = -(sum_dy_less_j0 - 0.5 * (dy[indices_less_j0] + dy[j0 - 1]) + dy[indices_less_j0])
        coord_y_tensor[~mask] = sum_dy_greater_j0 - 0.5 * dy[indices_greater_j0] + 0.5 * dy[j0 - 1]

    return coord_y_tensor


def calculate_coord_z_torch(z_matlab_tensor, m, k0, dz_tensor, device):
    """Calculate z-coordinate
    计算 z 方向坐标（与 x/y 同理，但分支定义略有不同）
    参数：
        z_matlab_tensor: Matlab 风格 z 索引（1-based）
        m: 类型标识
        k0: z 方向中心索引（1-based）
        dz_tensor: 每个 z 单元宽度
        device: 设备
    返回：
        coord_z_tensor: 相对 k0 的物理坐标（通常 z 向下为正或负，视定义而定）
    """
    # 确保输入为张量（并在指定 device 上）
    if not isinstance(z_matlab_tensor, torch.Tensor):
        z_matlab_tensor = torch.tensor(z_matlab_tensor, dtype=torch.float32, device=device)
    if not isinstance(dz_tensor, torch.Tensor):
        dz_tensor = torch.tensor(dz_tensor, dtype=torch.float32, device=device)

    coord_z_tensor = torch.zeros_like(z_matlab_tensor, dtype=torch.float32, device=device)
    cumsum_dz = torch.cumsum(dz_tensor, dim=0)

    # m 在 [3, 4, 5]：一种 z 向半单元处理方式
    if m in [3, 4, 5]:
        mask = z_matlab_tensor <= k0
        indices_less_k0 = (z_matlab_tensor[mask] - 1).to(torch.long)
        sum_dz_less_k0 = cumsum_dz[k0 - 1] - cumsum_dz[indices_less_k0]

        indices_greater_k0 = (z_matlab_tensor[~mask] - 1).to(torch.long)
        sum_dz_greater_k0 = cumsum_dz[indices_greater_k0] - cumsum_dz[k0 - 1]

        # 左侧（上方）为负，取左侧半单元 + 其余累计
        coord_z_tensor[mask] = -(sum_dz_less_k0 + 0.5 * dz_tensor[indices_less_k0])
        # 右侧（下方）为正，取右侧半单元
        coord_z_tensor[~mask] = sum_dz_greater_k0 - 0.5 * dz_tensor[indices_greater_k0]

    # 其他 m：采用不同的右侧偏移（-2）与整单元处理
    else:
        mask = z_matlab_tensor <= k0
        indices_less_k0 = (z_matlab_tensor[mask] - 1).to(torch.long)
        sum_dz_less_k0 = cumsum_dz[k0 - 1] - cumsum_dz[indices_less_k0]

        # 注意这里 -2（对应不同的网格中心/边界定义）
        indices_greater_k0 = (z_matlab_tensor[~mask] - 2).to(torch.long)
        sum_dz_greater_k0 = cumsum_dz[indices_greater_k0] - cumsum_dz[k0 - 1]

        # 左侧直接加整单元宽度
        coord_z_tensor[mask] = -(sum_dz_less_k0 + dz_tensor[indices_less_k0])
        # 右侧不做半单元扣除
        coord_z_tensor[~mask] = sum_dz_greater_k0

    return coord_z_tensor
