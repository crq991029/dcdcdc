import torch
from sub_initial_field_subloop import sub_initial_field_subloop_torch


def sub_initial_field_torch(L_loop, n_subloop, i0, j0, k0, XI, YJ, ZK, dx, dy, dz, Alpha, model_EC, device='cuda',
                            source_dir=2):
    # ... (前部分代码保持不变，直到生成 subgrid) ...

    # —— 参数张量化 (保持不变) ——
    L_loop = torch.tensor(L_loop, dtype=torch.float32, device=device)
    n_subloop = torch.tensor(n_subloop, dtype=torch.int32, device=device)
    dx = torch.as_tensor(dx, dtype=torch.float32, device=device)
    dy = torch.as_tensor(dy, dtype=torch.float32, device=device)
    dz = torch.as_tensor(dz, dtype=torch.float32, device=device)
    Alpha = torch.tensor(Alpha, dtype=torch.float32, device=device)

    # —— 初始化场 (保持不变) ——
    EX = torch.zeros((XI, YJ + 1, ZK + 1, 2), dtype=torch.float32, device=device)
    EY = torch.zeros((XI + 1, YJ, ZK + 1, 2), dtype=torch.float32, device=device)
    EZ = torch.zeros((XI + 1, YJ + 1, ZK, 2), dtype=torch.float32, device=device)
    HX = torch.zeros((XI + 1, YJ, ZK, 2), dtype=torch.float32, device=device)
    HY = torch.zeros((XI, YJ + 1, ZK, 2), dtype=torch.float32, device=device)
    HZ = torch.zeros((XI, YJ, ZK + 1, 2), dtype=torch.float32, device=device)

    # —— 子回线参数 ——
    L_subloop = L_loop / torch.sqrt(n_subloop.float())
    n_side = int(torch.sqrt(n_subloop.float()).item())

    subgrid = torch.linspace(-(L_loop - L_subloop) / 2, (L_loop - L_subloop) / 2, n_side, device=device)

    # 根据方向生成两个偏移量网格
    offset_1, offset_2 = torch.meshgrid(subgrid, subgrid, indexing='ij')

    for i in range(n_side):
        for j in range(n_side):
            # 传入 source_dir 和 对应的偏移量
            # 注意：sub_initial_field_subloop_torch 参数列表已变，需要更新调用
            EX_sub, EY_sub, EZ_sub, HX_sub, HY_sub, HZ_sub, t1_E, t1_H = sub_initial_field_subloop_torch(
                i0, j0, k0, XI, YJ, ZK, model_EC, dx, dy, dz,
                offset_1[i, j], offset_2[i, j], L_subloop, Alpha, device, source_dir
            )

            EX += EX_sub
            EY += EY_sub
            EZ += EZ_sub
            HX += HX_sub
            HY += HY_sub
            HZ += HZ_sub

    print(f'Initial field calculation (Direction {source_dir}) finished.')
    return EX, EY, EZ, HX, HY, HZ, t1_E, t1_H