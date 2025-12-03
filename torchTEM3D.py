from sub_initial_field import *
from sub_iteration import *

#%%
# calculate dbz/dt
def forward_3DTEM(L_loop, n_subloop, i0, j0, k0,
                  XI, YJ, ZK, dx, dy, dz, Alpha,
                  model_EC, Rx_st, Rx_end, iter_n_max, device='cuda', source_dir=2):
    # 调用初始场时传入 source_dir
    EX, EY, EZ, HX, HY, HZ, t1_E, t1_H = sub_initial_field_torch(
        L_loop, n_subloop, i0, j0, k0, XI, YJ, ZK, dx, dy, dz,
        Alpha, model_EC.detach(), device=device, source_dir=source_dir
    )
    
    # Iterative computation
    t_H, dBz = sub_iteration_torch(i0, j0, k0, XI, YJ, ZK, dx, dy, dz, Rx_st, Rx_end,
                             iter_n_max, Alpha, model_EC, EX, EY, EZ, HX, HY, HZ, t1_E, t1_H,device = device)
    
    return t_H, dBz