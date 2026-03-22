import torch

def cuda_detach_cpu(t: torch.Tensor):
    return t.clone().detach().to('cpu')

@torch.no_grad()
def bdot(t1: torch.Tensor, t2: torch.Tensor):
    """
    t1, t2: shape=(data_num * 3)
    output: shape=(datanum)
    """
    return torch.matmul(t1.unsqueeze(1), t2.unsqueeze(2)).squeeze(2).squeeze(1)

@torch.no_grad()
def bdot3d(t1: torch.Tensor, t2: torch.Tensor):
    """
    t1, t2: shape=(data_num * point_num(or edge_num) * 3)
    output: shape=(datanum * point_num(or edge_num))
    """
    return torch.matmul(t1.unsqueeze(2), t2.unsqueeze(3)).squeeze(3).squeeze(2)

def cross_subtract(t1: torch.Tensor, t2: torch.Tensor):
    """
    t1, t2: shape=(data_num * 3(point_num or edge_num) * 3(xyz))
    output: t1-t2 for all pairs across dim 1, shape=(datanum * 9 * 3(xyz))
    order: t1_0-t2_0, t1_0-t2_1, t1_0-t2_2, t1_1-t2_0, t1_0-t2_1, t1_0-t2_2, t1_2-t2_0, t1_2-t2_1, t1_2-t2_2
    """
    assert torch.__version__ >= '2.3.0+cu121' # Old versions are slow.
    t1_repeat = torch.repeat_interleave(t1, 3, dim=1)
    t2_repeat = t2.repeat(1, 3, 1)
    return t1_repeat - t2_repeat

# id_ij_temp = torch.tensor([[2, 2], [2, 0], [2, 1], [0, 2], [0, 0], [0, 1], [1, 2], [1, 0], [1, 1]], dtype=torch.int64)

def edge_edge_dist_all_torch(p: torch.Tensor, a: torch.Tensor, q: torch.Tensor, b: torch.Tensor):
    """
    https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/geomutils/src/sweep/GuSweepCapsuleCapsule.cpp
    p: origin of edge1, shape=(datanum * 3(points) * 3(xyz))
    a: vector of edge1 from p, shape=(datanum * 3(points) * 3(xyz))
    q: origin of edge2, shape=(datanum * 3(points) * 3(xyz))
    b: vector of edge2 from q, shape=(datanum * 3(points) * 3(xyz))
    x: closest point on (p, a), shape=(datanum * 9(all_pairs) * 3(xyz))
    y: closest point on (q, b), shape=(datanum * 9(all_pairs) * 3(xyz))
    """

    ### [point1, point2, point3] -> [point1, point1, point1, point2, point2, point2, point3, point3, point3], shape=(datanum * 9(all_pairs) * 3(xyz))
    p_repeat = torch.repeat_interleave(p, 3, dim=1) 
    a_repeat = torch.repeat_interleave(a, 3, dim=1)
    ### [point1, point2, point3] -> [point1, point2, point3, point1, point2, point3, point1, point2, point3], shape=(datanum * 9(all_pairs) * 3(xyz))
    q_repeat = q.repeat(1, 3, 1) 
    b_repeat = b.repeat(1, 3, 1)

    T = q_repeat - p_repeat # shape=(datanum * 9(all_pairs) * 3(xyz))
    ADotA = bdot3d(a_repeat, a_repeat) # shape=(datanum * 9(all_pairs))
    BDotB = bdot3d(b_repeat, b_repeat)
    ADotB = bdot3d(a_repeat, b_repeat)
    ADotT = bdot3d(a_repeat, T)
    BDotT = bdot3d(b_repeat, T)

    ### t parameterizes ray (p, a)
    ### u parameterizes ray (q, b)

    ### Compute denominator
    Denom = ADotA * BDotB - ADotB * ADotB
    Denom[Denom == 0.0] = torch.inf

    ### Compute t for the closest point on segment (p, a) to segment (q, b)
    t = torch.clamp((ADotT * BDotB - BDotT * ADotB) / Denom, 0, 1)

    ### Find u for point on ray (q, b) closest to point at t
    if_BDotB_0 = (BDotB == 0.0)
    BDotB[if_BDotB_0] = torch.inf
    u = (t * ADotB - BDotT) / BDotB

    ADotA[ADotA == 0] = torch.inf

    ### Clamp u if it's out of segment bounds, then recompute t if needed
    if_BDotB_not0 = torch.logical_not(if_BDotB_0)
    if_BDotB_not0_u_neg = torch.logical_and(if_BDotB_not0, u < 0)
    if_BDotB_not0_u_above1 = torch.logical_and(if_BDotB_not0, u > 1)

    u[if_BDotB_not0_u_neg] = 0
    t[if_BDotB_not0_u_neg] = torch.clamp(ADotT / ADotA, 0, 1)[if_BDotB_not0_u_neg]
    
    u[if_BDotB_not0_u_above1] = 1
    t[if_BDotB_not0_u_above1] = torch.clamp((ADotB + ADotT) / ADotA, 0, 1)[if_BDotB_not0_u_above1]
    
    t[if_BDotB_0] = torch.clamp(ADotT / ADotA, 0, 1)[if_BDotB_0]

    ### Compute closest points
    cp = p_repeat + a_repeat * t.unsqueeze(2).expand_as(p_repeat)
    cq = q_repeat + b_repeat * u.unsqueeze(2).expand_as(q_repeat)

    return cp, cq

def distance_triangle_triangle_torch(p: torch.Tensor, q: torch.Tensor):
    """
    https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/geomutils/src/distance/GuDistanceTriangleTriangle.cpp

    p: {p[0]: [[x00, y00, z00], [x01, y01, z01], [x02, y02, z02]]}, shape=(datanum * 3(point) * 3(xyz))
    q: {q[0]: [[x10, y10, z10], [x11, y11, z11], [x22, y22, z22]]}, shape=(datanum * 3(point) * 3(xyz))
    Sv: edges of p, shape=(datanum * 3(edge) * 3(xyz))
    Tv: edges of q, shape=(datanum * 3(edge) * 3(xyz))
    cp: closest point in p, shape=(datanum * 3(xyz))
    cq: closest point in q, shape=(datanum * 3(xyz))
    """

    # torch._C._cuda_clearCublasWorkspaces()
    # memory_before = torch.cuda.memory_allocated(p.device)
    # p_new = p.clone()
    # q_new = q.clone()
    # memory_after = torch.cuda.memory_allocated(p.device)
    # latent_size = memory_after - memory_before
    # print(latent_size)

    # print(p.element_size() * p.nelement() + q.element_size() * q.nelement())

    assert p.device == q.device
    assert p.dtype == q.dtype

    if_shown_disjoint = torch.full((p.shape[0],), False, dtype=torch.bool, device=p.device)
    
    cp_out = torch.zeros(p.shape[0], 3, dtype=p.dtype, device=p.device)
    cq_out = torch.zeros(p.shape[0], 3, dtype=p.dtype, device=p.device)

    ### roll: [0, 1, 2] -> [1, 2, 0]
    Sv = torch.roll(p, -1, dims=1) - p # edges of p, shape=(datanum * 3(edge) * 3(xyz))
    Tv = torch.roll(q, -1, dims=1) - q # edges of q, shape=(datanum * 3(edge) * 3(xyz))

    ### ----------------------------------------- Find return 1 -------------------------------------------
    #### dist for all edges (p1)
    cp_all, cq_all = edge_edge_dist_all_torch(p, Sv, q, Tv) # shape=(datanum * 9(all_pairs) * 3(xyz))

    V = cq_all - cp_all # shape=(datanum * 9(all_pairs) * 3(xyz))
    dd = bdot3d(V, V) # shape=(datanum * 9(all_pairs))

    min_dd, min_idx = torch.min(dd, dim=1) # shape=(datanum)
    if_min_idx = torch.zeros((p.shape[0], 9), dtype=torch.int, device=p.device)
    ones_temp = torch.ones((p.shape[0], 9), dtype=torch.int, device=p.device)
    if_min_idx.scatter_(1, min_idx.unsqueeze(1).expand_as(ones_temp), ones_temp)
    if_min_idx = if_min_idx.bool() # shape=(datanum, 9)

    ### roll: [0, 1, 2] -> [2, 0, 1]
    Zp = torch.repeat_interleave(torch.roll(p, 1, dims=1), 3, dim=1) - cp_all # repeat along p in cp_all and cq_all, shape=(datanum * 9(all_pairs) * 3(xyz))
    a = bdot3d(Zp, V) # shape=(datanum * 9(all_pairs))

    ### roll: [0, 1, 2] -> [2, 0, 1]
    Zq = torch.roll(q, 1, dims=1).repeat(1, 3, 1) - cq_all # repeat along q in cp_all and cq_all, shape=(datanum * 9(all_pairs) * 3(xyz))
    b = bdot3d(Zq, V)

    if_a_neg = (a <= 0.0) # shape=(datanum * 9(all_pairs))
    if_a_pos = torch.logical_not(if_a_neg)
    if_b_pos = (b >= 0.0)
    if_b_neg = torch.logical_not(if_b_pos)
    if_b_stpos = (b > 0.0)

    if_a_neg_b_pos = torch.logical_and(if_a_neg, if_b_pos) # shape=(datanum * 9(all_pairs))
    if_a_neg_b_neg = torch.logical_and(if_a_neg, if_b_neg) # shape=(datanum * 9(all_pairs))
    if_a_pos_b_stpos = torch.logical_and(if_a_pos, if_b_stpos) # shape=(datanum * 9(all_pairs))

    if_return1_cp_cq = torch.logical_and(if_a_neg_b_pos, if_min_idx) # shape=(datanum * 9(all_pairs))
    if_return1_cp_cq_out_temp = torch.any(if_return1_cp_cq, 1, keepdim=True) # shape=(datanum, 1)
    if_return1_cp_cq_out = if_return1_cp_cq_out_temp.expand_as(cp_out) # shape=(datanum, 3(xyz))
    if_return1_cp_cq_all = if_return1_cp_cq.unsqueeze(2).expand_as(cp_all) # shape=(datanum * 9(all_pairs), 3)

    cp_out[if_return1_cp_cq_out] = cp_all[if_return1_cp_cq_all] # a <=0 and b >= 0
    cq_out[if_return1_cp_cq_out] = cq_all[if_return1_cp_cq_all] # a <=0 and b >= 0

    if_not_return1_cp_cq_out = torch.logical_not(if_return1_cp_cq_out_temp)
    if_not_return1_cp_cq_out_all = if_not_return1_cp_cq_out.expand_as(if_return1_cp_cq) # shape=(datanum * 9(all_pairs))

    a[if_a_neg_b_neg] = 0.0 # a <= 0 and b < 0
    b[if_a_pos_b_stpos] = 0.0 # a > 0 and b > 0

    if_disjoint_a_b = (min_dd.unsqueeze(1) - a + b > 0.0) # shape=(datanum * 9(all_pairs))
    if_disjoint_a_b_out = torch.any(torch.logical_and(torch.logical_and(if_min_idx, if_not_return1_cp_cq_out_all), if_disjoint_a_b), dim=1)
    if_shown_disjoint[if_disjoint_a_b_out] = True
    
    if_not_out = if_not_return1_cp_cq_out.squeeze(1) # not out, shape=(datanum)
    ### ----------------------------------------- End of return 1 -----------------------------------------

    ### ----------------------------------------- Find return 2 -------------------------------------------
    ### Face normal of p 
    Sn = torch.cross(Sv[:, 0], Sv[:, 1], dim=1) # shape=(datanum * 3(xyz))
    Snl = bdot(Sn, Sn) # shape=(datanum)
    if_Snl_pos = (Snl > 1e-15)
    if_p_normal_in = torch.logical_and(if_not_out, if_Snl_pos) # shape=(datanum)

    Tp = bdot3d(p[:, [0]] - q, Sn.unsqueeze(1).expand(-1, 3, -1)) # shape=(datanum * 3(edgenum))
    if_Tp_pos_all = torch.all(Tp > 0, dim=1) # shape=(datanum * 3(edgenum))
    if_Tp_neg_all = torch.all(Tp < 0, dim=1) # shape=(datanum * 3(edgenum))
    if_Tp_pos_all_or_neg_all = torch.logical_or(if_Tp_pos_all, if_Tp_neg_all)

    Tp_min, Tp_argmin_idx = torch.min(Tp, dim=1) # shape=(datanum)
    Tp_max, Tp_argmax_idx = torch.max(Tp, dim=1) # shape=(datanum)

    if_Tp_idx_exist = torch.logical_and(if_p_normal_in, if_Tp_pos_all_or_neg_all) # shape=(datanum)
    if_shown_disjoint[if_Tp_idx_exist] = True

    q_from_Tp_argmin_idx = torch.take_along_dim(q, Tp_argmin_idx.reshape(-1, 1, 1), 1) # shape=(datanum, 1, 3(xyz))
    q_from_Tp_argmax_idx = torch.take_along_dim(q, Tp_argmax_idx.reshape(-1, 1, 1), 1)
    q_from_Tp_all = torch.cat([q_from_Tp_argmin_idx, q_from_Tp_argmax_idx], dim=1) # shape=(datanum, 2, 3(xyz))

    ### V_p: [q_min - p_0, q_min - p_1, q_min - p_2, q_max - p_0, q_max - p_1, q_max - p_2]
    V_p = torch.repeat_interleave(q_from_Tp_all, 3, dim=1) - p.repeat(1, 2, 1) # shape=(datanum, 6, 3(xyz))
    ### Z_p: [edge0, edge1, edge2, edge0, edge1, edge2]
    Z_p = torch.cross(Sn.unsqueeze(1), Sv, dim=2).repeat(1, 2, 1) # shape=(datanum, 6, 3(xyz))

    if_V_p_Z_p_pos = (bdot3d(V_p, Z_p) > 0) # shape=(datanum, 6)
    if_V_p_Z_p_pos_argmin = torch.all(if_V_p_Z_p_pos[:, :3], dim=1) # shape=(datanum)
    if_V_p_Z_p_pos_argmax = torch.all(if_V_p_Z_p_pos[:, 3:], dim=1) # shape=(datanum)
    
    if_return2_argmin = torch.logical_and(if_Tp_pos_all, torch.logical_and(if_Tp_idx_exist, if_V_p_Z_p_pos_argmin)) # shape=(datanum)
    if_return2_argmax = torch.logical_and(if_Tp_neg_all, torch.logical_and(if_Tp_idx_exist, if_V_p_Z_p_pos_argmax)) # shape=(datanum)

    if torch.any(if_return2_argmin):
        cp_out[if_return2_argmin] = q_from_Tp_all[if_return2_argmin, 0, :] + Sn[if_return2_argmin, :] * Tp_min[if_return2_argmin].unsqueeze(1) / Snl[if_return2_argmin].unsqueeze(1)
        cq_out[if_return2_argmin] = q_from_Tp_all[if_return2_argmin, 0, :]
    if torch.any(if_return2_argmax):
        cp_out[if_return2_argmax] = q_from_Tp_all[if_return2_argmax, 1, :] + Sn[if_return2_argmax, :] * Tp_max[if_return2_argmax].unsqueeze(1) / Snl[if_return2_argmax].unsqueeze(1)
        cq_out[if_return2_argmax] = q_from_Tp_all[if_return2_argmax, 1, :]

    if_not_out = torch.logical_and(if_not_out, torch.logical_not(torch.logical_or(if_return2_argmin, if_return2_argmax)))
    ### ----------------------------------------- End of return 2 -----------------------------------------

    ### ----------------------------------------- Find return 3 -------------------------------------------
    ### Face normal of q
    Tn = torch.cross(Tv[:, 0], Tv[:, 1], dim=1) # shape=(datanum * 3(xyz))
    Tnl = bdot(Tn, Tn) # shape=(datanum)
    if_Tnl_pos = (Tnl > 1e-15)
    if_q_normal_in = torch.logical_and(if_not_out, if_Tnl_pos) # shape=(datanum)

    Sp = bdot3d(q[:, [0]] - p, Tn.unsqueeze(1).expand(-1, 3, -1)) # shape=(datanum * 3(edgenum))
    if_Sp_pos_all = torch.all(Sp > 0, dim=1) # shape=(datanum * 3(edgenum))
    if_Sp_neg_all = torch.all(Sp < 0, dim=1) # shape=(datanum * 3(edgenum))
    if_Sp_pos_all_or_neg_all = torch.logical_or(if_Sp_pos_all, if_Sp_neg_all)

    Sp_min, Sp_argmin_idx = torch.min(Sp, dim=1) # shape=(datanum)
    Sp_max, Sp_argmax_idx = torch.max(Sp, dim=1) # shape=(datanum)

    if_Sp_idx_exist = torch.logical_and(if_q_normal_in, if_Sp_pos_all_or_neg_all) # shape=(datanum)
    if_shown_disjoint[if_Sp_idx_exist] = True

    p_from_Sp_argmin_idx = torch.take_along_dim(p, Sp_argmin_idx.reshape(-1, 1, 1), 1) # shape=(datanum, 1, 3(xyz))
    p_from_Sp_argmax_idx = torch.take_along_dim(p, Sp_argmax_idx.reshape(-1, 1, 1), 1)
    p_from_Sp_all = torch.cat([p_from_Sp_argmin_idx, p_from_Sp_argmax_idx], dim=1) # shape=(datanum, 2, 3(xyz))

    ### V_q: [p_min - q_0, p_min - q_1, p_min - q_2, p_max - q_0, p_max - q_1, p_max - q_2]
    V_q = torch.repeat_interleave(p_from_Sp_all, 3, dim=1) - q.repeat(1, 2, 1) # shape=(datanum, 6, 3(xyz))
    ### Z_q: [edge0, edge1, edge2, edge0, edge1, edge2]
    Z_q = torch.cross(Tn.unsqueeze(1), Tv, dim=2).repeat(1, 2, 1) # shape=(datanum, 6, 3(xyz))

    if_V_q_Z_q_pos = (bdot3d(V_q, Z_q) > 0) # shape=(datanum, 6)
    if_V_q_Z_q_pos_argmin = torch.all(if_V_q_Z_q_pos[:, :3], dim=1) # shape=(datanum)
    if_V_q_Z_q_pos_argmax = torch.all(if_V_q_Z_q_pos[:, 3:], dim=1) # shape=(datanum)
    
    if_return3_argmin = torch.logical_and(if_Sp_pos_all, torch.logical_and(if_Sp_idx_exist, if_V_q_Z_q_pos_argmin)) # shape=(datanum)
    if_return3_argmax = torch.logical_and(if_Sp_neg_all, torch.logical_and(if_Sp_idx_exist, if_V_q_Z_q_pos_argmax)) # shape=(datanum)

    if torch.any(if_return3_argmin):
        cp_out[if_return3_argmin] = p_from_Sp_all[if_return3_argmin, 0, :]
        cq_out[if_return3_argmin] = p_from_Sp_all[if_return3_argmin, 0, :] + Tn[if_return3_argmin, :] * Sp_min[if_return3_argmin].unsqueeze(1) / Tnl[if_return3_argmin].unsqueeze(1)
    if torch.any(if_return3_argmax):
        cp_out[if_return3_argmax] = p_from_Sp_all[if_return3_argmax, 1, :]
        cq_out[if_return3_argmax] = p_from_Sp_all[if_return3_argmax, 1, :] + Tn[if_return3_argmax, :] * Sp_max[if_return3_argmax].unsqueeze(1) / Tnl[if_return3_argmax].unsqueeze(1)

    if_not_out = torch.logical_and(if_not_out, torch.logical_not(torch.logical_or(if_return3_argmin, if_return3_argmax)))
    ### ----------------------------------------- End of return 3 -----------------------------------------

    ### ----------------------------------------- Find return 4 -------------------------------------------
    ### Final output (non-col)
    if_disjoint_final = torch.logical_and(if_not_out, if_shown_disjoint) # shape=(batch_num)
    if torch.any(if_disjoint_final):
        if_disjoint_final_out = if_disjoint_final.unsqueeze(1).expand(p.shape[0], 3) # shape=(batch_num, 3)
        if_disjoint_final_out2 = if_disjoint_final.unsqueeze(1).expand(p.shape[0], 9) # shape=(batch_num, 9)
        cp_out[if_disjoint_final_out] = cp_all[torch.logical_and(if_min_idx, if_disjoint_final_out2)].flatten()
        cq_out[if_disjoint_final_out] = cq_all[torch.logical_and(if_min_idx, if_disjoint_final_out2)].flatten()
    ### ----------------------------------------- End of return 4 -----------------------------------------

    ### ----------------------------------------- Find return 5 -------------------------------------------
    ### In-collision: cp_out = [0, 0, 0], cq_out = [0, 0, 0]
    ### ----------------------------------------- End of return 5 -----------------------------------------
    return cp_out, cq_out


if __name__ == "__main__":
    ### Setup
    start_gpu = torch.cuda.Event(enable_timing=True)
    end_gpu = torch.cuda.Event(enable_timing=True)
    datanum = 10000

    ### Warm-up
    distance_triangle_triangle_torch(torch.rand(datanum, 3, 3, device='cuda'), torch.rand(datanum, 3, 3, device='cuda'))

    ### Data generation
    verts1_cpu, verts2_cpu = torch.rand(datanum, 3, 3), torch.rand(datanum, 3, 3)
    verts1_cuda, verts2_cuda = verts1_cpu.to('cuda'), verts2_cpu.to('cuda')

    ### GPU-version test
    start_gpu.record()
    cp_torch, cq_torch = distance_triangle_triangle_torch(verts1_cuda, verts2_cuda)
    dists_gpu = (((cp_torch-cq_torch)**2).sum(dim=1))**0.5
    end_gpu.record()
    torch.cuda.synchronize()
    etime = start_gpu.elapsed_time(end_gpu)*1e-3
    print("GPU elapsed time:", etime)
    
    ### CPU-version test
    from triangle_dist_physx import distance_triangle_triangle
    import time
    start_cpu = time.time()
    dists_cpu = []
    for i in range(datanum):
        cp, cq = distance_triangle_triangle(verts1_cpu[i].numpy(), verts2_cpu[i].numpy())
        dist = ((cp[0]-cq[0])**2 + (cp[1]-cq[1])**2 + (cp[2]-cq[2])**2)**0.5
        dists_cpu.append(dist)
    etime_cpu = time.time() - start_cpu
    print("CPU elapsed time:", etime_cpu)
    print("max abs error:", torch.abs(cuda_detach_cpu(dists_gpu) - torch.tensor(dists_cpu)).max().item())
    print("mean avg error:", torch.abs(cuda_detach_cpu(dists_gpu) - torch.tensor(dists_cpu)).sum().item() / datanum)


