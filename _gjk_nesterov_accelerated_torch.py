from dataclasses import dataclass

import numpy as np
import torch

from distance3d.colliders import MeshGraph, Capsule, Sphere, Box, Cone, Cylinder, Ellipsoid

def _to_numpy(array):
    return array.detach().cpu().numpy()


@dataclass
class TorchMesh:
    mesh2origin: object
    vertices: object
    triangles: object

    def __post_init__(self):
        tensors = {
            "mesh2origin": self.mesh2origin,
            "vertices": self.vertices,
            "triangles": self.triangles,
        }
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")

        if self.mesh2origin.ndim not in (2, 3):
            raise ValueError("mesh2origin must have shape (4, 4) or (batch_size, 4, 4).")
        if self.vertices.ndim not in (2, 3):
            raise ValueError("vertices must have shape (n_vertices, 3) or (batch_size, n_vertices, 3).")
        if self.triangles.ndim not in (2, 3):
            raise ValueError("triangles must have shape (n_faces, 3) or (batch_size, n_faces, 3).")

        if self.mesh2origin.ndim == 2:
            if self.mesh2origin.shape != (4, 4):
                raise ValueError("mesh2origin must have shape (4, 4).")
            if self.vertices.shape[-1] != 3 or self.vertices.ndim != 2:
                raise ValueError("vertices must have shape (n_vertices, 3).")
            if self.triangles.shape[-1] != 3 or self.triangles.ndim != 2:
                raise ValueError("triangles must have shape (n_faces, 3).")
        else:
            if self.mesh2origin.shape[1:] != (4, 4):
                raise ValueError("mesh2origin must have shape (batch_size, 4, 4).")
            if self.vertices.ndim != 3 or self.vertices.shape[-1] != 3:
                raise ValueError("vertices must have shape (batch_size, n_vertices, 3).")
            if self.triangles.ndim != 3 or self.triangles.shape[-1] != 3:
                raise ValueError("triangles must have shape (batch_size, n_faces, 3).")
            batch_size = self.mesh2origin.shape[0]
            if self.vertices.shape[0] != batch_size or self.triangles.shape[0] != batch_size:
                raise ValueError("All batched tensors must share the same batch dimension.")

        if self.mesh2origin.device != self.vertices.device:
            raise ValueError("mesh2origin and vertices must be on the same device.")
        if self.vertices.device != self.triangles.device:
            raise ValueError("vertices and triangles must be on the same device.")
        if self.mesh2origin.dtype != self.vertices.dtype:
            raise ValueError("mesh2origin and vertices must use the same dtype.")
        if not torch.is_floating_point(self.mesh2origin):
            raise ValueError("mesh2origin must be floating point.")
        if not torch.is_floating_point(self.vertices):
            raise ValueError("vertices must be floating point.")

    @property
    def dtype(self):
        return self.vertices.dtype

    @property
    def device(self):
        return self.vertices.device

    @property
    def is_batched(self):
        return self.mesh2origin.ndim == 3

    @property
    def batch_size(self):
        if self.is_batched:
            return self.mesh2origin.shape[0]
        return 1

    def center(self):
        rotation = self.mesh2origin[..., :3, :3]
        translation = self.mesh2origin[..., :3, 3]
        local_center = self.vertices.mean(dim=-2)
        return translation + (rotation @ local_center.unsqueeze(-1)).squeeze(-1)

    def batch_item(self, index):
        if self.is_batched:
            return TorchMesh(
                self.mesh2origin[index],
                self.vertices[index],
                self.triangles[index],
            )
        if index != 0:
            raise IndexError("Unbatched TorchMesh only contains one item.")
        return self

    def to_numpy_collider(self):
        if self.is_batched:
            raise ValueError(
                "Batched TorchMesh cannot be converted with to_numpy_collider(); "
                "use batch_item(index) or to_numpy_colliders() instead."
            )
        return MeshGraph(
            _to_numpy(self.mesh2origin),
            _to_numpy(self.vertices),
            _to_numpy(self.triangles).astype(np.int64, copy=False),
        )

    def to_numpy_colliders(self):
        return [self.batch_item(i).to_numpy_collider() for i in range(self.batch_size)]


def _ensure_mesh_only(collider0, collider1):
    for collider in (collider0, collider1):
        if not isinstance(collider, TorchMesh):
            raise NotImplementedError("Only TorchMesh inputs are supported for now.")
    return collider0, collider1


def _reference_tensor(collider0, collider1):
    for collider in (collider0, collider1):
        if isinstance(collider, TorchMesh):
            return collider.vertices
    return None


def _to_torch_array(array, reference):
    return torch.as_tensor(array, dtype=reference.dtype, device=reference.device)


def _to_torch_scalar(value, reference):
    return torch.tensor(value, dtype=reference.dtype, device=reference.device)


def _mesh_world_vertices(mesh):
    rotation = mesh.mesh2origin[..., :3, :3]
    translation = mesh.mesh2origin[..., :3, 3]
    return torch.matmul(mesh.vertices, rotation.transpose(-1, -2)) + translation.unsqueeze(-2)


def _torch_norm_vector(v):
    norms = torch.linalg.norm(v, dim=-1, keepdim=True)
    safe_norms = torch.where(norms == 0.0, torch.ones_like(norms), norms)
    normalized = v / safe_norms
    return torch.where(norms == 0.0, v, normalized)


def _batched_mesh_support_points(world_vertices, search_directions):
    scores = torch.sum(world_vertices * search_directions.unsqueeze(1), dim=-1)
    support_indices = torch.argmax(scores, dim=1)
    batch_indices = torch.arange(world_vertices.shape[0], device=world_vertices.device)
    return world_vertices[batch_indices, support_indices]


def _project_point_simplex_batch(state, point_indices):
    if point_indices.numel() == 0:
        return
    state["ray"][point_indices] = state["support_point"][point_indices].clone()


def _project_line_simplex_batch(state, line_indices):
    if line_indices.numel() == 0:
        return

    simplex = state["simplex"][line_indices].clone()
    support_points0 = state["support_points0"][line_indices].clone()
    support_points1 = state["support_points1"][line_indices].clone()

    a = simplex[:, 1]
    b = simplex[:, 0]
    ab = b - a
    d = torch.sum(ab * (-a), dim=1)
    point_mask = d <= 0.0

    if bool(point_mask.any().item()):
        point_indices_local = line_indices[point_mask]
        a_point = a[point_mask]
        state["ray"][point_indices_local] = a_point
        state["simplex"][point_indices_local, 0] = a_point
        state["support_points0"][point_indices_local, 0] = support_points0[point_mask, 1]
        state["support_points1"][point_indices_local, 0] = support_points1[point_mask, 1]
        state["simplex_len"][point_indices_local] = 1
        state["inside"][point_indices_local] = (
            (d[point_mask] == 0.0) & torch.all(a_point == 0.0, dim=1)
        )

    segment_mask = ~point_mask
    if bool(segment_mask.any().item()):
        segment_indices = line_indices[segment_mask]
        a_segment = a[segment_mask]
        b_segment = b[segment_mask]
        ab_segment = ab[segment_mask]
        d_segment = d[segment_mask]
        denom = torch.sum(ab_segment * ab_segment, dim=1).unsqueeze(-1)
        ray = (
            torch.sum(ab_segment * b_segment, dim=1).unsqueeze(-1) * a_segment
            + d_segment.unsqueeze(-1) * b_segment
        ) / denom
        state["ray"][segment_indices] = ray
        state["simplex"][segment_indices, 0] = b_segment
        state["simplex"][segment_indices, 1] = a_segment
        state["support_points0"][segment_indices, 0] = support_points0[segment_mask, 0]
        state["support_points0"][segment_indices, 1] = support_points0[segment_mask, 1]
        state["support_points1"][segment_indices, 0] = support_points1[segment_mask, 0]
        state["support_points1"][segment_indices, 1] = support_points1[segment_mask, 1]
        state["simplex_len"][segment_indices] = 2
        state["inside"][segment_indices] = False


def _project_triangle_simplex_batch(state, triangle_indices):
    if triangle_indices.numel() == 0:
        return

    simplex = state["simplex"][triangle_indices].clone()
    support_points0 = state["support_points0"][triangle_indices].clone()
    support_points1 = state["support_points1"][triangle_indices].clone()

    a = simplex[:, 2]
    b = simplex[:, 1]
    c = simplex[:, 0]
    ab = b - a
    ac = c - a
    abc = torch.cross(ab, ac, dim=1)

    edge_ac2o = torch.sum(torch.cross(abc, ac, dim=1) * (-a), dim=1)
    towards_b = torch.sum(ab * (-a), dim=1)
    towards_c = torch.sum(ac * (-a), dim=1)
    edge_ab2o = torch.sum(torch.cross(ab, abc, dim=1) * (-a), dim=1)
    abc_dot_a0 = torch.sum(abc * (-a), dim=1)

    edge_ac_mask = edge_ac2o >= 0.0
    ac_segment_mask = edge_ac_mask & (towards_c >= 0.0)
    ab_region_mask = edge_ac_mask & (~(towards_c >= 0.0))
    ab_edge_mask = (~edge_ac_mask) & (edge_ab2o >= 0.0)
    point_mask = (ab_region_mask | ab_edge_mask) & (towards_b < 0.0)
    ab_segment_mask = (ab_region_mask | ab_edge_mask) & (~(towards_b < 0.0))
    triangle_mask = (~edge_ac_mask) & (~(edge_ab2o >= 0.0))

    if bool(point_mask.any().item()):
        point_indices_local = triangle_indices[point_mask]
        a_point = a[point_mask]
        state["ray"][point_indices_local] = a_point
        state["simplex"][point_indices_local, 0] = a_point
        state["support_points0"][point_indices_local, 0] = support_points0[point_mask, 2]
        state["support_points1"][point_indices_local, 0] = support_points1[point_mask, 2]
        state["simplex_len"][point_indices_local] = 1
        state["inside"][point_indices_local] = False

    if bool(ac_segment_mask.any().item()):
        segment_indices = triangle_indices[ac_segment_mask]
        a_segment = a[ac_segment_mask]
        c_segment = c[ac_segment_mask]
        ac_segment = ac[ac_segment_mask]
        towards_c_segment = towards_c[ac_segment_mask]
        denom = torch.sum(ac_segment * ac_segment, dim=1).unsqueeze(-1)
        ray = (
            torch.sum(ac_segment * c_segment, dim=1).unsqueeze(-1) * a_segment
            + towards_c_segment.unsqueeze(-1) * c_segment
        ) / denom
        state["ray"][segment_indices] = ray
        state["simplex"][segment_indices, 0] = c_segment
        state["simplex"][segment_indices, 1] = a_segment
        state["support_points0"][segment_indices, 0] = support_points0[ac_segment_mask, 0]
        state["support_points0"][segment_indices, 1] = support_points0[ac_segment_mask, 2]
        state["support_points1"][segment_indices, 0] = support_points1[ac_segment_mask, 0]
        state["support_points1"][segment_indices, 1] = support_points1[ac_segment_mask, 2]
        state["simplex_len"][segment_indices] = 2
        state["inside"][segment_indices] = False

    if bool(ab_segment_mask.any().item()):
        segment_indices = triangle_indices[ab_segment_mask]
        a_segment = a[ab_segment_mask]
        b_segment = b[ab_segment_mask]
        ab_segment = ab[ab_segment_mask]
        towards_b_segment = towards_b[ab_segment_mask]
        denom = torch.sum(ab_segment * ab_segment, dim=1).unsqueeze(-1)
        ray = (
            torch.sum(ab_segment * b_segment, dim=1).unsqueeze(-1) * a_segment
            + towards_b_segment.unsqueeze(-1) * b_segment
        ) / denom
        state["ray"][segment_indices] = ray
        state["simplex"][segment_indices, 0] = b_segment
        state["simplex"][segment_indices, 1] = a_segment
        state["support_points0"][segment_indices, 0] = support_points0[ab_segment_mask, 1]
        state["support_points0"][segment_indices, 1] = support_points0[ab_segment_mask, 2]
        state["support_points1"][segment_indices, 0] = support_points1[ab_segment_mask, 1]
        state["support_points1"][segment_indices, 1] = support_points1[ab_segment_mask, 2]
        state["simplex_len"][segment_indices] = 2
        state["inside"][segment_indices] = False

    if bool(triangle_mask.any().item()):
        tri_indices = triangle_indices[triangle_mask]
        abc_tri = abc[triangle_mask]
        abc_dot_a0_tri = abc_dot_a0[triangle_mask]
        inside_mask = abc_dot_a0_tri == 0.0
        cba_mask = abc_dot_a0_tri >= 0.0
        bca_mask = abc_dot_a0_tri < 0.0

        ray = (-abc_dot_a0_tri.unsqueeze(-1) / torch.sum(abc_tri * abc_tri, dim=1, keepdim=True)) * abc_tri
        if bool(inside_mask.any().item()):
            ray[inside_mask] = 0.0
        state["ray"][tri_indices] = ray
        state["simplex_len"][tri_indices] = 3
        state["inside"][tri_indices] = inside_mask

        if bool(cba_mask.any().item()):
            cba_indices = tri_indices[cba_mask]
            state["simplex"][cba_indices, 0] = c[triangle_mask][cba_mask]
            state["simplex"][cba_indices, 1] = b[triangle_mask][cba_mask]
            state["simplex"][cba_indices, 2] = a[triangle_mask][cba_mask]
            state["support_points0"][cba_indices, 0] = support_points0[triangle_mask][cba_mask, 0]
            state["support_points0"][cba_indices, 1] = support_points0[triangle_mask][cba_mask, 1]
            state["support_points0"][cba_indices, 2] = support_points0[triangle_mask][cba_mask, 2]
            state["support_points1"][cba_indices, 0] = support_points1[triangle_mask][cba_mask, 0]
            state["support_points1"][cba_indices, 1] = support_points1[triangle_mask][cba_mask, 1]
            state["support_points1"][cba_indices, 2] = support_points1[triangle_mask][cba_mask, 2]

        if bool(bca_mask.any().item()):
            bca_indices = tri_indices[bca_mask]
            state["simplex"][bca_indices, 0] = b[triangle_mask][bca_mask]
            state["simplex"][bca_indices, 1] = c[triangle_mask][bca_mask]
            state["simplex"][bca_indices, 2] = a[triangle_mask][bca_mask]
            state["support_points0"][bca_indices, 0] = support_points0[triangle_mask][bca_mask, 1]
            state["support_points0"][bca_indices, 1] = support_points0[triangle_mask][bca_mask, 0]
            state["support_points0"][bca_indices, 2] = support_points0[triangle_mask][bca_mask, 2]
            state["support_points1"][bca_indices, 0] = support_points1[triangle_mask][bca_mask, 1]
            state["support_points1"][bca_indices, 1] = support_points1[triangle_mask][bca_mask, 0]
            state["support_points1"][bca_indices, 2] = support_points1[triangle_mask][bca_mask, 2]


def _set_simplex_subset(state, indices, simplex_subset, support0_subset, support1_subset):
    if indices.numel() == 0:
        return

    simplex_count = simplex_subset.shape[1]
    state["simplex"][indices] = 0.0
    state["support_points0"][indices] = 0.0
    state["support_points1"][indices] = 0.0
    state["simplex"][indices, :simplex_count] = simplex_subset
    state["support_points0"][indices, :simplex_count] = support0_subset
    state["support_points1"][indices, :simplex_count] = support1_subset
    state["simplex_len"][indices] = simplex_count
    state["inside"][indices] = False


def _project_tetra_simplex_batch(state, tetra_indices):
    if tetra_indices.numel() == 0:
        return

    simplex = state["simplex"][tetra_indices].clone()
    support_points0 = state["support_points0"][tetra_indices].clone()
    support_points1 = state["support_points1"][tetra_indices].clone()

    a = simplex[:, 3]
    b = simplex[:, 2]
    c = simplex[:, 1]
    d = simplex[:, 0]

    aa = torch.sum(a * a, dim=1)

    da = torch.sum(d * a, dim=1)
    db = torch.sum(d * b, dim=1)
    dc = torch.sum(d * c, dim=1)
    dd = torch.sum(d * d, dim=1)
    da_aa = da - aa

    ca = torch.sum(c * a, dim=1)
    cb = torch.sum(c * b, dim=1)
    cc = torch.sum(c * c, dim=1)
    ca_aa = ca - aa

    ba = torch.sum(b * a, dim=1)
    bb = torch.sum(b * b, dim=1)
    bc = cb
    bd = db
    ba_aa = ba - aa
    ba_ca = ba - ca
    ca_da = ca - da
    da_ba = da - ba

    a_cross_b = torch.cross(a, b, dim=1)
    a_cross_c = torch.cross(a, c, dim=1)
    neg_d_dot_a_cross_b = -torch.sum(d * a_cross_b, dim=1)
    c_dot_a_cross_b = torch.sum(c * a_cross_b, dim=1)
    d_dot_a_cross_c = torch.sum(d * a_cross_c, dim=1)

    expr1 = ba * da_ba + bd * ba_aa - bb * da_aa <= 0.0
    expr2 = ba * ba_ca + bb * ca_aa - bc * ba_aa <= 0.0
    expr3 = ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0.0
    expr4 = ca * ca_da + cc * da_aa - dc * ca_aa <= 0.0
    expr5 = da * da_ba + dd * ba_aa - db * da_aa <= 0.0
    expr6 = da * ca_da + dc * da_aa - dd * ca_aa <= 0.0

    region_a = torch.zeros_like(ba_aa, dtype=torch.bool)
    region_ab = torch.zeros_like(ba_aa, dtype=torch.bool)
    region_ac = torch.zeros_like(ba_aa, dtype=torch.bool)
    region_ad = torch.zeros_like(ba_aa, dtype=torch.bool)
    region_abc = torch.zeros_like(ba_aa, dtype=torch.bool)
    region_acd = torch.zeros_like(ba_aa, dtype=torch.bool)
    region_adb = torch.zeros_like(ba_aa, dtype=torch.bool)
    inside_mask = torch.zeros_like(ba_aa, dtype=torch.bool)

    ba_case = ba_aa <= 0.0
    not_ba_case = ~ba_case
    neg_dab_case = neg_d_dot_a_cross_b <= 0.0
    c_ab_case = c_dot_a_cross_b <= 0.0
    d_ac_case = d_dot_a_cross_c <= 0.0
    da_case = da_aa <= 0.0
    ca_case = ca_aa <= 0.0
    c_ab_truth_case = c_dot_a_cross_b != 0.0

    left_branch = ba_case & neg_dab_case
    left_expr_branch = left_branch & expr1
    left_expr_da_branch = left_expr_branch & da_case
    region_abc |= left_expr_da_branch & expr2
    region_ab |= left_expr_da_branch & (~expr2)

    left_expr_not_da_branch = left_expr_branch & (~da_case)
    left_expr_not_da_expr2 = left_expr_not_da_branch & expr2
    left_expr_not_da_expr2_expr3 = left_expr_not_da_expr2 & expr3
    region_acd |= left_expr_not_da_expr2_expr3 & expr4
    region_ac |= left_expr_not_da_expr2_expr3 & (~expr4)
    region_abc |= left_expr_not_da_expr2 & (~expr3)
    region_ab |= left_expr_not_da_branch & (~expr2)

    left_not_expr_branch = left_branch & (~expr1)
    region_adb |= left_not_expr_branch & expr5
    left_not_expr_else = left_not_expr_branch & (~expr5)
    left_not_expr_else_expr4 = left_not_expr_else & expr4
    region_ad |= left_not_expr_else_expr4 & expr6
    region_acd |= left_not_expr_else_expr4 & (~expr6)
    left_not_expr_else_not_expr4 = left_not_expr_else & (~expr4)
    region_ad |= left_not_expr_else_not_expr4 & expr6
    region_ac |= left_not_expr_else_not_expr4 & (~expr6)

    left_else_branch = ba_case & (~neg_dab_case)
    left_else_cab = left_else_branch & c_ab_case
    left_else_cab_expr2 = left_else_cab & expr2
    left_else_cab_expr2_expr3 = left_else_cab_expr2 & expr3
    region_acd |= left_else_cab_expr2_expr3 & expr4
    region_ac |= left_else_cab_expr2_expr3 & (~expr4)
    region_abc |= left_else_cab_expr2 & (~expr3)
    region_ad |= left_else_cab & (~expr2)

    left_else_not_cab = left_else_branch & (~c_ab_case)
    left_else_not_cab_dac = left_else_not_cab & d_ac_case
    left_else_not_cab_dac_expr4 = left_else_not_cab_dac & expr4
    region_ad |= left_else_not_cab_dac_expr4 & expr6
    region_acd |= left_else_not_cab_dac_expr4 & (~expr6)
    left_else_not_cab_dac_not_expr4 = left_else_not_cab_dac & (~expr4)
    region_ac |= left_else_not_cab_dac_not_expr4 & ca_case
    region_ad |= left_else_not_cab_dac_not_expr4 & (~ca_case)
    inside_mask |= left_else_not_cab & (~d_ac_case)

    right_left_branch = not_ba_case & ca_case
    right_left_dac = right_left_branch & d_ac_case
    right_left_dac_da = right_left_dac & da_case
    right_left_dac_da_expr4 = right_left_dac_da & expr4
    right_left_dac_da_expr4_expr6 = right_left_dac_da_expr4 & expr6
    region_adb |= right_left_dac_da_expr4_expr6 & expr5
    region_ad |= right_left_dac_da_expr4_expr6 & (~expr5)
    region_acd |= right_left_dac_da_expr4 & (~expr6)
    right_left_dac_da_not_expr4 = right_left_dac_da & (~expr4)
    region_ac |= right_left_dac_da_not_expr4 & expr3
    region_abc |= right_left_dac_da_not_expr4 & (~expr3)

    right_left_dac_not_da = right_left_dac & (~da_case)
    right_left_dac_not_da_expr3 = right_left_dac_not_da & expr3
    region_acd |= right_left_dac_not_da_expr3 & expr4
    region_ac |= right_left_dac_not_da_expr3 & (~expr4)
    right_left_dac_not_da_not_expr3 = right_left_dac_not_da & (~expr3)
    region_abc |= right_left_dac_not_da_not_expr3 & c_ab_truth_case
    region_acd |= right_left_dac_not_da_not_expr3 & (~c_ab_truth_case)

    right_left_not_dac = right_left_branch & (~d_ac_case)
    right_left_not_dac_cab = right_left_not_dac & c_ab_case
    region_ac |= right_left_not_dac_cab & expr3
    region_abc |= right_left_not_dac_cab & (~expr3)
    right_left_not_dac_not_cab = right_left_not_dac & (~c_ab_case)
    right_left_not_dac_not_cab_negdab = right_left_not_dac_not_cab & neg_dab_case
    region_adb |= right_left_not_dac_not_cab_negdab & expr5
    region_ad |= right_left_not_dac_not_cab_negdab & (~expr5)
    inside_mask |= right_left_not_dac_not_cab & (~neg_dab_case)

    right_right_branch = not_ba_case & (~ca_case)
    right_right_da = right_right_branch & da_case
    right_right_da_negdab = right_right_da & neg_dab_case
    right_right_da_negdab_expr6 = right_right_da_negdab & expr6
    region_adb |= right_right_da_negdab_expr6 & expr5
    region_ad |= right_right_da_negdab_expr6 & (~expr5)
    right_right_da_negdab_not_expr6 = right_right_da_negdab & (~expr6)
    region_acd |= right_right_da_negdab_not_expr6 & d_ac_case
    region_adb |= right_right_da_negdab_not_expr6 & (~d_ac_case)

    right_right_da_not_negdab = right_right_da & (~neg_dab_case)
    right_right_da_not_negdab_dac = right_right_da_not_negdab & d_ac_case
    region_ad |= right_right_da_not_negdab_dac & expr6
    region_acd |= right_right_da_not_negdab_dac & (~expr6)
    inside_mask |= right_right_da_not_negdab & (~d_ac_case)

    region_a |= right_right_branch & (~da_case)

    coverage = (
        region_a.to(torch.int32)
        + region_ab.to(torch.int32)
        + region_ac.to(torch.int32)
        + region_ad.to(torch.int32)
        + region_abc.to(torch.int32)
        + region_acd.to(torch.int32)
        + region_adb.to(torch.int32)
        + inside_mask.to(torch.int32)
    )
    if bool((coverage == 0).any().item()):
        raise RuntimeError("Batched tetrahedron projection left some simplex states unassigned.")
    if bool((coverage > 1).any().item()):
        raise RuntimeError("Batched tetrahedron projection assigned multiple simplex regions.")

    if bool(region_a.any().item()):
        region_indices = tetra_indices[region_a]
        _set_simplex_subset(
            state,
            region_indices,
            a[region_a].unsqueeze(1),
            support_points0[region_a][:, [3]],
            support_points1[region_a][:, [3]],
        )
        _project_point_simplex_batch(state, region_indices)

    if bool(region_ab.any().item()):
        region_indices = tetra_indices[region_ab]
        _set_simplex_subset(
            state,
            region_indices,
            torch.stack((b[region_ab], a[region_ab]), dim=1),
            support_points0[region_ab][:, [2, 3]],
            support_points1[region_ab][:, [2, 3]],
        )
        _project_line_simplex_batch(state, region_indices)

    if bool(region_ac.any().item()):
        region_indices = tetra_indices[region_ac]
        _set_simplex_subset(
            state,
            region_indices,
            torch.stack((c[region_ac], a[region_ac]), dim=1),
            support_points0[region_ac][:, [1, 3]],
            support_points1[region_ac][:, [1, 3]],
        )
        _project_line_simplex_batch(state, region_indices)

    if bool(region_ad.any().item()):
        region_indices = tetra_indices[region_ad]
        _set_simplex_subset(
            state,
            region_indices,
            torch.stack((d[region_ad], a[region_ad]), dim=1),
            support_points0[region_ad][:, [0, 3]],
            support_points1[region_ad][:, [0, 3]],
        )
        _project_line_simplex_batch(state, region_indices)

    if bool(region_abc.any().item()):
        region_indices = tetra_indices[region_abc]
        _set_simplex_subset(
            state,
            region_indices,
            torch.stack((c[region_abc], b[region_abc], a[region_abc]), dim=1),
            support_points0[region_abc][:, [1, 2, 3]],
            support_points1[region_abc][:, [1, 2, 3]],
        )
        _project_triangle_simplex_batch(state, region_indices)

    if bool(region_acd.any().item()):
        region_indices = tetra_indices[region_acd]
        _set_simplex_subset(
            state,
            region_indices,
            torch.stack((d[region_acd], c[region_acd], a[region_acd]), dim=1),
            support_points0[region_acd][:, [0, 1, 3]],
            support_points1[region_acd][:, [0, 1, 3]],
        )
        _project_triangle_simplex_batch(state, region_indices)

    if bool(region_adb.any().item()):
        region_indices = tetra_indices[region_adb]
        _set_simplex_subset(
            state,
            region_indices,
            torch.stack((b[region_adb], d[region_adb], a[region_adb]), dim=1),
            support_points0[region_adb][:, [2, 0, 3]],
            support_points1[region_adb][:, [2, 0, 3]],
        )
        _project_triangle_simplex_batch(state, region_indices)

    if bool(inside_mask.any().item()):
        inside_indices = tetra_indices[inside_mask]
        state["ray"][inside_indices] = 0.0
        state["simplex_len"][inside_indices] = 4
        state["inside"][inside_indices] = True


def _clamp_distance(distance):
    if hasattr(distance, "clamp_min"):
        return distance.clamp_min(0.0)
    return max(distance, 0.0)


def _batch_size_of(collider0, collider1):
    batch_sizes = []
    for collider in (collider0, collider1):
        if isinstance(collider, TorchMesh):
            batch_sizes.append(collider.batch_size)

    if not batch_sizes:
        return 1

    batch_size = max(batch_sizes)
    for size in batch_sizes:
        if size not in (1, batch_size):
            raise ValueError(
                "Batched TorchMesh inputs must share the same batch size or be unbatched."
            )
    return batch_size


def _as_batched_mesh(collider):
    if collider.is_batched:
        return collider
    return TorchMesh(
        collider.mesh2origin.unsqueeze(0),
        collider.vertices.unsqueeze(0),
        collider.triangles.unsqueeze(0),
    )


def _init_batch_state(reference, batch_size, upper_bound, return_closest_points, use_nesterov_acceleration):
    state = {
        "normalize_support_direction": torch.ones(
            batch_size, dtype=torch.bool, device=reference.device),
        "inflation0": torch.zeros(
            batch_size, dtype=reference.dtype, device=reference.device),
        "inflation1": torch.zeros(
            batch_size, dtype=reference.dtype, device=reference.device),
        "inflation": torch.zeros(
            batch_size, dtype=reference.dtype, device=reference.device),
        "upper_bound": torch.full(
            (batch_size,), upper_bound, dtype=reference.dtype, device=reference.device),
        "alpha": torch.zeros(
            batch_size, dtype=reference.dtype, device=reference.device),
        "inside": torch.zeros(
            batch_size, dtype=torch.bool, device=reference.device),
        "simplex": torch.zeros(
            (batch_size, 4, 3), dtype=reference.dtype, device=reference.device),
        "support_points0": torch.zeros(
            (batch_size, 4, 3), dtype=reference.dtype, device=reference.device),
        "support_points1": torch.zeros(
            (batch_size, 4, 3), dtype=reference.dtype, device=reference.device),
        "simplex_len": torch.zeros(
            batch_size, dtype=torch.long, device=reference.device),
        "distance": torch.zeros(
            batch_size, dtype=reference.dtype, device=reference.device),
        "ray": torch.zeros(
            (batch_size, 3), dtype=reference.dtype, device=reference.device),
        "ray_len": torch.ones(
            batch_size, dtype=reference.dtype, device=reference.device),
        "ray_dir": torch.zeros(
            (batch_size, 3), dtype=reference.dtype, device=reference.device),
        "support_point": torch.zeros(
            (batch_size, 3), dtype=reference.dtype, device=reference.device),
        "iterations": torch.zeros(
            batch_size, dtype=torch.long, device=reference.device),
        "finished": torch.zeros(
            batch_size, dtype=torch.bool, device=reference.device),
        "nesterov_enabled": torch.full(
            (batch_size,), use_nesterov_acceleration,
            dtype=torch.bool, device=reference.device),
    }
    state["ray"][:, 0] = 1.0
    state["ray_dir"][:, 0] = 1.0
    state["support_point"][:, 0] = 1.0

    if return_closest_points:
        state["closest_point0"] = torch.zeros(
            (batch_size, 3), dtype=reference.dtype, device=reference.device)
        state["closest_point1"] = torch.zeros(
            (batch_size, 3), dtype=reference.dtype, device=reference.device)

    return state


def _finalize_batch_result(state, return_closest_points):
    if not return_closest_points:
        return (
            state["inside"],
            state["distance"],
            state["simplex"],
            state["iterations"],
        )
    return (
        state["inside"],
        state["distance"],
        state["simplex"],
        state["iterations"],
        state["closest_point0"],
        state["closest_point1"],
    )


def _squeeze_result(result, return_closest_points):
    if not return_closest_points:
        contact, distance, simplex, iterations = result
        return bool(contact[0].item()), distance[0], simplex[0], int(iterations[0].item())

    contact, distance, simplex, iterations, closest_point0, closest_point1 = result
    return (
        bool(contact[0].item()),
        distance[0],
        simplex[0],
        int(iterations[0].item()),
        closest_point0[0],
        closest_point1[0],
    )


def gjk_nesterov_accelerated_distance_and_points_torch(
        collider0, collider1, max_interations=128, upper_bound=1.79769e+308,
        tolerance=1e-6, use_nesterov_acceleration=False):
    """Torch-based batched Nesterov-accelerated GJK distance with closest points."""
    collider0, collider1 = _ensure_mesh_only(collider0, collider1)
    result = gjk_nesterov_accelerated(
        collider0,
        collider1,
        max_interations=max_interations,
        upper_bound=upper_bound,
        tolerance=tolerance,
        use_nesterov_acceleration=use_nesterov_acceleration,
        return_closest_points=True,
    )
    if _batch_size_of(collider0, collider1) == 1:
        contact, distance, simplex, iterations, closest_point0, closest_point1 = _squeeze_result(result, True)
    else:
        contact, distance, simplex, iterations, closest_point0, closest_point1 = result
    return (
        contact,
        _clamp_distance(distance),
        closest_point0,
        closest_point1,
        simplex,
        iterations,
    )


def gjk_nesterov_accelerated(
        collider0, collider1, max_interations=128, upper_bound=1.79769e+308,
        tolerance=1e-6, use_nesterov_acceleration=False,
        return_closest_points=False):
    """Batched torch GJK wrapper for TorchMesh inputs."""
    collider0, collider1 = _ensure_mesh_only(collider0, collider1)
    reference = _reference_tensor(collider0, collider1)
    batch_size = _batch_size_of(collider0, collider1)
    batch_collider0 = _as_batched_mesh(collider0)
    batch_collider1 = _as_batched_mesh(collider1)
    state = _init_batch_state(
        reference, batch_size, upper_bound, return_closest_points,
        use_nesterov_acceleration)

    state = _gjk_nesterov_accelerated_batch(
        batch_collider0,
        batch_collider1,
        state=state,
        reference=reference,
        max_interations=max_interations,
        tolerance=tolerance,
    )

    if return_closest_points:
        midpoints = 0.5 * (batch_collider0.center() + batch_collider1.center())
        closest_point0, closest_point1 = _compute_closest_points(
            state["simplex"],
            state["support_points0"],
            state["support_points1"],
            state["ray"],
            state["inside"],
            state["inflation0"],
            state["inflation1"],
            simplex_len=state["simplex_len"],
            midpoints=midpoints,
        )
        state["closest_point0"] = closest_point0
        state["closest_point1"] = closest_point1

    return _finalize_batch_result(state, return_closest_points)


def _gjk_nesterov_accelerated_batch(
        batch_collider0, batch_collider1, state, reference, max_interations=128,
        tolerance=1e-6):
    """Run the batched GJK while-loop with masked tolerance and support updates."""
    world_vertices0 = _mesh_world_vertices(batch_collider0) # vertices in world frame 
    world_vertices1 = _mesh_world_vertices(batch_collider1)
    active_mask = ~state["finished"]

    while bool(active_mask.any().item()):
        ### Check max iterations
        max_iteration_mask = active_mask & (state["iterations"] >= max_interations)
        if bool(max_iteration_mask.any().item()):
            state["finished"][max_iteration_mask] = True
            active_mask = ~state["finished"]
            if not bool(active_mask.any().item()):
                break

        ### Check tolerance
        tolerance_mask = active_mask & (state["ray_len"] < tolerance)
        if bool(tolerance_mask.any().item()):
            state["distance"][tolerance_mask] = -state["inflation"][tolerance_mask]
            state["inside"][tolerance_mask] = True
            state["finished"][tolerance_mask] = True
        active_mask = ~state["finished"]
        if not bool(active_mask.any().item()):
            break

        ### Nesterov acceleration update (normalize support direction)
        nesterov_mask = active_mask & state["nesterov_enabled"]
        normalized_nesterov_mask = nesterov_mask & state["normalize_support_direction"]
        if bool(normalized_nesterov_mask.any().item()):
            iterations = state["iterations"][normalized_nesterov_mask].to(dtype=state["ray"].dtype)
            momentum = (iterations + 2.0) / (iterations + 3.0)
            y = (
                momentum.unsqueeze(-1) * state["ray"][normalized_nesterov_mask]
                + (1.0 - momentum).unsqueeze(-1) * state["support_point"][normalized_nesterov_mask]
            )
            state["ray_dir"][normalized_nesterov_mask] = (
                momentum.unsqueeze(-1) * _torch_norm_vector(state["ray_dir"][normalized_nesterov_mask])
                + (1.0 - momentum).unsqueeze(-1) * _torch_norm_vector(y)
            )

        ### Nesterov acceleration update (not normalize support direction)
        unnormalized_nesterov_mask = nesterov_mask & (~state["normalize_support_direction"])
        if bool(unnormalized_nesterov_mask.any().item()):
            iterations = state["iterations"][unnormalized_nesterov_mask].to(dtype=state["ray"].dtype)
            momentum = (iterations + 1.0) / (iterations + 3.0)
            y = (
                momentum.unsqueeze(-1) * state["ray"][unnormalized_nesterov_mask]
                + (1.0 - momentum).unsqueeze(-1) * state["support_point"][unnormalized_nesterov_mask]
            )
            state["ray_dir"][unnormalized_nesterov_mask] = (
                momentum.unsqueeze(-1) * state["ray_dir"][unnormalized_nesterov_mask]
                + (1.0 - momentum).unsqueeze(-1) * y
            )

        ### No Nesterov acceleration update
        no_nesterov_mask = active_mask & (~state["nesterov_enabled"])
        if bool(no_nesterov_mask.any().item()):
            state["ray_dir"][no_nesterov_mask] = state["ray"][no_nesterov_mask]

        ### Get support points
        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        simplex_indices = state["simplex_len"][active_indices]
        ray_dir_active = state["ray_dir"][active_indices]
        support0 = _batched_mesh_support_points(world_vertices0[active_indices], -ray_dir_active)
        support1 = _batched_mesh_support_points(world_vertices1[active_indices], ray_dir_active)
        support_point = support0 - support1

        ### Update simplex and support points
        state["simplex"][active_indices, simplex_indices] = support_point
        state["support_points0"][active_indices, simplex_indices] = support0
        state["support_points1"][active_indices, simplex_indices] = support1
        state["support_point"][active_indices] = support_point
        state["simplex_len"][active_indices] = simplex_indices + 1

        ### Omega stopping check
        step_mask_local = torch.ones(active_indices.shape[0], dtype=torch.bool, device=reference.device)
        ray_dir_norm = torch.linalg.norm(ray_dir_active, dim=1)
        safe_ray_dir_norm = ray_dir_norm.clamp_min(torch.finfo(reference.dtype).eps)
        omega = torch.sum(ray_dir_active * support_point, dim=1) / safe_ray_dir_norm
        omega_stop_local = omega > state["upper_bound"][active_indices]
        if bool(omega_stop_local.any().item()):
            omega_stop_indices = active_indices[omega_stop_local]
            state["distance"][omega_stop_indices] = (
                omega[omega_stop_local] - state["inflation"][omega_stop_indices]
            )
            state["inside"][omega_stop_indices] = False
            state["finished"][omega_stop_indices] = True
            step_mask_local &= ~omega_stop_local

        ### Frank Wolfe duality stopping check
        frank_local = step_mask_local & state["nesterov_enabled"][active_indices]
        if bool(frank_local.any().item()):
            frank_gap = 2.0 * torch.sum(
                state["ray"][active_indices[frank_local]]
                * (state["ray"][active_indices[frank_local]] - support_point[frank_local]),
                dim=1,
            )
            frank_stop_local = torch.zeros_like(step_mask_local)
            frank_stop_local[frank_local] = (frank_gap - tolerance) <= 0.0
            if bool(frank_stop_local.any().item()):
                frank_stop_indices = active_indices[frank_stop_local]
                state["nesterov_enabled"][frank_stop_indices] = False
                state["simplex_len"][frank_stop_indices] -= 1
                step_mask_local &= ~frank_stop_local

        ### Convergence check
        alpha_local = step_mask_local.clone()
        if bool(alpha_local.any().item()):
            alpha_indices = active_indices[alpha_local]
            updated_alpha = torch.maximum(
                state["alpha"][alpha_indices],
                omega[alpha_local],
            )
            state["alpha"][alpha_indices] = updated_alpha
            diff = state["ray_len"][alpha_indices] - updated_alpha
            cv_check_passed = (diff - tolerance * state["ray_len"][alpha_indices]) <= 0.0
            cv_local = torch.zeros_like(step_mask_local)
            cv_local[alpha_local] = (
                (state["iterations"][alpha_indices] > 0) & cv_check_passed
            )
            if bool(cv_local.any().item()):
                cv_indices = active_indices[cv_local]
                cv_nesterov = state["nesterov_enabled"][cv_indices]
                state["simplex_len"][cv_indices] -= 1
                if bool(cv_nesterov.any().item()):
                    state["nesterov_enabled"][cv_indices[cv_nesterov]] = False
                cv_finish_indices = cv_indices[~cv_nesterov]
                if cv_finish_indices.numel() > 0:
                    state["distance"][cv_finish_indices] = (
                        state["ray_len"][cv_finish_indices] - state["inflation"][cv_finish_indices]
                    )
                    state["inside"][cv_finish_indices] = (state["distance"][cv_finish_indices] < tolerance)
                    state["finished"][cv_finish_indices] = True
                step_mask_local &= ~cv_local

        post_project_indices = active_indices[step_mask_local]
        if post_project_indices.numel() > 0:
            simplex_len_active = state["simplex_len"][post_project_indices]
            point_indices = post_project_indices[simplex_len_active == 1]
            line_indices = post_project_indices[simplex_len_active == 2]
            triangle_indices = post_project_indices[simplex_len_active == 3]
            tetra_indices = post_project_indices[simplex_len_active == 4]

            _project_point_simplex_batch(state, point_indices)
            _project_line_simplex_batch(state, line_indices)
            _project_triangle_simplex_batch(state, triangle_indices)
            _project_tetra_simplex_batch(state, tetra_indices)

        ### Post-projection (inside) checks
        if post_project_indices.numel() > 0:
            non_inside_indices = post_project_indices[~state["inside"][post_project_indices]]
            if non_inside_indices.numel() > 0:
                state["ray_len"][non_inside_indices] = torch.linalg.norm(
                    state["ray"][non_inside_indices], dim=1)

            finished_local = (
                state["inside"][post_project_indices] | (state["ray_len"][post_project_indices] == 0)
            )
            finished_indices = post_project_indices[finished_local]
            if finished_indices.numel() > 0:
                state["distance"][finished_indices] = -state["inflation"][finished_indices] - 1.0
                state["inside"][finished_indices] = True
                state["finished"][finished_indices] = True

            continue_indices = post_project_indices[~finished_local]
            if continue_indices.numel() > 0:
                state["iterations"][continue_indices] += 1
                state["finished"][continue_indices] = False

        active_mask = ~state["finished"]

    return state


def _inflation_of(collider):
    if type(collider) == Sphere or type(collider) == Capsule:
        return collider.radius
    return 0.0



def _barycentric_coordinates_from_simplex(simplex_points, closest_point):
    if len(simplex_points) == 1:
        return np.array([1.0])

    system = np.vstack((simplex_points.T, np.ones(len(simplex_points))))
    target = np.append(closest_point, 1.0)
    weights, _, _, _ = np.linalg.lstsq(system, target, rcond=None)
    weights[np.abs(weights) < 1e-12] = 0.0
    weights = np.clip(weights, 0.0, None)
    weight_sum = np.sum(weights)
    if weight_sum == 0.0:
        return np.ones(len(simplex_points), dtype=float) / len(simplex_points)
    return weights / weight_sum


def _barycentric_coordinates_from_simplex_batch(simplex_points, closest_points, simplex_len):
    dtype = simplex_points.dtype
    device = simplex_points.device

    batch_size = simplex_points.shape[0]
    weights = torch.zeros((batch_size, 4), dtype=dtype, device=device)
    target = torch.cat(
        (
            closest_points,
            torch.ones((batch_size, 1), dtype=dtype, device=device),
        ),
        dim=1,
    )
    abs_threshold = torch.tensor(1e-12, dtype=dtype, device=device)
    eps = torch.finfo(dtype).eps

    for active_count in range(1, 5):
        mask = simplex_len == active_count
        if not bool(mask.any().item()):
            continue

        if active_count == 1:
            weights[mask, 0] = 1.0
            continue

        simplex_group = simplex_points[mask, :active_count]
        system = torch.cat(
            (
                simplex_group.transpose(1, 2),
                torch.ones(
                    (simplex_group.shape[0], 1, active_count),
                    dtype=simplex_points.dtype,
                    device=simplex_points.device,
                ),
            ),
            dim=1,
        )
        solution = torch.matmul(
            torch.linalg.pinv(system),
            target[mask].unsqueeze(-1),
        ).squeeze(-1)
        solution = torch.where(
            solution.abs() < abs_threshold,
            torch.zeros_like(solution),
            solution,
        )
        solution = solution.clamp_min(0.0)
        weight_sum = solution.sum(dim=1, keepdim=True)
        fallback = torch.full_like(solution, 1.0 / active_count)
        solution = torch.where(
            weight_sum > 0.0,
            solution / weight_sum.clamp_min(eps),
            fallback,
        )
        weights[mask, :active_count] = solution

    return weights


def _compute_closest_points_single(
        simplex_points, support_points0, support_points1, ray, inside,
        inflation0, inflation1):
    weights = _barycentric_coordinates_from_simplex(simplex_points, ray)
    closest_point0 = weights.dot(support_points0)
    closest_point1 = weights.dot(support_points1)

    if inside:
        midpoint = 0.5 * (closest_point0 + closest_point1)
        return midpoint, midpoint

    ray_norm = np.linalg.norm(ray)
    if ray_norm > 0.0:
        normal = ray / ray_norm
        closest_point0 = closest_point0 - inflation0 * normal
        closest_point1 = closest_point1 + inflation1 * normal

    return closest_point0, closest_point1


def _compute_closest_points(
        simplex_points, support_points0, support_points1, ray, inside,
        inflation0, inflation1, simplex_len=None, midpoints=None):

    if not isinstance(simplex_points, torch.Tensor):
        return _compute_closest_points_single(
            simplex_points, support_points0, support_points1, ray, inside,
            inflation0, inflation1)

    if simplex_len is None or midpoints is None:
        raise ValueError(
            "Batched torch closest-point computation requires simplex_len and midpoints."
        )

    weights = _barycentric_coordinates_from_simplex_batch(
        simplex_points, ray, simplex_len)
    closest_point0 = torch.sum(weights.unsqueeze(-1) * support_points0, dim=1)
    closest_point1 = torch.sum(weights.unsqueeze(-1) * support_points1, dim=1)

    no_simplex_mask = simplex_len == 0
    if bool(no_simplex_mask.any().item()):
        closest_point0 = closest_point0.clone()
        closest_point1 = closest_point1.clone()
        closest_point0[no_simplex_mask] = midpoints[no_simplex_mask]
        closest_point1[no_simplex_mask] = midpoints[no_simplex_mask]

    inside_mask = inside & (~no_simplex_mask)
    if bool(inside_mask.any().item()):
        closest_point0 = closest_point0.clone()
        closest_point1 = closest_point1.clone()
        midpoint = 0.5 * (closest_point0[inside_mask] + closest_point1[inside_mask])
        closest_point0[inside_mask] = midpoint
        closest_point1[inside_mask] = midpoint

    ray_norm = torch.linalg.norm(ray, dim=1)
    valid_normal_mask = (~inside) & (~no_simplex_mask) & (ray_norm > 0.0)
    if bool(valid_normal_mask.any().item()):
        closest_point0 = closest_point0.clone()
        closest_point1 = closest_point1.clone()
        normal = ray[valid_normal_mask] / ray_norm[valid_normal_mask].unsqueeze(-1)
        closest_point0[valid_normal_mask] = (
            closest_point0[valid_normal_mask]
            - inflation0[valid_normal_mask].unsqueeze(-1) * normal
        )
        closest_point1[valid_normal_mask] = (
            closest_point1[valid_normal_mask]
            + inflation1[valid_normal_mask].unsqueeze(-1) * normal
        )

    return closest_point0, closest_point1
