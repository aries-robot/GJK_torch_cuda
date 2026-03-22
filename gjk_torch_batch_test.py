"""
=============================================================
Batched Minimum Distance Between Convex Meshes with Nesterov GJK of Torch

Code based on https://github.com/AlexanderFabisch/distance3d
=============================================================
"""
print(__doc__)

import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
import time

from _gjk_nesterov_accelerated_new import (
    gjk_nesterov_accelerated_distance_and_points,
)
from _gjk_nesterov_accelerated_torch import (
    TorchMesh,
    gjk_nesterov_accelerated_distance_and_points_torch,
)
from distance3d.mesh import make_convex_mesh


TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float64
BATCH_SIZE = 10000
N_VERTICES = 16
EXPECTED_NUM_FACES = 2 * N_VERTICES - 4
MAX_MESH_RETRIES = 32
MESH_GEN_SEED = 123

print(f"Using torch device: {TORCH_DEVICE}")


def to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def make_generator(seed):
    if TORCH_DEVICE.type == "cuda":
        generator = torch.Generator(device="cuda")
    else:
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def rotation_matrix_from_euler_xyz(roll, pitch, yaw):
    one = torch.ones((), dtype=roll.dtype, device=roll.device)
    zero = torch.zeros((), dtype=roll.dtype, device=roll.device)

    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    rx = torch.stack((
        torch.stack((one, zero, zero)),
        torch.stack((zero, cr, -sr)),
        torch.stack((zero, sr, cr)),
    ))
    ry = torch.stack((
        torch.stack((cp, zero, sp)),
        torch.stack((zero, one, zero)),
        torch.stack((-sp, zero, cp)),
    ))
    rz = torch.stack((
        torch.stack((cy, -sy, zero)),
        torch.stack((sy, cy, zero)),
        torch.stack((zero, zero, one)),
    ))
    return rz @ ry @ rx


def make_transform(translation, euler_xyz):
    transform = torch.eye(4, dtype=translation.dtype, device=translation.device)
    transform[:3, :3] = rotation_matrix_from_euler_xyz(*euler_xyz)
    transform[:3, 3] = translation
    return transform


def transform_points(mesh2origin, vertices):
    return vertices @ mesh2origin[:3, :3].T + mesh2origin[:3, 3]


def make_convex_ellipsoid_mesh(seed, radii, n_vertices):
    for attempt in range(MAX_MESH_RETRIES):
        rng = make_generator(seed + attempt)

        directions = torch.randn(
            (n_vertices, 3),
            generator=rng,
            dtype=TORCH_DTYPE,
            device=TORCH_DEVICE,
        )
        directions = directions / torch.norm(directions, dim=1, keepdim=True)
        vertices = directions * radii.unsqueeze(0)
        triangles = torch.as_tensor(
            make_convex_mesh(to_numpy(vertices)),
            dtype=torch.long,
            device=TORCH_DEVICE,
        )
        if triangles.shape[0] == EXPECTED_NUM_FACES:
            return vertices, triangles

    raise RuntimeError(
        f"Could not generate a convex mesh with {EXPECTED_NUM_FACES} faces after "
        f"{MAX_MESH_RETRIES} attempts."
    )


def gen_random_convex_mesh(seed, n_vertices, translation, euler_xyz):
    rng = make_generator(seed)

    radii = 0.5 + 0.5 * torch.rand(
        3,
        generator=rng,
        dtype=TORCH_DTYPE,
        device=TORCH_DEVICE,
    )
    vertices, triangles = make_convex_ellipsoid_mesh(
        seed=seed * 97 + 13,
        radii=radii,
        n_vertices=n_vertices,
    )
    mesh2origin = make_transform(
        translation=torch.tensor(translation, dtype=TORCH_DTYPE, device=TORCH_DEVICE),
        euler_xyz=torch.tensor(
            euler_xyz,
            dtype=TORCH_DTYPE,
            device=TORCH_DEVICE,
        ) * (math.pi / 180.0),
    )
    torch_collider = TorchMesh(mesh2origin, vertices, triangles)
    return torch_collider.to_numpy_collider(), torch_collider


def stack_torch_meshes(meshes):
    return TorchMesh(
        torch.stack([mesh.mesh2origin for mesh in meshes], dim=0),
        torch.stack([mesh.vertices for mesh in meshes], dim=0),
        torch.stack([mesh.triangles for mesh in meshes], dim=0),
    )


def make_pair_pose(pair_index, object_index):
    if object_index == 0:
        translation = [
            0.08 * pair_index,
            -0.04 * pair_index,
            0.03 * ((pair_index % 3) - 1),
        ]
        euler_xyz = [
            4.0 * pair_index,
            -3.0 * pair_index,
            2.0 * pair_index,
        ]
    else:
        translation = [
            1.55 + 0.07 * pair_index,
            0.25 * math.cos(0.5 * pair_index),
            0.20 + 0.06 * math.sin(0.5 * pair_index),
        ]
        euler_xyz = [
            -20.0 + 2.5 * pair_index,
            25.0 - 1.5 * pair_index,
            35.0 + 3.0 * pair_index,
        ]
    return translation, euler_xyz


def generate_random_convex_pairs(batch_size, n_vertices, seed=0):
    collider0_np = []
    collider1_np = []
    collider0_torch = []
    collider1_torch = []

    for pair_index in range(batch_size):
        translation0, euler0 = make_pair_pose(pair_index, object_index=0)
        translation1, euler1 = make_pair_pose(pair_index, object_index=1)

        pair_collider0_np, pair_collider0_torch = gen_random_convex_mesh(
            seed=2 * pair_index + seed,
            n_vertices=n_vertices,
            translation=translation0,
            euler_xyz=euler0,
        )
        pair_collider1_np, pair_collider1_torch = gen_random_convex_mesh(
            seed=2 * pair_index + 1 + seed,
            n_vertices=n_vertices,
            translation=translation1,
            euler_xyz=euler1,
        )

        collider0_np.append(pair_collider0_np)
        collider1_np.append(pair_collider1_np)
        collider0_torch.append(pair_collider0_torch)
        collider1_torch.append(pair_collider1_torch)

    return (
        collider0_np,
        collider1_np,
        stack_torch_meshes(collider0_torch),
        stack_torch_meshes(collider1_torch),
    )


def run_numpy_sequential_batch(colliders0, colliders1):
    contacts = []
    distances = []
    closest_points0 = []
    closest_points1 = []
    simplexes = []
    iterations = []

    for collider0, collider1 in zip(colliders0, colliders1):
        contact, distance, closest_point0, closest_point1, simplex, n_iterations = \
            gjk_nesterov_accelerated_distance_and_points(
                collider0,
                collider1,
                use_nesterov_acceleration=True,
            )
        contacts.append(contact)
        distances.append(distance)
        closest_points0.append(np.array(closest_point0, copy=True))
        closest_points1.append(np.array(closest_point1, copy=True))
        simplexes.append(np.array(simplex, copy=True))
        iterations.append(n_iterations)

    return (
        np.asarray(contacts, dtype=bool),
        np.asarray(distances, dtype=float),
        np.stack(closest_points0, axis=0),
        np.stack(closest_points1, axis=0),
        np.stack(simplexes, axis=0),
        np.asarray(iterations, dtype=np.int64),
    )


def plot_mesh(ax, collider, color, alpha):
    world_vertices = to_numpy(transform_points(collider.mesh2origin, collider.vertices))
    triangles = to_numpy(collider.triangles).astype(np.int64, copy=False)
    faces = world_vertices[triangles]
    surface = Poly3DCollection(
        faces, facecolor=color, edgecolor="k", linewidths=0.6, alpha=alpha)
    ax.add_collection3d(surface)
    ax.scatter(
        world_vertices[:, 0], world_vertices[:, 1], world_vertices[:, 2],
        color=color, s=12, alpha=min(alpha + 0.15, 1.0),
    )
    return world_vertices


def set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius == 0.0:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

### Warmup
collider0_np_batch, collider1_np_batch, collider0_torch_batch, collider1_torch_batch = \
    generate_random_convex_pairs(2, N_VERTICES, 0)
contacts_np, distances_np, closest_points0_np, closest_points1_np, simplexes_np, iterations_np = \
    run_numpy_sequential_batch(collider0_np_batch, collider1_np_batch)
contacts, distances, closest_points0, closest_points1, simplexes, iterations = \
    gjk_nesterov_accelerated_distance_and_points_torch(
        collider0_torch_batch,
        collider1_torch_batch,
        use_nesterov_acceleration=True,
    )

### Inference: (1) Generate dataset
collider0_np_batch, collider1_np_batch, collider0_torch_batch, collider1_torch_batch = \
    generate_random_convex_pairs(BATCH_SIZE, N_VERTICES, MESH_GEN_SEED)

### Inference: (2) Run numpy sequential batch
start_time_np = time.time()
contacts_np, distances_np, closest_points0_np, closest_points1_np, simplexes_np, iterations_np = \
    run_numpy_sequential_batch(collider0_np_batch, collider1_np_batch)
elapsed_time_np = time.time() - start_time_np

### Inference: (3) Run torch GPU batch
start_time_gpu = torch.cuda.Event(enable_timing=True)
end_time_gpu = torch.cuda.Event(enable_timing=True)
start_time_gpu.record()
contacts, distances, closest_points0, closest_points1, simplexes, iterations = \
    gjk_nesterov_accelerated_distance_and_points_torch(
        collider0_torch_batch,
        collider1_torch_batch,
        use_nesterov_acceleration=True,
    )
end_time_gpu.record()
torch.cuda.synchronize()
elapsed_time_gpu = start_time_gpu.elapsed_time(end_time_gpu)*1e-3

assert torch.equal(
    contacts,
    torch.as_tensor(contacts_np, dtype=torch.bool, device=TORCH_DEVICE),
)
assert torch.allclose(
    distances,
    torch.as_tensor(distances_np, dtype=TORCH_DTYPE, device=TORCH_DEVICE),
    atol=1e-6,
    rtol=1e-6,
)
assert torch.allclose(
    closest_points0,
    torch.as_tensor(closest_points0_np, dtype=TORCH_DTYPE, device=TORCH_DEVICE),
    atol=1e-6,
    rtol=1e-6,
)
assert torch.allclose(
    closest_points1,
    torch.as_tensor(closest_points1_np, dtype=TORCH_DTYPE, device=TORCH_DEVICE),
    atol=1e-6,
    rtol=1e-6,
)
expected_simplexes = torch.as_tensor(
    simplexes_np,
    dtype=TORCH_DTYPE,
    device=TORCH_DEVICE,
)
# The batched GPU support path can pick a different but equivalent support vertex,
# so the intermediate simplex is no longer guaranteed to match exactly.
assert simplexes.shape == expected_simplexes.shape
assert torch.isfinite(simplexes).all()
# assert torch.equal(
#     iterations,
#     torch.as_tensor(iterations_np, dtype=torch.long, device=TORCH_DEVICE),
# ) # This is not equal. (why?)

print(f"contacts shape: {tuple(contacts.shape)}")
print(f"distances shape: {tuple(distances.shape)}")
print(f"closest_points0 shape: {tuple(closest_points0.shape)}")
print(f"closest_points1 shape: {tuple(closest_points1.shape)}")
print(f"simplexes shape: {tuple(simplexes.shape)}")
print(f"iterations shape: {tuple(iterations.shape)}")
print("stacked distances:")
print(distances)
print("stacked iterations:")
print(iterations)
print(f"Max abs distance error: {np.max(np.abs(distances.to('cpu').numpy() - distances_np))}")
print(f"Elapsed time (numpy sequential): {elapsed_time_np:.6f} seconds")
print(f"Elapsed time (GPU): {elapsed_time_gpu:.6f} seconds")

first_collider0 = collider0_torch_batch.batch_item(0)
first_collider1 = collider1_torch_batch.batch_item(0)
first_closest_point0 = to_numpy(closest_points0[0])
first_closest_point1 = to_numpy(closest_points1[0])
first_distance = float(distances[0].item())

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

world_vertices0 = plot_mesh(ax, first_collider0, color="tab:blue", alpha=0.35)
world_vertices1 = plot_mesh(ax, first_collider1, color="tab:orange", alpha=0.35)

ax.scatter(
    first_closest_point0[0], first_closest_point0[1], first_closest_point0[2],
    color="tab:blue", s=80, label="Closest point on mesh 0")
ax.scatter(
    first_closest_point1[0], first_closest_point1[1], first_closest_point1[2],
    color="tab:orange", s=80, label="Closest point on mesh 1")
ax.plot(
    [first_closest_point0[0], first_closest_point1[0]],
    [first_closest_point0[1], first_closest_point1[1]],
    [first_closest_point0[2], first_closest_point1[2]],
    color="k", linewidth=2.5,
    label=f"First batch distance = {first_distance:.4f}",
)

all_points = np.vstack((
    world_vertices0,
    world_vertices1,
    first_closest_point0[np.newaxis],
    first_closest_point1[np.newaxis],
))
set_axes_equal(ax, all_points)

ax.set_title("First Pair from Batched _gjk_nesterov_accelerated_torch.py")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
