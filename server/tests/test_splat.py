import numpy as np
import pytest

from video3d import splat


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def _room(n=30000, seed=1):
    """A floor + two walls of random surface points, 6 x 3 x 5 m."""
    rng = np.random.default_rng(seed)
    floor = np.c_[rng.uniform(-3, 3, n), np.zeros(n), rng.uniform(-2.5, 2.5, n)]
    wall1 = np.c_[rng.uniform(-3, 3, n), rng.uniform(0, 3, n), np.full(n, -2.5)]
    wall2 = np.c_[np.full(n, 3.0), rng.uniform(0, 3, n), rng.uniform(-2.5, 2.5, n)]
    return np.vstack([floor, wall1, wall2])


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(500, 3))
    R, s, t = _rot("y", 40) @ _rot("x", 10), 2.5, np.array([1.0, -2.0, 0.5])
    s2, R2, t2 = splat.umeyama(src, s * src @ R.T + t)
    assert s2 == pytest.approx(s) and np.allclose(R2, R) and np.allclose(t2, t)


def test_align_recovers_unknown_yaw_scale_and_noise():
    mesh = _room()
    # the "COLMAP frame": arbitrary scale, yawed, flipped so camera-up is -Y (COLMAP convention), sparse + noisy
    s_true, R_true, t_true = 0.52, _rot("y", 70) @ _rot("x", 180), np.array([0.3, 1.1, 2.0])
    rng = np.random.default_rng(3)
    pts_scene = mesh[rng.choice(len(mesh), 3000, replace=False)] + rng.normal(0, 0.01, (3000, 3))
    pts_colmap = (pts_scene - t_true) @ R_true / s_true  # inverse of x_scene = s R x + t
    cam_up = np.array([0.0, -1.0, 0.0])  # world "up" in the COLMAP frame

    al = splat.align_to_mesh(pts_colmap, cam_up, mesh)
    assert al["s"] == pytest.approx(s_true, rel=0.05)  # planar rooms leave scale weakly constrained
    assert al["residual_median_m"] < 0.03
    assert al["inlier_frac_rel"] > 0.9


def test_bake_transforms_positions_scales_and_orientation():
    rng = np.random.default_rng(5)
    n = 50
    dt = [(k, "<f4") for k in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                               "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]]
    arr = np.zeros(n, dtype=dt)
    xyz = rng.uniform(-1, 1, (n, 3))
    arr["x"], arr["y"], arr["z"] = xyz.T
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for i in range(4):
        arr[f"rot_{i}"] = q[:, i]
    for i in range(3):
        arr[f"scale_{i}"] = np.log(0.05)
    arr["opacity"] = 2.0

    s, R, t = 1.7, _rot("y", 33) @ _rot("x", 20), np.array([0.5, 0.0, -1.0])
    out = splat.bake_and_clean(arr, s, R, t, np.array([-10.0] * 3), np.array([10.0] * 3))
    assert len(out) == n
    new = np.stack([out["x"], out["y"], out["z"]], 1)
    assert np.allclose(new, s * xyz @ R.T + t, atol=1e-4)
    assert np.allclose(np.exp(out["scale_0"]), 0.05 * s, rtol=1e-4)

    def quat_to_mat(qq):
        w, x, y, z = qq
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    for k in range(5):
        r_new = quat_to_mat(np.array([out[f"rot_{i}"][k] for i in range(4)], dtype=float))
        assert np.allclose(r_new, R @ quat_to_mat(q[k]), atol=1e-4)  # splat axes rotated by R


def test_bake_drops_floaters_and_transparent_splats():
    dt = [(k, "<f4") for k in ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                               "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"]]
    arr = np.zeros(3, dtype=dt)
    arr["rot_0"] = 1
    arr["scale_0"] = arr["scale_1"] = arr["scale_2"] = np.log(0.05)
    arr["opacity"] = [2.0, 2.0, -6.0]
    arr["x"] = [0.0, 50.0, 0.0]  # second is a far floater, third nearly transparent
    out = splat.bake_and_clean(arr, 1.0, np.eye(3), np.zeros(3), np.array([-2.0] * 3), np.array([2.0] * 3))
    assert len(out) == 1


def test_bake_rejects_higher_order_sh():
    arr = np.zeros(1, dtype=[("x", "<f4"), ("f_rest_0", "<f4")])
    with pytest.raises(ValueError):
        splat.bake_and_clean(arr, 1.0, np.eye(3), np.zeros(3), np.zeros(3), np.ones(3))


def test_ply_roundtrip(tmp_path):
    arr = np.zeros(4, dtype=[("x", "<f4"), ("y", "<f4"), ("opacity", "<f4")])
    arr["x"] = [1, 2, 3, 4]
    splat.write_ply(tmp_path / "a.ply", arr)
    back, _ = splat.read_ply(tmp_path / "a.ply")
    assert back["x"].tolist() == [1, 2, 3, 4] and back.dtype.names == ("x", "y", "opacity")
