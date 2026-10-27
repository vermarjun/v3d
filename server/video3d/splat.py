"""Photoreal track: video -> Gaussian splat aligned to the metric scene.glb.

Pipeline (all on-device, no torch / CUDA):
  1. ffmpeg           sample frames from data/scenes/<id>/input.mp4
  2. pycolmap (CPU)   SIFT + matching + incremental SfM  -> camera poses + sparse points
  3. Brush (wgpu)     train a Gaussian splat on the COLMAP dataset -> .ply
  4. alignment        similarity transform COLMAP frame -> scene.glb frame (metres, +Y up),
                      baked into the splat file, residual reported
  5. meta.json        ``meta["splat"] = {url, format, aligned}`` per CONTRACT.md

CLI: ``uv run --project server --extra splat video3d-splat <scene_id>``
Brush binary: $VIDEO3D_BRUSH, else <repo>/tools/brush/brush_app (macOS arm64 release from github.com/ArthurBrussee/brush).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger("video3d.splat")

from .config import FFMPEG, FFPROBE, ROOT, scenes_dir

BRUSH_DEFAULT = ROOT / "tools" / "brush" / "brush_app"


# --------------------------------------------------------------------------- frames


def video_duration(video: Path) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def extract_frames(video: Path, out_dir: Path, max_frames: int = 90, max_side: int = 1024) -> int:
    """Evenly sample about ``max_frames`` frames (fewer for very short clips)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = video_duration(video)
    n = int(min(max_frames, max(40, dur * 12)))
    vf = f"fps={n / dur:.4f},scale='min({max_side},iw)':-2"
    subprocess.run(
        [FFMPEG, "-loglevel", "error", "-y", "-i", str(video), "-vf", vf, "-q:v", "2",
         str(out_dir / "f%04d.jpg")], check=True)
    return len(list(out_dir.glob("f*.jpg")))


# --------------------------------------------------------------------------- SfM


def run_colmap(work: Path) -> "pycolmap.Reconstruction":  # noqa: F821
    import pycolmap

    images, db, sparse = work / "images", work / "database.db", work / "sparse"
    sparse.mkdir(exist_ok=True)
    if db.exists():
        db.unlink()
    n = len(list(images.glob("*.jpg")))
    pycolmap.extract_features(
        db, images, camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=pycolmap.ImageReaderOptions(camera_model="SIMPLE_RADIAL"))
    if n <= 50:
        pycolmap.match_exhaustive(db)
    else:  # video frames: neighbours are what matters, and exhaustive is O(n^2)
        pycolmap.match_sequential(db, pairing_options=pycolmap.SequentialPairingOptions(overlap=12))
    maps = pycolmap.incremental_mapping(db, images, sparse)
    if not maps:
        raise RuntimeError(
            "COLMAP could not reconstruct the video (camera must move and see a textured scene; "
            "a static camera has no parallax)")
    best = max(maps.values(), key=lambda m: m.num_reg_images())
    if best.num_reg_images() < 0.5 * n:
        raise RuntimeError(f"COLMAP registered only {best.num_reg_images()}/{n} frames")
    (sparse / "best").mkdir(exist_ok=True)
    best.write(sparse / "best")
    return best


# --------------------------------------------------------------------------- Brush


def run_brush(work: Path, steps: int, max_res: int, brush: Path, log_file: Path) -> Path:
    if not brush.exists():
        raise RuntimeError(f"Brush binary not found at {brush} (set VIDEO3D_BRUSH)")
    # Brush reads <dataset>/sparse/0; expose the chosen model there.
    s0 = work / "sparse" / "0"
    if s0.exists():
        shutil.rmtree(s0)
    shutil.copytree(work / "sparse" / "best", s0)
    out = work / "out"
    out.mkdir(exist_ok=True)
    cmd = [str(brush), str(work), "--total-steps", str(steps), "--sh-degree", "0",
           "--max-resolution", str(max_res), "--export-path", str(out), "--export-name", "splat.ply",
           "--export-every", str(steps), "--eval-every", "1000000000"]
    with open(log_file, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, "RUST_LOG": "info"})
    ply = out / "splat.ply"
    if not ply.exists():
        raise RuntimeError(f"Brush produced no output, see {log_file}")
    return ply


# --------------------------------------------------------------------------- PLY io


def read_ply(path: Path) -> tuple[np.ndarray, list[str]]:
    """Binary little-endian PLY with one vertex element -> structured array."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        body = f.read()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("only binary_little_endian PLY is supported")
    n = int(next(h for h in header if h.startswith("element vertex")).split()[2])
    tmap = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint": "<u4", "int": "<i4", "short": "<i2", "ushort": "<u2"}
    dt = [(h.split()[2], tmap[h.split()[1]]) for h in header if h.startswith("property")]
    arr = np.frombuffer(body, dtype=dt, count=n)
    return arr, header


def write_ply(path: Path, arr: np.ndarray) -> None:
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(arr)}"]
    head += [f"property float {n}" for n in arr.dtype.names]
    head += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(arr).tobytes())


# --------------------------------------------------------------------------- alignment


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity dst ~= s * R @ src + t  (R proper rotation)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / len(src)
    s = float((D * np.diag(S)).sum() / var)
    t = mu_d - s * R @ mu_s
    return s, R, t


def rot_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:
        axis = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def trimmed_icp(pts: np.ndarray, tree: cKDTree, s: float, R: np.ndarray, t: np.ndarray,
                iters: int, keep: float, back_keep: float = 0.85,
                anchor: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Bidirectional trimmed ICP with scale.

    Forward pairs (point -> nearest mesh vertex) alone let the point set shrink onto a dense patch of the
    mesh, so backward pairs (mesh vertex -> nearest point) are added. Both are trimmed because sparse SfM
    points cover only part of the mesh and the mesh has outliers. ``anchor`` = (src, dst) is one known
    correspondence, repeated to weigh as much as a fifth of the points, so a mostly-planar mesh cannot let
    the solution slide along its planes.
    """
    mesh = tree.data
    msub = mesh[np.random.default_rng(0).choice(len(mesh), min(len(mesh), 6000), replace=False)]
    cost = np.inf
    for _ in range(iters):
        moved = s * pts @ R.T + t
        d, idx = tree.query(moved)
        k = max(10, int(keep * len(pts)))
        sel = np.argsort(d)[:k]
        src, dst = [pts[sel]], [mesh[idx[sel]]]
        pt_tree = cKDTree(moved)
        db, ib = pt_tree.query(msub)
        kb = max(10, int(back_keep * len(msub)))
        selb = np.argsort(db)[:kb]
        src.append(pts[ib[selb]])
        dst.append(msub[selb])
        if anchor is not None:
            reps = max(1, len(pts) // 5)
            src.append(np.tile(anchor[0], (reps, 1)))
            dst.append(np.tile(anchor[1], (reps, 1)))
        s2, R2, t2 = umeyama(np.vstack(src), np.vstack(dst))
        if not (np.isfinite(s2) and 0.05 < s2 / s < 20 and np.isfinite(t2).all()):
            return s, R, t, np.inf  # diverged: reject this start
        s, R, t = s2, R2, t2
        cost = float(np.median(d[sel]) + np.median(db[selb]))
    return s, R, t, cost


def align_to_mesh(points: np.ndarray, cam_up: np.ndarray, mesh_vertices: np.ndarray,
                  seed: int = 0, prior: dict | None = None) -> dict:
    """Similarity (s, R, t) with  x_scene = s * R @ x_colmap + t.

    The COLMAP frame has arbitrary orientation, but the cameras were held roughly upright, so their mean
    "up" gives gravity. That leaves yaw + scale + translation, found by a multi-start trimmed ICP against the
    metric mesh vertices, then a free-rotation refinement.

    ``prior`` (optional) pins the first video camera, which the reconstruction backend reports in
    ``meta["cameras"][0]``: ``{"c": its centre in the COLMAP frame, "pos": the same in the scene frame,
    "R": a COLMAP->scene rotation making its forward axis match the reported look-at}``. Seeds then share that
    rotation and camera position and only scale is searched, and the camera position stays a soft anchor in
    every ICP step. Without it, yaw is brute-forced, which can lock onto a wrong basin in rooms of planes.
    """
    rng = np.random.default_rng(seed)
    mv = mesh_vertices[rng.choice(len(mesh_vertices), min(len(mesh_vertices), 300_000), replace=False)]
    tree = cKDTree(mv)
    pts = points[rng.choice(len(points), min(len(points), 4000), replace=False)]

    R_up = rot_from_to(cam_up, np.array([0.0, 1.0, 0.0]))
    p_up = pts @ R_up.T

    def spread(x):  # robust size: 2-98 percentile span, geometric mean over axes
        lo, hi = np.percentile(x, [2, 98], axis=0)
        return float(np.cbrt(np.prod(np.maximum(hi - lo, 1e-6))))

    s0 = spread(mv) / spread(p_up)
    c_m, c_p = np.median(mv, 0), np.median(p_up, 0)

    # coarse multi-start on small subsamples (cheap), then refine the best few on the full sets
    coarse_tree = cKDTree(mv[rng.choice(len(mv), min(len(mv), 20_000), replace=False)])
    cpts = pts[:1500]
    starts = []
    anchor = None
    if prior is not None:
        anchor = (np.asarray(prior["c"], float)[None], np.asarray(prior["pos"], float)[None])
        R = np.asarray(prior["R"], float)
        for sc in np.geomspace(0.5, 2.5, 21):
            t = np.asarray(prior["pos"], float) - sc * R @ np.asarray(prior["c"], float)
            starts.append(trimmed_icp(cpts, coarse_tree, sc, R, t, iters=8, keep=0.6, anchor=anchor))
    else:
        for mult in (0.7, 1.0, 1.4):
            for yaw in np.deg2rad(np.arange(0, 360, 15)):
                R = rot_y(yaw) @ R_up
                sc = s0 * mult
                t = c_m - sc * rot_y(yaw) @ c_p
                starts.append(trimmed_icp(cpts, coarse_tree, sc, R, t, iters=8, keep=0.6))
    starts.sort(key=lambda r: r[3])
    refined = [trimmed_icp(pts, tree, *r[:3], iters=15, keep=0.6, anchor=anchor) for r in starts[:4]]
    s, R, t, _ = min(refined, key=lambda r: r[3])
    allp = points[rng.choice(len(points), min(len(points), 20000), replace=False)]
    s, R, t, _ = trimmed_icp(allp, tree, s, R, t, iters=120, keep=0.8, anchor=anchor)  # point-to-point ICP crawls along planes

    d, _ = tree.query(s * allp @ R.T + t)
    diag = float(np.linalg.norm(mv.max(0) - mv.min(0)))
    out = {
        "s": s, "R": R, "t": t,
        "residual_median_m": float(np.median(d)),
        "residual_p90_m": float(np.percentile(d, 90)),
        "inlier_frac_5cm": float((d < 0.05).mean()),
        "inlier_frac_rel": float((d < 0.02 * diag).mean()),
        "scene_diag_m": diag,
    }
    if prior is not None:
        out["cam0_error_m"] = float(np.linalg.norm(s * R @ np.asarray(prior["c"], float) + t - np.asarray(prior["pos"], float)))
    return out


def raw_to_scene(raw_glb: Path, mesh_vertices: np.ndarray, cam: dict):
    """Exact rigid transform from MapAnything's raw.glb frame (= first camera's frame, y/z flipped by glTF)
    into the scene.glb frame. scene.glb is cleaned/moved raw.glb, so ICP from the reported first-camera pose
    lands at ~cm residual. Returns (R, t, median residual in metres, frame-0 points in the raw frame)."""
    import trimesh

    raw = trimesh.load(str(raw_glb), force="scene")
    node = next(n for n in raw.graph.nodes_geometry if raw.graph[n][1] == "geometry_0")  # frame 0's depth mesh
    T0 = raw.graph[node][0]
    g0 = np.asarray(raw.geometry["geometry_0"].vertices, float) @ T0[:3, :3].T + T0[:3, 3]
    pts = np.concatenate([np.asarray(g.vertices, float) for g in raw.dump()])[::20]
    pos, look = np.asarray(cam["position"], float), np.asarray(cam["look_at"], float)
    fwd = (look - pos) / np.linalg.norm(look - pos)
    down = -(np.array([0.0, 1.0, 0.0]) - fwd[1] * fwd)  # init: zero roll; ICP fixes the rest
    down /= np.linalg.norm(down)
    R, t = np.stack([np.cross(down, fwd), down, fwd], 1) @ np.diag([1.0, -1.0, -1.0]), pos
    tree = cKDTree(mesh_vertices[::2])
    for _ in range(30):
        d, idx = tree.query(pts @ R.T + t)
        k = np.argsort(d)[: int(0.6 * len(pts))]
        _, R, t = umeyama(pts[k], tree.data[idx[k]])
    return R, t, float(np.median(tree.query(pts @ R.T + t)[0])), g0


def align_with_raw(rec, src: np.ndarray, mesh_vertices: np.ndarray, raw_glb: Path, cam: dict) -> dict:
    """Pose from the backend's first camera (exact), scale from per-ray depths of that same photo.

    COLMAP image 0 and MapAnything frame 0 are the same photo, so COLMAP's frame maps to the scene by
    x_scene = s * (R_rs F R_cw) x + (s * R_rs F t_cw + t_rs), with F = diag(1,-1,-1) (glTF y/z flip).
    Only the scale s is unknown. Every COLMAP point seen in image 0 is a ray in camera 0 with a COLMAP depth;
    MapAnything's frame-0 depth mesh gives a metric depth along the same ray. The right s makes s * z_colmap
    match some mesh depth near that ray for many points at once (a sharp peak, ~7x chance on the office clip);
    "some neighbour" rather than the median because rays pass through gaps in the mesh.
    """
    R_rs, t_rs, raw_res, g0 = raw_to_scene(raw_glb, mesh_vertices, cam)
    im0 = min(rec.images.values(), key=lambda im: im.name)
    T = im0.cam_from_world()
    F = np.diag([1.0, -1.0, -1.0])
    R, t0 = R_rs @ F @ T.rotation.matrix(), R_rs @ F @ T.translation

    ids = [p.point3D_id for p in im0.points2D if p.has_point3D() and p.point3D_id in rec.points3D]
    pc = np.array([rec.points3D[i].xyz for i in ids]) @ T.rotation.matrix().T + T.translation  # camera-0 coords
    rc = np.linalg.norm(pc, axis=1)
    dir_c = (pc * np.array([1.0, -1.0, -1.0])) / rc[:, None]
    rg = np.linalg.norm(g0, axis=1)
    dd, idx = cKDTree(g0 / rg[:, None]).query(dir_c, k=40)
    depths = rg[idx]
    near = np.degrees(2 * np.arcsin(np.clip(dd / 2, 0, 1))) < 0.6

    def hit(sc):
        tgt = (sc * rc)[:, None]
        return float(((np.abs(depths - tgt) < 0.05 * tgt) & near).any(axis=1).mean())

    scales = np.linspace(0.2, 3.0, 141)
    hits = np.array([hit(sc) for sc in scales])
    s = float(scales[hits.argmax()])
    chance = float(np.median(hits))
    t = s * t0 + t_rs
    d, _ = cKDTree(mesh_vertices[::2]).query(s * src @ R.T + t, distance_upper_bound=0.1)
    return {"s": s, "R": R, "t": t, "method": "raw-cam0", "raw_to_scene_residual_m": raw_res,
            "scale_peak_hit": float(hits.max()), "scale_chance_hit": chance, "frame0_points": int(len(rc)),
            "frac_within_10cm": float(np.isfinite(d).mean()),
            "cam0_error_m": float(np.linalg.norm(s * R @ _cam_center(im0) + t - np.asarray(cam["position"], float)))}


def camera_up(rec) -> np.ndarray:
    ups = []
    for im in rec.images.values():
        R = im.cam_from_world().rotation.matrix()
        ups.append(-R[1, :])  # COLMAP camera y points down; world direction of cam y = R[1,:]
    up = np.mean(ups, axis=0)
    return up / np.linalg.norm(up)


def _cam_center(im) -> np.ndarray:
    T = im.cam_from_world()
    return -T.rotation.matrix().T @ T.translation


def sparse_points(rec) -> np.ndarray:
    return np.array([p.xyz for p in rec.points3D.values() if p.track.length() >= 3 and p.error < 2.0])


def load_mesh_vertices(glb: Path) -> np.ndarray:
    import trimesh

    scene = trimesh.load(str(glb), force="scene")
    return np.concatenate([g.vertices.view(np.ndarray) for g in scene.dump()], axis=0).astype(np.float64)


# --------------------------------------------------------------------------- bake


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, (..., 4) arrays in (w, x, y, z) order."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x, y, z = np.copysign(x, R[2, 1] - R[1, 2]), np.copysign(y, R[0, 2] - R[2, 0]), np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def bake_and_clean(arr: np.ndarray, s: float, R: np.ndarray, t: np.ndarray,
                   bbox_min: np.ndarray, bbox_max: np.ndarray, margin: float = 1.0) -> np.ndarray:
    """Apply the similarity to a 3DGS PLY (degree-0 SH) and drop floaters outside the scene box."""
    names = arr.dtype.names
    if any(n.startswith("f_rest_") for n in names):
        raise ValueError("expected --sh-degree 0 splats (SH rotation is not implemented)")
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
    xyz = s * xyz @ R.T + t
    out = arr.copy()
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for i in range(3):
        out[f"scale_{i}"] = arr[f"scale_{i}"] + np.log(s)  # PLY stores log-scales
    q = np.stack([arr[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = quat_mul(rot_to_quat(R)[None, :], q)
    for i in range(4):
        out[f"rot_{i}"] = q[:, i]
    keep = np.all((xyz > bbox_min - margin) & (xyz < bbox_max + margin), axis=1)
    keep &= arr["opacity"] > -2.5  # sigmoid^-1(0.075): drop near-transparent splats
    max_scale = np.exp(np.max([out[f"scale_{i}"] for i in range(3)], axis=0))
    keep &= max_scale < 0.5 * float(np.max(bbox_max - bbox_min))
    return out[keep]


# --------------------------------------------------------------------------- driver


def build_splat(scene_id: str, steps: int = 6000, max_frames: int = 90, max_res: int = 1024,
                keep_work: bool = False, force: bool = False, reuse: bool = False) -> dict:
    sdir = scenes_dir() / scene_id
    video, glb, meta_path = sdir / "input.mp4", sdir / "scene.glb", sdir / "meta.json"
    for p in (video, glb, meta_path):
        if not p.exists():
            raise FileNotFoundError(p)
    brush = Path(os.environ.get("VIDEO3D_BRUSH", BRUSH_DEFAULT))
    work = sdir / "splat_work"
    work.mkdir(exist_ok=True)
    timings: dict[str, float] = {}

    def stage(name, fn, *a, **k):
        t0 = time.time()
        r = fn(*a, **k)
        timings[name] = round(time.time() - t0, 1)
        log.info("%s done in %.1fs", name, timings[name])
        return r

    ply = work / "out" / "splat.ply"
    if reuse and ply.exists() and (work / "sparse" / "best").exists():
        import pycolmap  # a previous --keep-work run left the SfM model and trained splat behind

        rec = pycolmap.Reconstruction(str(work / "sparse" / "best"))
        log.info("reusing trained splat and %d-frame COLMAP model from %s", rec.num_reg_images(), work)
    else:
        n = stage("frames", extract_frames, video, work / "images", max_frames, max_res)
        rec = stage("colmap", run_colmap, work)
        log.info("colmap: %d/%d frames, %d points", rec.num_reg_images(), n, rec.num_points3D())
        ply = stage("train", run_brush, work, steps, max_res, brush, sdir / "splat_train.log")

    def align():
        mv = load_mesh_vertices(glb)
        arr, _ = read_ply(ply)
        # Dense source = confident splat centres: they cover untextured surfaces that SfM points miss, and
        # the metric mesh (16 frames) is too holey for the sparse points alone (used by the first-camera path).
        sc = np.exp(np.max([arr[f"scale_{i}"] for i in range(3)], axis=0))
        sel = (arr["opacity"] > 0) & (sc < np.percentile(sc, 90))
        src = np.stack([arr["x"], arr["y"], arr["z"]], 1)[sel].astype(np.float64)
        if len(src) < 2000:
            src = sparse_points(rec)
        cams = json.loads(meta_path.read_text()).get("cameras") or []
        raw_glb = sdir / "raw.glb"
        if cams and raw_glb.exists():  # exact pose from the backend's first camera, only scale searched
            al = align_with_raw(rec, src, mv, raw_glb, cams[0])
        else:
            src = sparse_points(rec)
            al = align_to_mesh(src, camera_up(rec), mv)
            al["method"] = "icp"
        d, _ = cKDTree(mv[:: max(1, len(mv) // 300_000)]).query(al["s"] * src @ al["R"].T + al["t"])
        diag = float(np.linalg.norm(mv.max(0) - mv.min(0)))
        al.update({"residual_median_m": float(np.median(d)), "residual_p90_m": float(np.percentile(d, 90)),
                   "inlier_frac_5cm": float((d < 0.05).mean()), "inlier_frac_rel": float((d < 0.02 * diag).mean()),
                   "scene_diag_m": diag})
        return al, mv

    (al, mv) = stage("align", align)
    if al["method"] == "icp":
        ok = al["residual_median_m"] < 0.03 * al["scene_diag_m"] and al["inlier_frac_rel"] > 0.5
    else:  # pose is exact; the scale peak must stand clearly above chance
        ok = al["raw_to_scene_residual_m"] < 0.05 and al["scale_peak_hit"] > 0.15 and al["scale_peak_hit"] > 3 * al["scale_chance_hit"]
    log.info("alignment (%s): scale=%.4f median residual=%.3fm p90=%.3fm (scene diag %.1fm) -> %s",
             al["method"], al["s"], al["residual_median_m"], al["residual_p90_m"], al["scene_diag_m"], "OK" if ok else "POOR")
    # sanity: a walkthrough camera must not sink below the floor (a wrong scale/tilt shows up here first)
    low = min((al["s"] * al["R"] @ _cam_center(im) + al["t"])[1] for im in rec.images.values())
    floor_y = json.loads(meta_path.read_text()).get("floor_y")
    if floor_y is not None and low < floor_y - 0.3:
        log.info("camera path dips %.2f m below the floor: alignment is not self-consistent", floor_y - low)
        ok = False
    al["lowest_camera_y_m"] = float(low)
    if not ok and not force:
        stale = json.loads(meta_path.read_text())
        stale["splat"] = None  # never leave a previous (possibly misaligned) splat advertised
        stale["splat_alignment"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in al.items() if k not in ("R", "t")}
        meta_path.write_text(json.dumps(stale, indent=2))
        raise RuntimeError(f"splat could not be aligned to scene.glb (residual {al['residual_median_m']:.2f} m); "
                           "rerun with --force to keep it anyway")

    arr, _ = read_ply(ply)
    baked = bake_and_clean(arr, al["s"], al["R"], al["t"], mv.min(0), mv.max(0))
    out = sdir / "splat.ply"
    write_ply(out, baked)
    log.info("splat.ply: %d -> %d splats, %.1f MB", len(arr), len(baked), out.stat().st_size / 1e6)

    meta = json.loads(meta_path.read_text())
    meta["splat"] = {"url": f"/scenes/{scene_id}/splat.ply", "format": "ply", "aligned": bool(ok)}
    meta.setdefault("timings_s", {})["splat"] = timings
    meta["splat_alignment"] = {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in al.items() if k not in ("R", "t")}
    # first/last camera in the scene frame (handy to frame a "view from the video" in the viewer)
    ims = sorted(rec.images.values(), key=lambda im: im.name)
    meta["splat_alignment"]["camera_path"] = [
        {"position": [round(float(x), 3) for x in al["s"] * al["R"] @ _cam_center(im) + al["t"]],
         "forward": [round(float(x), 3) for x in al["R"] @ im.cam_from_world().rotation.matrix().T @ [0, 0, 1]]}
        for im in (ims[0], ims[len(ims) // 2], ims[-1])]
    meta_path.write_text(json.dumps(meta, indent=2))
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return {"timings_s": timings, "alignment": meta["splat_alignment"], "splats": int(len(baked))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an aligned Gaussian splat for a scene (data/scenes/<id>)")
    ap.add_argument("scene_id")
    ap.add_argument("--steps", type=int, default=6000, help="Brush training steps (~7 it/s on M4)")
    ap.add_argument("--max-frames", type=int, default=90)
    ap.add_argument("--max-res", type=int, default=1024)
    ap.add_argument("--keep-work", action="store_true", help="keep splat_work/ (needed for --reuse)")
    ap.add_argument("--reuse", action="store_true", help="skip SfM + training if splat_work/ already has them")
    ap.add_argument("--force", action="store_true", help="write the splat even if alignment looks poor")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        res = build_splat(a.scene_id, a.steps, a.max_frames, a.max_res, a.keep_work, a.force, a.reuse)
    except Exception as e:  # CLI: report, non-zero exit
        print(f"video3d-splat failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
