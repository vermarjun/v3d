"""Raw MapAnything GLB -> clean, gravity-aligned scene.glb + meta.json.

The raw GLB from the Gradio Space is one vertex-coloured depth mesh per frame. Its scene graph carries the demo's
own OpenCV -> GL flip, so after applying the node transforms the frame is already roughly Y-up (a few degrees of
tilt remain). Steps: merge, drop stretched triangles (flying pixels), drop tiny islands, RANSAC the floor, rotate
it to +Y and y = 0, centre x/z, decimate by vertex clustering if over budget, write scene.glb + meta.json.
numpy + scipy + trimesh only.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse import csgraph

MODEL_NAME = "MapAnything (facebook/map-anything)"
MAX_FACES = 2_500_000
STRETCH_K = 2.0  # drop a face whose longest edge is > K x the local median longest edge
STRETCH_CELL = 0.5  # metres; cell size for the "local" median
MIN_ISLAND_FRAC = 0.001  # islands smaller than this fraction of all faces are dropped...
MIN_ISLAND_FACES = 300  # ...but never fewer than this many faces
UP_MAX_TILT_DEG = 40.0  # a floor candidate's normal must be within this angle of +Y
PLANE_TOL = 0.03  # metres; RANSAC inlier distance
MIN_FLOOR_FRAC = 0.04  # a floor candidate needs at least this share of the points


# ---------------------------------------------------------------- load / merge

def load_raw(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Merged (vertices float64 Nx3, faces int64 Mx3, colours uint8 Nx4, mesh count) with node transforms applied."""
    loaded = trimesh.load(str(path))
    meshes = loaded.dump() if isinstance(loaded, trimesh.Scene) else [loaded]
    vs, fs, cs, off = [], [], [], 0
    for m in meshes:
        if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
            continue
        vs.append(np.asarray(m.vertices, dtype=np.float64))
        fs.append(np.asarray(m.faces, dtype=np.int64) + off)
        cs.append(_colours(m))
        off += len(m.vertices)
    if not vs:
        raise ValueError("raw GLB contains no triangle meshes")
    return np.vstack(vs), np.vstack(fs), np.vstack(cs), len(vs)


def _colours(m: trimesh.Trimesh) -> np.ndarray:
    try:
        vis = m.visual if m.visual.kind == "vertex" else m.visual.to_color()
        c = np.asarray(vis.vertex_colors, dtype=np.uint8)
        if c.shape == (len(m.vertices), 4):
            return c
    except Exception:
        pass
    return np.full((len(m.vertices), 4), 200, dtype=np.uint8)


# ---------------------------------------------------------------- cleaning

def _face_max_edge(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return np.maximum.reduce([
        np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1)])


def drop_stretched(v: np.ndarray, f: np.ndarray, k: float = STRETCH_K, cell: float = STRETCH_CELL) -> np.ndarray:
    """Keep-mask over faces. Edge length grows with depth, so compare against the median in the face's own cell."""
    if len(f) == 0:
        return np.zeros(0, dtype=bool)
    edge = _face_max_edge(v, f)
    cen = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0
    q = np.floor((cen - cen.min(0)) / cell).astype(np.int64)
    dims = q.max(0) + 1
    key = q[:, 0] + dims[0] * (q[:, 1] + dims[1] * q[:, 2])
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    order = np.lexsort((edge, inv))  # sorted by cell, then edge
    starts = np.cumsum(counts) - counts
    med = edge[order][starts + counts // 2]
    local = med[inv]
    # tiny cells have no meaningful median: fall back to the global median
    local = np.where(counts[inv] >= 20, local, np.median(edge))
    return edge <= k * local


def drop_islands(n_verts: int, f: np.ndarray, min_faces: int) -> np.ndarray:
    """Keep-mask over faces: drop faces whose connected component has fewer than ``min_faces`` faces."""
    if len(f) == 0:
        return np.zeros(0, dtype=bool)
    g = sparse.coo_matrix((np.ones(len(f)), (f[:, 0], f[:, 1])), shape=(n_verts, n_verts))
    g = g + sparse.coo_matrix((np.ones(len(f)), (f[:, 1], f[:, 2])), shape=(n_verts, n_verts))
    _, labels = csgraph.connected_components(g, directed=False)
    per_comp = np.bincount(labels[f[:, 0]], minlength=labels.max() + 1)
    return per_comp[labels[f[:, 0]]] >= min_faces


def compact(v: np.ndarray, f: np.ndarray, c: np.ndarray):
    """Drop unreferenced vertices and renumber faces."""
    used = np.zeros(len(v), dtype=bool)
    used[f.ravel()] = True
    remap = np.cumsum(used) - 1
    return v[used], remap[f], c[used]


# ---------------------------------------------------------------- gravity alignment

def _ransac_planes(pts: np.ndarray, rng: np.random.Generator, n_planes: int = 8, iters: int = 500,
                   tol: float = PLANE_TOL):
    """Greedy sequential RANSAC. Yields (normal, d, inlier_count) with n.p + d = 0; counts are of ``pts``."""
    remaining = pts
    out = []
    for _ in range(n_planes):
        if len(remaining) < 100:
            break
        idx = rng.integers(0, len(remaining), size=(iters, 3))
        p0, p1, p2 = remaining[idx[:, 0]], remaining[idx[:, 1]], remaining[idx[:, 2]]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n, axis=1)
        ok = norm > 1e-9
        if not ok.any():
            break
        n = n[ok] / norm[ok, None]
        d = -np.einsum("ij,ij->i", n, p0[ok])
        best_count, best = -1, None
        # score every hypothesis on a subsample, then verify the winner on everything
        sub = remaining[rng.integers(0, len(remaining), size=min(len(remaining), 20000))]
        for i in range(len(n)):
            cnt = int((np.abs(sub @ n[i] + d[i]) < tol).sum())
            if cnt > best_count:
                best_count, best = cnt, i
        ni, di = n[best], d[best]
        inl = np.abs(remaining @ ni + di) < tol
        # refine with least squares on the inliers
        P = remaining[inl]
        cen = P.mean(0)
        nn = np.linalg.svd(P - cen, full_matrices=False)[2][-1]
        di = -float(nn @ cen)
        inl = np.abs(remaining @ nn + di) < tol
        out.append((nn, di, int(inl.sum())))
        remaining = remaining[~inl]
    return out


def find_floor(v: np.ndarray, seed: int = 0, tol: float = PLANE_TOL):
    """Find the floor/table plane. Returns (normal pointing up, d, inlier_fraction) or None.

    Candidates are planes whose normal is within UP_MAX_TILT_DEG of +Y. Among the well-supported ones the lowest wins,
    so a wall-dominant scene still resolves to the floor and not a shelf or a tabletop above it.
    """
    rng = np.random.default_rng(seed)
    n_sub = min(len(v), 200_000)
    pts = v[rng.choice(len(v), n_sub, replace=False)]
    cos_min = np.cos(np.radians(UP_MAX_TILT_DEG))
    cands = []
    for n, d, cnt in _ransac_planes(pts, rng, tol=tol):
        if n[1] < 0:
            n, d = -n, -d
        if n[1] >= cos_min:
            cands.append((n, d, cnt / n_sub))
    cands = [c for c in cands if c[2] >= MIN_FLOOR_FRAC]
    if not cands:
        return None
    biggest = max(c[2] for c in cands)
    cands = [c for c in cands if c[2] >= 0.4 * biggest]
    # lowest plane: height of the plane at the scene's median horizontal position
    mid = np.median(pts, axis=0)
    best = min(cands, key=lambda c: -(c[1] + c[0][0] * mid[0] + c[0][2] * mid[2]) / c[0][1])
    return best


def rotation_to_up(n: np.ndarray) -> np.ndarray:
    """Minimal rotation (Rodrigues) taking unit vector ``n`` to +Y."""
    n = n / np.linalg.norm(n)
    up = np.array([0.0, 1.0, 0.0])
    axis = np.cross(n, up)
    s, c = np.linalg.norm(axis), float(n @ up)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]) / s
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def align_to_floor(v: np.ndarray, plane, extra: np.ndarray | None = None):
    """Rotate so the plane normal is +Y, plane at y = 0, bbox centred on x/z.

    ``extra`` points (e.g. camera positions) get the identical rigid transform but do not affect the centring.
    Returns ``v`` (and ``extra`` if given).
    """
    n, d, _ = plane
    p0 = -d * n  # a point on the plane
    rot = rotation_to_up(n)
    out = (v - p0) @ rot.T
    shift = _xz_centre(out)
    out = out - shift
    if extra is None:
        return out
    return out, (extra - p0) @ rot.T - shift


def _xz_centre(v: np.ndarray) -> np.ndarray:
    """Offset that moves the bbox centre of ``v`` to x = z = 0 (y untouched)."""
    lo, hi = v.min(0), v.max(0)
    return np.array([(lo[0] + hi[0]) / 2, 0.0, (lo[2] + hi[2]) / 2])


# ---------------------------------------------------------------- decimation

def decimate_cluster(v: np.ndarray, f: np.ndarray, c: np.ndarray, max_faces: int):
    """Vertex clustering on a voxel grid until the face count fits. Cheap, numpy-only, welds frames together."""
    if len(f) <= max_faces:
        return v, f, c
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    size = float(np.sqrt(4 * area / (np.sqrt(3) * max_faces)))  # edge length of the target triangle
    lo = v.min(0)
    for _ in range(8):
        q = np.floor((v - lo) / size).astype(np.int64)
        dims = q.max(0) + 1
        key = q[:, 0] + dims[0] * (q[:, 1] + dims[1] * q[:, 2])
        uniq, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv).astype(np.float64)
        nv = np.stack([np.bincount(inv, v[:, i]) for i in range(3)], axis=1) / cnt[:, None]
        nc = np.stack([np.bincount(inv, c[:, i].astype(np.float64)) for i in range(4)], axis=1) / cnt[:, None]
        nf = inv[f]
        keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
        nf = nf[keep]
        # collapse duplicate faces (same three cells, any order)
        _, first = np.unique(np.sort(nf, axis=1), axis=0, return_index=True)
        nf = nf[np.sort(first)]
        if len(nf) <= max_faces:
            break
        size *= max(1.1, (len(nf) / max_faces) ** 0.5)
    nv, nf, nc = compact(nv, nf, np.rint(nc).astype(np.uint8))
    return nv, nf, nc


# ---------------------------------------------------------------- cameras

LOOK_DIST = 1.0  # metres from the camera to the look_at point written to meta.json
FRAME0_CAMERA = {"position": [0.0, 0.0, 0.0], "forward": [0.0, 0.0, -1.0]}


def _camera_points(cameras: list | None) -> np.ndarray:
    """(2n, 3): n positions followed by n look-at points, so one rigid transform maps both."""
    cams = cameras or [FRAME0_CAMERA]
    pos = np.array([c["position"] for c in cams], dtype=np.float64)
    fwd = np.array([c["forward"] for c in cams], dtype=np.float64)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    return np.vstack([pos, pos + LOOK_DIST * fwd])


def _cameras_out(pts: np.ndarray) -> list:
    n = len(pts) // 2
    return [{"position": [round(float(x), 4) for x in pts[i]], "look_at": [round(float(x), 4) for x in pts[n + i]]}
            for i in range(n)]


# ---------------------------------------------------------------- pipeline entry

def process(raw_glb: str | Path, out_dir: str | Path, scene_id: str, name: str, *, num_frames: int | None = None,
            backend: str | None = None, timings: dict | None = None, max_faces: int = MAX_FACES,
            cameras: list | None = None, extra_stats: dict | None = None) -> dict:
    """Run the full postprocess; writes out_dir/scene.glb and out_dir/meta.json and returns the meta dict.

    ``cameras``: recorded camera poses in the raw GLB frame, ``[{"position": [x, y, z], "forward": [x, y, z]}]``.
    The Space's GLB is built around the first camera, which sits at the origin looking down -Z, so that is the
    default when no poses are supplied.
    """
    t0 = time.time()
    out_dir = Path(out_dir)
    v, f, c, n_meshes = load_raw(raw_glb)
    stats = {"raw_faces": int(len(f))}

    f = f[drop_stretched(v, f)]
    stats["after_stretch_faces"] = int(len(f))
    f = f[drop_islands(len(v), f, max(MIN_ISLAND_FACES, int(MIN_ISLAND_FRAC * len(f))))]
    stats["after_islands_faces"] = int(len(f))
    v, f, c = compact(v, f, c)
    if len(f) == 0:
        raise ValueError("no geometry left after cleaning (all triangles were filtered out)")

    plane = find_floor(v)
    cam_pts = _camera_points(cameras)
    if plane is None:
        # a backend that hands over the raw OpenCV frame (y down, z forward) has the floor on the -Y side
        flipped = v * np.array([1.0, -1.0, -1.0])
        plane = find_floor(flipped)
        if plane is not None:
            v = flipped
            cam_pts = cam_pts * np.array([1.0, -1.0, -1.0])
    if plane is not None:
        v, cam_pts = align_to_floor(v, plane, cam_pts)
        floor_y = 0.0
    else:
        shift = _xz_centre(v)
        shift[1] = v[:, 1].min()
        v, cam_pts = v - shift, cam_pts - shift
        floor_y = None

    crop = float(os.environ.get("VIDEO3D_CROP") or 0)
    n_cam = len(cam_pts) // 2
    if crop > 0 and n_cam >= 4:
        # Object-centric captures: keep only the disc the camera orbit looked at, BEFORE decimating,
        # so the triangle budget goes to the object instead of the sparsely seen background.
        pos = cam_pts[:n_cam]
        centre = pos.mean(0)
        radius = crop * np.linalg.norm((pos - centre)[:, [0, 2]], axis=1).mean()
        keep_v = (np.hypot(v[:, 0] - centre[0], v[:, 2] - centre[2]) < radius) & (v[:, 1] > -0.4) & (v[:, 1] < pos[:, 1].max())
        v, f, c = compact(v, f[keep_v[f].all(1)], c)
        stats["crop_radius_m"] = round(float(radius), 3)
    v, f, c = decimate_cluster(v, f, c, max_faces)
    stats["final_faces"] = int(len(f))

    mesh = trimesh.Trimesh(vertices=v.astype(np.float32), faces=f, vertex_colors=c, process=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(out_dir / "scene.glb")

    meta_path = out_dir / "meta.json"
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    t = dict(timings or {})
    t["postprocess"] = round(time.time() - t0, 2)
    meta = {
        "id": scene_id, "name": name, "units": "meters", "up_axis": "+Y", "floor_y": floor_y,
        "bbox_min": [round(float(x), 4) for x in v.min(0)], "bbox_max": [round(float(x), 4) for x in v.max(0)],
        "scale_factor": old.get("scale_factor", 1.0),
        "num_frames": num_frames if num_frames is not None else n_meshes,
        "backend": backend or old.get("backend"), "model": MODEL_NAME, "timings_s": t,
        "cameras": _cameras_out(cam_pts), "splat": old.get("splat"),
        "stats": {**stats, **(extra_stats or {})},
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta
