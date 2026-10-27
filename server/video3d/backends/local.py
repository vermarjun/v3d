"""MapAnything on this machine (torch MPS, CPU fallback). Needs ``uv sync --extra local``.

Runs the repo's own ``model.infer`` on frames sampled from the video, then builds the same per-frame vertex-coloured
depth-mesh GLB the Hugging Face Space produces (``predictions_to_glb``, same masks and confidence cut), so the
existing postprocess consumes it unchanged. Unlike the Space it also yields the real camera poses.
"""
from __future__ import annotations

import os
import resource
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from ..config import FFMPEG, get_env
from . import BackendError, Progress

NAME = "local"
MODELS = ("facebook/map-anything", "facebook/map-anything-apache")  # CC-BY-NC first, Apache-2.0 as the fallback
CONF_PERCENTILE = 3.0  # the Space's default confidence cut
INSTALL_HINT = "run `uv sync --project server --extra local` to install torch and MapAnything"

_model = None
_model_device = None


def available() -> bool:
    """True when the optional ``local`` extra is installed (torch + mapanything importable)."""
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in ("torch", "mapanything"))


def _pick_device(torch) -> str:
    forced = get_env("VIDEO3D_LOCAL_DEVICE")
    if forced:
        return forced
    return "mps" if torch.backends.mps.is_available() else "cpu"


def extract_frames(video: Path, out_dir: Path, interval: float, n_frames: int, max_edge: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = f"scale='if(gt(iw,ih),min({max_edge},iw),-2)':'if(gt(iw,ih),-2,min({max_edge},ih))'"
    p = subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(video), "-vf", f"fps=1/{interval},{scale}",
                        "-frames:v", str(n_frames), "-q:v", "2", str(out_dir / "%04d.jpg")],
                       capture_output=True, text=True)
    frames = sorted(out_dir.glob("*.jpg"))
    if p.returncode != 0 or not frames:
        raise BackendError(f"Could not extract frames from the video: {p.stderr.strip()[-200:] or 'no frames'}")
    return frames


def _load_model(torch, device: str, progress: Progress):
    global _model, _model_device
    if _model is not None and _model_device == device:
        return _model
    from mapanything.models import MapAnything

    token = get_env("HF_TOKEN")
    last: Exception | None = None
    for repo in MODELS:
        try:
            progress("reconstructing", f"Loading {repo} weights", 0.12)
            model = MapAnything.from_pretrained(repo, token=token) if token else MapAnything.from_pretrained(repo)
            _model, _model_device = model.to(device).eval(), device
            return _model
        except Exception as e:  # noqa: BLE001 - try the next repo
            last = e
    raise BackendError(f"Could not load MapAnything weights ({last}). Check the network or HF_TOKEN.")


def to_glb_frame(poses: np.ndarray) -> np.ndarray:
    """Camera-to-world poses (S, 4, 4) -> the same poses in the GLB's frame (first camera at the origin, -Z forward).

    Mirrors ``apply_scene_alignment`` in mapanything.utils.hf_utils.viz: glb = F @ inv(P0) @ world, F = diag(1,-1,-1).
    """
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.einsum("ij,jk,skl->sil", flip, np.linalg.inv(poses[0]), poses)


def cameras_from_poses(poses: np.ndarray) -> list[dict]:
    """[{position, forward}] in the raw GLB frame, for postprocess.process(cameras=...)."""
    glb = to_glb_frame(poses)
    return [{"position": [float(x) for x in m[:3, 3]], "forward": [float(x) for x in m[:3, 2]]} for m in glb]


def reconstruct(video: Path, interval: float, num_frames: int, out_glb: Path, progress: Progress, work_dir: Path,
                resolution: int | None = None) -> dict:
    """Frames -> MapAnything -> raw GLB at ``out_glb``. Returns {num_frames, cameras, stats}."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be set before torch is imported
    if not available():
        raise BackendError(f"The local backend is not installed: {INSTALL_HINT}.")
    import torch
    from mapanything.utils.geometry import depthmap_to_world_frame
    from mapanything.utils.hf_utils.viz import predictions_to_glb
    from mapanything.utils.image import load_images

    resolution = resolution or int(get_env("VIDEO3D_LOCAL_RES") or 518)
    stats: dict = {"resolution": resolution}
    frames_dir = work_dir / "_frames"
    try:
        t = time.time()
        progress("uploading", "Sampling frames", 0.05)
        frames = extract_frames(video, frames_dir, interval, num_frames, max_edge=1024)
        stats["extract_s"] = round(time.time() - t, 2)

        t = time.time()
        device = _pick_device(torch)
        stats["device"] = device
        model = _load_model(torch, device, progress)
        stats["load_model_s"] = round(time.time() - t, 2)

        views = load_images([str(f) for f in frames], resolution_set=resolution, verbose=False)
        t = time.time()
        progress("reconstructing", f"Reconstructing {len(views)} frames on {device}", 0.2)
        try:
            with torch.no_grad():
                outputs = model.infer(views, apply_mask=True, mask_edges=True, memory_efficient_inference=True)
        except RuntimeError as e:  # MPS out of memory and friends
            raise BackendError(f"Local inference failed on {device}: {e}") from e
        stats["infer_s"] = round(time.time() - t, 2)

        t = time.time()
        progress("reconstructing", "Building the mesh", 0.85)
        pred: dict[str, list] = {k: [] for k in ("extrinsic", "intrinsic", "world_points", "conf", "depth", "images",
                                                 "final_mask")}
        for o in outputs:
            depth = o["depth_z"][0].squeeze(-1)
            intr, pose = o["intrinsics"][0], o["camera_poses"][0]
            pts, valid = depthmap_to_world_frame(depth, intr, pose)
            mask = o["mask"][0].squeeze(-1).cpu().numpy().astype(bool) if "mask" in o else np.ones(depth.shape, bool)
            pred["extrinsic"].append(pose.cpu().numpy())
            pred["intrinsic"].append(intr.cpu().numpy())
            pred["world_points"].append(pts.cpu().numpy())
            pred["conf"].append(o["conf"][0].squeeze(-1).cpu().numpy())
            pred["depth"].append(depth.cpu().numpy()[..., None])
            pred["images"].append(o["img_no_norm"][0].cpu().numpy())
            pred["final_mask"].append(mask & valid.cpu().numpy())
        predictions = {k: np.stack(v, axis=0) for k, v in pred.items()}
        scene = predictions_to_glb(predictions, filter_by_frames="All", show_cam=False, as_mesh=True,
                                   conf_percentile=CONF_PERCENTILE)
        scene.export(out_glb)
        stats["glb_s"] = round(time.time() - t, 2)
        stats["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)  # bytes on macOS
        return {"num_frames": len(frames), "cameras": cameras_from_poses(predictions["extrinsic"]), "stats": stats}
    except BackendError:
        raise
    except Exception as e:  # noqa: BLE001
        raise BackendError(f"The local backend failed: {type(e).__name__}: {e}") from e
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
