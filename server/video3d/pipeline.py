"""Job runner: video -> backend (HF Space, else Replicate) -> postprocess -> data/scenes/<id>/.

One background worker thread, jobs run one at a time. Every state change is written to status.json so a server
restart shows finished and failed scenes correctly.
"""
from __future__ import annotations

import os

import hashlib
import json
import logging
import queue
import shutil
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import postprocess
from .backends import BackendError, hf_space, local, replicate
from .config import FFMPEG, FFPROBE, get_env, scenes_dir

log = logging.getLogger("video3d")

TARGET_FRAMES = int(os.environ.get("VIDEO3D_FRAMES") or 16)
MIN_FRAMES, MAX_FRAMES = 8, max(24, TARGET_FRAMES)
MIN_INTERVAL, MAX_INTERVAL = 0.1, 5.0  # limits of the Space's interval slider
ACTIVE = ("queued", "uploading", "reconstructing", "postprocessing")


# ---------------------------------------------------------------- small helpers

def scene_id_for_file(path: Path) -> str:
    """First 12 hex chars of sha256 of the file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def scene_id_for_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def plan_frames(duration_s: float, target: int = TARGET_FRAMES) -> tuple[float, int]:
    """Choose the Space's sampling interval (seconds) for about ``target`` frames, clamped to 8..24 frames.

    Returns (interval, expected frame count).
    """
    duration_s = max(duration_s, 0.1)
    target = min(MAX_FRAMES, max(MIN_FRAMES, target))
    interval = duration_s / target
    interval = min(duration_s / MIN_FRAMES, max(duration_s / MAX_FRAMES, interval))  # keeps frames in 8..24
    interval = min(MAX_INTERVAL, max(MIN_INTERVAL, interval))  # the Space slider's own range
    frames = min(MAX_FRAMES, max(1, round(duration_s / interval)))
    return round(interval, 3), frames


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise BackendError(f"{Path(cmd[0]).name} failed: {p.stderr.strip()[-300:] or 'unknown error'}")
    return p.stdout


def probe_video(path: Path) -> dict:
    """{duration, fps, frames, width, height} from ffprobe."""
    out = _run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=width,height,r_frame_rate,duration,nb_frames:format=duration", "-of", "json", str(path)])
    info = json.loads(out)
    st = (info.get("streams") or [{}])[0]
    if not st:
        raise BackendError("The uploaded file has no video stream. Upload an mp4/mov/webm video.")
    num, _, den = str(st.get("r_frame_rate", "25/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 25.0
    duration = float(st.get("duration") or info.get("format", {}).get("duration") or 0)
    frames = int(st["nb_frames"]) if str(st.get("nb_frames", "")).isdigit() else round(duration * fps)
    return {"duration": duration, "fps": fps, "frames": frames, "width": st.get("width"), "height": st.get("height")}


def downscale_video(src: Path, dst: Path, max_edge: int = 1280) -> None:
    """Max ``max_edge`` px on the long edge, audio stripped, h264 mp4 (plays in the browser too)."""
    scale = (f"scale='if(gt(iw,ih),min({max_edge},iw),-2)':'if(gt(iw,ih),-2,min({max_edge},ih))'")
    _run([FFMPEG, "-v", "error", "-y", "-i", str(src), "-an", "-vf", scale, "-c:v", "libx264", "-preset", "veryfast",
          "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)])


def make_thumb(video: Path, dst: Path, duration_s: float) -> None:
    _run([FFMPEG, "-v", "error", "-y", "-ss", f"{max(duration_s, 0) / 2:.2f}", "-i", str(video), "-frames:v", "1",
          "-vf", "scale=480:-2", "-q:v", "4", str(dst)])


# ---------------------------------------------------------------- status persistence

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """status.json files under scenes_dir(). Writes are atomic; a lock serialises them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def dir(self, sid: str) -> Path:
        return scenes_dir() / sid

    def read(self, sid: str) -> dict | None:
        try:
            return json.loads((self.dir(sid) / "status.json").read_text())
        except (OSError, ValueError):
            return None

    def write(self, sid: str, **fields) -> dict:
        with self._lock:
            d = self.dir(sid)
            d.mkdir(parents=True, exist_ok=True)
            st = self.read(sid) or {"id": sid, "name": sid, "status": "queued", "stage_message": "", "progress": 0.0,
                                    "backend": None, "error": None, "created_at": now_iso()}
            st.update(fields)
            tmp = d / "status.json.tmp"
            tmp.write_text(json.dumps(st, indent=2))
            tmp.replace(d / "status.json")
            return st

    def list_all(self) -> list[dict]:
        root = scenes_dir()
        out = []
        for d in root.iterdir() if root.exists() else []:
            st = self.read(d.name) if d.is_dir() else None
            if st:
                out.append(st)
        return sorted(out, key=lambda s: s.get("created_at", ""), reverse=True)

    def mark_interrupted(self) -> int:
        """At startup: any job that was in flight when the server stopped can never finish."""
        n = 0
        for st in self.list_all():
            if st.get("status") in ACTIVE:
                self.write(st["id"], status="failed", progress=st.get("progress", 0.0),
                           stage_message="Interrupted", error="The server restarted while this job was running. "
                                                            "Upload the video again to retry.")
                n += 1
        return n


# ---------------------------------------------------------------- the job itself

def run_job(store: Store, sid: str, source: Path, raw_glb: Path | None = None) -> None:
    """Run one scene to completion. Never raises: failures are written to status.json.

    ``source`` is the uploaded video. If ``raw_glb`` is given the backend call is skipped (postprocess only).
    """
    d = store.dir(sid)
    timings: dict[str, float] = {}

    def progress(status: str, msg: str, frac: float, **extra) -> None:
        store.write(sid, status=status, stage_message=msg, progress=round(min(max(frac, 0.0), 0.99), 3), **extra)

    try:
        progress("uploading", "Preparing video", 0.02)
        t = time.time()
        video = d / "input.mp4"
        if source.resolve() != video.resolve():
            downscale_video(source, video)
            if source.parent == d:  # an upload we own: drop the original, input.mp4 replaces it
                source.unlink(missing_ok=True)
        info = probe_video(video)
        interval, n_frames = plan_frames(info["duration"])
        make_thumb(video, d / "thumb.jpg", info["duration"])
        timings["prepare"] = round(time.time() - t, 2)

        backend_name, cameras, extra_stats = None, None, None
        if raw_glb is not None:
            if raw_glb.resolve() != (d / "raw.glb").resolve():
                shutil.copy(raw_glb, d / "raw.glb")
            backend_name = hf_space.NAME  # the fixture was produced by the Space
            n_used = None
        else:
            t = time.time()
            res = _reconstruct(store, sid, video, info, interval, n_frames, d / "raw.glb", progress)
            backend_name, n_used = res["backend"], res["num_frames"]
            cameras, extra_stats = res.get("cameras"), res.get("stats")
            timings["reconstruct"] = round(time.time() - t, 2)

        progress("postprocessing", "Cleaning and aligning the mesh", 0.92, backend=backend_name)
        name = (store.read(sid) or {}).get("name") or sid
        postprocess.process(d / "raw.glb", d, sid, name, num_frames=n_used, backend=backend_name, timings=timings,
                            cameras=cameras, extra_stats=extra_stats)
        store.write(sid, status="done", stage_message="Done", progress=1.0, error=None)
    except BackendError as e:
        log.warning("job %s failed: %s", sid, e)
        store.write(sid, status="failed", stage_message="Failed", error=str(e))
    except Exception as e:  # noqa: BLE001 - last line of defence for the worker thread
        log.error("job %s crashed:\n%s", sid, traceback.format_exc())
        store.write(sid, status="failed", stage_message="Failed", error=f"Unexpected error: {type(e).__name__}: {e}")


# Order tried when VIDEO3D_BACKEND is "auto" (or unset). Unavailable backends are skipped.
AUTO_ORDER = ("hf-space", "replicate", "local")
MODES = ("auto", "local", "hf-space", "replicate")


def backend_order(mode: str | None = None) -> list[str]:
    """Backends to try, in order, for VIDEO3D_BACKEND=auto|local|hf-space|replicate.

    A forced mode runs that backend alone, except hf-space which still falls back to Replicate when a token is set
    (the original behaviour). In auto mode a backend that is not installed or configured is skipped.
    """
    mode = (mode or get_env("VIDEO3D_BACKEND") or "auto").strip().lower()
    if mode not in MODES:
        raise BackendError(f"VIDEO3D_BACKEND={mode!r} is not one of {', '.join(MODES)}.")
    has_rep = bool(get_env("REPLICATE_API_TOKEN"))
    if mode == "auto":
        return [b for b in AUTO_ORDER if (b != "local" or local.available()) and (b != "replicate" or has_rep)] \
            or ["hf-space"]
    if mode == "hf-space":
        return ["hf-space"] + (["replicate"] if has_rep else [])
    return [mode]


def _reconstruct(store: Store, sid: str, video: Path, info: dict, interval: float, n_frames: int, out_glb: Path,
                 progress) -> dict:
    """Run backends in order until one works. Returns {"backend", "num_frames", "cameras"?, "stats"?}."""

    def on(backend: str):
        def cb(status: str, msg: str, frac: float) -> None:
            progress(status, msg, frac, backend=backend)
        return cb

    runners = {
        "local": lambda cb: local.reconstruct(video, interval, n_frames, out_glb, cb, out_glb.parent),
        "hf-space": lambda cb: hf_space.reconstruct(video, interval, n_frames, out_glb, cb, out_glb.parent),
        "replicate": lambda cb: replicate.reconstruct(video, info["frames"], n_frames, out_glb, cb),
    }
    order = backend_order()
    errors: list[tuple[str, BackendError]] = []
    for name in order:
        progress("uploading", f"Starting the {name} backend", 0.1, backend=name)
        try:
            res = runners[name](on(name))
            return {**res, "backend": name}
        except BackendError as e:
            log.warning("backend %s failed: %s", name, e)
            errors.append((name, e))
    if len(errors) == 1:
        name, err = errors[0]
        hint = ""
        if name == "hf-space":
            hint = f" No fallback is configured: {replicate.TOKEN_HINT}."
        raise BackendError(f"{err}{hint}") from err
    raise BackendError("All backends failed. " + " ".join(f"{n}: {e}" for n, e in errors))


class Runner:
    """Single worker thread draining a FIFO queue of scene ids."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.q: queue.Queue[tuple[str, Path]] = queue.Queue()
        self.thread = threading.Thread(target=self._loop, name="video3d-worker", daemon=True)

    def start(self) -> None:
        self.store.mark_interrupted()
        self.thread.start()

    def submit(self, sid: str, source: Path) -> None:
        self.q.put((sid, source))

    def _loop(self) -> None:
        while True:
            sid, source = self.q.get()
            try:
                run_job(self.store, sid, source)
            finally:
                self.q.task_done()
