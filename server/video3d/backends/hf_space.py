"""MapAnything via the free facebook/map-anything ZeroGPU Space (gradio_client).

Never call the Space's /measure endpoint: it hangs.
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from ..config import FFMPEG, get_env
from . import BackendError, Progress, QuotaError

SPACE = "facebook/map-anything"
NAME = "hf-space"
# gradio_client's default httpx read timeout (~84 s) is too short for the Space's queue plus GPU run.
HTTP_TIMEOUT_S = 900
# We upload pre-extracted JPG frames, not the video: the Space's own video decoding took 220-613 s for a phone clip,
# the same frames as images take about 50 s.
FRAME_MAX_EDGE = 960

_QUOTA_RE = re.compile(
    r"quota|zerogpu|no gpu|gpu.*(unavailable|busy)|queue is full|too many requests|rate.?limit|429",
    re.IGNORECASE)


def classify_error(exc: BaseException) -> BackendError:
    """Turn whatever gradio_client raised into a BackendError (QuotaError for quota/queue problems)."""
    text = f"{type(exc).__name__}: {exc}"
    if _QUOTA_RE.search(text):
        hint = "" if get_env("HF_TOKEN") else " Set HF_TOKEN (env or server/.env) to a Hugging Face token for more quota."
        return QuotaError(f"The Hugging Face Space refused the job (GPU quota or queue full): {exc}{hint}")
    return BackendError(f"The Hugging Face Space failed: {exc}")


def extract_frames(video: Path, out_dir: Path, interval: float, num_frames: int, max_edge: int = FRAME_MAX_EDGE
                   ) -> list[Path]:
    """JPG frames at ``interval`` seconds, at most ``num_frames``, long edge ``max_edge`` (autorotate applies)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = f"scale='if(gt(iw,ih),min({max_edge},iw),-2)':'if(gt(iw,ih),-2,min({max_edge},ih))'"
    p = subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(video), "-vf", f"fps=1/{interval},{scale}",
                        "-frames:v", str(num_frames), "-q:v", "3", str(out_dir / "f_%03d.jpg")],
                       capture_output=True, text=True)
    frames = sorted(out_dir.glob("f_*.jpg"))
    if p.returncode != 0 or not frames:
        raise BackendError(f"Could not extract frames from the video: {p.stderr.strip()[-200:] or 'no frames'}")
    return frames


def _fraction(elapsed: float, lo: float, hi: float, tau: float) -> float:
    """Asymptotic progress between lo and hi: the Space gives no real percentage."""
    return lo + (hi - lo) * (1 - math.exp(-elapsed / tau))


def _wait(job, status: str, label: str, lo: float, hi: float, tau: float, progress: Progress,
          marks: dict | None = None) -> object:
    """Poll until the job is done. ``marks["finished_s"]`` records when the Space reported FINISHED, so the
    time after that (result download) can be told apart from queue + GPU time."""
    start = time.time()
    while not job.done():
        msg = f"{label} ({int(time.time() - start)} s)"
        try:
            st = job.status()
            if marks is not None and "finished_s" not in marks and getattr(st.code, "name", "") == "FINISHED":
                marks["finished_s"] = time.time() - start
            if st.rank is not None and st.rank > 0:
                msg = f"{label} (queue position {st.rank + 1} of {st.queue_size}, {int(time.time() - start)} s)"
        except Exception:
            pass
        progress(status, msg, _fraction(time.time() - start, lo, hi, tau))
        time.sleep(1.0)
    return job.result()


def reconstruct(video: Path, interval: float, num_frames: int, out_glb: Path, progress: Progress,
                work_dir: Path) -> dict:
    """Upload sampled frames, run the Space, copy its GLB to ``out_glb``. Returns ``{"num_frames": int}``."""
    try:
        from gradio_client import Client, handle_file
    except ImportError as e:  # pragma: no cover
        raise BackendError(f"gradio_client is not installed: {e}")

    token = get_env("HF_TOKEN")
    dl = work_dir / "_hf_download"
    dl.mkdir(parents=True, exist_ok=True)
    try:
        progress("uploading", "Connecting to the Hugging Face Space", 0.12)
        client = Client(SPACE, token=token, verbose=False, download_files=str(dl),  # same Client = same session state
                        httpx_kwargs={"timeout": httpx.Timeout(HTTP_TIMEOUT_S, connect=30)})
        frames = extract_frames(video, work_dir / "_frames", interval, num_frames)
        # `interval` only matters for video uploads; the Space ignores it for images
        job = client.submit(files=[handle_file(str(f)) for f in frames], interval=1.0,
                            api_name="/update_gallery_on_unified_upload")
        t = time.time()
        _wait(job, "uploading", f"Uploading {len(frames)} frames to the Space", 0.12, 0.3, 40, progress)
        stats = {"gallery_s": round(time.time() - t, 1)}

        job = client.submit(frame_filter="All", show_cam=False, filter_black_bg=False, filter_white_bg=False,
                            apply_mask=True, show_mesh=True, api_name="/gradio_demo")
        t, marks = time.time(), {}
        result = _wait(job, "reconstructing", "Reconstructing 3D on the Space GPU", 0.3, 0.9, 80, progress, marks)
        total = time.time() - t
        stats["gpu_s"] = round(marks.get("finished_s", total), 1)      # queue + GPU, until the Space says FINISHED
        stats["download_s"] = round(total - marks.get("finished_s", total), 1)  # then fetching the GLB
        glb = result[0] if isinstance(result, (list, tuple)) else result
        if not glb or not Path(str(glb)).exists():
            raise BackendError(f"The Space returned no GLB file (got {str(result)[:200]!r}).")
        shutil.copy(str(glb), out_glb)
        stats["glb_mb"] = round(out_glb.stat().st_size / 1e6, 1)
        return {"num_frames": len(frames), "stats": {"space": stats}}
    except BackendError:
        raise
    except Exception as e:
        raise classify_error(e) from e
    finally:
        shutil.rmtree(dl, ignore_errors=True)
        shutil.rmtree(work_dir / "_frames", ignore_errors=True)
