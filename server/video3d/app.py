"""FastAPI app: the HTTP API from CONTRACT.md, the scene cache at /scenes/, and web/ at /."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import WEB_DIR, scenes_dir
from .pipeline import ACTIVE, Runner, Store, now_iso

def photos_to_video(photo_dir: Path, out: Path) -> None:
    """Stitch photos (sorted by name) into a 4 fps H.264 clip, letterboxed to one size."""
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    files = sorted(p for p in photo_dir.iterdir() if p.name.startswith("p_"))
    listing = photo_dir / "list.txt"
    # concat demuxer: one 0.25 s frame per photo (4 fps); mixed JPG/PNG is fine.
    listing.write_text("".join(f"file '{p}'\nduration 0.25\n" for p in files) + f"file '{files[-1]}'\n")
    cmd = [ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
           "-vf", "scale=1280:960:force_original_aspect_ratio=decrease,pad=1280:960:(ow-iw)/2:(oh-ih)/2,fps=4,format=yuv420p",
           "-c:v", "libx264", "-crf", "18", "-f", "mp4", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise HTTPException(400, f"Could not read the photos (use JPG or PNG): {r.stderr.strip()[:300]}")


PLACEHOLDER ="<!doctype html><title>Video3D</title><p>Video3D server is running. The viewer (web/) is not built yet."


class Calibration(BaseModel):
    scale_factor: float = Field(gt=0, lt=1000)


def _meta(sid: str) -> dict | None:
    try:
        return json.loads((scenes_dir() / sid / "meta.json").read_text())
    except (OSError, ValueError):
        return None


def _url(sid: str, filename: str) -> str | None:
    return f"/scenes/{sid}/{filename}" if (scenes_dir() / sid / filename).exists() else None


def _splat_url(sid: str, meta: dict | None) -> str | None:
    splat = (meta or {}).get("splat")
    if splat and splat.get("url"):
        return splat["url"] if (scenes_dir() / sid / Path(splat["url"]).name).exists() else None
    return None


def create_app() -> FastAPI:
    store = Store()
    runner = Runner(store)
    submit_lock = threading.Lock()
    scenes_dir().mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runner.start()
        yield

    app = FastAPI(title="Video3D", lifespan=lifespan)

    @app.post("/api/scenes")
    async def create_scene(video: UploadFile | None = None, photos: list[UploadFile] | None = None,
                           name: str | None = Form(None)):
        if not video and not photos:
            raise HTTPException(400, "Upload a video or at least 3 photos.")
        incoming = scenes_dir() / ".incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        tmp = incoming / uuid.uuid4().hex
        h, size = hashlib.sha256(), 0
        try:
            if photos:
                # Photos become a short slideshow video, so the rest of the pipeline is unchanged.
                if len(photos) < 3:
                    raise HTTPException(400, "Upload at least 3 photos of the scene from different angles.")
                pdir = incoming / (tmp.name + "_photos")
                pdir.mkdir()
                try:
                    for i, p in enumerate(sorted(photos, key=lambda p: p.filename or "")):
                        data = await p.read()
                        h.update(data)
                        size += len(data)
                        (pdir / f"p_{i:04d}{Path(p.filename or '').suffix.lower() or '.jpg'}").write_bytes(data)
                    photos_to_video(pdir, tmp)
                finally:
                    shutil.rmtree(pdir, ignore_errors=True)
                filename = (photos[0].filename or "photos")
            else:
                with open(tmp, "wb") as fh:
                    while chunk := await video.read(1 << 20):
                        h.update(chunk)
                        fh.write(chunk)
                        size += len(chunk)
                filename = video.filename or ""
            if size == 0:
                raise HTTPException(400, "The upload is empty.")
            sid = h.hexdigest()[:12]
            display = (name or "").strip() or Path(filename).stem or sid
            d = scenes_dir() / sid
            with submit_lock:
                st = store.read(sid)
                if (d / "scene.glb").exists() and st and st.get("status") == "done":
                    return JSONResponse({"id": sid, "status": "done", "cached": True}, status_code=200)
                if st and st.get("status") in ACTIVE:
                    return JSONResponse({"id": sid, "status": st["status"], "cached": False}, status_code=202)
                d.mkdir(parents=True, exist_ok=True)
                source = d / ("upload.mp4" if photos else f"upload{Path(filename).suffix.lower() or '.mp4'}")
                shutil.move(str(tmp), source)
                store.write(sid, id=sid, name=display, status="queued", stage_message="Queued", progress=0.0,
                            backend=None, error=None, created_at=now_iso())
                runner.submit(sid, source)
            return JSONResponse({"id": sid, "status": "queued", "cached": False}, status_code=202)
        finally:
            tmp.unlink(missing_ok=True)

    @app.get("/api/scenes")
    def list_scenes():
        out = []
        for st in store.list_all():
            sid = st["id"]
            has_splat = _splat_url(sid, _meta(sid)) is not None
            out.append({"id": sid, "name": st.get("name", sid), "status": st["status"],
                        "created_at": st.get("created_at"), "thumbnail_url": _url(sid, "thumb.jpg"),
                        "has_splat": has_splat})
        return out

    @app.get("/api/scenes/{sid}")
    def get_scene(sid: str):
        st = store.read(sid)
        if not st:
            raise HTTPException(404, "Unknown scene.")
        done = st["status"] == "done"
        meta = _meta(sid) if done else None
        return {
            "id": sid, "name": st.get("name", sid), "status": st["status"],
            "stage_message": st.get("stage_message", ""), "progress": st.get("progress", 0.0),
            "backend": st.get("backend"), "error": st.get("error"),
            "glb_url": _url(sid, "scene.glb") if done else None,
            "meta_url": _url(sid, "meta.json") if done else None,
            "splat_url": _splat_url(sid, meta),
            "video_url": _url(sid, "input.mp4"), "thumbnail_url": _url(sid, "thumb.jpg"),
        }

    @app.post("/api/scenes/{sid}/calibration")
    def calibrate(sid: str, body: Calibration):
        meta = _meta(sid)
        if meta is None:
            raise HTTPException(404, "Scene has no meta.json yet.")
        meta["scale_factor"] = body.scale_factor
        path = scenes_dir() / sid / "meta.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(path)
        return meta

    app.mount("/scenes", StaticFiles(directory=scenes_dir()), name="scenes")

    WEB_DIR.mkdir(exist_ok=True)

    @app.get("/", include_in_schema=False)
    def index():
        page = WEB_DIR / "index.html"  # checked per request: the viewer may be built while the server runs
        return FileResponse(page) if page.exists() else HTMLResponse(PLACEHOLDER)

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
