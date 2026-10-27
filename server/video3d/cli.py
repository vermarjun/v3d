"""``video3d-ingest``: run the pipeline without the server, e.g. to pre-build the demo cache.

    video3d-ingest clip.mp4 --name "Lab bench"
    video3d-ingest --raw-glb fixtures/sample_raw.glb --video fixtures/sample.mp4 --name "Robot lab (sample)"
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .pipeline import Store, now_iso, run_job, scene_id_for_file


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="video3d-ingest", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", type=Path, help="video file to reconstruct (calls the backends)")
    ap.add_argument("--name", help="display name for the scene")
    ap.add_argument("--raw-glb", type=Path, help="skip the backend: postprocess this existing raw GLB only")
    ap.add_argument("--video", dest="video_opt", type=Path, help="the video the raw GLB came from (with --raw-glb)")
    args = ap.parse_args(argv)

    video = args.video_opt or args.video
    if video is None or not video.exists():
        ap.error("a video file is required (positional, or --video with --raw-glb)")
    if args.raw_glb and not args.raw_glb.exists():
        ap.error(f"--raw-glb not found: {args.raw_glb}")

    sid = scene_id_for_file(video)
    store = Store()
    d = store.dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    name = args.name or video.stem
    old = store.read(sid) or {}
    store.write(sid, id=sid, name=name, status="queued", stage_message="Queued", progress=0.0, error=None,
                created_at=old.get("created_at") or now_iso())
    source = d / f"upload{video.suffix.lower() or '.mp4'}"
    shutil.copy(video, source)

    print(f"scene {sid} -> {d}", file=sys.stderr)
    run_job(store, sid, source, raw_glb=args.raw_glb)
    st = store.read(sid)
    if st["status"] != "done":
        print(f"FAILED: {st['error']}", file=sys.stderr)
        return 1
    print(json.dumps(json.loads((d / "meta.json").read_text()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
