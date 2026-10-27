import sys
import types
from pathlib import Path

import pytest

from video3d import pipeline
from video3d.backends import BackendError, hf_space

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "sample.mp4"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture video missing")
def test_extract_frames_makes_requested_jpgs(tmp_path):
    info = pipeline.probe_video(FIXTURE)
    interval, n = pipeline.plan_frames(info["duration"])
    frames = hf_space.extract_frames(FIXTURE, tmp_path / "f", interval, n)
    assert abs(len(frames) - n) <= 1 and len(frames) <= n
    assert [f.name for f in frames][:2] == ["f_001.jpg", "f_002.jpg"]
    assert all(f.stat().st_size > 1000 for f in frames)
    from PIL import Image

    assert max(Image.open(frames[0]).size) <= hf_space.FRAME_MAX_EDGE


def test_extract_frames_fails_readably(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"nope")
    with pytest.raises(BackendError, match="extract frames"):
        hf_space.extract_frames(bad, tmp_path / "f", 0.5, 8)


def test_space_gets_frames_not_video_and_long_timeout(tmp_path, monkeypatch):
    seen = {}

    class FakeJob:
        def __init__(self, result):
            self._result = result

        def done(self):
            return True

        def result(self):
            return self._result

    class FakeClient:
        def __init__(self, src, **kw):
            seen["src"], seen["kw"] = src, kw

        def submit(self, **kw):
            seen.setdefault("calls", []).append(kw)
            if kw["api_name"] == "/gradio_demo":
                glb = tmp_path / "space.glb"
                glb.write_bytes(b"glb")
                return FakeJob([str(glb), "log"])
            return FakeJob(None)

    fake = types.SimpleNamespace(Client=FakeClient, handle_file=lambda p: ("file", p))
    monkeypatch.setitem(sys.modules, "gradio_client", fake)
    monkeypatch.setattr(hf_space, "get_env", lambda name: "hf_tok" if name == "HF_TOKEN" else None)

    def fake_extract(video, out_dir, interval, n, max_edge=960):
        out_dir.mkdir()
        files = [out_dir / f"f_{i:03d}.jpg" for i in range(1, 4)]
        for f in files:
            f.write_bytes(b"x")
        seen["extract"] = (video, interval, n)
        return files

    monkeypatch.setattr(hf_space, "extract_frames", fake_extract)
    out = tmp_path / "raw.glb"
    res = hf_space.reconstruct(tmp_path / "in.mp4", 0.57, 16, out, lambda *a: None, tmp_path)
    assert res["num_frames"] == 3 and out.read_bytes() == b"glb"
    assert set(res["stats"]["space"]) == {"gallery_s", "gpu_s", "download_s", "glb_mb"}
    assert seen["extract"] == (tmp_path / "in.mp4", 0.57, 16)
    timeout = seen["kw"]["httpx_kwargs"]["timeout"]
    assert timeout.read == 900 and timeout.connect == 30 and seen["kw"]["token"] == "hf_tok"
    first = seen["calls"][0]
    assert first["api_name"] == "/update_gallery_on_unified_upload" and first["interval"] == 1.0
    assert [f[1].endswith(".jpg") for f in first["files"]] == [True] * 3
    assert all(c["api_name"] != "/measure" for c in seen["calls"])
    assert not (tmp_path / "_frames").exists()  # temp frames are cleaned up
