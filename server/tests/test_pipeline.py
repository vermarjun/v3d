import hashlib
import json

import pytest

from video3d import pipeline
from video3d.backends import BackendError, QuotaError, hf_space, local, replicate


def test_scene_id_is_first_12_hex_of_sha256(tmp_path):
    data = b"hello video"
    expect = hashlib.sha256(data).hexdigest()[:12]
    p = tmp_path / "a.mp4"
    p.write_bytes(data)
    assert pipeline.scene_id_for_bytes(data) == expect == pipeline.scene_id_for_file(p)
    assert len(expect) == 12


@pytest.mark.parametrize("duration", [3.5, 9.2, 30, 0.5, 600])
def test_plan_frames_is_about_16_and_clamped(duration):
    interval, frames = pipeline.plan_frames(duration)
    assert 0.1 <= interval <= 5.0
    assert frames == round(duration / interval) or interval in (0.1, 5.0)
    if duration in (3.5, 9.2, 30):
        assert 15 <= frames <= 17
    assert frames <= pipeline.MAX_FRAMES == 24 and pipeline.TARGET_FRAMES == 16


def test_status_persists_and_interrupted_jobs_fail(data_dir):
    store = pipeline.Store()
    store.write("aaa", name="Done one", status="done", progress=1.0)
    store.write("bbb", name="Was running", status="reconstructing", progress=0.5, created_at="2099-01-01")
    store.write("ccc", name="Queued", status="queued")

    fresh = pipeline.Store()  # "after a restart"
    assert fresh.read("aaa")["status"] == "done"
    assert fresh.mark_interrupted() == 2
    assert fresh.read("aaa")["status"] == "done"
    b = fresh.read("bbb")
    assert b["status"] == "failed" and "restarted" in b["error"]
    assert fresh.list_all()[0]["id"] == "bbb"  # newest first
    assert json.loads((data_dir / "scenes" / "bbb" / "status.json").read_text())["status"] == "failed"


def test_bad_video_fails_with_readable_error(data_dir):
    store = pipeline.Store()
    store.write("bad", name="Bad", status="queued")
    src = data_dir / "scenes" / "bad" / "upload.mp4"
    src.write_bytes(b"this is not a video")
    pipeline.run_job(store, "bad", src)
    st = store.read("bad")
    assert st["status"] == "failed" and st["error"]
    assert "ffmpeg" in st["error"] and "Traceback" not in st["error"]


def test_hf_quota_errors_are_typed():
    assert isinstance(hf_space.classify_error(RuntimeError("You have exceeded your GPU quota (60s requested)")), QuotaError)
    assert isinstance(hf_space.classify_error(Exception("ZeroGPU worker error: No GPU was available")), QuotaError)
    other = hf_space.classify_error(ValueError("boom"))
    assert isinstance(other, BackendError) and not isinstance(other, QuotaError)


def test_no_replicate_token_gives_actionable_error(data_dir, monkeypatch):
    monkeypatch.setattr(pipeline, "get_env", lambda name: "hf-space" if name == "VIDEO3D_BACKEND" else None)
    monkeypatch.setattr(hf_space, "reconstruct", lambda *a, **k: (_ for _ in ()).throw(QuotaError("quota gone")))
    with pytest.raises(BackendError) as e:
        pipeline._reconstruct(pipeline.Store(), "x", data_dir / "v.mp4", {"frames": 100}, 0.2, 24, data_dir / "raw.glb",
                              lambda *a, **k: None)
    assert "REPLICATE_API_TOKEN" in str(e.value) and "quota gone" in str(e.value)


def test_falls_back_to_replicate_when_token_set(data_dir, monkeypatch):
    env = {"REPLICATE_API_TOKEN": "tok", "VIDEO3D_BACKEND": "hf-space"}
    monkeypatch.setattr(pipeline, "get_env", env.get)
    monkeypatch.setattr(hf_space, "reconstruct", lambda *a, **k: (_ for _ in ()).throw(QuotaError("quota gone")))
    monkeypatch.setattr(replicate, "reconstruct", lambda *a, **k: {"num_frames": 24})
    res = pipeline._reconstruct(pipeline.Store(), "x", data_dir / "v.mp4", {"frames": 100}, 0.2, 24,
                                data_dir / "raw.glb", lambda *a, **k: None)
    assert (res["backend"], res["num_frames"]) == ("replicate", 24)


def test_backend_order(monkeypatch):
    env: dict = {}
    monkeypatch.setattr(pipeline, "get_env", env.get)
    monkeypatch.setattr(local, "available", lambda: True)
    assert pipeline.backend_order() == ["hf-space", "local"]            # no Replicate token: skipped
    env["REPLICATE_API_TOKEN"] = "t"
    assert pipeline.backend_order("auto") == ["hf-space", "replicate", "local"]
    assert pipeline.backend_order("local") == ["local"]
    assert pipeline.backend_order("replicate") == ["replicate"]
    assert pipeline.backend_order("hf-space") == ["hf-space", "replicate"]
    monkeypatch.setattr(local, "available", lambda: False)
    assert pipeline.backend_order("auto") == ["hf-space", "replicate"]
    with pytest.raises(BackendError):
        pipeline.backend_order("gpu-please")


def test_auto_tries_space_first_then_falls_through_to_local(data_dir, monkeypatch):
    monkeypatch.setattr(pipeline, "get_env", {}.get)
    monkeypatch.setattr(local, "available", lambda: True)
    calls = []

    def space(*a, **k):
        calls.append("hf-space")
        raise QuotaError("quota gone")

    def on_device(*a, **k):
        calls.append("local")
        return {"num_frames": 5}

    monkeypatch.setattr(hf_space, "reconstruct", space)
    monkeypatch.setattr(local, "reconstruct", on_device)
    res = pipeline._reconstruct(pipeline.Store(), "x", data_dir / "v.mp4", {"frames": 100}, 0.2, 24,
                                data_dir / "raw.glb", lambda *a, **k: None)
    assert calls == ["hf-space", "local"] and res["backend"] == "local"


def test_local_pose_conversion_matches_glb_frame():
    import numpy as np
    from video3d.backends.local import cameras_from_poses

    p0 = np.eye(4)
    p0[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # first camera rotated 90 degrees about z, at (2, 0, 0)
    p0[:3, 3] = [2, 0, 0]
    p1 = np.eye(4)
    p1[:3, 3] = [2, 1, 0]
    cams = cameras_from_poses(np.stack([p0, p1]))
    assert np.allclose(cams[0]["position"], [0, 0, 0]) and np.allclose(cams[0]["forward"], [0, 0, -1])
    assert len(cams) == 2 and np.isclose(np.linalg.norm(cams[1]["forward"]), 1.0)


def test_replicate_request_building():
    schema = {"inputs", "sampling_rate", "return_mesh", "return_pcd", "normals", "to_base64"}
    inp = replicate.build_input("https://api.replicate.com/v1/files/x", total_frames=175, target_frames=24,
                                schema_input_names=schema)
    assert inp == {"inputs": ["https://api.replicate.com/v1/files/x"], "sampling_rate": 7, "return_mesh": True,
                   "return_pcd": False, "normals": False}
    # keys the live schema doesn't know are not sent
    assert "sampling_rate" not in replicate.build_input("u", 100, 24, {"inputs"})
    assert replicate.build_input("u", 10, 24, {"inputs", "sampling_rate"})["sampling_rate"] == 1
    assert replicate.build_prediction("v123", inp) == {"version": "v123", "input": inp}


def test_replicate_output_parsing():
    assert replicate.pick_mesh_url({"mesh": "https://x/mesh.glb", "data": [], "point_cloud": None}) == "https://x/mesh.glb"
    assert replicate.pick_mesh_url({"data": ["https://x/a.json", "https://x/b.glb"]}) == "https://x/b.glb"
    with pytest.raises(BackendError):
        replicate.pick_mesh_url({"data": ["https://x/a.json"], "mesh": None})
