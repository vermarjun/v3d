import json
import time

import pytest
from fastapi.testclient import TestClient

from video3d import pipeline
from video3d.app import create_app


@pytest.fixture
def client(data_dir):
    with TestClient(create_app()) as c:
        yield c


def test_list_is_empty_json(client):
    r = client.get("/api/scenes")
    assert r.status_code == 200 and r.json() == []


def test_unknown_scene_404(client):
    assert client.get("/api/scenes/nope").status_code == 404


def test_cached_scene_returns_200(client, data_dir):
    data = b"pretend video bytes"
    sid = pipeline.scene_id_for_bytes(data)
    d = data_dir / "scenes" / sid
    d.mkdir(parents=True)
    (d / "scene.glb").write_bytes(b"glb")
    (d / "meta.json").write_text(json.dumps({"id": sid, "scale_factor": 1.0}))
    pipeline.Store().write(sid, name="Cached", status="done", progress=1.0)

    r = client.post("/api/scenes", files={"video": ("a.mp4", data, "video/mp4")}, data={"name": "x"})
    assert r.status_code == 200
    assert r.json() == {"id": sid, "status": "done", "cached": True}
    got = client.get(f"/api/scenes/{sid}").json()
    assert got["glb_url"] == f"/scenes/{sid}/scene.glb" and got["meta_url"] == f"/scenes/{sid}/meta.json"
    assert got["splat_url"] is None and got["video_url"] is None
    assert client.get(f"/scenes/{sid}/scene.glb").content == b"glb"
    listed = client.get("/api/scenes").json()
    assert listed[0]["id"] == sid and listed[0]["has_splat"] is False


def test_bad_upload_becomes_failed_with_error(client):
    r = client.post("/api/scenes", files={"video": ("junk.mp4", b"not a video at all", "video/mp4")})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued" and body["cached"] is False
    for _ in range(100):
        st = client.get(f"/api/scenes/{body['id']}").json()
        if st["status"] in ("failed", "done"):
            break
        time.sleep(0.1)
    assert st["status"] == "failed"
    assert st["error"] and st["glb_url"] is None
    assert st["name"] == "junk"


def test_empty_upload_rejected(client):
    assert client.post("/api/scenes", files={"video": ("e.mp4", b"", "video/mp4")}).status_code == 400


def test_calibration_persists(client, data_dir):
    d = data_dir / "scenes" / "abc"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"id": "abc", "scale_factor": 1.0}))
    r = client.post("/api/scenes/abc/calibration", json={"scale_factor": 1.25})
    assert r.status_code == 200 and r.json()["scale_factor"] == 1.25
    assert json.loads((d / "meta.json").read_text())["scale_factor"] == 1.25
    assert client.post("/api/scenes/abc/calibration", json={"scale_factor": -1}).status_code == 422
    assert client.post("/api/scenes/zzz/calibration", json={"scale_factor": 1.0}).status_code == 404
