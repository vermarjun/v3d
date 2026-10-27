"""Fallback: MapAnything on Replicate (vufinder/map-anything, L40S) over plain HTTP.

UNTESTED LIVE (no REPLICATE_API_TOKEN was available when this was written). The input/output schema below comes from
the model's public API page: input ``inputs`` (list of file URLs, required), ``sampling_rate`` (every n-th frame, video
only), ``return_mesh`` / ``return_pcd`` / ``normals`` booleans; output ``{"data": [json urls], "mesh": glb url,
"point_cloud": glb url}``. ``schema_input_names`` lets us confirm the keys against the live schema before sending.
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx

from ..config import get_env
from . import BackendError, Progress, QuotaError

API = "https://api.replicate.com/v1"
MODEL = "vufinder/map-anything"
NAME = "replicate"
TOKEN_HINT = "set REPLICATE_API_TOKEN (env or server/.env) to enable the Replicate fallback"


def build_input(video_url: str, total_frames: int, target_frames: int, schema_input_names: set[str] | None = None
                ) -> dict:
    """Request ``input`` object. ``sampling_rate`` = every n-th frame so about ``target_frames`` are used."""
    payload: dict = {"inputs": [video_url]}
    optional = {
        "sampling_rate": max(1, round(total_frames / max(1, target_frames))),
        "return_mesh": True,
        "return_pcd": False,
        "normals": False,
    }
    for key, value in optional.items():
        if schema_input_names is None or key in schema_input_names:
            payload[key] = value
    return payload


def build_prediction(version: str, input_obj: dict) -> dict:
    return {"version": version, "input": input_obj}


def pick_mesh_url(output: dict | list | str | None) -> str:
    """Find the GLB mesh URL in a prediction output."""
    if isinstance(output, dict):
        for key in ("mesh", "point_cloud"):
            if isinstance(output.get(key), str) and output[key]:
                return output[key]
        for item in output.get("data") or []:
            if isinstance(item, str) and item.lower().endswith(".glb"):
                return item
    if isinstance(output, str) and output:
        return output
    raise BackendError(f"Replicate returned no mesh (output was {str(output)[:200]!r}).")


def _check(resp: httpx.Response, what: str) -> None:
    if resp.status_code in (402, 429):
        raise QuotaError(f"Replicate refused {what} (HTTP {resp.status_code}: billing or rate limit): {resp.text[:200]}")
    if resp.status_code >= 400:
        raise BackendError(f"Replicate {what} failed (HTTP {resp.status_code}): {resp.text[:300]}")


def reconstruct(video: Path, total_frames: int, target_frames: int, out_glb: Path, progress: Progress,
                timeout_s: float = 900) -> dict:
    token = get_env("REPLICATE_API_TOKEN")
    if not token:
        raise BackendError(f"No Replicate token: {TOKEN_HINT}.")
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(headers=headers, timeout=120, follow_redirects=True) as http:
        progress("uploading", "Looking up the Replicate model", 0.1)
        r = http.get(f"{API}/models/{MODEL}")
        _check(r, "model lookup")
        model = r.json()
        version = (model.get("latest_version") or {}).get("id")
        if not version:
            raise BackendError("Replicate model has no latest version.")
        schemas = ((model["latest_version"].get("openapi_schema") or {}).get("components") or {}).get("schemas") or {}
        props = (schemas.get("Input") or {}).get("properties")
        names = set(props) if props else None

        progress("uploading", "Uploading video to Replicate", 0.15)
        with open(video, "rb") as fh:
            r = http.post(f"{API}/files", files={"content": (video.name, fh, "video/mp4")})
        _check(r, "file upload")
        file_url = r.json()["urls"]["get"]

        r = http.post(f"{API}/predictions",
                      json=build_prediction(version, build_input(file_url, total_frames, target_frames, names)))
        _check(r, "prediction")
        pred = r.json()
        start = time.time()
        while pred["status"] not in ("succeeded", "failed", "canceled"):
            if time.time() - start > timeout_s:
                raise BackendError(f"Replicate did not finish within {int(timeout_s)} s.")
            progress("reconstructing", "Reconstructing 3D on Replicate (L40S)",
                     0.3 + 0.6 * min(1.0, (time.time() - start) / 240))
            time.sleep(2)
            pred = http.get(pred["urls"]["get"]).json()
        if pred["status"] != "succeeded":
            raise BackendError(f"Replicate prediction {pred['status']}: {str(pred.get('error'))[:300]}")
        mesh_url = pick_mesh_url(pred.get("output"))
        progress("reconstructing", "Downloading mesh from Replicate", 0.9)
        with http.stream("GET", mesh_url) as resp:
            _check(resp, "mesh download")
            with open(out_glb, "wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
    return {"num_frames": target_frames}
