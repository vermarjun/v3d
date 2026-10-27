"""Paths and environment lookup shared by the server, pipeline and CLI."""
from __future__ import annotations

import os
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
ROOT = SERVER_DIR.parent
WEB_DIR = ROOT / "web"
FFMPEG = "/opt/homebrew/bin/ffmpeg" if Path("/opt/homebrew/bin/ffmpeg").exists() else "ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe" if Path("/opt/homebrew/bin/ffprobe").exists() else "ffprobe"


def scenes_dir() -> Path:
    """data/scenes, or $VIDEO3D_DATA/scenes (used by tests)."""
    base = Path(os.environ["VIDEO3D_DATA"]) if os.environ.get("VIDEO3D_DATA") else ROOT / "data"
    return base / "scenes"


def _dotenv() -> dict[str, str]:
    path = SERVER_DIR / ".env"
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    return out


def get_env(name: str) -> str | None:
    """Environment variable first, then server/.env. Empty strings count as unset."""
    return os.environ.get(name) or _dotenv().get(name) or None
