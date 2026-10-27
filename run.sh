#!/usr/bin/env bash
# Start Video3D on http://127.0.0.1:8000 (serves web/ and the API). Ctrl-C stops it.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
command -v uv >/dev/null || { echo "uv not found: install it from https://docs.astral.sh/uv/" >&2; exit 1; }
[ -d server/.venv ] || uv sync --project server --extra local
mkdir -p data/scenes
URL=http://127.0.0.1:8000
# open the browser once the server answers
( for _ in $(seq 1 50); do curl -sf "$URL/api/scenes" >/dev/null 2>&1 && { open "$URL" 2>/dev/null || true; break; }; sleep 0.2; done ) &
export HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1
exec uv run --project server --extra local uvicorn video3d.app:app --host 127.0.0.1 --port 8000
