"""Reconstruction backends. Each exposes ``reconstruct(video, interval, num_frames, out_glb, progress) -> dict``."""
from __future__ import annotations

from typing import Callable

# progress(status, stage_message, fraction 0..1)
Progress = Callable[[str, str, float], None]


class BackendError(RuntimeError):
    """A backend could not produce a reconstruction. The message is shown to the user as-is."""


class QuotaError(BackendError):
    """The backend refused because of GPU quota, a full queue or rate limiting."""
