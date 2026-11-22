"""Atomic file replacement that tolerates readers.

Every output is written to a temporary file first and then moved over the
target, so a reader never sees a half-written file.  On Windows the move fails
with ``PermissionError`` while another process has the target open (the
scheduler reading a worker's status file, a monitor reading the progress
file); such readers hold the file for a moment only, so the move is retried.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

#: How long a replacement keeps retrying while the target is open elsewhere.
RETRY_SECONDS = 10.0
#: Pause between two attempts.
RETRY_INTERVAL = 0.05


def replace_file(tmp: Path, path: Path, retry_seconds: float = RETRY_SECONDS) -> None:
    """``os.replace(tmp, path)``, retried for up to ``retry_seconds`` while ``path`` is busy."""
    deadline = time.monotonic() + retry_seconds
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(RETRY_INTERVAL)
