"""Atomic file replace that survives a reader holding the old file for a moment (Windows)."""

from __future__ import annotations

import os
import time
from pathlib import Path


def replace_file(tmp: Path, path: Path, tries: int = 10, wait_s: float = 0.5) -> None:
    """os.replace(tmp, path), retried while Windows refuses it because another process (the
    dev server streaming the old file, an antivirus scan) still has `path` open."""
    for i in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == tries - 1:
                raise PermissionError(
                    f"cannot replace {path}: another program has it open. Stop the viewer's "
                    f"dev server (npm run dev) or close the file, then rerun; the new file is "
                    f"{tmp}"
                ) from None
            time.sleep(wait_s)
