"""Folder / file naming and safe writing."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_name(name: str, max_len: int = 80, fallback: str = "untitled") -> str:
    """Make ``name`` usable as a file/folder name on Windows, macOS and Linux."""
    s = _INVALID.sub(" ", name)
    s = s.replace("|", " ").replace("#", "")
    s = re.sub(r"\s+", " ", s).strip(" .")
    if len(s) > max_len:
        cut = s[:max_len]
        # Prefer cutting on a word boundary.
        s = cut.rsplit(" ", 1)[0] if " " in cut[max_len // 2 :] else cut
        s = s.strip(" .-")
    if not s:
        s = fallback
    if s.split(".")[0].upper() in _RESERVED:
        s = f"_{s}"
    return s


def video_folder(parent: Path, title: str, video_id: str, prefix: str = "") -> Path:
    """Folder for one video. Reuses the folder if it already belongs to this video,
    otherwise adds the video id so two videos with the same title never collide."""
    base = safe_name(f"{prefix}{title}")
    folder = parent / base
    if folder.exists():
        meta = folder / "metadata.json"
        try:
            owner = json.loads(meta.read_text(encoding="utf-8")).get("video", {}).get("video_id")
        except (OSError, ValueError):
            owner = None
        if owner != video_id:
            folder = parent / f"{base} [{video_id}]"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + rename so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
