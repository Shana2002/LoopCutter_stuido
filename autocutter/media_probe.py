from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .models import VideoInfo


def _subprocess_flags() -> int:
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        return subprocess.CREATE_NO_WINDOW
    return 0


def find_executable(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    raise FileNotFoundError(
        f"'{name}' executable was not found on PATH. Please install FFmpeg and ensure '{name}' is accessible."
    )


def _parse_fraction(value: str) -> float:
    if not value or value == "0/0":
        return 30.0
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            d = float(den)
            return float(num) / d if d != 0 else 30.0
        except ValueError:
            return 30.0
    try:
        return float(value)
    except ValueError:
        return 30.0


def probe_video(video_path: str | Path) -> VideoInfo:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    ffprobe = find_executable("ffprobe")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_subprocess_flags(),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip() or 'Unknown error'}")

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise ValueError(f"No video stream found in: {path.name}")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)

    # Respect rotation metadata if present (e.g. 90 / 270 deg)
    rotation = 0
    for sd in video_stream.get("side_data_list", []):
        if "rotation" in sd:
            try:
                rotation = int(round(float(sd["rotation"])))
            except (TypeError, ValueError):
                pass
    tags = video_stream.get("tags") or {}
    if "rotate" in tags and rotation == 0:
        try:
            rotation = int(round(float(tags["rotate"])))
        except (TypeError, ValueError):
            pass

    if abs(rotation) in (90, 270):
        width, height = height, width

    duration_raw = video_stream.get("duration") or fmt.get("duration") or 0
    try:
        duration_sec = float(duration_raw)
    except (TypeError, ValueError):
        duration_sec = 0.0

    fps = _parse_fraction(
        video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "30/1"
    )
    if fps <= 0 or fps > 240:
        fps = 30.0

    size_bytes = 0
    try:
        size_bytes = int(fmt.get("size") or path.stat().st_size)
    except OSError:
        pass

    return VideoInfo(
        path=path,
        duration_sec=duration_sec,
        width=width,
        height=height,
        fps=round(fps, 3),
        has_audio=has_audio,
        codec_name=str(video_stream.get("codec_name") or ""),
        file_size_bytes=size_bytes,
    )


def extract_frame_png(
    video_path: str | Path,
    timestamp_sec: float = 0.0,
    max_width: Optional[int] = 720,
) -> bytes:
    """Extract a single frame at `timestamp_sec` as PNG bytes."""
    ffmpeg = find_executable("ffmpeg")
    ts = max(0.0, float(timestamp_sec))
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{ts:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
    ]
    if max_width and max_width > 0:
        cmd.extend(["-vf", f"scale={max_width}:-2"])
    cmd.extend(["-f", "image2pipe", "-vcodec", "png", "pipe:1"])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        creationflags=_subprocess_flags(),
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Failed to extract preview frame: {stderr or 'No image data'}")
    return proc.stdout
