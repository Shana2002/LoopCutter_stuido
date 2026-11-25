from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


OUTPUT_WIDTH = 720
OUTPUT_HEIGHT = 1280

FitMode = Literal["cover", "contain", "stretch"]
BackgroundMode = Literal["solid", "blur"]
BadgePosition = Literal[
    "top_area",
    "above_video",
    "video_top_left",
    "video_top_right",
    "below_video",
    "bottom_area",
]


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration_sec: float
    width: int
    height: int
    fps: float
    has_audio: bool
    codec_name: str = ""
    file_size_bytes: int = 0

    @property
    def aspect_ratio(self) -> float:
        if self.height <= 0:
            return 16 / 9
        return self.width / self.height

    @property
    def is_landscape(self) -> bool:
        return self.width >= self.height


@dataclass
class PartSegment:
    index: int  # 1-based part number
    total_parts: int
    start_sec: float
    end_sec: float
    selected: bool = True
    status: str = "Pending"  # Pending | Exporting | Done | Failed | Cancelled
    progress: float = 0.0  # 0.0 .. 1.0
    output_path: Optional[Path] = None
    error_message: str = ""

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(frozen=True)
class LayoutGeometry:
    canvas_width: int
    canvas_height: int
    video_x: int
    video_y: int
    video_width: int
    video_height: int

    @property
    def top_area_height(self) -> int:
        return max(0, self.video_y)

    @property
    def bottom_area_y(self) -> int:
        return self.video_y + self.video_height

    @property
    def bottom_area_height(self) -> int:
        return max(0, self.canvas_height - self.bottom_area_y)


@dataclass
class LayoutSettings:
    top_image_path: Optional[Path] = None
    bottom_image_path: Optional[Path] = None
    fit_mode: FitMode = "cover"
    background_mode: BackgroundMode = "blur"
    background_color: str = "#0F1117"
    vertical_position: float = 0.5  # 0.0 (top) .. 0.5 (center) .. 1.0 (bottom)
    divider_height: int = 0  # optional thin separator bar in px (0..12)
    divider_color: str = "#1E2433"


@dataclass
class BadgeSettings:
    enabled: bool = True
    template: str = "Part {n}"
    position: BadgePosition = "above_video"
    font_size: int = 34
    text_color: str = "#FFFFFF"
    bg_color: str = "#E11D48"
    bg_opacity: float = 0.92
    border_color: str = "#FFFFFF"
    border_width: int = 2
    corner_radius: int = 18
    margin: int = 20


@dataclass
class ExportSettings:
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "exports")
    filename_prefix: str = ""
    crf: int = 21
    preset: str = "veryfast"
    audio_bitrate: str = "192k"
    overwrite: bool = True


def even_round(value: float, minimum: int = 2) -> int:
    rounded = int(round(value / 2.0) * 2)
    return max(minimum, rounded)


def compute_layout_geometry(
    src_width: int,
    src_height: int,
    vertical_position: float = 0.5,
    canvas_width: int = OUTPUT_WIDTH,
    canvas_height: int = OUTPUT_HEIGHT,
) -> LayoutGeometry:
    """Compute exact even-aligned 9:16 720p layout boxes for a given source video size."""
    if src_width <= 0 or src_height <= 0:
        src_width, src_height = 1920, 1080

    aspect = src_width / src_height
    video_width = even_round(canvas_width)
    video_height = even_round(video_width / aspect)

    if video_height > canvas_height:
        video_height = even_round(canvas_height)
        video_width = even_round(min(canvas_width, video_height * aspect))

    pos = max(0.0, min(1.0, vertical_position))
    max_y = max(0, canvas_height - video_height)
    video_y = int(round((max_y * pos) / 2.0) * 2)
    video_y = max(0, min(max_y, video_y))

    max_x = max(0, canvas_width - video_width)
    video_x = int(round((max_x * 0.5) / 2.0) * 2)

    return LayoutGeometry(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        video_x=video_x,
        video_y=video_y,
        video_width=video_width,
        video_height=video_height,
    )


def calculate_parts(
    total_duration_sec: float,
    segment_duration_sec: float,
    start_offset_sec: float = 0.0,
    end_limit_sec: Optional[float] = None,
    min_tail_sec: float = 1.0,
) -> list[PartSegment]:
    """Split a video duration into sequential PartSegments of `segment_duration_sec`."""
    if total_duration_sec <= 0 or segment_duration_sec <= 0:
        return []

    effective_start = max(0.0, min(start_offset_sec, total_duration_sec))
    effective_end = (
        total_duration_sec
        if end_limit_sec is None or end_limit_sec <= 0
        else min(end_limit_sec, total_duration_sec)
    )

    if effective_end - effective_start <= 0.1:
        return []

    raw_ranges: list[tuple[float, float]] = []
    cursor = effective_start
    epsilon = 1e-6

    while cursor + epsilon < effective_end:
        next_end = min(cursor + segment_duration_sec, effective_end)
        remaining = effective_end - next_end
        if 0 < remaining < min_tail_sec and raw_ranges:
            # Tiny trailing sliver gets absorbed into current segment
            next_end = effective_end
        elif next_end - cursor < min_tail_sec and raw_ranges:
            # Merge tiny last slice into previous segment
            prev_start, _ = raw_ranges[-1]
            raw_ranges[-1] = (prev_start, effective_end)
            break

        raw_ranges.append((round(cursor, 3), round(next_end, 3)))
        if next_end >= effective_end - epsilon:
            break
        cursor = next_end

    total_parts = len(raw_ranges)
    return [
        PartSegment(
            index=idx + 1,
            total_parts=total_parts,
            start_sec=start,
            end_sec=end,
        )
        for idx, (start, end) in enumerate(raw_ranges)
    ]


def format_timestamp(seconds: float, include_ms: bool = False) -> str:
    seconds = max(0.0, seconds)
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    if include_ms:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration_short(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_part_label(template: str, part_num: int, total_parts: int) -> str:
    cleaned = (template or "Part {n}").strip()
    if not cleaned:
        cleaned = "Part {n}"
    try:
        return cleaned.format(n=part_num, total=total_parts)
    except (KeyError, ValueError, IndexError):
        return f"{cleaned} {part_num}"


def sanitize_filename(name: str) -> str:
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if c in invalid else c for c in name).strip()
    return cleaned or "video"
