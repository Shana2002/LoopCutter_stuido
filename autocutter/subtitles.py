from __future__ import annotations

import re
from pathlib import Path

from .models import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    LayoutGeometry,
    SubtitleCue,
    SubtitlePosition,
    SubtitleSettings,
)


_SRT_TIME_RE = re.compile(
    r"(?:(\d{1,3}):)?(\d{1,2}):(\d{2})[,.](\d{1,3})\s*-->\s*(?:(\d{1,3}):)?(\d{1,2}):(\d{2})[,.](\d{1,3})"
)
_HTML_TAG_RE = re.compile(r"</?(?:i|b|u|s|font|c|ruby|rt|v|lang)\b[^>]*>", re.IGNORECASE)
_ASS_TAG_RE = re.compile(r"\{[^}]*\}")


def _decode_subtitle_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _to_seconds(hours: str | None, minutes: str, seconds: str, millis: str) -> float:
    h = int(hours) if hours else 0
    m = int(minutes)
    s = int(seconds)
    ms = int(millis.ljust(3, "0")[:3])
    return h * 3600.0 + m * 60.0 + s + ms / 1000.0


def _clean_srt_vtt_text(lines: list[str]) -> str:
    joined = "\n".join(line.strip() for line in lines if line.strip())
    joined = _HTML_TAG_RE.sub("", joined)
    return joined.strip()


def _parse_srt_or_vtt(text: str) -> list[SubtitleCue]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", normalized)
    cues: list[SubtitleCue] = []

    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if not lines:
            continue
        first_upper = lines[0].strip().upper()
        if first_upper.startswith("WEBVTT") or first_upper.startswith("NOTE") or first_upper.startswith("STYLE"):
            continue

        time_idx = -1
        match = None
        for i, ln in enumerate(lines[:3]):
            m = _SRT_TIME_RE.search(ln)
            if m:
                time_idx = i
                match = m
                break

        if match is None or time_idx < 0:
            continue

        start = _to_seconds(match.group(1), match.group(2), match.group(3), match.group(4))
        end = _to_seconds(match.group(5), match.group(6), match.group(7), match.group(8))
        if end <= start:
            continue

        cue_text = _clean_srt_vtt_text(lines[time_idx + 1 :])
        if cue_text:
            cues.append(SubtitleCue(start_sec=start, end_sec=end, text=cue_text))

    cues.sort(key=lambda c: (c.start_sec, c.end_sec))
    return cues


def _parse_ass_time(raw: str) -> float:
    parts = raw.strip().split(":")
    if len(parts) != 3:
        return 0.0
    h = int(parts[0])
    m = int(parts[1])
    sec_parts = parts[2].split(".")
    s = int(sec_parts[0])
    cs = int(sec_parts[1].ljust(2, "0")[:2]) if len(sec_parts) > 1 else 0
    return h * 3600.0 + m * 60.0 + s + cs / 100.0


def _parse_ass(text: str) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    in_events = False
    format_fields: list[str] = []

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("["):
            in_events = line.lower() == "[events]"
            continue
        if not in_events:
            continue

        lower = line.lower()
        if lower.startswith("format:"):
            format_fields = [f.strip().lower() for f in line.split(":", 1)[1].split(",")]
            continue

        if lower.startswith("dialogue:"):
            payload = line.split(":", 1)[1].lstrip()
            num_cols = len(format_fields) if format_fields else 10
            cols = payload.split(",", num_cols - 1)
            if len(cols) < 4:
                continue

            if format_fields and "start" in format_fields and "end" in format_fields and "text" in format_fields:
                start_raw = cols[format_fields.index("start")]
                end_raw = cols[format_fields.index("end")]
                txt_raw = cols[format_fields.index("text")]
            else:
                start_raw = cols[1]
                end_raw = cols[2]
                txt_raw = cols[-1]

            start = _parse_ass_time(start_raw)
            end = _parse_ass_time(end_raw)
            if end <= start:
                continue

            clean_txt = _ASS_TAG_RE.sub("", txt_raw)
            clean_txt = clean_txt.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ").strip()
            if clean_txt:
                cues.append(SubtitleCue(start_sec=start, end_sec=end, text=clean_txt))

    cues.sort(key=lambda c: (c.start_sec, c.end_sec))
    return cues


def load_subtitle_cues(path: str | Path) -> list[SubtitleCue]:
    sub_path = Path(path)
    if not sub_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {sub_path}")

    text = _decode_subtitle_bytes(sub_path.read_bytes())
    ext = sub_path.suffix.lower()
    if ext in (".ass", ".ssa"):
        return _parse_ass(text)
    return _parse_srt_or_vtt(text)


def slice_cues_for_part(
    cues: list[SubtitleCue],
    part_start_sec: float,
    part_end_sec: float,
    sync_offset_sec: float = 0.0,
) -> list[SubtitleCue]:
    """Extract and time-shift cues overlapping [part_start_sec, part_end_sec] to 0-based part time."""
    if part_end_sec <= part_start_sec or not cues:
        return []

    sliced: list[SubtitleCue] = []
    for cue in cues:
        shifted_start = cue.start_sec + sync_offset_sec
        shifted_end = cue.end_sec + sync_offset_sec

        clipped_start = max(part_start_sec, shifted_start)
        clipped_end = min(part_end_sec, shifted_end)
        if clipped_end - clipped_start <= 0.03:
            continue

        sliced.append(
            SubtitleCue(
                start_sec=round(clipped_start - part_start_sec, 3),
                end_sec=round(clipped_end - part_start_sec, 3),
                text=cue.text,
            )
        )
    return sliced


def find_active_cue_text(
    cues: list[SubtitleCue],
    timestamp_sec: float,
    sync_offset_sec: float = 0.0,
) -> str:
    """Return active subtitle text at `timestamp_sec` (in source movie timeline)."""
    if not cues:
        return ""
    active: list[str] = []
    for cue in cues:
        s = cue.start_sec + sync_offset_sec
        e = cue.end_sec + sync_offset_sec
        if s <= timestamp_sec <= e:
            active.append(cue.text)
    return "\n".join(active)


def _hex_to_ass_color(hex_color: str, alpha_0_to_255: int = 0) -> str:
    """Convert #RRGGBB to ASS &HAABBGGRR (note: ASS alpha 00=opaque, FF=transparent)."""
    cleaned = (hex_color or "#FFFFFF").strip().lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join(c * 2 for c in cleaned)
    if len(cleaned) != 6:
        return "&H00FFFFFF"
    r = cleaned[0:2].upper()
    g = cleaned[2:4].upper()
    b = cleaned[4:6].upper()
    a = f"{max(0, min(255, alpha_0_to_255)):02X}"
    return f"&H{a}{b}{g}{r}"


def _format_ass_time(seconds: float) -> str:
    total_cs = max(0, int(round(seconds * 100)))
    hours = total_cs // 360_000
    minutes = (total_cs % 360_000) // 6000
    secs = (total_cs % 6000) // 100
    cs = total_cs % 100
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def compute_subtitle_ass_placement(
    position: SubtitlePosition,
    geometry: LayoutGeometry,
) -> tuple[int, int]:
    """Return ASS (Alignment, MarginV) in 720x1280 coordinates."""
    if position == "below_video":
        if geometry.bottom_area_height >= 50:
            return (8, geometry.bottom_area_y + 18)
        return (2, 24)
    if position == "bottom_area":
        if geometry.bottom_area_height >= 50:
            center_margin = max(20, geometry.bottom_area_height // 2 - 20)
            return (2, center_margin)
        return (2, 24)
    if position == "video_center":
        video_mid_y = geometry.video_y + geometry.video_height // 2
        return (5, max(0, OUTPUT_HEIGHT - video_mid_y))
    # Default: "video_bottom"
    dist_from_bottom = max(12, OUTPUT_HEIGHT - geometry.bottom_area_y + 16)
    return (2, dist_from_bottom)


def write_part_ass_file(
    cues: list[SubtitleCue],
    out_path: Path,
    settings: SubtitleSettings,
    geometry: LayoutGeometry,
) -> Path:
    alignment, margin_v = compute_subtitle_ass_placement(settings.position, geometry)
    primary = _hex_to_ass_color(settings.text_color, alpha_0_to_255=0)
    outline = _hex_to_ass_color(settings.outline_color, alpha_0_to_255=0)
    back = _hex_to_ass_color(settings.outline_color, alpha_0_to_255=90 if settings.bg_box else 140)
    border_style = 3 if settings.bg_box else 1
    outline_px = 6 if settings.bg_box else 3
    shadow_px = 0 if settings.bg_box else 1

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUTPUT_WIDTH}
PlayResY: {OUTPUT_HEIGHT}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Segoe UI,{max(14, settings.font_size)},{primary},&H000000FF,{outline},{back},-1,0,0,0,100,100,0,0,{border_style},{outline_px},{shadow_px},{alignment},32,32,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    use_center_pos = settings.position == "video_center"
    center_y = geometry.video_y + geometry.video_height // 2

    for cue in cues:
        escaped_text = cue.text.replace("\n", "\\N")
        if use_center_pos:
            escaped_text = f"{{\\pos({OUTPUT_WIDTH // 2},{center_y})}}{escaped_text}"
        lines.append(
            f"Dialogue: 0,{_format_ass_time(cue.start_sec)},{_format_ass_time(cue.end_sec)},Default,,0,0,0,,{escaped_text}\n"
        )

    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


def write_part_srt_file(cues: list[SubtitleCue], out_path: Path) -> Path:
    blocks: list[str] = []
    for idx, cue in enumerate(cues, start=1):
        blocks.append(
            f"{idx}\n{_format_srt_time(cue.start_sec)} --> {_format_srt_time(cue.end_sec)}\n{cue.text}\n"
        )
    out_path.write_text("\n".join(blocks), encoding="utf-8")
    return out_path


def escape_ffmpeg_filter_path(path: Path) -> str:
    """Escape a filesystem path for single-quoted FFmpeg filter arguments on Windows/POSIX."""
    posix_like = str(path).replace("\\", "/")
    posix_like = posix_like.replace(":", "\\:")
    posix_like = posix_like.replace("'", "\\'")
    return posix_like
