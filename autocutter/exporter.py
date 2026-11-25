from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .compositor import render_badge_overlay, render_zone_image
from .media_probe import find_executable
from .models import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    BadgeSettings,
    ExportSettings,
    LayoutGeometry,
    LayoutSettings,
    PartSegment,
    VideoInfo,
    compute_layout_geometry,
    sanitize_filename,
)


def _subprocess_flags() -> int:
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        return subprocess.CREATE_NO_WINDOW
    return 0


def _ffmpeg_hex(hex_color: str) -> str:
    cleaned = (hex_color or "#000000").strip().lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join(c * 2 for c in cleaned)
    if len(cleaned) != 6:
        return "0x000000"
    return f"0x{cleaned.lower()}"


def build_part_output_path(
    video_info: VideoInfo,
    part: PartSegment,
    export_settings: ExportSettings,
) -> Path:
    stem = sanitize_filename(export_settings.filename_prefix or video_info.path.stem)
    digits = max(2, len(str(part.total_parts)))
    filename = f"{stem}_part_{part.index:0{digits}d}_of_{part.total_parts:0{digits}d}.mp4"
    return export_settings.output_dir / filename


def build_ffmpeg_command(
    ffmpeg_exe: str,
    video_info: VideoInfo,
    part: PartSegment,
    output_path: Path,
    geometry: LayoutGeometry,
    layout: LayoutSettings,
    export_settings: ExportSettings,
    top_png_path: Optional[Path],
    bottom_png_path: Optional[Path],
    badge_png_path: Optional[Path],
) -> list[str]:
    cmd: list[str] = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if export_settings.overwrite else "-n",
        "-ss",
        f"{part.start_sec:.3f}",
        "-t",
        f"{part.duration_sec:.3f}",
        "-i",
        str(video_info.path),
    ]

    input_idx = 1
    top_idx: Optional[int] = None
    bottom_idx: Optional[int] = None
    badge_idx: Optional[int] = None

    if top_png_path is not None:
        cmd.extend(["-i", str(top_png_path)])
        top_idx = input_idx
        input_idx += 1

    if bottom_png_path is not None:
        cmd.extend(["-i", str(bottom_png_path)])
        bottom_idx = input_idx
        input_idx += 1

    if badge_png_path is not None:
        cmd.extend(["-i", str(badge_png_path)])
        badge_idx = input_idx
        input_idx += 1

    filters: list[str] = []
    stage = 0

    if layout.background_mode == "blur":
        filters.append("[0:v]split=2[v_main][v_bg]")
        filters.append(
            f"[v_main]scale={geometry.video_width}:{geometry.video_height}:flags=lanczos,setsar=1[vid_fg]"
        )
        filters.append(
            f"[v_bg]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},boxblur=25:3,eq=brightness=-0.08,setsar=1[bg]"
        )
        filters.append(
            f"[bg][vid_fg]overlay={geometry.video_x}:{geometry.video_y}[s{stage}]"
        )
    else:
        bg_hex = _ffmpeg_hex(layout.background_color)
        filters.append(
            f"[0:v]scale={geometry.video_width}:{geometry.video_height}:flags=lanczos,"
            f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:{geometry.video_x}:{geometry.video_y}:color={bg_hex},"
            f"setsar=1[s{stage}]"
        )

    if layout.divider_height > 0:
        div_hex = _ffmpeg_hex(layout.divider_color)
        dh = layout.divider_height
        if geometry.video_y >= dh:
            next_s = stage + 1
            filters.append(
                f"[s{stage}]drawbox=x=0:y={geometry.video_y - dh}:w={OUTPUT_WIDTH}:h={dh}:color={div_hex}:t=fill[s{next_s}]"
            )
            stage = next_s
        if geometry.bottom_area_height >= dh:
            next_s = stage + 1
            filters.append(
                f"[s{stage}]drawbox=x=0:y={geometry.bottom_area_y}:w={OUTPUT_WIDTH}:h={dh}:color={div_hex}:t=fill[s{next_s}]"
            )
            stage = next_s

    if top_idx is not None:
        next_s = stage + 1
        filters.append(f"[{top_idx}:v]format=rgba[top_ov]")
        filters.append(f"[s{stage}][top_ov]overlay=0:0:eof_action=repeat[s{next_s}]")
        stage = next_s

    if bottom_idx is not None:
        next_s = stage + 1
        filters.append(f"[{bottom_idx}:v]format=rgba[bot_ov]")
        filters.append(
            f"[s{stage}][bot_ov]overlay=0:{geometry.bottom_area_y}:eof_action=repeat[s{next_s}]"
        )
        stage = next_s

    if badge_idx is not None:
        next_s = stage + 1
        filters.append(f"[{badge_idx}:v]format=rgba[bdg_ov]")
        filters.append(f"[s{stage}][bdg_ov]overlay=0:0:eof_action=repeat[s{next_s}]")
        stage = next_s

    filters.append(f"[s{stage}]format=yuv420p[vout]")

    cmd.extend(["-filter_complex", ";".join(filters)])
    cmd.extend(["-map", "[vout]"])

    if video_info.has_audio:
        cmd.extend(["-map", "0:a:0?", "-c:a", "aac", "-b:a", export_settings.audio_bitrate])
    else:
        cmd.append("-an")

    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            export_settings.preset,
            "-crf",
            str(export_settings.crf),
            "-movflags",
            "+faststart",
            "-progress",
            "pipe:1",
            "-nostats",
            str(output_path),
        ]
    )
    return cmd


class ExportCancelledError(Exception):
    pass


def export_parts(
    video_info: VideoInfo,
    parts: list[PartSegment],
    layout: LayoutSettings,
    badge: BadgeSettings,
    export_settings: ExportSettings,
    cancel_event: Optional[threading.Event] = None,
    on_part_start: Optional[Callable[[PartSegment, Path], None]] = None,
    on_part_progress: Optional[Callable[[PartSegment, float, str], None]] = None,
    on_part_done: Optional[Callable[[PartSegment], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
) -> list[PartSegment]:
    """Export all selected `parts` sequentially into 720x1280 (9:16 720p) MP4 files."""
    ffmpeg_exe = find_executable("ffmpeg")
    export_settings.output_dir.mkdir(parents=True, exist_ok=True)

    geometry = compute_layout_geometry(
        video_info.width,
        video_info.height,
        vertical_position=layout.vertical_position,
    )

    selected_parts = [p for p in parts if p.selected]
    if not selected_parts:
        return []

    with tempfile.TemporaryDirectory(prefix="autocutter_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # Pre-render static top and bottom zone images once for all parts
        top_png_path: Optional[Path] = None
        top_img = render_zone_image(
            layout.top_image_path,
            geometry.canvas_width,
            geometry.top_area_height,
            layout.fit_mode,
            layout.background_mode,
            layout.background_color,
        )
        if top_img is not None:
            top_png_path = tmp_dir / "top_zone.png"
            top_img.save(top_png_path, format="PNG")

        bottom_png_path: Optional[Path] = None
        bottom_img = render_zone_image(
            layout.bottom_image_path,
            geometry.canvas_width,
            geometry.bottom_area_height,
            layout.fit_mode,
            layout.background_mode,
            layout.background_color,
        )
        if bottom_img is not None:
            bottom_png_path = tmp_dir / "bottom_zone.png"
            bottom_img.save(bottom_png_path, format="PNG")

        for part in selected_parts:
            if cancel_event and cancel_event.is_set():
                part.status = "Cancelled"
                continue

            output_path = build_part_output_path(video_info, part, export_settings)
            part.output_path = output_path
            part.status = "Exporting"
            part.progress = 0.0
            part.error_message = ""

            if on_part_start:
                on_part_start(part, output_path)

            badge_png_path: Optional[Path] = None
            if badge.enabled:
                badge_img = render_badge_overlay(part.index, part.total_parts, badge, geometry)
                badge_png_path = tmp_dir / f"badge_{part.index}.png"
                badge_img.save(badge_png_path, format="PNG")

            cmd = build_ffmpeg_command(
                ffmpeg_exe=ffmpeg_exe,
                video_info=video_info,
                part=part,
                output_path=output_path,
                geometry=geometry,
                layout=layout,
                export_settings=export_settings,
                top_png_path=top_png_path,
                bottom_png_path=bottom_png_path,
                badge_png_path=badge_png_path,
            )

            if on_log:
                on_log(f"[Part {part.index}/{part.total_parts}] Exporting -> {output_path.name}")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=_subprocess_flags(),
            )

            duration_us = max(1.0, part.duration_sec * 1_000_000.0)
            speed_str = ""
            cancelled = False
            stderr_lines: list[str] = []

            def _drain_stderr() -> None:
                if proc.stderr is None:
                    return
                for err_line in proc.stderr:
                    stderr_lines.append(err_line)

            stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    if cancel_event and cancel_event.is_set():
                        cancelled = True
                        proc.terminate()
                        break

                    line = raw_line.strip()
                    if "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    if key == "speed":
                        speed_str = val.strip()
                    elif key == "out_time_us":
                        try:
                            out_us = float(val)
                            frac = max(0.0, min(0.999, out_us / duration_us))
                            part.progress = frac
                            if on_part_progress:
                                on_part_progress(part, frac, speed_str)
                        except ValueError:
                            pass
                    elif key == "progress" and val == "end":
                        part.progress = 1.0

                if cancelled:
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
                    stderr_thread.join(timeout=1)
                    if output_path.exists():
                        try:
                            output_path.unlink()
                        except OSError:
                            pass
                    part.status = "Cancelled"
                    part.progress = 0.0
                    if on_part_done:
                        on_part_done(part)
                    raise ExportCancelledError("Export cancelled by user.")

                returncode = proc.wait()
                stderr_thread.join(timeout=2)
                stderr_text = "".join(stderr_lines)

                if returncode != 0:
                    part.status = "Failed"
                    part.error_message = stderr_text.strip() or f"FFmpeg exited with code {returncode}"
                    if on_log:
                        on_log(f"[Part {part.index}] ERROR: {part.error_message}")
                else:
                    part.status = "Done"
                    part.progress = 1.0
                    if on_log:
                        on_log(f"[Part {part.index}] Done ({output_path.name})")

            except ExportCancelledError:
                raise
            except Exception as exc:
                part.status = "Failed"
                part.error_message = str(exc)
                if proc.poll() is None:
                    proc.kill()
                if on_log:
                    on_log(f"[Part {part.index}] ERROR: {exc}")

            if on_part_done:
                on_part_done(part)

            # Yield briefly between jobs
            time.sleep(0.02)

    return selected_parts
