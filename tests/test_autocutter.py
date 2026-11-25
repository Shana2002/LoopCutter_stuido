from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from autocutter.compositor import (
    fit_image_to_box,
    render_badge_overlay,
    render_preview_frame,
)
from autocutter.exporter import export_parts
from autocutter.media_probe import extract_frame_png, find_executable, probe_video
from autocutter.models import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    BadgeSettings,
    ExportSettings,
    LayoutSettings,
    calculate_parts,
    compute_layout_geometry,
    format_part_label,
    format_timestamp,
)


def test_calculate_parts_ten_minute_movie_split() -> None:
    # 25-minute movie cut into 10-minute parts -> 10m, 10m, 5m
    parts = calculate_parts(total_duration_sec=1500.0, segment_duration_sec=600.0)
    assert len(parts) == 3
    assert [(p.index, p.total_parts, p.start_sec, p.end_sec) for p in parts] == [
        (1, 3, 0.0, 600.0),
        (2, 3, 600.0, 1200.0),
        (3, 3, 1200.0, 1500.0),
    ]


def test_calculate_parts_absorbs_tiny_tail() -> None:
    # 20m + 0.4s -> should remain 2 parts instead of creating a 0.4s Part 3
    parts = calculate_parts(total_duration_sec=1200.4, segment_duration_sec=600.0, min_tail_sec=1.0)
    assert len(parts) == 2
    assert parts[-1].end_sec == 1200.4


def test_calculate_parts_with_start_offset() -> None:
    parts = calculate_parts(
        total_duration_sec=120.0,
        segment_duration_sec=30.0,
        start_offset_sec=30.0,
    )
    assert len(parts) == 3
    assert parts[0].start_sec == 30.0
    assert parts[-1].end_sec == 120.0


def test_layout_geometry_720p_vertical_even_dimensions() -> None:
    geo = compute_layout_geometry(1920, 1080, vertical_position=0.5)
    assert geo.canvas_width == 720
    assert geo.canvas_height == 1280
    assert geo.video_width == 720
    assert geo.video_height % 2 == 0
    assert geo.video_y % 2 == 0
    assert geo.top_area_height + geo.video_height + geo.bottom_area_height == 1280


def test_format_helpers() -> None:
    assert format_timestamp(3661.0) == "01:01:01"
    assert format_part_label("Part {n} of {total}", 2, 12) == "Part 2 of 12"
    assert format_part_label("Part {n}", 4, 10) == "Part 4"


def test_compositor_preview_and_badge_render(tmp_path: Path) -> None:
    top_path = tmp_path / "top.png"
    bot_path = tmp_path / "bot.png"
    Image.new("RGB", (800, 300), (220, 40, 40)).save(top_path)
    Image.new("RGB", (800, 300), (40, 120, 220)).save(bot_path)

    geo = compute_layout_geometry(1920, 1080, vertical_position=0.5)
    layout = LayoutSettings(
        top_image_path=top_path,
        bottom_image_path=bot_path,
        fit_mode="cover",
        background_mode="solid",
        background_color="#101010",
    )

    for pos in (
        "top_area",
        "above_video",
        "video_top_left",
        "video_top_right",
        "below_video",
        "bottom_area",
    ):
        badge = BadgeSettings(enabled=True, template="Part {n}/{total}", position=pos)  # type: ignore[arg-type]
        ov = render_badge_overlay(2, 5, badge, geo)
        assert ov.size == (OUTPUT_WIDTH, OUTPUT_HEIGHT)
        # Ensure badge actually drew non-transparent pixels
        assert ov.getbbox() is not None

        preview = render_preview_frame(None, geo, layout, badge, part_num=2, total_parts=5)
        assert preview.size == (OUTPUT_WIDTH, OUTPUT_HEIGHT)


def test_fit_image_to_box_modes() -> None:
    src = Image.new("RGBA", (400, 200), (10, 200, 30, 255))
    for mode in ("cover", "contain", "stretch"):
        out = fit_image_to_box(src, 720, 400, fit_mode=mode)  # type: ignore[arg-type]
        assert out.size == (720, 400)


def test_end_to_end_ffmpeg_export_720p_9_16(tmp_path: Path) -> None:
    ffmpeg = find_executable("ffmpeg")
    src_video = tmp_path / "sample_landscape.mp4"
    top_img = tmp_path / "top_banner.png"
    bot_img = tmp_path / "bottom_banner.png"
    out_dir = tmp_path / "exports"

    # Generate synthetic 4-second 1280x720 landscape video with audio
    gen_cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=4:size=1280x720:rate=24",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=4",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(src_video),
    ]
    res = subprocess.run(gen_cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Synthetic video creation failed: {res.stderr}"

    Image.new("RGB", (720, 400), (240, 180, 40)).save(top_img)
    Image.new("RGB", (720, 400), (30, 140, 210)).save(bot_img)

    info = probe_video(src_video)
    assert info.width == 1280
    assert info.height == 720
    assert info.is_landscape is True
    assert info.has_audio is True

    frame_png = extract_frame_png(src_video, timestamp_sec=1.0, max_width=720)
    assert frame_png.startswith(b"\x89PNG")

    parts = calculate_parts(info.duration_sec, segment_duration_sec=2.0)
    assert len(parts) == 2

    layout = LayoutSettings(
        top_image_path=top_img,
        bottom_image_path=bot_img,
        fit_mode="cover",
        background_mode="blur",
    )
    badge = BadgeSettings(enabled=True, template="Part {n}", position="above_video")
    export_cfg = ExportSettings(output_dir=out_dir, preset="ultrafast", crf=24)

    exported = export_parts(
        video_info=info,
        parts=parts,
        layout=layout,
        badge=badge,
        export_settings=export_cfg,
    )
    assert len(exported) == 2

    for idx, part in enumerate(exported, start=1):
        assert part.status == "Done", f"Part {idx} failed: {part.error_message}"
        assert part.output_path is not None and part.output_path.exists()
        out_info = probe_video(part.output_path)
        assert out_info.width == 720
        assert out_info.height == 1280
        assert out_info.has_audio is True
        assert 1.5 <= out_info.duration_sec <= 2.5

    # Also verify solid-color background + divider bar mode on 1 selected part
    parts[0].selected = True
    parts[1].selected = False
    solid_dir = tmp_path / "exports_solid"
    solid_exported = export_parts(
        video_info=info,
        parts=parts,
        layout=LayoutSettings(
            top_image_path=top_img,
            bottom_image_path=bot_img,
            fit_mode="contain",
            background_mode="solid",
            background_color="#112233",
            divider_height=4,
        ),
        badge=BadgeSettings(enabled=True, template="Part {n}/{total}", position="top_area"),
        export_settings=ExportSettings(output_dir=solid_dir, preset="ultrafast", crf=24),
    )
    assert len(solid_exported) == 1
    assert solid_exported[0].status == "Done"
    solid_info = probe_video(solid_exported[0].output_path)  # type: ignore[arg-type]
    assert (solid_info.width, solid_info.height) == (720, 1280)


def test_qt_main_window_smoke() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    from autocutter.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    assert win.windowTitle().startswith("AutoCutter Studio")
    win.close()
    app.processEvents()
