from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from .models import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    BadgeSettings,
    FitMode,
    LayoutGeometry,
    LayoutSettings,
    SubtitleSettings,
    format_part_label,
)


_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\bahnschrift.ttf",
    r"C:\Windows\Fonts\impact.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def parse_hex_color(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    cleaned = (hex_color or "#000000").strip().lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join(c * 2 for c in cleaned)
    if len(cleaned) != 6:
        return (0, 0, 0, alpha)
    try:
        r = int(cleaned[0:2], 16)
        g = int(cleaned[2:4], 16)
        b = int(cleaned[4:6], 16)
        return (r, g, b, max(0, min(255, alpha)))
    except ValueError:
        return (0, 0, 0, alpha)


@lru_cache(maxsize=32)
def get_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(10, int(size))
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    for fallback in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(fallback, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_image_to_box(
    image: Image.Image,
    target_w: int,
    target_h: int,
    fit_mode: FitMode = "cover",
    bg_rgba: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Image.Image:
    """Resize/crop `image` to exactly (target_w, target_h) according to `fit_mode`."""
    if target_w <= 0 or target_h <= 0:
        return Image.new("RGBA", (max(1, target_w), max(1, target_h)), bg_rgba)

    src = ImageOps.exif_transpose(image).convert("RGBA")
    src_w, src_h = src.size
    if src_w <= 0 or src_h <= 0:
        return Image.new("RGBA", (target_w, target_h), bg_rgba)

    if fit_mode == "stretch":
        return src.resize((target_w, target_h), Image.Resampling.LANCZOS)

    if fit_mode == "contain":
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = src.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (target_w, target_h), bg_rgba)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y), resized)
        return canvas

    # Default: "cover" (proportional scale + center crop)
    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(target_w, int(round(src_w * scale)))
    new_h = max(target_h, int(round(src_h * scale)))
    resized = src.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def render_zone_image(
    image_path: Optional[Path],
    target_w: int,
    target_h: int,
    fit_mode: FitMode,
    background_mode: str,
    background_color: str,
) -> Optional[Image.Image]:
    """Render either the top or bottom banner image fitted to its exact zone dimensions."""
    if not image_path or target_w <= 0 or target_h <= 0:
        return None
    path = Path(image_path)
    if not path.exists():
        return None

    with Image.open(path) as img:
        bg_rgba = (
            (0, 0, 0, 0)
            if background_mode == "blur"
            else parse_hex_color(background_color, alpha=255)
        )
        return fit_image_to_box(img, target_w, target_h, fit_mode=fit_mode, bg_rgba=bg_rgba)


def render_badge_overlay(
    part_num: int,
    total_parts: int,
    badge: BadgeSettings,
    geometry: LayoutGeometry,
) -> Image.Image:
    """Create a transparent 720x1280 RGBA overlay containing the 'Part X' pill badge."""
    overlay = Image.new("RGBA", (geometry.canvas_width, geometry.canvas_height), (0, 0, 0, 0))
    if not badge.enabled:
        return overlay

    text = format_part_label(badge.template, part_num, total_parts)
    if not text:
        return overlay

    draw = ImageDraw.Draw(overlay)
    font = get_bold_font(badge.font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])

    pad_x = max(18, int(round(badge.font_size * 0.72)))
    pad_y = max(10, int(round(badge.font_size * 0.38)))
    pill_w = text_w + pad_x * 2
    pill_h = text_h + pad_y * 2
    margin = max(4, badge.margin)

    # Clamp pill width if user sets very long text
    pill_w = min(pill_w, geometry.canvas_width - 16)

    pos = badge.position
    if pos == "top_area":
        x = (geometry.canvas_width - pill_w) // 2
        area_h = geometry.top_area_height
        y = (area_h - pill_h) // 2 if area_h >= pill_h else margin
        y = max(margin, min(max(margin, area_h - pill_h - margin), y))
    elif pos == "above_video":
        x = (geometry.canvas_width - pill_w) // 2
        y = max(margin, geometry.video_y - pill_h - margin)
    elif pos == "video_top_left":
        x = geometry.video_x + margin
        y = geometry.video_y + margin
    elif pos == "video_top_right":
        x = geometry.video_x + geometry.video_width - pill_w - margin
        y = geometry.video_y + margin
    elif pos == "below_video":
        x = (geometry.canvas_width - pill_w) // 2
        max_y = geometry.canvas_height - pill_h - margin
        y = min(max_y, geometry.bottom_area_y + margin)
    elif pos == "bottom_area":
        x = (geometry.canvas_width - pill_w) // 2
        area_h = geometry.bottom_area_height
        y = geometry.bottom_area_y + ((area_h - pill_h) // 2 if area_h >= pill_h else margin)
        y = min(geometry.canvas_height - pill_h - margin, y)
    else:
        x = (geometry.canvas_width - pill_w) // 2
        y = margin

    x = max(8, min(geometry.canvas_width - pill_w - 8, x))
    y = max(8, min(geometry.canvas_height - pill_h - 8, y))

    radius = max(0, min(badge.corner_radius, pill_h // 2))
    alpha = int(round(max(0.0, min(1.0, badge.bg_opacity)) * 255))
    bg_rgba = parse_hex_color(badge.bg_color, alpha=alpha)
    text_rgba = parse_hex_color(badge.text_color, alpha=255)
    border_rgba = parse_hex_color(badge.border_color, alpha=min(255, alpha + 30))

    # Subtle drop shadow
    if alpha > 20:
        shadow_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        shadow_draw.rounded_rectangle(
            [x, y + 4, x + pill_w, y + pill_h + 4],
            radius=radius,
            fill=(0, 0, 0, min(130, alpha)),
        )
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=6))
        overlay = Image.alpha_composite(overlay, shadow_layer)
        draw = ImageDraw.Draw(overlay)

    # Pill background + optional border
    draw.rounded_rectangle(
        [x, y, x + pill_w, y + pill_h],
        radius=radius,
        fill=bg_rgba,
        outline=border_rgba if badge.border_width > 0 else None,
        width=max(0, badge.border_width),
    )

    # Exact visual centering using textbbox offsets
    text_x = x + (pill_w - text_w) / 2.0 - bbox[0]
    text_y = y + (pill_h - text_h) / 2.0 - bbox[1]
    draw.text((text_x, text_y), text, font=font, fill=text_rgba)

    return overlay


def _make_placeholder_video_frame(width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height), (28, 34, 48))
    draw = ImageDraw.Draw(img)
    # Subtle diagonal stripe & center label
    draw.rectangle([12, 12, width - 12, height - 12], outline=(68, 82, 110), width=2)
    font = get_bold_font(28)
    label = "16:9 LANDSCAPE VIDEO AREA"
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((width - tw) / 2 - bbox[0], (height - th) / 2 - bbox[1]),
        label,
        font=font,
        fill=(148, 163, 184),
    )
    return img


def _wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        if not words:
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if (bbox[2] - bbox[0]) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def render_subtitle_overlay(
    subtitle_text: str,
    subtitle: SubtitleSettings,
    geometry: LayoutGeometry,
) -> Image.Image:
    """Render wrapped subtitle text onto a 720x1280 RGBA overlay for live preview."""
    overlay = Image.new("RGBA", (geometry.canvas_width, geometry.canvas_height), (0, 0, 0, 0))
    if not subtitle.enabled or not subtitle_text.strip():
        return overlay

    draw = ImageDraw.Draw(overlay)
    font = get_bold_font(subtitle.font_size)
    max_w = geometry.canvas_width - 64
    lines = _wrap_text_to_width(draw, subtitle_text, font, max_w)
    if not lines:
        return overlay

    line_boxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
    line_heights = [max(1, b[3] - b[1]) for b in line_boxes]
    line_widths = [max(1, b[2] - b[0]) for b in line_boxes]
    line_gap = max(6, int(round(subtitle.font_size * 0.25)))
    total_h = sum(line_heights) + line_gap * (len(lines) - 1)
    block_w = max(line_widths)

    pos = subtitle.position
    if pos == "below_video" and geometry.bottom_area_height >= 50:
        top_y = geometry.bottom_area_y + 18
    elif pos == "bottom_area" and geometry.bottom_area_height >= 50:
        top_y = geometry.bottom_area_y + max(16, (geometry.bottom_area_height - total_h) // 2)
    elif pos == "video_center":
        top_y = geometry.video_y + max(8, (geometry.video_height - total_h) // 2)
    else:
        # video_bottom
        top_y = geometry.bottom_area_y - total_h - 18

    top_y = max(12, min(geometry.canvas_height - total_h - 12, top_y))

    text_rgba = parse_hex_color(subtitle.text_color, alpha=255)
    outline_rgba = parse_hex_color(subtitle.outline_color, alpha=255)

    if subtitle.bg_box:
        pad_x, pad_y = 16, 10
        bx = (geometry.canvas_width - block_w) // 2 - pad_x
        by = top_y - pad_y
        bw = block_w + pad_x * 2
        bh = total_h + pad_y * 2
        box_rgba = parse_hex_color(subtitle.outline_color, alpha=175)
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=10, fill=box_rgba)

    cur_y = top_y
    stroke_w = 0 if subtitle.bg_box else max(2, subtitle.font_size // 12)
    for ln, bbox, lw, lh in zip(lines, line_boxes, line_widths, line_heights):
        tx = (geometry.canvas_width - lw) / 2.0 - bbox[0]
        ty = cur_y - bbox[1]
        draw.text(
            (tx, ty),
            ln,
            font=font,
            fill=text_rgba,
            stroke_width=stroke_w,
            stroke_fill=outline_rgba if stroke_w > 0 else None,
        )
        cur_y += lh + line_gap

    return overlay


def render_preview_frame(
    frame_png_bytes: Optional[bytes],
    geometry: LayoutGeometry,
    layout: LayoutSettings,
    badge: BadgeSettings,
    part_num: int = 1,
    total_parts: int = 1,
    subtitle_text: Optional[str] = None,
    subtitle_settings: Optional[SubtitleSettings] = None,
) -> Image.Image:
    """Compose a full 720x1280 preview image matching the final FFmpeg export."""
    if frame_png_bytes:
        with Image.open(io.BytesIO(frame_png_bytes)) as raw_frame:
            video_rgb = ImageOps.exif_transpose(raw_frame).convert("RGB")
    else:
        video_rgb = _make_placeholder_video_frame(geometry.video_width, geometry.video_height)

    # 1. Base canvas (solid or blurred video backdrop)
    if layout.background_mode == "blur":
        bg = fit_image_to_box(
            video_rgb,
            geometry.canvas_width,
            geometry.canvas_height,
            fit_mode="cover",
        ).convert("RGB")
        bg = bg.filter(ImageFilter.GaussianBlur(radius=24))
        bg = ImageEnhance.Brightness(bg).enhance(0.84)
        canvas = bg.convert("RGBA")
    else:
        canvas = Image.new(
            "RGBA",
            (geometry.canvas_width, geometry.canvas_height),
            parse_hex_color(layout.background_color, alpha=255),
        )

    # 2. Scaled foreground video
    vid_resized = video_rgb.resize(
        (geometry.video_width, geometry.video_height), Image.Resampling.LANCZOS
    ).convert("RGBA")
    canvas.paste(vid_resized, (geometry.video_x, geometry.video_y))

    # 3. Optional divider lines above & below video
    if layout.divider_height > 0:
        draw = ImageDraw.Draw(canvas)
        div_color = parse_hex_color(layout.divider_color, alpha=255)
        dh = layout.divider_height
        if geometry.video_y >= dh:
            draw.rectangle(
                [0, geometry.video_y - dh, geometry.canvas_width, geometry.video_y],
                fill=div_color,
            )
        if geometry.bottom_area_height >= dh:
            draw.rectangle(
                [0, geometry.bottom_area_y, geometry.canvas_width, geometry.bottom_area_y + dh],
                fill=div_color,
            )

    # 4. Top & Bottom images
    top_img = render_zone_image(
        layout.top_image_path,
        geometry.canvas_width,
        geometry.top_area_height,
        layout.fit_mode,
        layout.background_mode,
        layout.background_color,
    )
    if top_img is not None:
        canvas.paste(top_img, (0, 0), top_img)

    bottom_img = render_zone_image(
        layout.bottom_image_path,
        geometry.canvas_width,
        geometry.bottom_area_height,
        layout.fit_mode,
        layout.background_mode,
        layout.background_color,
    )
    if bottom_img is not None:
        canvas.paste(bottom_img, (0, geometry.bottom_area_y), bottom_img)

    # 5. Subtitle preview overlay
    if subtitle_settings and subtitle_settings.enabled and subtitle_text:
        sub_layer = render_subtitle_overlay(subtitle_text, subtitle_settings, geometry)
        canvas = Image.alpha_composite(canvas, sub_layer)

    # 6. Part badge overlay
    if badge.enabled:
        badge_layer = render_badge_overlay(part_num, total_parts, badge, geometry)
        canvas = Image.alpha_composite(canvas, badge_layer)

    return canvas.convert("RGB")
