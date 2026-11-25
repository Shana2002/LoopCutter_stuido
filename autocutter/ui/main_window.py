from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Optional

from PIL import Image
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QDragEnterEvent, QDropEvent, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..compositor import render_preview_frame
from ..exporter import ExportCancelledError, build_part_output_path, export_parts
from ..media_probe import extract_frame_png, probe_video
from ..models import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    BadgePosition,
    BadgeSettings,
    ExportSettings,
    FitMode,
    LayoutSettings,
    PartSegment,
    VideoInfo,
    calculate_parts,
    compute_layout_geometry,
    format_duration_short,
    format_part_label,
    format_timestamp,
)


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".flv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0B0F17;
    color: #E2E8F0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #131A26;
    border: 1px solid #222C3F;
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
    color: #F8FAFC;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #93C5FD;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
QPushButton {
    background-color: #1E293B;
    color: #F1F5F9;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #283548;
    border-color: #475569;
}
QPushButton:pressed {
    background-color: #172030;
}
QPushButton:disabled {
    background-color: #131A26;
    color: #64748B;
    border-color: #1E293B;
}
QPushButton#primaryBtn {
    background-color: #E11D48;
    border-color: #FB7185;
    color: #FFFFFF;
    padding: 10px 16px;
    font-size: 14px;
    border-radius: 8px;
}
QPushButton#primaryBtn:hover {
    background-color: #BE123C;
}
QPushButton#primaryBtn:disabled {
    background-color: #4C1D2B;
    border-color: #5C2234;
    color: #94A3B8;
}
QPushButton#presetBtn {
    padding: 4px 8px;
    font-size: 12px;
    background-color: #182234;
}
QPushButton#presetBtn:checked {
    background-color: #2563EB;
    border-color: #60A5FA;
    color: #FFFFFF;
}
QLineEdit, QComboBox, QSpinBox {
    background-color: #0E1420;
    border: 1px solid #2A354A;
    border-radius: 6px;
    padding: 5px 8px;
    color: #F8FAFC;
    selection-background-color: #2563EB;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border-color: #3B82F6;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #1E293B;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #3B82F6;
    border: 2px solid #93C5FD;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 8px;
}
QTableWidget {
    background-color: #0E1420;
    border: 1px solid #222C3F;
    border-radius: 8px;
    gridline-color: #1B2434;
    alternate-background-color: #111827;
}
QTableWidget::item {
    padding: 4px 6px;
}
QTableWidget::item:selected {
    background-color: #1D4ED8;
    color: #FFFFFF;
}
QHeaderView::section {
    background-color: #151E2E;
    color: #94A3B8;
    border: none;
    border-bottom: 1px solid #232E42;
    padding: 6px;
    font-weight: 600;
    font-size: 12px;
}
QProgressBar {
    background-color: #0E1420;
    border: 1px solid #253147;
    border-radius: 6px;
    text-align: center;
    height: 18px;
    color: #F8FAFC;
    font-weight: 600;
}
QProgressBar::chunk {
    background-color: #2563EB;
    border-radius: 5px;
}
QPlainTextEdit {
    background-color: #0A0E16;
    border: 1px solid #222C3F;
    border-radius: 6px;
    font-family: "Consolas", monospace;
    font-size: 11px;
    color: #94A3B8;
}
QScrollArea {
    border: none;
}
"""


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    rgb = img.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimg = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ProbeWorker(QThread):
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def run(self) -> None:
        try:
            info = probe_video(self.path)
            self.finished_ok.emit(info)
        except Exception as exc:
            self.failed.emit(str(exc))


class FrameExtractWorker(QThread):
    frame_ready = pyqtSignal(int, float, bytes)
    frame_failed = pyqtSignal(int, str)

    def __init__(self, request_id: int, video_path: Path, timestamp_sec: float) -> None:
        super().__init__()
        self.request_id = request_id
        self.video_path = video_path
        self.timestamp_sec = timestamp_sec

    def run(self) -> None:
        try:
            png_bytes = extract_frame_png(self.video_path, self.timestamp_sec, max_width=720)
            self.frame_ready.emit(self.request_id, self.timestamp_sec, png_bytes)
        except Exception as exc:
            self.frame_failed.emit(self.request_id, str(exc))


class ExportWorker(QThread):
    part_started = pyqtSignal(int, str)
    part_progress = pyqtSignal(int, float, str)
    part_finished = pyqtSignal(int, str, str)
    log_message = pyqtSignal(str)
    all_finished = pyqtSignal(bool, str)

    def __init__(
        self,
        video_info: VideoInfo,
        parts: list[PartSegment],
        layout: LayoutSettings,
        badge: BadgeSettings,
        export_settings: ExportSettings,
    ) -> None:
        super().__init__()
        self.video_info = video_info
        self.parts = parts
        self.layout = layout
        self.badge = badge
        self.export_settings = export_settings
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            export_parts(
                video_info=self.video_info,
                parts=self.parts,
                layout=self.layout,
                badge=self.badge,
                export_settings=self.export_settings,
                cancel_event=self.cancel_event,
                on_part_start=lambda part, out_p: self.part_started.emit(part.index, str(out_p)),
                on_part_progress=lambda part, frac, spd: self.part_progress.emit(
                    part.index, frac, spd
                ),
                on_part_done=lambda part: self.part_finished.emit(
                    part.index, part.status, part.error_message
                ),
                on_log=lambda msg: self.log_message.emit(msg),
            )
            failed = sum(1 for p in self.parts if p.selected and p.status == "Failed")
            done = sum(1 for p in self.parts if p.selected and p.status == "Done")
            if failed:
                self.all_finished.emit(
                    False, f"Completed with errors: {done} exported, {failed} failed."
                )
            else:
                self.all_finished.emit(True, f"Successfully exported {done} part(s) to 720p 9:16!")
        except ExportCancelledError:
            self.all_finished.emit(False, "Export cancelled.")
        except Exception as exc:
            self.all_finished.emit(False, f"Export error: {exc}")


class AspectPreviewLabel(QLabel):
    """Displays the 9:16 (720x1280) preview scaled crisply inside available space."""

    def __init__(self) -> None:
        super().__init__()
        self._full_pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(270, 480)
        self.setStyleSheet(
            "background-color: #070A10; border: 1px solid #243047; border-radius: 12px;"
        )

    def set_preview_image(self, img: Image.Image) -> None:
        self._full_pixmap = pil_to_qpixmap(img)
        self._rescale()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._full_pixmap is None:
            return
        avail_w = max(120, self.width() - 16)
        avail_h = max(210, self.height() - 16)
        scaled = self._full_pixmap.scaled(
            avail_w, avail_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)


class ColorSwatchButton(QPushButton):
    color_changed = pyqtSignal(str)

    def __init__(self, initial_hex: str, label: str = "") -> None:
        super().__init__(label or initial_hex)
        self._hex = initial_hex
        self._update_style()
        self.clicked.connect(self._pick_color)

    @property
    def hex_color(self) -> str:
        return self._hex

    def set_hex_color(self, hex_color: str) -> None:
        self._hex = hex_color
        self.setText(hex_color.upper())
        self._update_style()

    def _update_style(self) -> None:
        c = QColor(self._hex)
        luminance = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
        fg = "#0F172A" if luminance > 150 else "#F8FAFC"
        self.setStyleSheet(
            f"background-color: {self._hex}; color: {fg}; border: 1px solid #475569; "
            "border-radius: 6px; padding: 4px 10px; font-weight: 600;"
        )

    def _pick_color(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._hex), self, "Select Color")
        if chosen.isValid():
            self.set_hex_color(chosen.name().upper())
            self.color_changed.emit(self._hex)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AutoCutter Studio — Landscape to 720p 9:16 Shorts Splitter")
        self.resize(1420, 880)
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLESHEET)

        self.video_info: Optional[VideoInfo] = None
        self.parts: list[PartSegment] = []
        self.active_part_index: int = 1
        self.current_frame_bytes: Optional[bytes] = None
        self.frame_request_counter: int = 0
        self.probe_worker: Optional[ProbeWorker] = None
        self._active_frame_workers: set[FrameExtractWorker] = set()
        self.export_worker: Optional[ExportWorker] = None

        self.top_image_path: Optional[Path] = None
        self.bottom_image_path: Optional[Path] = None

        self._frame_debounce_timer = QTimer(self)
        self._frame_debounce_timer.setSingleShot(True)
        self._frame_debounce_timer.setInterval(120)
        self._frame_debounce_timer.timeout.connect(self._request_preview_frame)

        self._build_ui()
        self._refresh_preview()

    # ------------------------------------------------------------------ UI Build
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # Top header bar
        header = QHBoxLayout()
        title_lbl = QLabel("AutoCutter Studio")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: 700; color: #F8FAFC;")
        subtitle_lbl = QLabel(
            "Cut landscape movies into timed parts · Convert to 720×1280 (9:16 720p) · Top/Bottom images · Burn-in Part Labels"
        )
        subtitle_lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
        header.addWidget(title_lbl)
        header.addSpacing(10)
        header.addWidget(subtitle_lbl, 1)
        root_layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)
        root_layout.addWidget(splitter, 1)

        # LEFT PANEL (Scrollable controls)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(380)
        left_scroll.setMaximumWidth(460)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(2, 2, 8, 8)
        left_layout.setSpacing(8)

        left_layout.addWidget(self._build_source_group())
        left_layout.addWidget(self._build_timeframe_group())
        left_layout.addWidget(self._build_images_layout_group())
        left_layout.addWidget(self._build_badge_group())
        left_layout.addWidget(self._build_export_settings_group())
        left_layout.addStretch(1)
        left_scroll.setWidget(left_container)

        # CENTER PANEL (Live 9:16 Preview)
        center_panel = self._build_preview_panel()

        # RIGHT PANEL (Parts Breakdown + Progress + Actions)
        right_panel = self._build_parts_panel()

        splitter.addWidget(left_scroll)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)

    def _build_source_group(self) -> QGroupBox:
        box = QGroupBox("1. Landscape Source Video")
        lay = QVBoxLayout(box)
        lay.setSpacing(6)

        row = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setPlaceholderText("Drop a video file here or click Browse...")
        browse_btn = QPushButton("Browse Video")
        browse_btn.clicked.connect(self._browse_video)
        row.addWidget(self.video_path_edit, 1)
        row.addWidget(browse_btn)
        lay.addLayout(row)

        self.video_meta_lbl = QLabel("No video loaded · Supports MP4, MKV, MOV, AVI, WebM")
        self.video_meta_lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
        self.video_meta_lbl.setWordWrap(True)
        lay.addWidget(self.video_meta_lbl)
        return box

    def _build_timeframe_group(self) -> QGroupBox:
        box = QGroupBox("2. Cut Time Frame")
        lay = QVBoxLayout(box)
        lay.setSpacing(8)

        presets_row = QHBoxLayout()
        presets_row.setSpacing(4)
        self.preset_buttons: list[tuple[int, QPushButton]] = []
        for minutes in (1, 3, 5, 10, 15, 30):
            btn = QPushButton(f"{minutes}m")
            btn.setObjectName("presetBtn")
            btn.setCheckable(True)
            btn.setChecked(minutes == 10)
            btn.clicked.connect(lambda _checked, m=minutes: self._apply_duration_preset(m))
            presets_row.addWidget(btn)
            self.preset_buttons.append((minutes, btn))
        lay.addLayout(presets_row)

        grid = QGridLayout()
        grid.addWidget(QLabel("Cut Every:"), 0, 0)
        self.seg_min_spin = QSpinBox()
        self.seg_min_spin.setRange(0, 600)
        self.seg_min_spin.setValue(10)
        self.seg_min_spin.setSuffix(" min")
        self.seg_min_spin.valueChanged.connect(self._on_timeframe_changed)

        self.seg_sec_spin = QSpinBox()
        self.seg_sec_spin.setRange(0, 59)
        self.seg_sec_spin.setValue(0)
        self.seg_sec_spin.setSuffix(" sec")
        self.seg_sec_spin.valueChanged.connect(self._on_timeframe_changed)

        grid.addWidget(self.seg_min_spin, 0, 1)
        grid.addWidget(self.seg_sec_spin, 0, 2)

        grid.addWidget(QLabel("Start Offset:"), 1, 0)
        self.offset_min_spin = QSpinBox()
        self.offset_min_spin.setRange(0, 600)
        self.offset_min_spin.setValue(0)
        self.offset_min_spin.setSuffix(" min")
        self.offset_min_spin.valueChanged.connect(self._on_timeframe_changed)

        self.offset_sec_spin = QSpinBox()
        self.offset_sec_spin.setRange(0, 59)
        self.offset_sec_spin.setValue(0)
        self.offset_sec_spin.setSuffix(" sec")
        self.offset_sec_spin.valueChanged.connect(self._on_timeframe_changed)

        grid.addWidget(self.offset_min_spin, 1, 1)
        grid.addWidget(self.offset_sec_spin, 1, 2)
        lay.addLayout(grid)
        return box

    def _build_images_layout_group(self) -> QGroupBox:
        box = QGroupBox("3. Top & Bottom Images (9:16 Layout)")
        lay = QVBoxLayout(box)
        lay.setSpacing(8)

        # Top Image Row
        top_row = QHBoxLayout()
        self.top_thumb = QLabel("TOP")
        self.top_thumb.setFixedSize(46, 46)
        self.top_thumb.setAlignment(Qt.AlignCenter)
        self.top_thumb.setStyleSheet(
            "background-color: #0E1420; border: 1px dashed #334155; border-radius: 6px; font-size: 10px; color: #64748B;"
        )
        self.top_path_lbl = QLabel("No top image")
        self.top_path_lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
        top_browse = QPushButton("Choose Top")
        top_browse.clicked.connect(lambda: self._browse_image("top"))
        top_clear = QPushButton("✕")
        top_clear.setFixedWidth(30)
        top_clear.clicked.connect(lambda: self._clear_image("top"))
        top_row.addWidget(self.top_thumb)
        top_row.addWidget(self.top_path_lbl, 1)
        top_row.addWidget(top_browse)
        top_row.addWidget(top_clear)
        lay.addLayout(top_row)

        # Bottom Image Row
        bot_row = QHBoxLayout()
        self.bot_thumb = QLabel("BOT")
        self.bot_thumb.setFixedSize(46, 46)
        self.bot_thumb.setAlignment(Qt.AlignCenter)
        self.bot_thumb.setStyleSheet(
            "background-color: #0E1420; border: 1px dashed #334155; border-radius: 6px; font-size: 10px; color: #64748B;"
        )
        self.bot_path_lbl = QLabel("No bottom image")
        self.bot_path_lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
        bot_browse = QPushButton("Choose Bottom")
        bot_browse.clicked.connect(lambda: self._browse_image("bottom"))
        bot_clear = QPushButton("✕")
        bot_clear.setFixedWidth(30)
        bot_clear.clicked.connect(lambda: self._clear_image("bottom"))
        bot_row.addWidget(self.bot_thumb)
        bot_row.addWidget(self.bot_path_lbl, 1)
        bot_row.addWidget(bot_browse)
        bot_row.addWidget(bot_clear)
        lay.addLayout(bot_row)

        # Options Grid
        grid = QGridLayout()
        grid.addWidget(QLabel("Image Fit:"), 0, 0)
        self.fit_combo = QComboBox()
        self.fit_combo.addItem("Cover (Fill & Crop)", "cover")
        self.fit_combo.addItem("Contain (Keep Full Image)", "contain")
        self.fit_combo.addItem("Stretch", "stretch")
        self.fit_combo.currentIndexChanged.connect(self._refresh_preview)
        grid.addWidget(self.fit_combo, 0, 1, 1, 2)

        grid.addWidget(QLabel("Background:"), 1, 0)
        self.bg_mode_combo = QComboBox()
        self.bg_mode_combo.addItem("Blurred Video", "blur")
        self.bg_mode_combo.addItem("Solid Color", "solid")
        self.bg_mode_combo.currentIndexChanged.connect(self._refresh_preview)
        self.bg_color_btn = ColorSwatchButton("#0F1117")
        self.bg_color_btn.color_changed.connect(lambda _c: self._refresh_preview())
        grid.addWidget(self.bg_mode_combo, 1, 1)
        grid.addWidget(self.bg_color_btn, 1, 2)

        grid.addWidget(QLabel("Video Y Pos:"), 2, 0)
        self.vpos_slider = QSlider(Qt.Horizontal)
        self.vpos_slider.setRange(0, 100)
        self.vpos_slider.setValue(50)
        self.vpos_slider.valueChanged.connect(self._refresh_preview)
        self.vpos_lbl = QLabel("Center")
        self.vpos_lbl.setFixedWidth(48)
        grid.addWidget(self.vpos_slider, 2, 1)
        grid.addWidget(self.vpos_lbl, 2, 2)

        grid.addWidget(QLabel("Divider Bar:"), 3, 0)
        self.divider_spin = QSpinBox()
        self.divider_spin.setRange(0, 12)
        self.divider_spin.setValue(0)
        self.divider_spin.setSuffix(" px")
        self.divider_spin.valueChanged.connect(self._refresh_preview)
        self.divider_color_btn = ColorSwatchButton("#1E2433")
        self.divider_color_btn.color_changed.connect(lambda _c: self._refresh_preview())
        grid.addWidget(self.divider_spin, 3, 1)
        grid.addWidget(self.divider_color_btn, 3, 2)

        lay.addLayout(grid)
        return box

    def _build_badge_group(self) -> QGroupBox:
        box = QGroupBox("4. Part Indicator in Video")
        lay = QVBoxLayout(box)
        lay.setSpacing(6)

        self.badge_enabled_chk = QCheckBox('Show which part ("Part X") inside exported video')
        self.badge_enabled_chk.setChecked(True)
        self.badge_enabled_chk.toggled.connect(self._refresh_preview)
        lay.addWidget(self.badge_enabled_chk)

        tpl_row = QHBoxLayout()
        tpl_row.addWidget(QLabel("Format:"))
        self.badge_template_edit = QLineEdit("Part {n}")
        self.badge_template_edit.textChanged.connect(self._on_badge_text_changed)
        tpl_row.addWidget(self.badge_template_edit, 1)
        for label, tpl in (("Part 1", "Part {n}"), ("1 / N", "Part {n} of {total}"), ("EP 1", "EP {n}")):
            b = QPushButton(label)
            b.setObjectName("presetBtn")
            b.clicked.connect(lambda _c, t=tpl: self.badge_template_edit.setText(t))
            tpl_row.addWidget(b)
        lay.addLayout(tpl_row)

        grid = QGridLayout()
        grid.addWidget(QLabel("Position:"), 0, 0)
        self.badge_pos_combo = QComboBox()
        self.badge_pos_combo.addItem("Above Video", "above_video")
        self.badge_pos_combo.addItem("Top Area Center", "top_area")
        self.badge_pos_combo.addItem("On Video — Top Left", "video_top_left")
        self.badge_pos_combo.addItem("On Video — Top Right", "video_top_right")
        self.badge_pos_combo.addItem("Below Video", "below_video")
        self.badge_pos_combo.addItem("Bottom Area Center", "bottom_area")
        self.badge_pos_combo.currentIndexChanged.connect(self._refresh_preview)
        grid.addWidget(self.badge_pos_combo, 0, 1, 1, 2)

        grid.addWidget(QLabel("Font Size:"), 1, 0)
        self.badge_size_spin = QSpinBox()
        self.badge_size_spin.setRange(16, 72)
        self.badge_size_spin.setValue(34)
        self.badge_size_spin.setSuffix(" px")
        self.badge_size_spin.valueChanged.connect(self._refresh_preview)
        grid.addWidget(self.badge_size_spin, 1, 1)

        self.badge_opacity_spin = QSpinBox()
        self.badge_opacity_spin.setRange(20, 100)
        self.badge_opacity_spin.setValue(92)
        self.badge_opacity_spin.setSuffix("% bg")
        self.badge_opacity_spin.valueChanged.connect(self._refresh_preview)
        grid.addWidget(self.badge_opacity_spin, 1, 2)

        grid.addWidget(QLabel("Colors:"), 2, 0)
        self.badge_bg_btn = ColorSwatchButton("#E11D48")
        self.badge_bg_btn.color_changed.connect(lambda _c: self._refresh_preview())
        self.badge_fg_btn = ColorSwatchButton("#FFFFFF")
        self.badge_fg_btn.color_changed.connect(lambda _c: self._refresh_preview())
        grid.addWidget(self.badge_bg_btn, 2, 1)
        grid.addWidget(self.badge_fg_btn, 2, 2)

        lay.addLayout(grid)
        return box

    def _build_export_settings_group(self) -> QGroupBox:
        box = QGroupBox("5. 720p (720×1280) Export Settings")
        lay = QVBoxLayout(box)
        lay.setSpacing(6)

        out_row = QHBoxLayout()
        self.out_dir_edit = QLineEdit(str(Path.cwd() / "exports"))
        out_btn = QPushButton("Folder")
        out_btn.clicked.connect(self._browse_output_dir)
        out_row.addWidget(self.out_dir_edit, 1)
        out_row.addWidget(out_btn)
        lay.addLayout(out_row)

        grid = QGridLayout()
        grid.addWidget(QLabel("File Prefix:"), 0, 0)
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("Auto from video name")
        self.prefix_edit.textChanged.connect(self._refresh_parts_table_names)
        grid.addWidget(self.prefix_edit, 0, 1, 1, 2)

        grid.addWidget(QLabel("Quality:"), 1, 0)
        self.crf_combo = QComboBox()
        self.crf_combo.addItem("High Quality (CRF 18)", 18)
        self.crf_combo.addItem("Balanced (CRF 21)", 21)
        self.crf_combo.addItem("Compact Fast (CRF 24)", 24)
        self.crf_combo.setCurrentIndex(1)
        grid.addWidget(self.crf_combo, 1, 1)

        self.preset_combo = QComboBox()
        for p in ("ultrafast", "veryfast", "fast", "medium"):
            self.preset_combo.addItem(p, p)
        self.preset_combo.setCurrentText("veryfast")
        grid.addWidget(self.preset_combo, 1, 2)
        lay.addLayout(grid)
        return box

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(6)

        top_bar = QHBoxLayout()
        self.preview_part_lbl = QLabel("Live 9:16 Preview (720×1280 · 720p)")
        self.preview_part_lbl.setStyleSheet("font-weight: 700; color: #F8FAFC; font-size: 14px;")
        self.preview_geo_lbl = QLabel("720×1280")
        self.preview_geo_lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
        top_bar.addWidget(self.preview_part_lbl, 1)
        top_bar.addWidget(self.preview_geo_lbl)
        lay.addLayout(top_bar)

        self.preview_widget = AspectPreviewLabel()
        lay.addWidget(self.preview_widget, 1)

        scrub_row = QHBoxLayout()
        scrub_row.addWidget(QLabel("Preview Frame:"))
        self.scrub_slider = QSlider(Qt.Horizontal)
        self.scrub_slider.setRange(0, 100)
        self.scrub_slider.setValue(25)
        self.scrub_slider.setEnabled(False)
        self.scrub_slider.valueChanged.connect(self._on_scrub_changed)
        self.scrub_ts_lbl = QLabel("00:00:00")
        self.scrub_ts_lbl.setFixedWidth(70)
        self.scrub_ts_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        scrub_row.addWidget(self.scrub_slider, 1)
        scrub_row.addWidget(self.scrub_ts_lbl)
        lay.addLayout(scrub_row)
        return panel

    def _build_parts_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(410)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(4, 2, 2, 2)
        lay.setSpacing(8)

        # Header + select buttons
        hdr = QHBoxLayout()
        self.parts_summary_lbl = QLabel("Parts Queue (0 parts)")
        self.parts_summary_lbl.setStyleSheet("font-weight: 700; font-size: 14px; color: #F8FAFC;")
        sel_all_btn = QPushButton("Select All")
        sel_all_btn.clicked.connect(lambda: self._set_all_parts_selected(True))
        sel_none_btn = QPushButton("None")
        sel_none_btn.clicked.connect(lambda: self._set_all_parts_selected(False))
        hdr.addWidget(self.parts_summary_lbl, 1)
        hdr.addWidget(sel_all_btn)
        hdr.addWidget(sel_none_btn)
        lay.addLayout(hdr)

        # Parts Table
        self.parts_table = QTableWidget(0, 5)
        self.parts_table.setHorizontalHeaderLabels(
            ["Export", "Part", "Time Frame", "Duration", "Status"]
        )
        self.parts_table.setAlternatingRowColors(True)
        self.parts_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.parts_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.parts_table.verticalHeader().setVisible(False)
        hdr_view = self.parts_table.horizontalHeader()
        hdr_view.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr_view.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr_view.setSectionResizeMode(2, QHeaderView.Stretch)
        hdr_view.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr_view.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.parts_table.itemChanged.connect(self._on_table_item_changed)
        self.parts_table.cellClicked.connect(self._on_part_row_clicked)
        self.parts_table.cellDoubleClicked.connect(self._on_part_row_double_clicked)
        lay.addWidget(self.parts_table, 1)

        hint_lbl = QLabel(
            "Tip: Click any part to preview its frame & badge. Double-click a finished part to play it."
        )
        hint_lbl.setStyleSheet("color: #64748B; font-size: 11px;")
        lay.addWidget(hint_lbl)

        # Progress & Export Action Box
        prog_box = QFrame()
        prog_box.setStyleSheet(
            "QFrame { background-color: #131A26; border: 1px solid #222C3F; border-radius: 10px; }"
        )
        prog_lay = QVBoxLayout(prog_box)
        prog_lay.setContentsMargins(10, 10, 10, 10)
        prog_lay.setSpacing(8)

        self.progress_status_lbl = QLabel("Ready to export 720p (720×1280) 9:16 parts")
        self.progress_status_lbl.setStyleSheet("border: none; color: #CBD5E1; font-weight: 600;")
        self.overall_progress = QProgressBar()
        self.overall_progress.setValue(0)
        prog_lay.addWidget(self.progress_status_lbl)
        prog_lay.addWidget(self.overall_progress)

        btn_row = QHBoxLayout()
        self.export_btn = QPushButton("Export Selected Parts (720p 9:16)")
        self.export_btn.setObjectName("primaryBtn")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._start_export)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_export)

        self.open_folder_btn = QPushButton("Open Output")
        self.open_folder_btn.clicked.connect(self._open_output_folder)

        btn_row.addWidget(self.export_btn, 1)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.open_folder_btn)
        prog_lay.addLayout(btn_row)
        lay.addWidget(prog_box)

        # Compact log output
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(115)
        self.log_box.setPlaceholderText("FFmpeg export log will appear here...")
        lay.addWidget(self.log_box)
        return panel

    # --------------------------------------------------------- Drag & Drop
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        urls = event.mimeData().urls()
        if not urls:
            return
        for url in urls:
            local = Path(url.toLocalFile())
            if not local.exists():
                continue
            ext = local.suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                self._load_video(local)
                break
            if ext in IMAGE_EXTENSIONS:
                if self.top_image_path is None:
                    self._set_image("top", local)
                else:
                    self._set_image("bottom", local)

    # --------------------------------------------------------- File Selection
    def _browse_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Landscape Video",
            "",
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.ts);;All Files (*.*)",
        )
        if file_path:
            self._load_video(Path(file_path))

    def _load_video(self, path: Path) -> None:
        self.video_path_edit.setText(str(path))
        self.video_meta_lbl.setText("Inspecting video with ffprobe...")
        self.export_btn.setEnabled(False)

        self.probe_worker = ProbeWorker(path)
        self.probe_worker.finished_ok.connect(self._on_video_probed)
        self.probe_worker.failed.connect(self._on_video_probe_failed)
        self.probe_worker.start()

    def _on_video_probed(self, info: VideoInfo) -> None:
        self.video_info = info
        orientation = "Landscape" if info.is_landscape else "Portrait/Square"
        audio_txt = "Audio" if info.has_audio else "No Audio"
        size_mb = info.file_size_bytes / (1024 * 1024) if info.file_size_bytes else 0
        self.video_meta_lbl.setText(
            f"{info.width}×{info.height} ({orientation}) · {format_timestamp(info.duration_sec)} · "
            f"{info.fps:.2f} fps · {audio_txt} · {size_mb:.1f} MB"
        )
        # Set default output dir alongside video if still default
        default_dir = Path.cwd() / "exports"
        if Path(self.out_dir_edit.text().strip()) == default_dir:
            self.out_dir_edit.setText(str(info.path.parent / f"{info.path.stem}_shorts_720p"))

        self.scrub_slider.setEnabled(True)
        self._recalculate_parts()
        self._schedule_frame_extract()

    def _on_video_probe_failed(self, err: str) -> None:
        self.video_meta_lbl.setText(f"Failed to read video: {err}")
        QMessageBox.critical(self, "Video Error", f"Could not inspect video:\n{err}")

    def _browse_image(self, slot: str) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {'Top' if slot == 'top' else 'Bottom'} Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All Files (*.*)",
        )
        if file_path:
            self._set_image(slot, Path(file_path))

    def _set_image(self, slot: str, path: Path) -> None:
        pix = QPixmap(str(path))
        if pix.isNull():
            QMessageBox.warning(self, "Invalid Image", f"Could not open image:\n{path}")
            return
        thumb = pix.scaled(44, 44, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        if slot == "top":
            self.top_image_path = path
            self.top_thumb.setPixmap(thumb)
            self.top_path_lbl.setText(path.name)
            self.top_path_lbl.setToolTip(str(path))
        else:
            self.bottom_image_path = path
            self.bot_thumb.setPixmap(thumb)
            self.bot_path_lbl.setText(path.name)
            self.bot_path_lbl.setToolTip(str(path))
        self._refresh_preview()

    def _clear_image(self, slot: str) -> None:
        if slot == "top":
            self.top_image_path = None
            self.top_thumb.clear()
            self.top_thumb.setText("TOP")
            self.top_path_lbl.setText("No top image")
        else:
            self.bottom_image_path = None
            self.bot_thumb.clear()
            self.bot_thumb.setText("BOT")
            self.bot_path_lbl.setText("No bottom image")
        self._refresh_preview()

    def _browse_output_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.out_dir_edit.text().strip()
        )
        if chosen:
            self.out_dir_edit.setText(chosen)

    # --------------------------------------------------------- Settings Gathering
    def _current_layout_settings(self) -> LayoutSettings:
        return LayoutSettings(
            top_image_path=self.top_image_path,
            bottom_image_path=self.bottom_image_path,
            fit_mode=self.fit_combo.currentData(),
            background_mode=self.bg_mode_combo.currentData(),
            background_color=self.bg_color_btn.hex_color,
            vertical_position=self.vpos_slider.value() / 100.0,
            divider_height=self.divider_spin.value(),
            divider_color=self.divider_color_btn.hex_color,
        )

    def _current_badge_settings(self) -> BadgeSettings:
        return BadgeSettings(
            enabled=self.badge_enabled_chk.isChecked(),
            template=self.badge_template_edit.text(),
            position=self.badge_pos_combo.currentData(),
            font_size=self.badge_size_spin.value(),
            text_color=self.badge_fg_btn.hex_color,
            bg_color=self.badge_bg_btn.hex_color,
            bg_opacity=self.badge_opacity_spin.value() / 100.0,
        )

    def _current_export_settings(self) -> ExportSettings:
        out_dir = Path(self.out_dir_edit.text().strip() or (Path.cwd() / "exports"))
        return ExportSettings(
            output_dir=out_dir,
            filename_prefix=self.prefix_edit.text().strip(),
            crf=int(self.crf_combo.currentData()),
            preset=str(self.preset_combo.currentData()),
        )

    # --------------------------------------------------------- Parts & Table
    def _apply_duration_preset(self, minutes: int) -> None:
        self.seg_min_spin.blockSignals(True)
        self.seg_sec_spin.blockSignals(True)
        self.seg_min_spin.setValue(minutes)
        self.seg_sec_spin.setValue(0)
        self.seg_min_spin.blockSignals(False)
        self.seg_sec_spin.blockSignals(False)
        self._sync_preset_buttons()
        self._recalculate_parts()

    def _sync_preset_buttons(self) -> None:
        m = self.seg_min_spin.value()
        s = self.seg_sec_spin.value()
        for preset_m, btn in self.preset_buttons:
            btn.setChecked(s == 0 and m == preset_m)

    def _on_timeframe_changed(self) -> None:
        self._sync_preset_buttons()
        self._recalculate_parts()

    def _on_badge_text_changed(self) -> None:
        self._refresh_parts_table_names()
        self._refresh_preview()

    def _recalculate_parts(self) -> None:
        if self.video_info is None:
            self.parts = []
            self._populate_parts_table()
            self._refresh_preview()
            return

        seg_sec = self.seg_min_spin.value() * 60 + self.seg_sec_spin.value()
        if seg_sec <= 0:
            seg_sec = 60

        offset_sec = self.offset_min_spin.value() * 60 + self.offset_sec_spin.value()
        self.parts = calculate_parts(
            total_duration_sec=self.video_info.duration_sec,
            segment_duration_sec=float(seg_sec),
            start_offset_sec=float(offset_sec),
        )
        self.active_part_index = 1 if self.parts else 0
        self._populate_parts_table()
        self.export_btn.setEnabled(bool(self.parts))
        self._schedule_frame_extract()
        self._refresh_preview()

    def _populate_parts_table(self) -> None:
        self.parts_table.blockSignals(True)
        self.parts_table.setRowCount(len(self.parts))
        template = self.badge_template_edit.text()

        for row, part in enumerate(self.parts):
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk_item.setCheckState(Qt.Checked if part.selected else Qt.Unchecked)
            self.parts_table.setItem(row, 0, chk_item)

            label_txt = format_part_label(template, part.index, part.total_parts)
            part_item = QTableWidgetItem(label_txt)
            part_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.parts_table.setItem(row, 1, part_item)

            range_txt = f"{format_timestamp(part.start_sec)} – {format_timestamp(part.end_sec)}"
            range_item = QTableWidgetItem(range_txt)
            range_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.parts_table.setItem(row, 2, range_item)

            dur_item = QTableWidgetItem(format_duration_short(part.duration_sec))
            dur_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.parts_table.setItem(row, 3, dur_item)

            status_item = QTableWidgetItem(part.status)
            status_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.parts_table.setItem(row, 4, status_item)

        if self.parts:
            self.parts_table.selectRow(max(0, self.active_part_index - 1))

        self.parts_table.blockSignals(False)
        self._update_parts_summary()

    def _refresh_parts_table_names(self) -> None:
        if not self.parts:
            return
        self.parts_table.blockSignals(True)
        template = self.badge_template_edit.text()
        for row, part in enumerate(self.parts):
            item = self.parts_table.item(row, 1)
            if item is not None:
                item.setText(format_part_label(template, part.index, part.total_parts))
        self.parts_table.blockSignals(False)

    def _update_parts_summary(self) -> None:
        selected_count = sum(1 for p in self.parts if p.selected)
        total_count = len(self.parts)
        total_dur = sum(p.duration_sec for p in self.parts if p.selected)
        self.parts_summary_lbl.setText(
            f"Parts Queue ({selected_count}/{total_count} selected · {format_duration_short(total_dur)})"
        )
        if self.export_worker is None or not self.export_worker.isRunning():
            self.export_btn.setEnabled(selected_count > 0 and self.video_info is not None)

    def _set_all_parts_selected(self, selected: bool) -> None:
        self.parts_table.blockSignals(True)
        for row, part in enumerate(self.parts):
            part.selected = selected
            item = self.parts_table.item(row, 0)
            if item is not None:
                item.setCheckState(Qt.Checked if selected else Qt.Unchecked)
        self.parts_table.blockSignals(False)
        self._update_parts_summary()

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        row = item.row()
        if 0 <= row < len(self.parts):
            self.parts[row].selected = item.checkState() == Qt.Checked
            self._update_parts_summary()

    def _on_part_row_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self.parts):
            self.active_part_index = self.parts[row].index
            self._schedule_frame_extract()
            self._refresh_preview()

    def _on_part_row_double_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self.parts):
            part = self.parts[row]
            if part.output_path and part.output_path.exists():
                if os.name == "nt":
                    os.startfile(str(part.output_path))
                else:
                    subprocess.Popen(["xdg-open", str(part.output_path)])

    # --------------------------------------------------------- Live Preview
    def _active_part(self) -> Optional[PartSegment]:
        if not self.parts:
            return None
        idx = max(0, min(len(self.parts) - 1, self.active_part_index - 1))
        return self.parts[idx]

    def _on_scrub_changed(self) -> None:
        self._schedule_frame_extract()

    def _schedule_frame_extract(self) -> None:
        if self.video_info is None:
            return
        self._frame_debounce_timer.start()

    def _request_preview_frame(self) -> None:
        if self.video_info is None:
            return
        part = self._active_part()
        if part is not None:
            frac = self.scrub_slider.value() / 100.0
            ts = part.start_sec + part.duration_sec * frac
        else:
            ts = self.video_info.duration_sec * (self.scrub_slider.value() / 100.0)

        self.scrub_ts_lbl.setText(format_timestamp(ts))
        self.frame_request_counter += 1
        req_id = self.frame_request_counter
        worker = FrameExtractWorker(req_id, self.video_info.path, ts)
        worker.frame_ready.connect(self._on_frame_ready)
        worker.frame_failed.connect(self._on_frame_failed)
        worker.finished.connect(lambda w=worker: self._active_frame_workers.discard(w))
        self._active_frame_workers.add(worker)
        worker.start()

    def _on_frame_ready(self, request_id: int, ts: float, png_bytes: bytes) -> None:
        if request_id != self.frame_request_counter:
            return
        self.current_frame_bytes = png_bytes
        self.scrub_ts_lbl.setText(format_timestamp(ts))
        self._refresh_preview()

    def _on_frame_failed(self, request_id: int, _err: str) -> None:
        if request_id != self.frame_request_counter:
            return
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        layout = self._current_layout_settings()
        badge = self._current_badge_settings()

        val = self.vpos_slider.value()
        if val == 50:
            self.vpos_lbl.setText("Center")
        elif val < 50:
            self.vpos_lbl.setText(f"Top {val}%")
        else:
            self.vpos_lbl.setText(f"Bot {val}%")

        src_w = self.video_info.width if self.video_info else 1920
        src_h = self.video_info.height if self.video_info else 1080
        geo = compute_layout_geometry(src_w, src_h, vertical_position=layout.vertical_position)

        part = self._active_part()
        if part is not None:
            part_num = part.index
            total_parts = part.total_parts
            self.preview_part_lbl.setText(
                f"Previewing: {format_part_label(badge.template, part_num, total_parts)} "
                f"({format_timestamp(part.start_sec)} – {format_timestamp(part.end_sec)})"
            )
        else:
            part_num = 1
            total_parts = 12
            self.preview_part_lbl.setText("Live 9:16 Preview (720×1280 · 720p)")

        self.preview_geo_lbl.setText(
            f"720×1280 · Video {geo.video_width}×{geo.video_height} · "
            f"Top {geo.top_area_height}px · Bottom {geo.bottom_area_height}px"
        )

        img = render_preview_frame(
            frame_png_bytes=self.current_frame_bytes,
            geometry=geo,
            layout=layout,
            badge=badge,
            part_num=part_num,
            total_parts=total_parts,
        )
        self.preview_widget.set_preview_image(img)

    # --------------------------------------------------------- Export Execution
    def _start_export(self) -> None:
        if self.video_info is None:
            return
        selected = [p for p in self.parts if p.selected]
        if not selected:
            QMessageBox.information(self, "Nothing Selected", "Select at least one part to export.")
            return

        layout = self._current_layout_settings()
        badge = self._current_badge_settings()
        export_settings = self._current_export_settings()

        # Reset statuses on selected rows
        for row, part in enumerate(self.parts):
            if part.selected:
                part.status = "Queued"
                part.progress = 0.0
                item = self.parts_table.item(row, 4)
                if item:
                    item.setText("Queued")

        self.log_box.clear()
        self.overall_progress.setValue(0)
        self.export_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.export_worker = ExportWorker(
            video_info=self.video_info,
            parts=self.parts,
            layout=layout,
            badge=badge,
            export_settings=export_settings,
        )
        self.export_worker.part_started.connect(self._on_export_part_started)
        self.export_worker.part_progress.connect(self._on_export_part_progress)
        self.export_worker.part_finished.connect(self._on_export_part_finished)
        self.export_worker.log_message.connect(self._append_log)
        self.export_worker.all_finished.connect(self._on_export_all_finished)
        self.export_worker.start()

    def _cancel_export(self) -> None:
        if self.export_worker and self.export_worker.isRunning():
            self.cancel_btn.setEnabled(False)
            self.progress_status_lbl.setText("Cancelling export...")
            self.export_worker.cancel()

    def _on_export_part_started(self, part_index: int, output_path: str) -> None:
        row = part_index - 1
        if 0 <= row < self.parts_table.rowCount():
            item = self.parts_table.item(row, 4)
            if item:
                item.setText("Exporting 0%")
        self.progress_status_lbl.setText(f"Exporting Part {part_index} -> {Path(output_path).name}")

    def _on_export_part_progress(self, part_index: int, frac: float, speed: str) -> None:
        row = part_index - 1
        pct = int(round(frac * 100))
        if 0 <= row < self.parts_table.rowCount():
            item = self.parts_table.item(row, 4)
            if item:
                item.setText(f"Exporting {pct}%")

        selected = [p for p in self.parts if p.selected]
        if selected:
            total_frac = sum(
                1.0 if p.status == "Done" else (p.progress if p.index == part_index else 0.0)
                for p in selected
            ) / len(selected)
            self.overall_progress.setValue(int(round(total_frac * 100)))

        speed_txt = f" · {speed}" if speed else ""
        self.progress_status_lbl.setText(f"Exporting Part {part_index}: {pct}%{speed_txt}")

    def _on_export_part_finished(self, part_index: int, status: str, error_msg: str) -> None:
        row = part_index - 1
        if 0 <= row < self.parts_table.rowCount():
            item = self.parts_table.item(row, 4)
            if item:
                display = "Done ✓" if status == "Done" else status
                item.setText(display)
                if error_msg:
                    item.setToolTip(error_msg)

    def _append_log(self, msg: str) -> None:
        self.log_box.appendPlainText(msg)

    def _on_export_all_finished(self, ok: bool, summary: str) -> None:
        self.cancel_btn.setEnabled(False)
        self._update_parts_summary()
        if ok:
            self.overall_progress.setValue(100)
        self.progress_status_lbl.setText(summary)
        self._append_log(summary)
        if ok:
            QMessageBox.information(self, "Export Complete", summary)
        elif "cancelled" not in summary.lower():
            QMessageBox.warning(self, "Export Finished", summary)

    def _open_output_folder(self) -> None:
        out_dir = Path(self.out_dir_edit.text().strip() or (Path.cwd() / "exports"))
        out_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(out_dir))
        else:
            subprocess.Popen(["xdg-open", str(out_dir)])

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.cancel()
            self.export_worker.wait(3000)
        for worker in list(self._active_frame_workers):
            if worker.isRunning():
                worker.wait(1000)
        super().closeEvent(event)
