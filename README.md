# AutoCutter Studio

Desktop application that takes a landscape video (e.g., a full movie), cuts it into timed parts (e.g., 10-minute shorts), converts each part to **720p vertical 9:16 (`720×1280`)**, adds custom **top and bottom images**, and displays **which part is in the video** both inside the app's parts queue/preview and as a customizable burned-in `"Part X"` badge on the exported video.

## Requirements

- **Python 3.10+**
- **FFmpeg & FFprobe** installed and available on `PATH`
- Python packages: `PyQt5`, `Pillow`

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Or double-click `run.bat` on Windows.

## Features

- **Time-Frame Splitting**: Presets (`1m`, `3m`, `5m`, `10m`, `15m`, `30m`) plus custom minutes/seconds and start offset.
- **9:16 720p Conversion (`720×1280`)**: Scales the landscape video proportionally to `720px` width and places it cleanly between top and bottom banner zones.
- **Top & Bottom Images**: Separate top and bottom image pickers with `Cover`, `Contain`, or `Stretch` fit modes, adjustable vertical video position, divider bar, and `Blurred Video` or `Solid Color` backdrop.
- **Part Indicator ("Which Part")**:
  - **In the App**: Interactive Parts Queue table (`Part 1`, time frame, duration, export status) + Live 9:16 preview showing the exact frame and badge for whichever part you click.
  - **In the Exported Video**: Crisp burned-in `"Part 1"` / `"Part 1 of N"` pill badge with customizable position, font size, colors, and opacity.
- **Subtitle Merging (`.srt`, `.ass`, `.ssa`, `.vtt`)**:
  - Automatically slices and shifts subtitle timestamps for each cut part so Part 2, Part 3, etc. stay perfectly synced.
  - Supports **Burn-in** (visible on video), **Soft Track** (`mov_text` CC stream inside MP4), or **Both**, with live preview, position presets, font size, color, box background, and sync offset (`±s`).
- **Selective & Fast Export**: Check/uncheck specific parts to export, real-time FFmpeg progress & speed readout, cancel support, and double-click finished rows to play.
