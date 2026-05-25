#!/usr/bin/env python3
"""
build_strips.py — bundle per-hour TD frames into one sprite-sheet PNG per set.

Layout: 6 columns × 4 rows (24 frames total). Index → (row, col):
    row = hour // 6
    col = hour %  6

A horizontal strip (24×1) would exceed the 16 384 px texture limit common on
laptop GPUs (24 × 1024 = 24 576). The 6×4 grid sits at 6144 × 3892, which
fits well within an 8 192 limit.

For each set we write:
    docs/aqi/<set>/strip.png      — sprite sheet (LA mode preserved)
    docs/aqi/<set>/manifest.json  — frame metadata + grid layout + vmin/vmax

A tiny docs/index.html lists the URLs for sanity-checking once Pages is live.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

SOURCE_ROOT = Path("png")
OUT_ROOT    = Path("docs/aqi")
SETS        = ("hourly", "avoided", "net")

GRID_COLS = 6
GRID_ROWS = 4
FRAME_COUNT = GRID_COLS * GRID_ROWS  # 24


def build_one(name: str) -> None:
    src_dir = SOURCE_ROOT / name / "td"
    range_path = src_dir / "range.json"
    if not range_path.exists():
        print(f"  [skip] {name}: no {range_path}")
        return

    range_meta = json.loads(range_path.read_text())

    frame_paths = sorted(src_dir.glob("[0-9][0-9].png"))
    if len(frame_paths) != FRAME_COUNT:
        raise RuntimeError(
            f"{name}: expected {FRAME_COUNT} frames, found {len(frame_paths)}"
        )

    frames = [Image.open(p) for p in frame_paths]
    mode = frames[0].mode
    fw, fh = frames[0].size
    for p, im in zip(frame_paths, frames):
        if im.size != (fw, fh) or im.mode != mode:
            raise RuntimeError(
                f"{name}: inconsistent frame {p} ({im.mode}, {im.size}) vs ({mode}, {(fw, fh)})"
            )

    strip = Image.new(mode, (fw * GRID_COLS, fh * GRID_ROWS))
    for i, im in enumerate(frames):
        col = i % GRID_COLS
        row = i // GRID_COLS
        strip.paste(im, (col * fw, row * fh))

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    strip_path = out_dir / "strip.png"
    strip.save(strip_path, optimize=True)

    manifest = {
        "name": name,
        "frame_count": FRAME_COUNT,
        "grid": {"cols": GRID_COLS, "rows": GRID_ROWS},
        "frame_w": fw,
        "frame_h": fh,
        "strip_w": fw * GRID_COLS,
        "strip_h": fh * GRID_ROWS,
        "mode": mode,
        "vmin": range_meta["vmin"],
        "vmax": range_meta["vmax"],
        "unit": range_meta["unit"],
        "mask_extent": range_meta.get("mask_extent"),
        "source_grid_wh": range_meta.get("source_grid_wh"),
        "pixel_mapping": "value = (gray/255) * (vmax - vmin) + vmin; alpha=0 outside polygon",
        "frame_at": "row = hour // cols, col = hour % cols",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    size_kb = strip_path.stat().st_size / 1024
    print(f"  {name:8} → {strip_path}  ({strip.size[0]}×{strip.size[1]}, {size_kb:.0f} KB)")


def write_index_html() -> None:
    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    body_rows = "\n".join(
        f'      <tr><td><code>{s}</code></td>'
        f'<td><a href="aqi/{s}/strip.png">strip.png</a></td>'
        f'<td><a href="aqi/{s}/manifest.json">manifest.json</a></td></tr>'
        for s in SETS
    )
    setup_link = '<p><a href="td_setup.py">td_setup.py</a> — paste into TouchDesigner Textport (Alt-T) to build the ingestion network inside <code>/project1/aqi_animation</code>.</p>'
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BLR Metro AQI — TD assets</title>
  <style>
    body {{ font: 14px/1.5 ui-monospace, monospace; max-width: 720px; margin: 40px auto; padding: 0 16px; color: #222; }}
    h1   {{ font-size: 18px; margin-bottom: 4px; }}
    p    {{ color: #555; }}
    table{{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    td, th {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }}
    th   {{ font-weight: 600; color: #666; }}
    a    {{ color: #0a52d6; }}
    code {{ background: #f6f6f6; padding: 1px 5px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>BLR Metro AQI — TouchDesigner assets</h1>
  <p>24-hour PM2.5 animations clipped to the Bengaluru boundary. Each strip is a 6×4 grid (row = hour ÷ 6, col = hour mod 6). Pull <code>strip.png</code> once with <code>Movie File In TOP</code>, fetch <code>manifest.json</code> with <code>Web Client DAT</code> for <code>vmin/vmax</code> and grid layout.</p>
  <table>
    <thead><tr><th>set</th><th>strip</th><th>manifest</th></tr></thead>
    <tbody>
{body_rows}
    </tbody>
  </table>
  {setup_link}
  <p style="margin-top:24px">Source: <a href="https://github.com/diagram-chasing/metro-aqi-exhibition">diagram-chasing/metro-aqi-exhibition</a></p>
</body>
</html>
"""
    (OUT_ROOT.parent / "index.html").write_text(html)
    print(f"  index    → {OUT_ROOT.parent / 'index.html'}")


def copy_td_setup() -> None:
    src = Path("td_setup.py")
    if not src.exists():
        return
    dest = OUT_ROOT.parent / "td_setup.py"
    dest.write_bytes(src.read_bytes())
    print(f"  td_setup → {dest}")


def main() -> None:
    print("=" * 60)
    print(f"Building {GRID_COLS}×{GRID_ROWS} sprite strips → {OUT_ROOT}/")
    print("=" * 60)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in SETS:
        build_one(name)
    write_index_html()
    copy_td_setup()
    print("\nDone.")


if __name__ == "__main__":
    main()
