"""
td_export.py — write raw greyscale PNG frames sized to the data grid.

Frames are intended for TouchDesigner ingestion: no axes, titles, colorbars,
margins, or padding. Pixel dimensions equal the input array dimensions.

Convention:
  * pixel value 0   → vmin
  * pixel value 255 → vmax
  * arrays are flipped along latitude so frame Y matches screen Y
    (matplotlib origin="lower" → image origin="upper")

A `range.json` sidecar in each output directory records vmin/vmax and the
physical unit so the mapping back to data is preserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image

# Shared long-edge size for TouchDesigner frame output. Underlying fields are
# smooth Gaussian kernels, so Lanczos upsampling is faithful.
TD_LONG_EDGE = 1024

# Default boundary used to clip TD frames to the actual Bengaluru polygon
# instead of the rectangular GBA bbox. Falsy → no clipping.
TD_BOUNDARY_GEOJSON: Path | None = Path("blr-boundary.geojson")


def _polygon_rings(geojson_path: Path) -> list[np.ndarray]:
    """Return [outer_ring, *holes] as Nx2 (lon, lat) arrays. Picks the first
    Polygon/MultiPolygon feature found."""
    gj = json.loads(Path(geojson_path).read_text())
    for feat in gj["features"]:
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            return [np.asarray(r, dtype=np.float64) for r in geom["coordinates"]]
        if geom["type"] == "MultiPolygon":
            # take the polygon with the largest outer ring (by vertex count)
            polys = geom["coordinates"]
            outer = max(polys, key=lambda p: len(p[0]))
            return [np.asarray(r, dtype=np.float64) for r in outer]
    raise ValueError(f"no Polygon/MultiPolygon feature in {geojson_path}")


def rasterize_polygon(
    geojson_path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Return a (len(lats), len(lons)) bool mask — True inside the polygon
    (after subtracting holes). lats/lons are 1-D coordinate arrays."""
    rings = _polygon_rings(geojson_path)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    outer = MplPath(rings[0])
    mask = outer.contains_points(pts).reshape(lat_grid.shape)
    for hole in rings[1:]:
        inside_hole = MplPath(hole).contains_points(pts).reshape(lat_grid.shape)
        mask &= ~inside_hole
    return mask


def _crop_to_mask(mask: np.ndarray) -> tuple[slice, slice]:
    """Return (row_slice, col_slice) trimming all-False borders. If the mask
    is empty, returns the full extent."""
    if not mask.any():
        return slice(None), slice(None)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    return slice(r0, r1), slice(c0, c1)


def _to_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    span = max(vmax - vmin, 1e-12)
    scaled = (np.asarray(arr, dtype=np.float64) - vmin) / span
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled = np.where(np.isnan(scaled), 0.0, scaled)
    return (scaled * 255.0 + 0.5).astype(np.uint8)


def _resize_to_long_edge(img: Image.Image, target_long_edge: int) -> Image.Image:
    """Lanczos-resize so the longer edge equals target_long_edge, preserving
    aspect ratio. Returns the input untouched if it already matches or if
    target_long_edge is falsy."""
    if not target_long_edge:
        return img
    w, h = img.size
    long_edge = max(w, h)
    if long_edge == target_long_edge:
        return img
    scale = target_long_edge / long_edge
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(new_size, Image.LANCZOS)


def _compose_image(
    arr: np.ndarray,
    vmin: float,
    vmax: float,
    mask: np.ndarray | None,
    crop: tuple[slice, slice] | None,
) -> Image.Image:
    """Quantise + mask + flip + crop, return a PIL image (L or LA)."""
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {arr.shape}")
    gray = _to_uint8(arr, vmin, vmax)
    if mask is None:
        flipped = np.flipud(gray)
        return Image.fromarray(flipped, mode="L")

    if mask.shape != arr.shape:
        raise ValueError(f"mask shape {mask.shape} != arr shape {arr.shape}")
    alpha = (mask.astype(np.uint8) * 255)
    gray = np.where(mask, gray, 0).astype(np.uint8)
    if crop is not None:
        rs, cs = crop
        gray = gray[rs, cs]
        alpha = alpha[rs, cs]
    la = np.stack([np.flipud(gray), np.flipud(alpha)], axis=-1)
    return Image.fromarray(la, mode="LA")


def write_frame(
    arr: np.ndarray,
    path: Path,
    vmin: float,
    vmax: float,
    target_long_edge: int | None = None,
    mask: np.ndarray | None = None,
    crop: tuple[slice, slice] | None = None,
) -> None:
    """Write a single 2D array as an axis-flipped PNG.

    - If `mask` is provided, outputs an LA PNG with alpha=0 outside the mask.
    - If `crop` is provided, trims to those row/col slices before saving.
    - If `target_long_edge` is provided, Lanczos-resizes after cropping.
    """
    img = _compose_image(arr, vmin, vmax, mask, crop)
    img = _resize_to_long_edge(img, target_long_edge or 0)
    img.save(path)


def write_sequence(
    frames: Iterable[np.ndarray],
    out_dir: Path,
    vmin: float,
    vmax: float,
    unit: str,
    name_template: str = "{i:02d}.png",
    target_long_edge: int | None = None,
    mask: np.ndarray | None = None,
    lats: np.ndarray | None = None,
    lons: np.ndarray | None = None,
    geojson_path: Path | None = None,
) -> None:
    """Write a sequence of frames + a range.json sidecar.

    To mask + crop to a polygon, pass `mask` (or `geojson_path`) plus `lats`
    and `lons` describing the input grid.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if mask is None and geojson_path is not None:
        if lats is None or lons is None:
            raise ValueError("lats and lons required when geojson_path is given")
        mask = rasterize_polygon(geojson_path, lats, lons)

    crop: tuple[slice, slice] | None = None
    crop_extent: dict | None = None
    if mask is not None:
        crop = _crop_to_mask(mask)
        if lats is not None and lons is not None:
            rs, cs = crop
            crop_extent = {
                "lat_min": float(lats[rs][0]),
                "lat_max": float(lats[rs][-1]),
                "lon_min": float(lons[cs][0]),
                "lon_max": float(lons[cs][-1]),
            }

    n = 0
    last_arr_shape: tuple[int, int] | None = None
    for i, frame in enumerate(frames):
        write_frame(
            frame, out_dir / name_template.format(i=i),
            vmin, vmax, target_long_edge=target_long_edge,
            mask=mask, crop=crop,
        )
        n += 1
        last_arr_shape = (frame.shape[1], frame.shape[0])

    sidecar = {
        "vmin": float(vmin),
        "vmax": float(vmax),
        "unit": unit,
        "frame_count": n,
        "source_grid_wh": list(last_arr_shape) if last_arr_shape else None,
        "target_long_edge": target_long_edge,
        "resample": "Lanczos" if target_long_edge else "none",
        "masked": mask is not None,
        "mask_extent": crop_extent,
        "pixel_mapping": "0 → vmin, 255 → vmax (linear); alpha=0 outside mask",
        "axis_order": "image: row=lat (top=north), col=lon (left=west)",
        "note": "arrays were flipped along latitude before save",
    }
    (out_dir / "range.json").write_text(json.dumps(sidecar, indent=2))


def cube_percentile_range(
    cube: np.ndarray,
    lo: float = 2.0,
    hi: float = 98.0,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Robust vmin/vmax across an entire cube (skips NaN). If a 2D mask is
    given (lat, lon), the percentile is computed only over masked pixels —
    use this when frames are clipped to a polygon, so the dark outside doesn't
    swing the contrast."""
    if mask is not None:
        if mask.shape != cube.shape[-2:]:
            raise ValueError(f"mask {mask.shape} != frame {cube.shape[-2:]}")
        values = cube[..., mask]
    else:
        values = cube
    vmin = float(np.nanpercentile(values, lo))
    vmax = float(np.nanpercentile(values, hi))
    if vmin >= vmax:
        vmin = 0.0
        vmax = max(float(np.nanmax(values)), 1.0)
    return vmin, vmax
