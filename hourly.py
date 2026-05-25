#!/usr/bin/env python3
"""
hourly.py — BLR hourly PM2.5 from oaq.notf.in

Flow:
  1. Fetch station metadata (airnet, aurassure, cpcb)
  2. Filter to GBA bounding box
  3. Fetch last-24h readings per station
  4. Average readings by hour-of-day (0–23) per station
  5. Spatially interpolate to a 0.01° storage grid with a Gaussian-IDW kernel
  6. Save: NetCDF, gridded CSV, hourly-average CSV
  7. Generate PNGs:
       • with_labels / without_labels — matplotlib reference views
       • td/                          — raw greyscale frames sized to the
                                        higher-resolution render grid for
                                        TouchDesigner ingestion

Interpolation
-------------
Each grid cell is a weighted average of all station readings with weights

    w_s(cell) = exp(−distance(cell, s)² / (2·σ²))     (Gaussian, σ in km)
    value(cell) = Σ_s w_s · pm25_s / Σ_s w_s

mirroring the kernel structure used in metro.py (so both visualisations have
the same "smooth bumps" aesthetic). Distances are computed in UTM zone 43N
(EPSG:32643). σ = HOURLY_SIGMA_KM defaults to 2 km, slightly broader than
the metro decay so sparse station coverage doesn't show triangle facets.
"""

import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pyproj

import td_export

warnings.filterwarnings('ignore')

# ── API credentials ────────────────────────────────────────────────────────
_AUTH = (
    "URLPrefix=aHR0cHM6Ly9vYXEubm90Zi5pbi92MS8="
    "&Expires=1778585853"
    "&KeyName=prod-key-1"
    "&Signature=FQhbt-v-pGNwwovPlSOL5EMcmOo="
)
_BASE = "https://oaq.notf.in/v1"
PROVIDERS = ["airnet", "aurassure", "cpcb"]


def _url(path: str) -> str:
    return f"{_BASE}/{path}?{_AUTH}"


# ── GBA bounding box + 0.01° padding ─────────────────────────────────────
LAT_MIN, LAT_MAX = 12.8235, 13.1526
LON_MIN, LON_MAX = 77.4499, 77.7941

# Storage grid (NetCDF/CSV) — kept at 0.01° for backward compatibility
FINE_LAT = np.arange(LAT_MIN, LAT_MAX + 0.005, 0.01)
FINE_LON = np.arange(LON_MIN, LON_MAX + 0.005, 0.01)

# Render grid (PNG / TD frames) — finer so smoothness is visible
RENDER_STEP_DEG = 0.002
RENDER_LAT = np.arange(LAT_MIN, LAT_MAX + RENDER_STEP_DEG / 2, RENDER_STEP_DEG)
RENDER_LON = np.arange(LON_MIN, LON_MAX + RENDER_STEP_DEG / 2, RENDER_STEP_DEG)

# Coarser grid for text labels (one label per ~5 km cell)
COARSE_LAT = np.arange(LAT_MIN + 0.025, LAT_MAX, 0.05)
COARSE_LON = np.arange(LON_MIN + 0.025, LON_MAX, 0.05)

# Gaussian IDW kernel width (km). 2 km gives a soft, continuously varying field
# over ~20–30 sparse stations without obvious facets.
HOURLY_SIGMA_KM = 2.0

OUT     = Path("./")
OUT_RAW = OUT / "raw/hourly"
OUT_NC  = OUT / "nc/hourly"
OUT_PNG = OUT / "png/hourly"

_UTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)


# ── Data fetching ──────────────────────────────────────────────────────────

def fetch_stations() -> List[dict]:
    """Return all BLR stations from all providers."""
    stations = []
    for prov in PROVIDERS:
        url = _url(f"provider={prov}/live/global/all_stations_latest.json")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            for s in resp.json().get("stations", []):
                lat, lon = s.get("lat"), s.get("lon")
                if lat and lon and LAT_MIN <= float(lat) <= LAT_MAX and LON_MIN <= float(lon) <= LON_MAX:
                    stations.append({
                        "id": str(s["id"]),
                        "name": s.get("name", ""),
                        "provider": prov,
                        "lat": float(lat),
                        "lon": float(lon),
                    })
        except Exception as exc:
            print(f"  [warn] {prov} station list: {exc}")
    print(f"Found {len(stations)} BLR stations.")
    return stations


def fetch_readings(station: dict) -> pd.DataFrame:
    """Return DataFrame(hour, pm25) for one station; empty DataFrame if unavailable."""
    url = _url(f"provider={station['provider']}/live/sensors/{station['id']}/last24h.json")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        rows = []
        for r in resp.json().get("data", []):
            if r[1] is not None:
                try:
                    rows.append({"hour": pd.Timestamp(r[0]).hour, "pm25": float(r[1])})
                except Exception:
                    pass
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["hour", "pm25"])
    except Exception as exc:
        print(f"  [warn] {station['provider']} sensor {station['id']}: {exc}")
        return pd.DataFrame(columns=["hour", "pm25"])


def collect_all_readings(
    stations: List[dict],
) -> Tuple[Dict[int, List[Tuple[float, float, float]]], pd.DataFrame]:
    """
    Fetches time-series for every station.

    Returns:
      hour_pts: {hour: [(lat, lon, pm25_avg), ...]}
      raw_df:   all individual readings with station metadata
    """
    hour_pts: Dict[int, list] = {h: [] for h in range(24)}
    raw_rows = []

    for i, st in enumerate(stations):
        df = fetch_readings(st)
        if df.empty:
            continue
        for _, row in df.iterrows():
            raw_rows.append({
                "station_id": st["id"],
                "station_name": st["name"],
                "provider": st["provider"],
                "lat": st["lat"],
                "lon": st["lon"],
                "hour": int(row["hour"]),
                "pm25": row["pm25"],
            })
        for hour, grp in df.groupby("hour"):
            hour_pts[int(hour)].append((st["lat"], st["lon"], grp["pm25"].mean()))
        time.sleep(0.1)
        if (i + 1) % 10 == 0:
            print(f"  Fetched {i + 1}/{len(stations)} stations…")

    return hour_pts, pd.DataFrame(raw_rows)


# ── Interpolation: Gaussian IDW ────────────────────────────────────────────

def _gaussian_idw(
    pts: List[Tuple[float, float, float]],
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    sigma_km: float,
) -> np.ndarray:
    """
    Smoothly interpolate sparse station points onto a regular lat/lon grid.

    weight_s(cell) = exp(−d(cell, s)² / (2·σ²))     # d in km
    value(cell)    = Σ_s w_s · v_s / Σ_s w_s

    Returns a (len(grid_lat), len(grid_lon)) float32 array; NaN only if every
    station weight underflows (i.e. all stations astronomically far away).
    """
    n_lat, n_lon = len(grid_lat), len(grid_lon)
    if not pts:
        return np.full((n_lat, n_lon), np.nan, dtype=np.float32)

    src_lat, src_lon, vals = zip(*pts)
    st_e, st_n = _UTM.transform(list(src_lon), list(src_lat))
    st_e = np.asarray(st_e)
    st_n = np.asarray(st_n)
    vals = np.asarray(vals, dtype=np.float64)

    two_sigma_sq_m2 = 2.0 * (sigma_km * 1000.0) ** 2

    out = np.empty((n_lat, n_lon), dtype=np.float32)
    for i, lat in enumerate(grid_lat):
        cell_e, cell_n = _UTM.transform(grid_lon, np.full(n_lon, lat))
        cell_e = np.asarray(cell_e)
        cell_n = np.asarray(cell_n)
        d_sq = (cell_e[:, None] - st_e[None, :]) ** 2 + (cell_n[:, None] - st_n[None, :]) ** 2
        w = np.exp(-d_sq / two_sigma_sq_m2)
        w_sum = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(w_sum > 0, (w @ vals) / w_sum, np.nan).astype(np.float32)
    return out


def build_storage_dataset(hour_pts: Dict[int, list]) -> xr.Dataset:
    """xr.Dataset with PM25(hour, lat, lon) on the coarse 0.01° storage grid."""
    cube = np.stack([
        _gaussian_idw(hour_pts[h], FINE_LAT, FINE_LON, HOURLY_SIGMA_KM)
        for h in range(24)
    ])
    return xr.Dataset(
        {"PM25": (["hour", "lat", "lon"], cube)},
        coords={"hour": np.arange(24), "lat": FINE_LAT, "lon": FINE_LON},
        attrs={
            "title": "BLR Hourly Average PM2.5 (Gaussian IDW, σ=%.1fkm)" % HOURLY_SIGMA_KM,
            "source": "oaq.notf.in (airnet, aurassure, cpcb)",
            "generated_at": datetime.now().isoformat(),
            "lat_units": "degrees_north",
            "lon_units": "degrees_east",
            "pm25_units": "ug/m3",
            "sigma_km": HOURLY_SIGMA_KM,
        },
    )


def build_render_cube(hour_pts: Dict[int, list]) -> np.ndarray:
    """High-resolution cube used for PNG output."""
    return np.stack([
        _gaussian_idw(hour_pts[h], RENDER_LAT, RENDER_LON, HOURLY_SIGMA_KM)
        for h in range(24)
    ])


# ── Output helpers ─────────────────────────────────────────────────────────

def save_outputs(ds: xr.Dataset) -> None:
    nc_path = OUT_NC / "gridded.nc"
    ds.to_netcdf(nc_path)
    print(f"  NetCDF       → {nc_path}")

    df = ds["PM25"].to_dataframe().reset_index().dropna(subset=["PM25"])
    csv_path = OUT_RAW / "gridded.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Gridded CSV  → {csv_path}")


def generate_pngs(render_cube: np.ndarray) -> None:
    boundary_mask = (
        td_export.rasterize_polygon(td_export.TD_BOUNDARY_GEOJSON, RENDER_LAT, RENDER_LON)
        if td_export.TD_BOUNDARY_GEOJSON else None
    )
    vmin, vmax = td_export.cube_percentile_range(render_cube, mask=boundary_mask)

    rglon, rglat = np.meshgrid(RENDER_LON, RENDER_LAT)
    cglon, cglat = np.meshgrid(COARSE_LON, COARSE_LAT)

    for with_labels in (False, True):
        tag = "with_labels" if with_labels else "without_labels"
        out_dir = OUT_PNG / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        for hour in range(24):
            pm25 = render_cube[hour]
            fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
            im = ax.imshow(
                pm25,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                aspect="auto",
                interpolation="bilinear",
            )

            if with_labels:
                label_vals = griddata(
                    np.column_stack([rglat.ravel(), rglon.ravel()]),
                    pm25.ravel(),
                    np.column_stack([cglat.ravel(), cglon.ravel()]),
                    method="nearest",
                ).reshape(cglat.shape)
                for i in range(cglat.shape[0]):
                    for j in range(cglat.shape[1]):
                        v = label_vals[i, j]
                        if np.isnan(v):
                            continue
                        brightness = (v - vmin) / max(vmax - vmin, 1.0)
                        color = "black" if brightness > 0.5 else "white"
                        ax.text(
                            cglon[i, j], cglat[i, j],
                            f"{v:.0f}",
                            ha="center", va="center",
                            fontsize=6, color=color, fontweight="bold",
                        )

            ax.set_title(f"BLR PM2.5  —  {hour:02d}:00 IST", fontsize=12)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.colorbar(im, ax=ax, label="PM2.5 (µg/m³)", fraction=0.035, pad=0.04)
            plt.tight_layout()
            plt.savefig(
                out_dir / f"{hour:02d}.png",
                dpi=150, bbox_inches="tight", facecolor="white",
            )
            plt.close()

        print(f"  24 PNGs ({tag:<14}) → {out_dir}/")

    td_export.write_sequence(
        (render_cube[h] for h in range(24)),
        OUT_PNG / "td",
        vmin=vmin,
        vmax=vmax,
        unit="ug_per_m3",
        target_long_edge=td_export.TD_LONG_EDGE,
        lats=RENDER_LAT,
        lons=RENDER_LON,
        mask=boundary_mask,
    )
    print(f"  24 PNGs ({'td':<14}) → {OUT_PNG / 'td'}/  ({td_export.TD_LONG_EDGE}px long edge, source {render_cube.shape[2]}×{render_cube.shape[1]})")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("BLR Hourly PM2.5  |  oaq.notf.in  |  Gaussian IDW σ=%.1f km" % HOURLY_SIGMA_KM)
    print("=" * 60)
    for d in (OUT, OUT_RAW, OUT_NC, OUT_PNG):
        d.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Fetching station list…")
    stations = fetch_stations()
    if not stations:
        print("No stations found. Exiting.")
        return

    print(f"\n[2/5] Fetching last-24h time-series for {len(stations)} stations…")
    hour_pts, raw_df = collect_all_readings(stations)
    raw_csv = OUT_RAW / "hourly.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"  Raw readings → {raw_csv}  ({len(raw_df)} rows)")

    if raw_df.empty:
        print("No readings returned from any station. Exiting.")
        return

    print("\n[3/5] Interpolating to 0.01° storage grid (Gaussian IDW)…")
    ds = build_storage_dataset(hour_pts)

    print("\n[4/5] Saving NetCDF + CSVs…")
    save_outputs(ds)

    print(f"\n[5/5] Rendering at 0.002° ({len(RENDER_LAT)}×{len(RENDER_LON)}) + writing PNGs…")
    render_cube = build_render_cube(hour_pts)
    generate_pngs(render_cube)

    print(f"\nDone. All outputs in: {OUT.resolve()}")


if __name__ == "__main__":
    main()
