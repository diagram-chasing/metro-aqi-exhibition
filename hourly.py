#!/usr/bin/env python3
"""
hourly.py — BLR hourly PM2.5 from oaq.notf.in

Flow:
  1. Fetch station metadata (airnet, aurassure, cpcb)
  2. Filter to GBA bounding box
  3. Fetch last-24h readings per station
  4. Average readings by hour-of-day (0–23) per station
  5. Spatially interpolate to a 0.01° grid for each hour
  6. Save: NetCDF, gridded CSV, hourly-average CSV
  7. Generate 24 PNG heatmaps — one version with numbers, one without

Interpolation:
  scipy.interpolate.griddata with method="linear" is used when ≥ 3 station
  points are available for an hour; otherwise falls back to "nearest".

  Linear interpolation works in two stages:
    (a) Delaunay triangulation — the station locations are triangulated so
        the city is tiled by non-overlapping triangles.
    (b) Barycentric interpolation — for each 0.01° grid point P inside a
        triangle with vertices A, B, C, the PM2.5 value is the area-weighted
        average of the three station values:

          PM2.5(P) = λ_A·v_A + λ_B·v_B + λ_C·v_C
          where λ_i = area of sub-triangle opposite vertex i / area(ABC)
          and λ_A + λ_B + λ_C = 1

  Grid points outside the convex hull of stations (no enclosing triangle)
  are filled with the nearest-neighbour value to avoid edge NaNs.
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
#   S 12.8334905  N 13.1426196  W 77.4598797  E 77.7840639
LAT_MIN, LAT_MAX = 12.8235, 13.1526
LON_MIN, LON_MAX = 77.4499, 77.7941

# Fine grid for interpolated heatmap (~55×55 points)
FINE_LAT = np.arange(LAT_MIN, LAT_MAX + 0.005, 0.01)
FINE_LON = np.arange(LON_MIN, LON_MAX + 0.005, 0.01)

# Coarser grid for text labels (one label per ~5 km cell)
COARSE_LAT = np.arange(LAT_MIN + 0.025, LAT_MAX, 0.05)
COARSE_LON = np.arange(LON_MIN + 0.025, LON_MAX, 0.05)

OUT     = Path("./")
OUT_RAW = OUT / "raw/hourly"
OUT_NC  = OUT / "nc/hourly"
OUT_PNG = OUT / "png/hourly"


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


# ── Interpolation ──────────────────────────────────────────────────────────

def _interp_hour(
    pts: List[Tuple[float, float, float]],
    grid_lat_2d: np.ndarray,
    grid_lon_2d: np.ndarray,
) -> np.ndarray:
    """Interpolate sparse station points onto a 2-D grid."""
    if not pts:
        return np.full(grid_lat_2d.shape, np.nan)

    src_lat, src_lon, vals = zip(*pts)
    src = np.column_stack([src_lat, src_lon])
    dst = np.column_stack([grid_lat_2d.ravel(), grid_lon_2d.ravel()])

    method = "nearest" if len(pts) < 3 else "linear"
    result = griddata(src, vals, dst, method=method).reshape(grid_lat_2d.shape)

    if method == "linear" and np.any(np.isnan(result)):
        nearest = griddata(src, vals, dst, method="nearest").reshape(grid_lat_2d.shape)
        result = np.where(np.isnan(result), nearest, result)

    return result.astype(np.float32)


def build_dataset(hour_pts: Dict[int, list]) -> xr.Dataset:
    """Build xr.Dataset with PM25(hour, lat, lon)."""
    glon, glat = np.meshgrid(FINE_LON, FINE_LAT)
    cube = np.stack([_interp_hour(hour_pts[h], glat, glon) for h in range(24)])
    return xr.Dataset(
        {"PM25": (["hour", "lat", "lon"], cube)},
        coords={"hour": np.arange(24), "lat": FINE_LAT, "lon": FINE_LON},
        attrs={
            "title": "BLR Hourly Average PM2.5",
            "source": "oaq.notf.in (airnet, aurassure, cpcb)",
            "generated_at": datetime.now().isoformat(),
            "lat_units": "degrees_north",
            "lon_units": "degrees_east",
            "pm25_units": "ug/m3",
        },
    )


# ── Output helpers ─────────────────────────────────────────────────────────

def save_outputs(ds: xr.Dataset) -> None:
    nc_path = OUT_NC / "gridded.nc"
    ds.to_netcdf(nc_path)
    print(f"  NetCDF       → {nc_path}")

    df = ds["PM25"].to_dataframe().reset_index().dropna(subset=["PM25"])
    csv_path = OUT_RAW / "gridded.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Gridded CSV  → {csv_path}")


def generate_pngs(ds: xr.Dataset) -> None:
    cube = ds["PM25"].values  # (24, lat, lon)
    vmin = float(np.nanpercentile(cube, 2))
    vmax = float(np.nanpercentile(cube, 98))
    if vmin >= vmax:
        vmin, vmax = 0.0, max(float(np.nanmax(cube)), 1.0)

    glon, glat = np.meshgrid(FINE_LON, FINE_LAT)
    cglon, cglat = np.meshgrid(COARSE_LON, COARSE_LAT)

    for with_labels in (False, True):
        tag = "with_labels" if with_labels else "without_labels"
        out_dir = OUT_PNG / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        for hour in range(24):
            fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
            pm25 = cube[hour]

            im = ax.imshow(
                pm25,
                cmap="gray",          # pixel brightness ≡ PM2.5 level
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                aspect="auto",
                interpolation="nearest",
            )

            if with_labels:
                label_vals = griddata(
                    np.column_stack([glat.ravel(), glon.ravel()]),
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

        print(f"  24 PNGs ({tag:<12}) → {out_dir}/")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("BLR Hourly PM2.5  |  oaq.notf.in")
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

    print("\n[3/5] Interpolating to 0.01° grid…")
    ds = build_dataset(hour_pts)

    print("\n[4/5] Saving NetCDF + CSVs…")
    save_outputs(ds)

    print("\n[5/5] Generating PNGs…")
    generate_pngs(ds)

    print(f"\nDone. All outputs in: {OUT.resolve()}")


if __name__ == "__main__":
    main()
