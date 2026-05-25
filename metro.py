#!/usr/bin/env python3
"""
metro.py — BLR metro PM2.5 ridership-weighted reduction fraction CUBE

Flow:
  1. Load hourly entry/exit ridership from Parquet files
  2. Average ridership per (station, hour) across all observed days
  3. Match stations to coordinates from metro.geojson
  4. For each hour h, build a ridership-weighted exponential kernel:
       score_h[cell] = Σ_s  ridership_s,h × exp(−distance(cell, s) / DECAY_KM)
  5. Normalise so the GLOBAL peak across all 24 hours = NEAR_METRO_REDUCTION
     (avoids per-hour brightness flicker in the animation)
  6. Add a uniform baseline so the CUBE-WIDE mean = CITY_TARGET_REDUCTION
  7. Save: NetCDF, gridded CSV, station CSV
  8. Generate matplotlib PNGs (with/without label overlays) + raw TD frames

Methodology
-----------
Produces ReductionFraction[hour, lat, lon] used in main.py as:

    NetPM25[hour] = ObsPM25[hour] / (1 − ReductionFraction[hour])

ObsPM25 already reflects the benefit of current metro ridership. Dividing
recovers the PM2.5 level that would exist if metro users drove instead.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pyproj

import td_export

warnings.filterwarnings('ignore')

# ── Reduction parameters (literature-calibrated) ──────────────────────────────
NEAR_METRO_REDUCTION  = 0.25   # peak reduction at the highest-influence cell
DECAY_KM              = 1.5    # exponential decay distance of each station [km]
CITY_TARGET_REDUCTION = 0.05   # cube-wide average target

# ── Bounding box (same as hourly.py) ─────────────────────────────────────────
LAT_MIN, LAT_MAX = 12.8235, 13.1526
LON_MIN, LON_MAX = 77.4499, 77.7941

FINE_LAT = np.arange(LAT_MIN, LAT_MAX + 0.005, 0.01)
FINE_LON = np.arange(LON_MIN, LON_MAX + 0.005, 0.01)

# Higher-resolution grid for PNG / TD rendering (matches hourly.py RENDER_*)
RENDER_STEP_DEG = 0.002
RENDER_LAT = np.arange(LAT_MIN, LAT_MAX + RENDER_STEP_DEG / 2, RENDER_STEP_DEG)
RENDER_LON = np.arange(LON_MIN, LON_MAX + RENDER_STEP_DEG / 2, RENDER_STEP_DEG)

COARSE_LAT = np.arange(LAT_MIN + 0.025, LAT_MAX, 0.05)
COARSE_LON = np.arange(LON_MIN + 0.025, LON_MAX, 0.05)

OUT     = Path("./")
OUT_RAW = OUT / "raw/avoided"
OUT_NC  = OUT / "nc/avoided"
OUT_PNG = OUT / "png/avoided"

NAME_MAP = {
    "Sri Balagangadharanatha Swamiji Station, Hosahalli": "Sri Balagangadharanatha Swamiji Stn., Hosahalli",
    "Vijayanagar": "Vijayanagara",
}

_UTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)


def load_hourly_ridership() -> pd.DataFrame:
    """Return DataFrame(Station, Hour, ridership) — entries+exits averaged
    over all observed days, so each (station, hour) has one number."""
    df_entries = pd.read_parquet("raw/station-hourly.parquet")
    df_exits   = pd.read_parquet("raw/station-hourly-exits.parquet")

    e = df_entries.groupby(["Station", "Hour"])["Ridership"].mean().rename("entries")
    x = df_exits.groupby(["Station", "Hour"])["Ridership"].mean().rename("exits")
    df = pd.concat([e, x], axis=1).reset_index()
    df["entries"]   = df["entries"].fillna(0)
    df["exits"]     = df["exits"].fillna(0)
    df["ridership"] = df["entries"] + df["exits"]
    return df[["Station", "Hour", "ridership"]]


def load_station_coords() -> dict:
    with open("raw/metro.geojson") as f:
        gj = json.load(f)
    coords = {}
    for feat in gj["features"]:
        if feat["geometry"]["type"] != "Point":
            continue
        name = feat["properties"]["fullName"]
        lon, lat = feat["geometry"]["coordinates"]
        coords[name] = (lat, lon)
    return coords


def build_station_table() -> tuple[pd.DataFrame, np.ndarray]:
    """Return:
      stations_df: (station, lat, lon) per metro station
      ridership:   (n_stations, 24) hourly ridership array, row-aligned with stations_df
    """
    hourly = load_hourly_ridership()
    coords = load_station_coords()

    matched_rows, unmatched = [], set()
    for st in hourly["Station"].unique():
        name = NAME_MAP.get(st, st)
        if name not in coords:
            unmatched.add(st)
            continue
        lat, lon = coords[name]
        matched_rows.append({"station": st, "lat": lat, "lon": lon})

    if unmatched:
        print(f"  [warn] No coordinates for {len(unmatched)} station(s): {sorted(unmatched)}")

    stations_df = pd.DataFrame(matched_rows)

    # Build (n_stations × 24) ridership matrix in the same order as stations_df
    pivot = (
        hourly.pivot_table(index="Station", columns="Hour", values="ridership", fill_value=0.0)
        .reindex(index=stations_df["station"], columns=range(24), fill_value=0.0)
    )
    ridership = pivot.to_numpy(dtype=np.float64)
    return stations_df, ridership


def _score_on_grid(
    grid_lat: np.ndarray, grid_lon: np.ndarray,
    st_e: np.ndarray, st_n: np.ndarray, ridership: np.ndarray,
) -> np.ndarray:
    """Return (24, n_lat, n_lon) un-normalised score cube on the given grid."""
    n_lat, n_lon = len(grid_lat), len(grid_lon)
    n_stations = len(st_e)
    kernel = np.empty((n_lat, n_lon, n_stations), dtype=np.float32)
    for i, lat in enumerate(grid_lat):
        cell_e, cell_n = _UTM.transform(grid_lon, np.full(n_lon, lat))
        d_km = np.sqrt(
            (np.asarray(cell_e)[:, None] - st_e[None, :]) ** 2 +
            (np.asarray(cell_n)[:, None] - st_n[None, :]) ** 2
        ) / 1000.0
        kernel[i] = np.exp(-d_km / DECAY_KM).astype(np.float32)

    flat = kernel.reshape(-1, n_stations)
    score = (flat @ ridership.astype(np.float32)).reshape(n_lat, n_lon, 24)
    return np.transpose(score, (2, 0, 1))  # (24, lat, lon)


def compute_reduction_cubes(
    stations_df: pd.DataFrame, ridership: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the reduction cube on both the storage grid (NetCDF) and the render
    grid (PNG/TD). Both cubes share the same global normalisation + baseline so
    they're directly comparable to each other and to main.py's NetPM25 cube.

    Returns (storage_cube, render_cube), each (24, n_lat, n_lon) float32.
    """
    # Project all stations once
    st_e, st_n = _UTM.transform(stations_df["lon"].to_numpy(),
                                stations_df["lat"].to_numpy())
    st_e = np.asarray(st_e)
    st_n = np.asarray(st_n)

    print(f"  Storage kernel : {len(FINE_LAT)}×{len(FINE_LON)}×{len(st_e)} cells…")
    score_storage = _score_on_grid(FINE_LAT, FINE_LON, st_e, st_n, ridership)
    print(f"  Render kernel  : {len(RENDER_LAT)}×{len(RENDER_LON)}×{len(st_e)} cells…")
    score_render  = _score_on_grid(RENDER_LAT, RENDER_LON, st_e, st_n, ridership)

    # Normalise both by the render-grid peak so the calibration anchors to the
    # actual maximum the eye sees in the animation.
    score_max = float(score_render.max())
    if score_max == 0:
        return np.zeros_like(score_storage), np.zeros_like(score_render)

    storage = score_storage / score_max * NEAR_METRO_REDUCTION
    render  = score_render  / score_max * NEAR_METRO_REDUCTION

    # Use the render cube's mean for baseline so the city-wide target is met
    # exactly on the resolution we actually look at.
    baseline = max(0.0, CITY_TARGET_REDUCTION - float(render.mean()))
    storage = np.clip(storage + baseline, 0.0, NEAR_METRO_REDUCTION)
    render  = np.clip(render  + baseline, 0.0, NEAR_METRO_REDUCTION)

    print(f"  Peak reduction   : {float(render.max()) * 100:.1f}%  (target ≤{NEAR_METRO_REDUCTION*100:.0f}%)")
    print(f"  Baseline added   : {baseline * 100:.2f}%")
    print(f"  Cube-wide average: {float(render.mean()) * 100:.2f}%  (target {CITY_TARGET_REDUCTION*100:.0f}%)")
    print(f"  Per-hour means   : {[f'{float(render[h].mean())*100:.1f}%' for h in (0, 6, 9, 12, 18, 21)]}  (hours 0,6,9,12,18,21)")
    return storage.astype(np.float32), render.astype(np.float32)


def build_dataset(cube: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {"ReductionFraction": (["hour", "lat", "lon"], cube)},
        coords={"hour": np.arange(24), "lat": FINE_LAT, "lon": FINE_LON},
        attrs={
            "title":                  "BLR Metro PM2.5 Hourly Reduction Fraction Cube (ridership-weighted)",
            "source":                 "station-hourly.parquet, station-hourly-exits.parquet, metro.geojson",
            "generated_at":           datetime.now().isoformat(),
            "lat_units":              "degrees_north",
            "lon_units":              "degrees_east",
            "near_metro_reduction":   NEAR_METRO_REDUCTION,
            "decay_km":               DECAY_KM,
            "city_target_reduction":  CITY_TARGET_REDUCTION,
            "note": (
                "ReductionFraction[hour, cell] = normalised ridership-weighted kernel + baseline. "
                "Applied in main.py as: NetPM25 = ObsPM25 / (1 − ReductionFraction). "
                "Global peak calibrated to 25%; cube-wide mean to 5%."
            ),
        },
    )


def save_outputs(ds: xr.Dataset, stations_df: pd.DataFrame, ridership: np.ndarray) -> None:
    nc_path = OUT_NC / "avoided_gridded.nc"
    ds.to_netcdf(nc_path)
    print(f"  NetCDF         → {nc_path}")

    grid_df = ds["ReductionFraction"].to_dataframe().reset_index().dropna(subset=["ReductionFraction"])
    grid_df.to_csv(OUT_RAW / "avoided_gridded.csv", index=False)
    print(f"  Gridded CSV    → {OUT_RAW / 'avoided_gridded.csv'}")

    out_stations = stations_df.copy()
    for h in range(24):
        out_stations[f"ridership_h{h:02d}"] = ridership[:, h]
    out_stations["ridership_peak"] = ridership.max(axis=1)
    out_stations["ridership_total"] = ridership.sum(axis=1)
    out_stations.to_csv(OUT_RAW / "avoided_stations.csv", index=False)
    print(f"  Stations CSV   → {OUT_RAW / 'avoided_stations.csv'}")


def generate_pngs(render_cube: np.ndarray) -> None:
    vmin, vmax = 0.0, NEAR_METRO_REDUCTION

    rglon, rglat = np.meshgrid(RENDER_LON, RENDER_LAT)
    cglon, cglat = np.meshgrid(COARSE_LON, COARSE_LAT)

    for with_labels in (False, True):
        tag     = "with_labels" if with_labels else "without_labels"
        out_dir = OUT_PNG / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        for hour in range(24):
            grid = render_cube[hour]
            fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
            im = ax.imshow(
                grid,
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
                    grid.ravel(),
                    np.column_stack([cglat.ravel(), cglon.ravel()]),
                    method="nearest",
                ).reshape(cglat.shape)
                for i in range(cglat.shape[0]):
                    for j in range(cglat.shape[1]):
                        v = label_vals[i, j]
                        if np.isnan(v):
                            continue
                        brightness = (v - vmin) / max(vmax - vmin, 1e-6)
                        color = "black" if brightness > 0.5 else "white"
                        ax.text(
                            cglon[i, j], cglat[i, j],
                            f"{v * 100:.0f}%",
                            ha="center", va="center",
                            fontsize=6, color=color, fontweight="bold",
                        )

            ax.set_title(f"BLR Metro Reduction Fraction  —  {hour:02d}:00 IST", fontsize=12)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.colorbar(im, ax=ax, label="PM2.5 Reduction Fraction", fraction=0.035, pad=0.04)
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
        unit="reduction_fraction",
        target_long_edge=td_export.TD_LONG_EDGE,
        lats=RENDER_LAT,
        lons=RENDER_LON,
        geojson_path=td_export.TD_BOUNDARY_GEOJSON,
    )
    print(f"  24 PNGs ({'td':<14}) → {OUT_PNG / 'td'}/  ({td_export.TD_LONG_EDGE}px long edge, source {render_cube.shape[2]}×{render_cube.shape[1]})")


def main() -> None:
    print("=" * 60)
    print(f"BLR Metro Reduction Fraction Cube  |  peak={NEAR_METRO_REDUCTION*100:.0f}%  "
          f"decay={DECAY_KM} km  city-avg≈{CITY_TARGET_REDUCTION*100:.0f}%")
    print("=" * 60)
    for d in (OUT_RAW, OUT_NC, OUT_PNG):
        d.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading hourly ridership and station coordinates…")
    stations_df, ridership = build_station_table()
    print(f"  {len(stations_df)} stations  |  total ridership (sum of 24h means): {ridership.sum():,.0f}")

    print("\n[2/4] Computing hourly ridership-weighted reduction cubes…")
    storage_cube, render_cube = compute_reduction_cubes(stations_df, ridership)
    ds = build_dataset(storage_cube)

    print("\n[3/4] Saving outputs…")
    save_outputs(ds, stations_df, ridership)

    print("\n[4/4] Generating PNGs…")
    generate_pngs(render_cube)

    print(f"\nDone. All outputs in: {OUT.resolve()}")


if __name__ == "__main__":
    main()
