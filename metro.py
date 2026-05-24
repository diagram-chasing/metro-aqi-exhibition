#!/usr/bin/env python3
"""
metro.py — BLR metro PM2.5 ridership-weighted reduction fraction map

Flow:
  1. Load hourly entry/exit ridership from Parquet files
  2. Compute peak ridership (max entries + max exits) per station
  3. Match stations to coordinates from metro.geojson
  4. Build a ridership-weighted exponential kernel: each station contributes
       ridership_s × exp(−distance_to_s / DECAY_KM)
     summed over all stations for each 0.01° grid cell
  5. Normalise so the peak cell reaches NEAR_METRO_REDUCTION (25%)
  6. Add a uniform baseline so the city-wide average equals CITY_TARGET_REDUCTION (5%)
  7. Save: NetCDF, gridded CSV, station CSV
  8. Generate PNG heatmap of the reduction fraction

Methodology
-----------
Produces a static spatial map ReductionFraction[lat, lon] used in main.py as:

    NetPM25 = ObsPM25 / (1 − ReductionFraction)

ObsPM25 already reflects the benefit of metro ridership. Dividing recovers the
PM2.5 level that would exist if metro users drove instead (observed + avoided).

Each metro station s has a peak ridership R_s (max hourly entries + exits).
The influence at a grid cell is a ridership-weighted exponential kernel:

    score[cell] = Σ_s  R_s × exp(−distance(cell, s) / DECAY_KM)

The score map is then normalised and a uniform baseline is added to meet two
literature-calibrated targets:

  • Peak ReductionFraction = 25%
  • City-wide mean = 5%

High ridership stations produce stronger and wider local reductions.
Less used stations contribute less.
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

warnings.filterwarnings('ignore')

# ── Reduction parameters (literature-calibrated) ──────────────────────────────
NEAR_METRO_REDUCTION  = 0.25   # peak reduction at the highest-influence cell (midpoint 20–30%)
DECAY_KM              = 1.5    # exponential decay distance of each station's influence [km]
CITY_TARGET_REDUCTION = 0.05   # city-wide average target

# ── Bounding box (same as hourly.py) ─────────────────────────────────────────
LAT_MIN, LAT_MAX = 12.8235, 13.1526
LON_MIN, LON_MAX = 77.4499, 77.7941

FINE_LAT = np.arange(LAT_MIN, LAT_MAX + 0.005, 0.01)
FINE_LON = np.arange(LON_MIN, LON_MAX + 0.005, 0.01)

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


def load_station_maxima() -> pd.DataFrame:
    df_entries = pd.read_parquet("raw/station-hourly.parquet")
    df_exits   = pd.read_parquet("raw/station-hourly-exits.parquet")
    max_entries = df_entries.groupby("Station")["Ridership"].max().rename("max_entries")
    max_exits   = df_exits.groupby("Station")["Ridership"].max().rename("max_exits")
    df = pd.concat([max_entries, max_exits], axis=1).reset_index()
    df["max_entries"] = df["max_entries"].fillna(0)
    df["max_exits"]   = df["max_exits"].fillna(0)
    return df


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


def build_station_df() -> pd.DataFrame:
    df     = load_station_maxima()
    coords = load_station_coords()

    rows, unmatched = [], []
    for _, row in df.iterrows():
        name = NAME_MAP.get(row["Station"], row["Station"])
        if name not in coords:
            unmatched.append(row["Station"])
            continue
        lat, lon = coords[name]
        rows.append({
            "station":     row["Station"],
            "lat":         lat,
            "lon":         lon,
            "max_entries": row["max_entries"],
            "max_exits":   row["max_exits"],
            "ridership":   row["max_entries"] + row["max_exits"],
        })

    if unmatched:
        print(f"  [warn] No coordinates for {len(unmatched)} station(s): {unmatched}")

    return pd.DataFrame(rows)


def compute_reduction_grid(station_df: pd.DataFrame) -> np.ndarray:
    """
    Ridership-weighted exponential kernel summed over all stations.

    score[i,j] = Σ_s  ridership_s × exp(−d_{s,i,j} / DECAY_KM)

    Normalised so the peak cell = NEAR_METRO_REDUCTION, then a uniform
    baseline is added so the city-wide mean equals CITY_TARGET_REDUCTION.
    """
    n_lat, n_lon = len(FINE_LAT), len(FINE_LON)
    score = np.zeros((n_lat, n_lon), dtype=np.float64)

    # Pre-project all grid columns once per latitude row
    for i, lat in enumerate(FINE_LAT):
        cell_e, cell_n = _UTM.transform(FINE_LON, np.full(n_lon, lat))
        for _, row in station_df.iterrows():
            s_e, s_n = _UTM.transform(row["lon"], row["lat"])
            d_km = np.sqrt((cell_e - s_e) ** 2 + (cell_n - s_n) ** 2) / 1000.0
            score[i] += row["ridership"] * np.exp(-d_km / DECAY_KM)

    # Normalise peak to NEAR_METRO_REDUCTION
    score_max = score.max()
    reduction = (score / score_max * NEAR_METRO_REDUCTION) if score_max > 0 else np.zeros_like(score)

    # Add uniform baseline so city average reaches CITY_TARGET_REDUCTION
    current_avg = reduction.mean()
    baseline = max(0.0, CITY_TARGET_REDUCTION - current_avg)
    reduction = np.clip(reduction + baseline, 0.0, NEAR_METRO_REDUCTION)

    achieved_avg = float(reduction.mean())
    achieved_peak = float(reduction.max())
    print(f"  Peak reduction   : {achieved_peak * 100:.1f}%  (target ≤{NEAR_METRO_REDUCTION*100:.0f}%)")
    print(f"  Baseline added   : {baseline * 100:.1f}%")
    print(f"  City-wide average: {achieved_avg * 100:.2f}%  (target {CITY_TARGET_REDUCTION*100:.0f}%)")

    return reduction.astype(np.float32)


def build_dataset(grid: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {"ReductionFraction": (["lat", "lon"], grid)},
        coords={"lat": FINE_LAT, "lon": FINE_LON},
        attrs={
            "title":                  "BLR Metro PM2.5 Reduction Fraction Map (ridership-weighted)",
            "source":                 "station-hourly.parquet, station-hourly-exits.parquet, metro.geojson",
            "generated_at":           datetime.now().isoformat(),
            "lat_units":              "degrees_north",
            "lon_units":              "degrees_east",
            "near_metro_reduction":   NEAR_METRO_REDUCTION,
            "decay_km":               DECAY_KM,
            "city_target_reduction":  CITY_TARGET_REDUCTION,
            "note": (
                "ReductionFraction[cell] = normalised ridership-weighted kernel + baseline. "
                "Applied in main.py as: NetPM25 = ObsPM25 / (1 − ReductionFraction). "
                "Peak calibrated to 25%"
            ),
        },
    )


def save_outputs(ds: xr.Dataset, station_df: pd.DataFrame) -> None:
    nc_path = OUT_NC / "avoided_gridded.nc"
    ds.to_netcdf(nc_path)
    print(f"  NetCDF         → {nc_path}")

    grid_df = ds["ReductionFraction"].to_dataframe().reset_index().dropna(subset=["ReductionFraction"])
    grid_df.to_csv(OUT_RAW / "avoided_gridded.csv", index=False)
    print(f"  Gridded CSV    → {OUT_RAW / 'avoided_gridded.csv'}")

    station_df.to_csv(OUT_RAW / "avoided_stations.csv", index=False)
    print(f"  Stations CSV   → {OUT_RAW / 'avoided_stations.csv'}")


def generate_pngs(ds: xr.Dataset) -> None:
    grid = ds["ReductionFraction"].values
    vmin, vmax = 0.0, NEAR_METRO_REDUCTION

    glon, glat   = np.meshgrid(FINE_LON, FINE_LAT)
    cglon, cglat = np.meshgrid(COARSE_LON, COARSE_LAT)

    for with_labels in (False, True):
        tag     = "with_labels" if with_labels else "without_labels"
        out_dir = OUT_PNG / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        im = ax.imshow(
            grid,
            cmap="gray",
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

        ax.set_title("BLR Metro  —  PM2.5 Reduction Fraction", fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(im, ax=ax, label="PM2.5 Reduction Fraction", fraction=0.035, pad=0.04)
        plt.tight_layout()
        plt.savefig(
            out_dir / "avoided_pm25.png",
            dpi=150, bbox_inches="tight", facecolor="white",
        )
        plt.close()

        print(f"  PNG ({tag:<12}) → {out_dir}/avoided_pm25.png")


def main() -> None:
    print("=" * 60)
    print(f"BLR Metro Reduction Fraction  |  peak={NEAR_METRO_REDUCTION*100:.0f}%  "
          f"decay={DECAY_KM} km  city-avg≈{CITY_TARGET_REDUCTION*100:.0f}%")
    print("=" * 60)
    for d in (OUT_RAW, OUT_NC, OUT_PNG):
        d.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading ridership and station coordinates…")
    station_df = build_station_df()
    total_ridership = station_df["ridership"].sum()
    print(f"  {len(station_df)} stations  |  total peak ridership: {total_ridership:,.0f}")

    print("\n[2/4] Computing ridership-weighted reduction grid…")
    grid = compute_reduction_grid(station_df)
    ds   = build_dataset(grid)

    print("\n[3/4] Saving outputs…")
    save_outputs(ds, station_df)

    print("\n[4/4] Generating PNGs…")
    generate_pngs(ds)

    print(f"\nDone. All outputs in: {OUT.resolve()}")


if __name__ == "__main__":
    main()
