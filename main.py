#!/usr/bin/env python3
"""
main.py — Net PM2.5: what observed PM2.5 would be if metro users drove instead

Flow:
  1. (Optional) Refetch hourly PM2.5 via hourly.py pipeline
  2. Load hourly PM2.5 grid from nc/hourly/gridded.nc       — PM25(hour, lat, lon)
  3. Load reduction fraction cube from nc/avoided/avoided_gridded.nc
                                                            — ReductionFraction(hour, lat, lon)
  4. Combine:   NetPM25[h] = PM25[h] / (1 - ReductionFraction[h])
  5. Save: nc/net/net_gridded.nc, raw/net/net_gridded.csv
  6. Generate PNGs:
       • with_labels / without_labels — matplotlib reference views
       • td/                          — raw greyscale frames sized to the
                                        higher-resolution render grid
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import hourly as hourly_mod
import td_export

warnings.filterwarnings('ignore')

NC_HOURLY  = Path("nc/hourly/gridded.nc")
NC_AVOIDED = Path("nc/avoided/avoided_gridded.nc")

OUT_NC  = Path("nc/net")
OUT_RAW = Path("raw/net")
OUT_PNG = Path("png/net")

LAT_MIN, LAT_MAX = 12.8235, 13.1526
LON_MIN, LON_MAX = 77.4499, 77.7941

COARSE_LAT = np.arange(LAT_MIN + 0.025, LAT_MAX, 0.05)
COARSE_LON = np.arange(LON_MIN + 0.025, LON_MAX, 0.05)


def refetch_hourly() -> None:
    print("\n[refetch] Fetching station list…")
    stations = hourly_mod.fetch_stations()
    if not stations:
        raise RuntimeError("No stations found — aborting refetch.")

    print(f"\n[refetch] Fetching last-24h time-series for {len(stations)} stations…")
    hour_pts, raw_df = hourly_mod.collect_all_readings(stations)

    hourly_mod.OUT_RAW.mkdir(parents=True, exist_ok=True)
    raw_csv = hourly_mod.OUT_RAW / "hourly.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"  Raw readings → {raw_csv}  ({len(raw_df)} rows)")

    if raw_df.empty:
        raise RuntimeError("No readings returned — aborting refetch.")

    print("\n[refetch] Interpolating to 0.01° grid…")
    ds = hourly_mod.build_storage_dataset(hour_pts)

    hourly_mod.OUT_NC.mkdir(parents=True, exist_ok=True)
    hourly_mod.save_outputs(ds)
    print("[refetch] Done.\n")


def load_inputs() -> tuple[xr.DataArray, xr.DataArray]:
    if not NC_HOURLY.exists():
        raise FileNotFoundError(
            f"{NC_HOURLY} not found. Run hourly.py first, or pass --refetch-hourly."
        )
    if not NC_AVOIDED.exists():
        raise FileNotFoundError(
            f"{NC_AVOIDED} not found. Run metro.py first."
        )

    ds_h = xr.open_dataset(NC_HOURLY)
    ds_a = xr.open_dataset(NC_AVOIDED)
    return ds_h["PM25"], ds_a["ReductionFraction"]


def build_net_dataset(pm25: xr.DataArray, fraction: xr.DataArray) -> xr.Dataset:
    """NetPM25 = ObsPM25 / (1 − ReductionFraction). Broadcasts hour-to-hour
    when fraction has a 'hour' dim, or static if 2D."""
    net = pm25 / (1.0 - fraction)
    return xr.Dataset(
        {"NetPM25": net},
        attrs={
            "title": "BLR Hourly Net PM2.5 (observed + avoided; what PM2.5 would be without metro)",
            "source": f"{NC_HOURLY}, {NC_AVOIDED}",
            "generated_at": datetime.now().isoformat(),
            "lat_units": "degrees_north",
            "lon_units": "degrees_east",
            "pm25_units": "ug/m3",
        },
    )


def save_outputs(ds: xr.Dataset) -> None:
    nc_path = OUT_NC / "net_gridded.nc"
    ds.to_netcdf(nc_path)
    print(f"  NetCDF       → {nc_path}")

    df = ds["NetPM25"].to_dataframe().reset_index().dropna(subset=["NetPM25"])
    csv_path = OUT_RAW / "net_gridded.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Gridded CSV  → {csv_path}")


def _resample_to_render_grid(cube: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Bilinear resample a (hour, lat, lon) cube to the hourly render grid."""
    rlat = hourly_mod.RENDER_LAT
    rlon = hourly_mod.RENDER_LON

    da = xr.DataArray(
        cube,
        coords={"hour": np.arange(cube.shape[0]), "lat": lats, "lon": lons},
        dims=["hour", "lat", "lon"],
    )
    resampled = da.interp(lat=rlat, lon=rlon, method="linear")
    # Edge cells beyond the source grid corners can be NaN; fall back to nearest.
    if bool(resampled.isnull().any()):
        nearest = da.interp(lat=rlat, lon=rlon, method="nearest")
        resampled = resampled.fillna(nearest)
    return resampled.to_numpy()


def generate_pngs(ds: xr.Dataset) -> None:
    cube = ds["NetPM25"].values  # (24, lat, lon) on storage grid
    lats = ds["lat"].values
    lons = ds["lon"].values

    render_cube = _resample_to_render_grid(cube, lats, lons)
    boundary_mask = (
        td_export.rasterize_polygon(
            td_export.TD_BOUNDARY_GEOJSON,
            hourly_mod.RENDER_LAT, hourly_mod.RENDER_LON,
        ) if td_export.TD_BOUNDARY_GEOJSON else None
    )
    vmin, vmax = td_export.cube_percentile_range(render_cube, mask=boundary_mask)

    rglon, rglat = np.meshgrid(hourly_mod.RENDER_LON, hourly_mod.RENDER_LAT)
    cglon, cglat = np.meshgrid(COARSE_LON, COARSE_LAT)

    for with_labels in (False, True):
        tag = "with_labels" if with_labels else "without_labels"
        out_dir = OUT_PNG / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        for hour in range(24):
            net = render_cube[hour]
            fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
            im = ax.imshow(
                net,
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
                    net.ravel(),
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

            ax.set_title(f"BLR Net PM2.5  —  {hour:02d}:00 IST", fontsize=12)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.colorbar(im, ax=ax, label="Net PM2.5 (µg/m³)", fraction=0.035, pad=0.04)
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
        lats=hourly_mod.RENDER_LAT,
        lons=hourly_mod.RENDER_LON,
        mask=boundary_mask,
    )
    print(f"  24 PNGs ({'td':<14}) → {OUT_PNG / 'td'}/  ({td_export.TD_LONG_EDGE}px long edge, source {render_cube.shape[2]}×{render_cube.shape[1]})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute net PM2.5 = hourly observed + metro avoided (PM2.5 without metro)"
    )
    parser.add_argument(
        "--refetch-hourly",
        action="store_true",
        help="Re-fetch hourly PM2.5 from oaq.notf.in before computing net values",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BLR Net PM2.5  |  observed + metro-avoided (PM2.5 without metro)")
    print("=" * 60)

    if args.refetch_hourly:
        refetch_hourly()

    for d in (OUT_NC, OUT_RAW, OUT_PNG):
        d.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading hourly PM2.5 and metro reduction fraction…")
    pm25, fraction = load_inputs()
    print(f"  Hourly shape:   {pm25.shape}  (hour × lat × lon)")
    print(f"  Fraction shape: {fraction.shape}  ({'hour × ' if fraction.ndim == 3 else ''}lat × lon)")
    print(f"  Fraction range: {float(fraction.min()):.3f} – {float(fraction.max()):.3f}")

    print("\n[2/3] Computing net PM2.5 and saving…")
    ds_net = build_net_dataset(pm25, fraction)
    save_outputs(ds_net)

    print("\n[3/3] Generating PNGs…")
    generate_pngs(ds_net)

    print(f"\nDone. All outputs in: {OUT_NC.parent.resolve()}")


if __name__ == "__main__":
    main()
