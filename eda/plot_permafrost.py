"""Sample plots of the Permafrost_cci v5.0 variables (Alaska 2000-2020 + pan-Arctic overview).

Reads the Alaska native-grid subsets written by pipeline/subset.py and, for the
pan-Arctic overview, one year of the raw NH files. Writes PNGs to eda/plots/.

Usage:
    python -I eda/plot_permafrost.py
    python -I eda/plot_permafrost.py --only timeseries nh_overview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from rasterio.features import geometry_mask
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "eda"))
import plotstyle as ps  # noqa: E402
from config import load_config, local_path  # noqa: E402

MAP_YEAR = 2020
TREND_YEARS = (2003, 2020)  # Case A (MODIS LST) only, avoids the 2002/2003 method change
GT_DEPTHS = [("GST", "Ground surface"), ("T1m", "1 m"), ("T2m", "2 m"), ("T5m", "5 m"), ("T10m", "10 m")]
TEMP_NORM = TwoSlopeNorm(vmin=-12, vcenter=0, vmax=8)
SOURCE_NOTE = "Source: ESA Permafrost_cci v5.0 (CRDP v4), 0.01° grid, shown in Alaska Albers for display."


def open_subset(cfg: dict, key: str) -> xr.Dataset:
    files = sorted((cfg["subset_dir"] / key).glob("*.nc"))
    return xr.open_mfdataset(files, combine="by_coords", chunks={"time": 1},
                             data_vars="minimal", coords="minimal", compat="override")


def alaska_mask(ds: xr.Dataset) -> xr.DataArray:
    """True inside the Alaska state polygon (Natural Earth admin-1, 10 m)."""
    shp = shpreader.natural_earth(resolution="10m", category="cultural", name="admin_1_states_provinces")
    geoms = [r.geometry for r in shpreader.Reader(shp).records()
             if r.attributes.get("name") == "Alaska" and r.attributes.get("admin") == "United States of America"]
    lat, lon = ds["lat"].values, ds["lon"].values
    res_lat, res_lon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    north_first = lat[0] > lat[-1]
    transform = from_origin(lon.min() - res_lon / 2, lat.max() + res_lat / 2, res_lon, res_lat)
    inside = ~geometry_mask(geoms, out_shape=(lat.size, lon.size), transform=transform)
    if not north_first:
        inside = inside[::-1]
    return xr.DataArray(inside, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


def pixel_area_km2(ds: xr.Dataset) -> xr.DataArray:
    dlat = abs(float(ds["lat"][1] - ds["lat"][0]))
    dlon = abs(float(ds["lon"][1] - ds["lon"][0]))
    return (111.32 * dlat) * (111.32 * dlon) * np.cos(np.deg2rad(ds["lat"]))


def year_of(ds: xr.Dataset, year: int) -> xr.Dataset:
    return ds.sel(time=str(year)).isel(time=0)


def value_and_uncertainty(cfg, key, var, label, units, vmax, unc_vmax, fname):
    ds = open_subset(cfg, key)
    yr = year_of(ds, MAP_YEAR).load()
    fig = plt.figure(figsize=(11, 5.2))
    for i, (name, cmap, top, lab) in enumerate([
        (var, ps.SEQ, vmax, f"{label} ({units})"),
        (f"{var}_uncertainty", ps.SEQ_UNC, unc_vmax, f"Within-pixel spatial uncertainty ({units})"),
    ]):
        ax = ps.ak_axes(fig, (1, 2, i + 1), f"{lab.split(' (')[0]}, {MAP_YEAR}")
        im = ps.show_grid(ax, yr[name], cmap, vmin=0, vmax=top)
        ps.colorbar(fig, im, ax, lab, extend="max")
    ps.save(fig, fname, SOURCE_NOTE + " Gray = no data (non-permafrost or ocean).")


def ground_temperature_maps(cfg):
    ds = open_subset(cfg, "permafrost_gt")
    yr = year_of(ds, MAP_YEAR)
    for suffix, cmap, norm, label in [
        ("", ps.DIV, TEMP_NORM, "Mean annual temperature (°C)"),
        ("_uncertainty", ps.SEQ_UNC, None, "Within-pixel spatial uncertainty (°C)"),
    ]:
        fig = plt.figure(figsize=(13, 7.4))
        for i, (var, name) in enumerate(GT_DEPTHS):
            ax = ps.ak_axes(fig, (2, 3, i + 1), f"{name}, {MAP_YEAR}")
            im = ps.show_grid(ax, yr[var + suffix].load(), cmap, norm=norm,
                              vmin=None if norm else 0, vmax=None if norm else 3)
        fig.subplots_adjust(wspace=0.08, hspace=0.12)
        cax = fig.add_axes([0.70, 0.28, 0.25, 0.025])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both" if norm else "max")
        cb.set_label(label, color=ps.INK_2)
        cb.outline.set_edgecolor(ps.AXIS)
        cb.ax.tick_params(labelcolor=ps.MUTED, color=ps.AXIS, labelsize=8)
        fig.text(0.70, 0.36, "Permafrost_cci ground temperature\n(GST and four depths)", fontsize=10,
                 color=ps.INK, va="bottom")
        ps.save(fig, f"ak_ground_temperature{suffix}_{MAP_YEAR}.png", SOURCE_NOTE)


def timeseries(cfg):
    alt, pfr, gt = (open_subset(cfg, k) for k in ("permafrost_alt", "permafrost_pfr", "permafrost_gt"))
    mask = alaska_mask(pfr)
    area = pixel_area_km2(pfr).broadcast_like(mask).where(mask)

    def wmean(da):
        w = area.where(da.notnull())
        return ((da * w).sum(("lat", "lon")) / w.sum(("lat", "lon"))).compute()

    series = [
        ("Permafrost area in Alaska", "thousand km²",
         ((pfr["PFR"] / 100) * area).sum(("lat", "lon")).compute() / 1e3),
        ("Mean active layer thickness (permafrost pixels)", "m", wmean(alt["ALT"])),
        ("Mean ground surface temperature", "°C", wmean(gt["GST"])),
        ("Mean ground temperature at 2 m", "°C", wmean(gt["T2m"])),
    ]
    years = pfr["time"].dt.year.values
    fig, axes = plt.subplots(len(series), 1, figsize=(8, 9.5), sharex=True)
    for ax, (title, units, s) in zip(axes, series):
        ax.axvspan(years.min() - 0.5, 2002.5, color=ps.NEUTRAL, zorder=0, lw=0)
        ax.plot(years, s.values, color=ps.LINE, marker="o", markersize=4, zorder=3)
        ax.set_title(title, loc="left")
        ax.set_ylabel(units)
        ax.grid(axis="x", visible=False)
        lo, hi = float(s.min()), float(s.max())
        ax.annotate(f"{float(s[-1]):.2f}", (years[-1], float(s[-1])), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=ps.INK_2)
        ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.25 * (hi - lo))
    axes[0].text(years.min() - 0.3, axes[0].get_ylim()[1], "ERA5-based\n(Case B)", va="top",
                 fontsize=7, color=ps.MUTED)
    axes[-1].set_xlim(years.min() - 0.5, years.max() + 1.5)
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    ps.save(fig, "ak_timeseries_2000_2020.png",
            SOURCE_NOTE.replace(", shown in Alaska Albers for display", "")
            + " Area-weighted over the Alaska state polygon. Shaded = 1997–2002 product (ERA5, bias-corrected).")
    return series


def trends(cfg):
    gt, alt = open_subset(cfg, "permafrost_gt"), open_subset(cfg, "permafrost_alt")
    sel = slice(str(TREND_YEARS[0]), str(TREND_YEARS[1]))
    fig = plt.figure(figsize=(11, 5.2))
    for i, (da, label, units, lim) in enumerate([
        (gt["T2m"].sel(time=sel), "Trend in ground temperature at 2 m", "°C per decade", 1.5),
        (alt["ALT"].sel(time=sel), "Trend in active layer thickness", "m per decade", 0.3),
    ]):
        da = da.astype("float32").load()
        t = da["time"].dt.year.astype("float32")
        tc = t - t.mean()
        n = da.notnull().sum("time")
        slope = ((da - da.mean("time")) * tc).sum("time") / (tc ** 2).where(da.notnull()).sum("time")
        slope = (slope * 10).where(n >= 15)
        ax = ps.ak_axes(fig, (1, 2, i + 1), f"{label}, {TREND_YEARS[0]}–{TREND_YEARS[1]}")
        im = ps.show_grid(ax, slope, ps.DIV, vmin=-lim, vmax=lim)
        ps.colorbar(fig, im, ax, f"Linear trend ({units})", extend="both")
    ps.save(fig, f"ak_trends_{TREND_YEARS[0]}_{TREND_YEARS[1]}.png",
            SOURCE_NOTE + " OLS slope per pixel, Case A years only (≥15 valid years).")


def distributions(cfg):
    alt, gt = open_subset(cfg, "permafrost_alt"), open_subset(cfg, "permafrost_gt")
    mask = alaska_mask(alt)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (da, title, units, bins) in zip(axes, [
        (year_of(alt, MAP_YEAR)["ALT"], "Active layer thickness", "m", np.arange(-0.05, 4.0, 0.1)),  # ALT is on 0.1 m steps
        (year_of(gt, MAP_YEAR)["T2m"], "Ground temperature at 2 m", "°C", np.arange(-12, 8.25, 0.25)),
    ]):
        v = da.where(mask).values.ravel()
        v = v[np.isfinite(v)]
        ax.hist(v, bins=bins, color=ps.LINE, edgecolor=ps.SURFACE, linewidth=0.5)
        ax.set_title(f"{title}, Alaska {MAP_YEAR}", loc="left")
        ax.set_xlabel(units)
        ax.set_ylabel("pixels (≈1 km²)")
        ax.grid(axis="x", visible=False)
        med = float(np.median(v))
        ax.axvline(med, color=ps.INK_2, lw=1, ls="--")
        ax.annotate(f"median {med:.2f} {units}", (med, ax.get_ylim()[1] * 0.92), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=ps.INK_2)
        beyond = np.mean((v < bins[0]) | (v > bins[-1])) * 100
        if beyond >= 0.05:
            ax.text(0.98, 0.80, f"{beyond:.1f}% of pixels outside\nplotted range (max {v.max():.1f} {units})",
                    transform=ax.transAxes, ha="right", fontsize=7, color=ps.MUTED)
    ps.save(fig, f"ak_distributions_{MAP_YEAR}.png", SOURCE_NOTE.split(",")[0] + ".")


def nh_overview(cfg):
    paths = {k: local_path(cfg, cfg["products"][k]["archive_path"]) / str(MAP_YEAR)
             for k in ("permafrost_pfr", "permafrost_alt", "permafrost_gt")}
    panels = [
        ("permafrost_pfr", "PFR", "Permafrost fraction", "%", ps.SEQ, dict(vmin=0, vmax=100)),
        ("permafrost_alt", "ALT", "Active layer thickness", "m", ps.SEQ, dict(vmin=0, vmax=3)),
        ("permafrost_gt", "T2m", "Ground temperature at 2 m", "°C", ps.DIV, dict(norm=TEMP_NORM)),
    ]
    fig = plt.figure(figsize=(15, 5.8))
    for i, (key, var, title, units, cmap, kw) in enumerate(panels):
        f = next(paths[key].glob("*.nc"))
        with xr.open_dataset(f) as ds:
            lat_slice = slice(40, 85) if ds["lat"][0] < ds["lat"][-1] else slice(85, 40)
            da = ds[var].isel(time=0).sel(lat=lat_slice).coarsen(lat=10, lon=10, boundary="trim").mean().load()
        ax = ps.nh_axes(fig, (1, 3, i + 1), f"{title}, {MAP_YEAR}", min_lat=45)
        im = ps.show_grid(ax, da, cmap, **kw)
        ps.colorbar(fig, im, ax, f"{title} ({units})", extend="max" if cmap is ps.SEQ else "both")
    ps.save(fig, f"nh_overview_{MAP_YEAR}.png",
            "Source: ESA Permafrost_cci v5.0 (CRDP v4), NH files, block-averaged to 0.1° for display.")


def maps(cfg):
    value_and_uncertainty(cfg, "permafrost_alt", "ALT", "Active layer thickness", "m", 3, 1,
                          f"ak_active_layer_thickness_{MAP_YEAR}.png")
    value_and_uncertainty(cfg, "permafrost_pfr", "PFR", "Permafrost fraction", "%", 100, 50,
                          f"ak_permafrost_fraction_{MAP_YEAR}.png")
    ground_temperature_maps(cfg)


def print_timeseries(cfg):
    for title, units, s in timeseries(cfg):
        print(f"  {title}: " + ", ".join(f"{int(y)} {float(v):.2f}" for y, v in zip(s["time"].dt.year, s)) + f" {units}")


STEPS = {"maps": maps, "timeseries": print_timeseries, "trends": trends,
         "distributions": distributions, "nh_overview": nh_overview}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(STEPS), help="run only these figure groups")
    args = ap.parse_args()
    ps.apply_style()
    cfg = load_config()
    for name in args.only or STEPS:
        STEPS[name](cfg)


if __name__ == "__main__":
    main()
