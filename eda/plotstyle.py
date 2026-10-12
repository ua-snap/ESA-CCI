"""Shared plotting style for the EDA figures.

Color by job (validated with the dataviz skill's palette validator):
  * sequential magnitude -> one blue hue, light -> dark (SEQ)
  * second sequential context (uncertainty) -> one orange hue (SEQ_UNC)
  * signed values around a meaningful zero (deg C, trends) -> blue <-> gray <-> red (DIV)
  * categories on maps -> at most 5 hues + neutral gray ("other"); the 5-hue set
    passes all-pairs CVD and normal-vision checks
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.path as mpath
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

PLOT_DIR = Path(__file__).resolve().parent / "plots"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"    # diverging midpoint
NODATA = "#d9d8d2"     # flagged / masked pixels on maps

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fbe0d3", "#f6b597", "#f08a5e", "#eb6834", "#c24f1f", "#923a15", "#62270d"]
RED_ARM = ["#fbdad9", "#f4a9a8", "#ec7877", "#e34948", "#bb3332", "#8f2423", "#621817"]

SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
SEQ_UNC = LinearSegmentedColormap.from_list("seq_orange", ORANGE_RAMP)
DIV = LinearSegmentedColormap.from_list("div_blue_red", BLUE_RAMP[::-1] + [NEUTRAL] + RED_ARM)
for cmap in (SEQ, SEQ_UNC, DIV):
    cmap.set_bad((0, 0, 0, 0))  # transparent: the gray land layer below shows "no data"

CAT = {"blue": "#2a78d6", "green": "#008300", "yellow": "#eda100", "magenta": "#e87ba4",
       "violet": "#4a3aa7", "other": "#a8a69f"}
LINE = CAT["blue"]

AK_PROJ = ccrs.AlbersEqualArea(central_longitude=-154, central_latitude=50,
                               standard_parallels=(55, 65))  # EPSG:3338, display only
AK_EXTENT = [-171, -129, 52.5, 71.8]  # lon/lat view window
NH_PROJ = ccrs.NorthPolarStereo(central_longitude=-45)
PLATE = ccrs.PlateCarree()


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK, "font.size": 9, "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def ak_axes(fig, pos, title: str | None = None):
    ax = fig.add_subplot(*pos, projection=AK_PROJ)
    ax.reset_view = lambda: ax.set_extent(AK_EXTENT, crs=PLATE)
    ax.reset_view()
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=NODATA, edgecolor="none", zorder=0)
    ax.coastlines("50m", linewidth=0.5, color=INK_2, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor=MUTED, zorder=3)
    ax.spines["geo"].set_edgecolor(AXIS)
    if title:
        ax.set_title(title, loc="left")
    return ax


def nh_axes(fig, pos, title: str | None = None, min_lat: float = 45):
    ax = fig.add_subplot(*pos, projection=NH_PROJ)
    # set_extent([-180, 180, ...]) collapses to one quadrant here, so set projected limits directly
    r = abs(NH_PROJ.transform_point(-45, min_lat, PLATE)[1])
    ax.reset_view = lambda: (ax.set_xlim(-r, r), ax.set_ylim(-r, r))
    ax.reset_view()
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=NODATA, edgecolor="none", zorder=0)
    ax.coastlines("110m", linewidth=0.4, color=INK_2, zorder=3)
    ax.spines["geo"].set_edgecolor(AXIS)
    if title:
        ax.set_title(title, loc="left")
    return ax


def show_grid(ax, da, cmap, vmin=None, vmax=None, norm=None):
    """Draw a lat/lon DataArray on a projected axis (image reprojection, fast for 1 km grids)."""
    lat, lon = da["lat"].values, da["lon"].values
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    extent = [lon.min() - dlon / 2, lon.max() + dlon / 2, lat.min() - dlat / 2, lat.max() + dlat / 2]
    data = da.values if lat[0] > lat[-1] else da.values[::-1]
    im = ax.imshow(data, origin="upper", extent=extent, transform=PLATE, cmap=cmap,
                   vmin=vmin, vmax=vmax, norm=norm, interpolation="nearest",
                   regrid_shape=1600, zorder=1)
    if hasattr(ax, "reset_view"):  # imshow resets the view to the image bounds
        ax.reset_view()
    return im


def colorbar(fig, im, ax, label: str, **kw):
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04, **kw)
    cb.set_label(label, color=INK_2)
    cb.outline.set_edgecolor(AXIS)
    cb.ax.tick_params(color=AXIS, labelcolor=MUTED, labelsize=8)
    return cb


def save(fig, name: str, note: str | None = None) -> Path:
    if note:
        fig.text(0.01, -0.01, note, color=MUTED, fontsize=7, ha="left", va="top")
    PLOT_DIR.mkdir(exist_ok=True)
    out = PLOT_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out.relative_to(PLOT_DIR.parent.parent))
    return out


def categorical_cmap(colors: list[str]) -> ListedColormap:
    cmap = ListedColormap(colors)
    cmap.set_bad((0, 0, 0, 0))
    return cmap
