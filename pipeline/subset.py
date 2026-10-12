"""Clip downloaded products to the Alaska bbox on their native grid (no resampling).

Works for products on a regular 1-D lat/lon grid (all Permafrost_cci v5.0 files).
Each input file becomes one output file under <subset_dir>/<product>/, with the same
filename, the original packed dtypes, scale factors and fill values, and the
grid_mapping variable kept.

Usage:
    python -I pipeline/subset.py --theme permafrost
    python -I pipeline/subset.py --product permafrost_alt --years 2000 2020
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config, local_path  # noqa: E402

YEAR = re.compile(r"[-_](\d{4})-fv")
KEEP_ENCODING = ("dtype", "scale_factor", "add_offset", "_FillValue", "units", "calendar")


def clip(ds: xr.Dataset, bbox: dict) -> xr.Dataset:
    lat = ds["lat"]
    lat_slice = (
        slice(bbox["north"], bbox["south"]) if lat[0] > lat[-1] else slice(bbox["south"], bbox["north"])
    )
    return ds.sel(lat=lat_slice, lon=slice(bbox["west"], bbox["east"]))


def write(ds: xr.Dataset, out: Path) -> None:
    encoding = {}
    for name, var in ds.variables.items():
        enc = {k: v for k, v in var.encoding.items() if k in KEEP_ENCODING}
        if var.ndim >= 2:
            enc.update(zlib=True, complevel=4, shuffle=True)
        encoding[name] = enc
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".nc.part")
    ds.to_netcdf(tmp, encoding=encoding)
    tmp.rename(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product", action="append", default=[])
    ap.add_argument("--theme", action="append", default=[])
    ap.add_argument("--years", nargs=2, type=int, help="first and last year (default: config alaska.years)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    bbox = cfg["alaska"]["bbox"]
    y0, y1 = args.years or cfg["alaska"]["years"]
    keys = args.product + [k for k, p in cfg["products"].items() if p["theme"] in args.theme]
    if not keys:
        ap.error("give --product or --theme")

    for key in dict.fromkeys(keys):
        src_dir = local_path(cfg, cfg["products"][key]["archive_path"])
        files = sorted(f for f in src_dir.rglob("*.nc") if y0 <= int(YEAR.search(f.name)[1]) <= y1)
        if not files:
            print(f"{key}: no downloaded files in {src_dir} for {y0}-{y1}", file=sys.stderr)
            continue
        out_dir = cfg["subset_dir"] / key
        for f in files:
            out = out_dir / f.name
            if out.exists() and not args.overwrite:
                continue
            with xr.open_dataset(f, mask_and_scale=True, decode_times=True) as ds:
                sub = clip(ds, bbox)
                sub.attrs["subset"] = f"Alaska bbox {bbox} clipped from {f.name} on native grid"
                write(sub.load(), out)
            print(f"{key}: {out.name} {tuple(sub.sizes.values())}")


if __name__ == "__main__":
    main()
