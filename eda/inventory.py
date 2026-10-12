"""Inventory every product in pipeline/config.yaml for EDA.md.

For each product this script:
  * summarizes the archive manifest (file count, size, years, directory pattern)
  * reports local raw / Alaska-subset holdings for downloaded products
  * opens one sample file (local copy if present, otherwise remote via HTTP
    byte-range reads, which fetch only the header) and records dimensions,
    extent, grid spacing, CRS, variables and key global attributes

Outputs: eda/metadata/<product>.json and eda/inventory.md (markdown tables).

Usage:
    python -I eda/inventory.py               # all products
    python -I eda/inventory.py --product snow_swe
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))
from ceda import url_for  # noqa: E402
from config import load_config, local_path  # noqa: E402

OUT_DIR = REPO / "eda" / "metadata"
GLOBAL_KEYS = [
    "title", "summary", "institution", "source", "platform", "sensor", "spatial_resolution",
    "geospatial_lat_min", "geospatial_lat_max", "geospatial_lon_min", "geospatial_lon_max",
    "time_coverage_start", "time_coverage_end", "product_version", "doi", "license", "references",
]
# Prefer a mid-record sample date when picking the example file.
SAMPLE_HINTS = ["20100315", "-2010-", "AVHRR_NOAA18"]


def human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def file_year(name: str) -> int | None:
    m = re.match(r"(\d{4})\d{4}-", name) or re.search(r"-(\d{4})(?:-v|-fv|b?\.nc)", name)
    return int(m[1]) if m else None


def dir_pattern(path: str, root: str) -> str:
    rel = path[len(root) + 1 :].rsplit("/", 1)[0] if "/" in path[len(root) + 1 :] else ""
    rel = re.sub(r"(?<=/)\d{4}(?=/|$)|^\d{4}(?=/|$)", "{YYYY}", rel)
    rel = re.sub(r"(?<=/)\d{2}(?=/|$)", "{MM}", rel)
    return rel or "."


def summarize_manifest(cfg: dict, key: str) -> dict:
    prod = cfg["products"][key]
    rows = list(csv.DictReader(open(cfg["manifest_dir"] / f"{key}.csv")))
    names = [r["path"].rsplit("/", 1)[1] for r in rows]
    years = sorted({y for y in map(file_year, names) if y})
    patterns = sorted({dir_pattern(r["path"], prod["archive_path"]) for r in rows})
    sizes = [int(r["size_bytes"]) for r in rows]
    return {
        "n_files": len(rows),
        "total_bytes": sum(sizes),
        "total_human": human(sum(sizes)),
        "median_file_human": human(float(np.median(sizes))) if sizes else None,
        "first_year": years[0] if years else None,
        "last_year": years[-1] if years else None,
        "dir_patterns": patterns[:10],
        "example_files": [names[0], names[-1]] if names else [],
        "paths": [r["path"] for r in rows],
    }


def local_holdings(cfg: dict, key: str) -> dict:
    def stats(d: Path) -> tuple[int, int]:
        files = list(d.rglob("*.nc")) if d.exists() else []
        return len(files), sum(f.stat().st_size for f in files)

    raw_dir = local_path(cfg, cfg["products"][key]["archive_path"])
    sub_dir = cfg["subset_dir"] / key
    (nr, br), (ns, bs) = stats(raw_dir), stats(sub_dir)
    return {
        "raw_dir": str(raw_dir), "raw_files": nr, "raw_human": human(br),
        "subset_dir": str(sub_dir), "subset_files": ns, "subset_human": human(bs),
    }


def pick_sample(cfg: dict, key: str, paths: list[str]) -> tuple[str, bool]:
    ncs = [p for p in paths if p.endswith(".nc")]
    for hint in SAMPLE_HINTS:
        hits = [p for p in ncs if hint in p]
        if hits:
            ncs = hits
    archive = ncs[len(ncs) // 2] if len(ncs) > 1 else ncs[0]
    local = local_path(cfg, archive)
    return (str(local), False) if local.exists() else (url_for(archive) + "#mode=bytes", True)


def crs_of(ds: xr.Dataset) -> dict:
    gm_names = {ds[v].attrs.get("grid_mapping") for v in ds.data_vars} - {None}
    for name in list(gm_names) + ["spatial_ref", "crs"]:
        if name in ds.variables:
            attrs = ds[name].attrs
            wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref") or attrs.get("wkt")
            try:
                crs = CRS.from_wkt(wkt) if wkt else CRS.from_cf(attrs)
                return {"grid_mapping_var": name, "epsg": crs.to_epsg(), "name": crs.name,
                        "geotransform": attrs.get("GeoTransform")}
            except Exception as exc:  # noqa: BLE001
                return {"grid_mapping_var": name, "error": str(exc), "attrs": {k: str(v) for k, v in attrs.items()}}
    return {"note": "no grid_mapping variable; regular lat/lon assumed"}


def describe(src: str) -> dict:
    ds = xr.open_dataset(src, decode_times=True, mask_and_scale=False)
    lat = next((c for c in ("lat", "latitude", "y") if c in ds.coords or c in ds.variables), None)
    lon = next((c for c in ("lon", "longitude", "x") if c in ds.coords or c in ds.variables), None)
    grid = {}
    for name in (lat, lon):
        if name and ds[name].ndim == 1 and ds[name].size > 1:
            v = ds[name].values
            grid[name] = {"size": int(v.size), "first": float(v[0]), "last": float(v[-1]),
                          "step": float(np.round(np.median(np.diff(v)), 6))}
    variables = []
    for name, var in ds.data_vars.items():
        if var.ndim == 0 or name.endswith("_bnds"):
            continue
        a = var.attrs
        variables.append({
            "name": name, "dims": list(var.dims), "dtype": str(var.dtype),
            "units": a.get("units", ""), "long_name": a.get("long_name") or a.get("standard_name", ""),
            "standard_name": a.get("standard_name", ""),
            "scale_factor": a.get("scale_factor"), "add_offset": a.get("add_offset"),
            "fill_value": None if a.get("_FillValue") is None else str(a.get("_FillValue")),
            "flag_meanings": a.get("flag_meanings", ""),
        })
    time = None
    if "time" in ds.coords and ds["time"].size:
        time = [str(ds["time"].values[0])[:19], str(ds["time"].values[-1])[:19]]
    out = {
        "sample": src.replace("#mode=bytes", ""),
        "dims": {k: int(v) for k, v in ds.sizes.items()},
        "grid": grid, "time_in_sample": time, "crs": crs_of(ds), "variables": variables,
        "global_attrs": {k: str(ds.attrs[k]) for k in GLOBAL_KEYS if k in ds.attrs},
    }
    ds.close()
    return out


def md_tables(results: dict) -> str:
    lines = ["# Inventory (generated by eda/inventory.py; do not edit by hand)", ""]
    lines += ["## Holdings", "",
              "| Product | Archive files | Archive size | Years | Downloaded | Alaska subset |",
              "|---|---:|---:|---|---|---|"]
    for key, r in results.items():
        m, loc = r["manifest"], r.get("local", {})
        dl = f"{loc['raw_files']} files, {loc['raw_human']}" if loc.get("raw_files") else "catalog only"
        sub = f"{loc['subset_files']} files, {loc['subset_human']}" if loc.get("subset_files") else "-"
        lines.append(f"| `{key}` | {m['n_files']:,} | {m['total_human']} | {m['first_year']}–{m['last_year']} | {dl} | {sub} |")
    for key, r in results.items():
        m, d = r["manifest"], r["sample"]
        lines += ["", f"## `{key}`: {r['title']}", ""]
        lines.append(f"- Archive: `{r['archive_path']}`")
        lines.append(f"- Directory pattern(s) below archive path: {', '.join(f'`{p}`' for p in m['dir_patterns'])}")
        lines.append(f"- Example files: `{m['example_files'][0]}` … `{m['example_files'][-1]}` (median file {m['median_file_human']})")
        lines.append(f"- Sample inspected: `{d['sample'].rsplit('/', 1)[-1]}` ({'remote header read' if r['remote'] else 'local file'})")
        lines.append(f"- Dimensions: {', '.join(f'{k}={v:,}' for k, v in d['dims'].items())}")
        for name, g in d["grid"].items():
            lines.append(f"- `{name}`: {g['first']:.4f} → {g['last']:.4f}, step {g['step']} ({g['size']:,} cells)")
        crs = d["crs"]
        if crs.get("epsg"):
            crs_txt = f"EPSG:{crs['epsg']} ({crs.get('name')})"
        elif crs.get("name"):
            crs_txt = f"{crs['name']} geographic lat/lon (WKT in `{crs['grid_mapping_var']}` has no EPSG code; equivalent to EPSG:4326)"
        else:
            crs_txt = "no grid_mapping variable; regular WGS84 lat/lon grid per product docs (EPSG:4326)"
        lines.append(f"- CRS: {crs_txt}" + (f"; GeoTransform `{crs['geotransform']}`" if crs.get("geotransform") else ""))
        if d["time_in_sample"]:
            lines.append(f"- Time stamp(s) in sample: {d['time_in_sample'][0]}" + (
                f" … {d['time_in_sample'][1]}" if d["time_in_sample"][1] != d["time_in_sample"][0] else ""))
        lines += ["", "| Variable | Dims | Stored dtype | Units | Description | Scale / offset | Fill |",
                  "|---|---|---|---|---|---|---|"]
        for v in d["variables"]:
            so = "" if v["scale_factor"] is None else f"{v['scale_factor']} / {v['add_offset'] or 0}"
            desc = (v["long_name"] or "").replace("|", "/")
            if v["flag_meanings"]:
                desc += f" (flags: {v['flag_meanings'][:120]}{'…' if len(v['flag_meanings']) > 120 else ''})"
            lines.append(f"| `{v['name']}` | {', '.join(v['dims'])} | {v['dtype']} | {v['units']} | {desc} | {so} | {v['fill_value'] or ''} |")
        if d["global_attrs"]:
            lines += ["", "<details><summary>Key global attributes</summary>", ""]
            lines += [f"- **{k}**: {str(v)[:400]}" for k, v in d["global_attrs"].items()]
            lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config()
    keys = args.product or list(cfg["products"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in keys:
        prod = cfg["products"][key]
        man = summarize_manifest(cfg, key)
        src, remote = pick_sample(cfg, key, man.pop("paths"))
        print(f"{key}: {man['n_files']} files, {man['total_human']}; sample {'(remote)' if remote else '(local)'}")
        res = {"title": prod["title"], "theme": prod["theme"], "archive_path": prod["archive_path"],
               "catalogue": prod.get("catalogue"), "manifest": man, "remote": remote, "sample": describe(src)}
        if prod.get("download"):
            res["local"] = local_holdings(cfg, key)
        (OUT_DIR / f"{key}.json").write_text(json.dumps(res, indent=2, default=str))
        results[key] = res

    if not args.product:
        (REPO / "eda" / "inventory.md").write_text(md_tables(results))
        print("wrote eda/inventory.md")


if __name__ == "__main__":
    main()
