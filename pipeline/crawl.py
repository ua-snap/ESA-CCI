"""Crawl CEDA archive paths and write file manifests (path, size, mtime) as CSV.

Usage:
    python pipeline/crawl.py /neodc/esacci/snow/data            # ad-hoc path -> stdout summary
    python pipeline/crawl.py --product permafrost_alt           # product from config.yaml
    python pipeline/crawl.py --all                              # every product in config.yaml

Manifests go to <manifest_dir>/<name>.csv (see config.yaml) and are reused by download.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ceda import crawl  # noqa: E402


def human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def write_manifest(out_dir: Path, name: str, entries) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "size_bytes", "mtime"])
        for e in sorted(entries, key=lambda e: e.path):
            w.writerow([e.path, e.size, e.mtime])
    return out


def summarize(root: str, entries, depth: int) -> None:
    """Print file count/size grouped by the first `depth` path components below root."""
    groups: dict[str, list] = defaultdict(lambda: [0, 0])
    for e in entries:
        rel = e.path[len(root.rstrip("/")) + 1 :].split("/")
        key = "/".join(rel[: min(depth, len(rel) - 1)]) or "."
        groups[key][0] += 1
        groups[key][1] += e.size
    for key in sorted(groups):
        n, size = groups[key]
        print(f"{key:<70} {n:>7} files {human(size):>10}")
    print(f"{'TOTAL':<70} {len(entries):>7} files {human(sum(e.size for e in entries)):>10}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="archive path to crawl, e.g. /neodc/esacci/snow/data")
    ap.add_argument("--product", action="append", default=[], help="product key from config.yaml")
    ap.add_argument("--all", action="store_true", help="crawl every product in config.yaml")
    ap.add_argument("--save-as", help="write ad-hoc crawl to manifests/<name>.csv")
    ap.add_argument("--depth", type=int, default=3, help="grouping depth for the summary")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.path:
        entries = crawl(args.path, workers=args.workers)
        summarize(args.path, entries, args.depth)
        if args.save_as:
            from config import load_config  # needs PyYAML (project env)

            print("wrote", write_manifest(load_config()["manifest_dir"], args.save_as, entries))
        return

    from config import load_config  # needs PyYAML (project env)

    cfg = load_config()
    keys = list(cfg["products"]) if args.all else args.product
    if not keys:
        ap.error("give a path, --product or --all")
    for key in keys:
        prod = cfg["products"][key]
        entries = crawl(prod["archive_path"], workers=args.workers)
        if prod.get("include"):
            entries = [e for e in entries if any(s in e.path for s in prod["include"])]
        print(f"== {key}: {prod['archive_path']}")
        summarize(prod["archive_path"], entries, args.depth)
        print("wrote", write_manifest(cfg["manifest_dir"], key, entries))


if __name__ == "__main__":
    main()
