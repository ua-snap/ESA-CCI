"""Download products listed in config.yaml from the CEDA archive.

Uses the manifest written by crawl.py (<manifest_dir>/<product>.csv), crawling
first if it is missing. Files already on disk with the expected size are skipped,
so the script can be re-run to resume. Products marked `download: false` in the
config are catalog-only and are refused unless --force is given.

Usage:
    python pipeline/download.py --product permafrost_alt --product permafrost_pfr
    python pipeline/download.py --theme permafrost
    python pipeline/download.py --theme permafrost --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from ceda import crawl, url_for  # noqa: E402
from config import load_config, local_path  # noqa: E402
from crawl import human, write_manifest  # noqa: E402


def read_manifest(cfg: dict, key: str, refresh: bool) -> list[tuple[str, int]]:
    prod = cfg["products"][key]
    path = cfg["manifest_dir"] / f"{key}.csv"
    if refresh or not path.exists():
        entries = crawl(prod["archive_path"])
        if prod.get("include"):
            entries = [e for e in entries if any(s in e.path for s in prod["include"])]
        write_manifest(cfg["manifest_dir"], key, entries)
    with open(path) as f:
        return [(r["path"], int(r["size_bytes"])) for r in csv.DictReader(f)]


def fetch(url: str, dest: Path, size: int) -> str:
    if dest.exists() and dest.stat().st_size == size:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
    if tmp.stat().st_size != size:
        raise IOError(f"size mismatch for {url}: got {tmp.stat().st_size}, expected {size}")
    tmp.rename(dest)
    return "downloaded"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product", action="append", default=[])
    ap.add_argument("--theme", action="append", default=[])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--refresh-manifest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="allow catalog-only (download: false) products")
    args = ap.parse_args()

    cfg = load_config()
    keys = args.product + [k for k, p in cfg["products"].items() if p["theme"] in args.theme]
    if not keys:
        ap.error("give --product or --theme")

    jobs = []
    for key in dict.fromkeys(keys):
        prod = cfg["products"][key]
        if not prod.get("download") and not args.force:
            sys.exit(f"{key} is catalog-only (download: false); pass --force to override")
        files = read_manifest(cfg, key, args.refresh_manifest)
        total = sum(s for _, s in files)
        print(f"{key}: {len(files)} files, {human(total)} -> {local_path(cfg, prod['archive_path'])}")
        jobs += [(url_for(p), local_path(cfg, p), s) for p, s in files]

    if args.dry_run:
        return

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch, *job): job for job in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), unit="file"):
            try:
                fut.result()
            except Exception as exc:
                failed.append((futs[fut][0], exc))
    for url, exc in failed:
        print("FAILED", url, exc, file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
