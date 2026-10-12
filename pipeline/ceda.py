"""Helpers for crawling the CEDA archive (https://dap.ceda.ac.uk).

CEDA serves nginx autoindex pages, which list each entry's name, modification time
and size in bytes ("-" for directories). Only the standard library is used, so the
crawler can run before the project environment exists.
"""

from __future__ import annotations

import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import unquote, urljoin

DAP_ROOT = "https://dap.ceda.ac.uk"

_ENTRY = re.compile(
    r'<a href="(?P<href>[^"]+)">[^<]*</a>\s+'
    r"(?P<date>\d{2}-\w{3}-\d{4} \d{2}:\d{2})\s+(?P<size>\d+|-)"
)


@dataclass
class Entry:
    path: str  # archive path, e.g. /neodc/esacci/permafrost/.../file.nc
    size: int  # bytes (files only)
    mtime: str


def _get(url: str, retries: int = 4, timeout: int = 60) -> str:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def list_dir(path: str) -> tuple[list[str], list[Entry]]:
    """Return (subdirectory paths, file entries) for one archive directory."""
    path = path if path.endswith("/") else path + "/"
    html = _get(DAP_ROOT + path)
    dirs, files = [], []
    for m in _ENTRY.finditer(html):
        href = m["href"]
        if href.startswith("..") or href.startswith("?"):
            continue
        full = unquote(urljoin(path, href))
        if href.endswith("/"):
            dirs.append(full)
        else:
            files.append(Entry(full, int(m["size"]), m["date"]))
    return dirs, files


def crawl(path: str, workers: int = 8, max_depth: int = 12) -> list[Entry]:
    """Recursively list all files below an archive path (breadth-first, threaded)."""
    out: list[Entry] = []
    frontier = [path]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(max_depth):
            if not frontier:
                break
            next_frontier: list[str] = []
            for dirs, files in pool.map(list_dir, frontier):
                next_frontier.extend(dirs)
                out.extend(files)
            frontier = next_frontier
    return out


def url_for(path: str) -> str:
    return DAP_ROOT + path
