#!/usr/bin/env python3
"""
Fast parallel bulk download of PRIMVS lightcurves.

Strategy: Split missing files by their top-level directory prefix (e.g., 512/xxx/xxx.csv)
and run multiple rsync processes in parallel. This dramatically reduces per-file overhead
compared to a single rsync with a giant manifest.

Usage:
    python fits_bulk_download_fast.py <input_fits> [--workers 8] [--dry-run]

Examples:
    python fits_bulk_download_fast.py ../catalog/PRIMVS_P01.fits
    python fits_bulk_download_fast.py ../catalog/PRIMVS_P01.fits --workers 16
    python fits_bulk_download_fast.py ../catalog/PRIMVS_P01.fits --dry-run
"""

import subprocess
import sys
import os
import argparse
import tempfile
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Configuration ──────────────────────────────────────────────────────────
REMOTE_HOST = "uhhpc"
REMOTE_BASE = "/beegfs/car/njm/virac_lightcurves/"
LOCAL_BASE = "/project/galacticbulge/PRIMVS/light_curves/"
DEFAULT_WORKERS = 8
# ───────────────────────────────────────────────────────────────────────────


def sid_to_relpath(sid) -> str:
    s = str(int(sid))
    return f"{s[:3]}/{s[3:6]}/{s}.csv"


def scan_existing_sids(base_dir: str) -> set:
    """Walk the directory tree once and return a set of source IDs already on disk."""
    existing = set()
    base = Path(base_dir)
    if not base.exists():
        return existing
    for csv_file in base.rglob("*.csv"):
        existing.add(csv_file.stem)  # filename without .csv extension
    return existing


def rsync_prefix(prefix: str, relpaths: list, tmpdir: str, dry_run: bool = False) -> dict:
    """Run rsync for a single top-level prefix directory. Returns stats dict."""
    manifest = os.path.join(tmpdir, f"manifest_{prefix}.txt")
    with open(manifest, "w") as f:
        f.write("\n".join(relpaths) + "\n")

    cmd = [
        "rsync", "-avh", "--compress",
        "--files-from=" + manifest,
        f"{REMOTE_HOST}:{REMOTE_BASE}",
        LOCAL_BASE,
    ]
    if dry_run:
        cmd.insert(2, "--dry-run")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - t0
        success = result.returncode == 0
        if not success:
            err_snippet = (result.stderr or "")[:200]
        else:
            err_snippet = ""
        return {
            "prefix": prefix,
            "requested": len(relpaths),
            "success": success,
            "elapsed": elapsed,
            "error": err_snippet,
        }
    except subprocess.TimeoutExpired:
        return {
            "prefix": prefix,
            "requested": len(relpaths),
            "success": False,
            "elapsed": time.time() - t0,
            "error": "TIMEOUT after 3600s",
        }
    except Exception as e:
        return {
            "prefix": prefix,
            "requested": len(relpaths),
            "success": False,
            "elapsed": time.time() - t0,
            "error": str(e)[:200],
        }


def main():
    parser = argparse.ArgumentParser(description="Fast parallel PRIMVS lightcurve download")
    parser.add_argument("input_fits", help="Input FITS file with sourceid column")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel rsync processes (default: {DEFAULT_WORKERS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be transferred without downloading")
    parser.add_argument("--id-column", default="sourceid",
                        help="Column name for source IDs (default: sourceid)")
    args = parser.parse_args()

    # Suppress TF warnings
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    from astropy.table import Table

    # ── 1. Read source IDs ─────────────────────────────────────────────────
    print(f"Reading {args.input_fits}...")
    tbl = Table.read(args.input_fits, hdu=1)
    source_ids = tbl[args.id_column].data
    total = len(source_ids)
    print(f"  Total source IDs: {total:,}")

    # ── 2. Identify missing files ──────────────────────────────────────────
    print("Scanning local filesystem for existing files...")
    t0 = time.time()
    existing_sids = scan_existing_sids(LOCAL_BASE)
    scan_time = time.time() - t0
    print(f"  Found {len(existing_sids):,} existing files in {scan_time:.1f}s")

    missing_by_prefix = defaultdict(list)
    already_exist = 0

    for sid in source_ids:
        s = str(int(sid))
        if s in existing_sids:
            already_exist += 1
        else:
            relpath = sid_to_relpath(sid)
            prefix = relpath.split("/")[0]
            missing_by_prefix[prefix].append(relpath)

    n_missing = sum(len(v) for v in missing_by_prefix.values())
    n_prefixes = len(missing_by_prefix)
    check_time = time.time() - t0

    print(f"  Already exist locally: {already_exist:,}")
    print(f"  Missing: {n_missing:,}")
    print(f"  Spread across {n_prefixes} top-level directories")
    print(f"  Check took {check_time:.1f}s\n")

    if n_missing == 0:
        print("Nothing to download.")
        return

    # ── 3. Parallel rsync by prefix ────────────────────────────────────────
    workers = min(args.workers, n_prefixes)
    mode = "DRY RUN" if args.dry_run else "DOWNLOADING"
    print(f"{mode} with {workers} parallel rsync workers...")
    print(f"  {n_missing:,} files across {n_prefixes} prefix groups\n")

    t_start = time.time()
    completed = 0
    files_done = 0
    failed_prefixes = []

    with tempfile.TemporaryDirectory(prefix="primvs_dl_") as tmpdir:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(rsync_prefix, prefix, paths, tmpdir, args.dry_run): prefix
                for prefix, paths in missing_by_prefix.items()
            }

            for future in as_completed(futures):
                result = future.result()
                completed += 1
                files_done += result["requested"]
                elapsed_total = time.time() - t_start

                if result["success"]:
                    status = "✓"
                else:
                    status = "✗"
                    failed_prefixes.append(result)

                pct = files_done / n_missing * 100
                rate = files_done / elapsed_total if elapsed_total > 0 else 0
                eta = (n_missing - files_done) / rate if rate > 0 else 0

                print(
                    f"  {status} prefix {result['prefix']:>3s}: "
                    f"{result['requested']:>5,} files in {result['elapsed']:.0f}s  "
                    f"[{completed}/{n_prefixes} groups, {pct:.1f}%, "
                    f"~{rate:.0f} files/s, ETA {eta/60:.0f}m]"
                )

    # ── 4. Summary ─────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Total time:       {total_time/60:.1f} minutes")
    print(f"  Already existed:  {already_exist:,}")
    print(f"  Requested:        {n_missing:,}")
    print(f"  Failed groups:    {len(failed_prefixes)}")
    print(f"  Avg rate:         {n_missing/total_time:.0f} files/s")

    if failed_prefixes:
        print(f"\nFailed prefixes:")
        for fp in failed_prefixes:
            print(f"  {fp['prefix']}: {fp['error']}")

    # ── 5. Verify ──────────────────────────────────────────────────────────
    if not args.dry_run:
        print("\nVerifying...")
        existing_after = scan_existing_sids(LOCAL_BASE)
        still_missing = sum(1 for sid in source_ids if str(int(sid)) not in existing_after)
        if still_missing:
            print(f"  WARNING: Still missing {still_missing:,} files")
        else:
            print(f"  ✓ All {total:,} lightcurves present locally.")


if __name__ == "__main__":
    main()
