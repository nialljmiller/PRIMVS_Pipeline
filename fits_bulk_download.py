import subprocess
from pathlib import Path
from primvs_pipeline.primvs_api import PrimvsCatalog

# Configuration
REMOTE_HOST = "uhhpc"
REMOTE_BASE = "/beegfs/car/njm/virac_lightcurves/"  # note trailing slash not required
LOCAL_BASE = "/project/galacticbulge/PRIMVS/light_curves/"

cat = PrimvsCatalog(LOCAL_BASE)

def sid_to_relpath(sid) -> str:
    s = str(int(sid))
    return f"{s[:3]}/{s[3:6]}/{s}.csv"

def bulk_rsync_missing(source_ids, manifest_path="transfer_list.txt") -> int:
    # 1) Build missing list
    missing_paths = []
    for sid in source_ids:
        if not cat.source_exists(sid):
            missing_paths.append(sid_to_relpath(sid))

    if not missing_paths:
        print("No missing files detected.")
        return 0

    # 2) Write manifest
    with open(manifest_path, "w") as f:
        f.write("\n".join(missing_paths) + "\n")

    # 3) Execute rsync once
    print(f"Requesting {len(missing_paths)} files via rsync...")
    subprocess.run(
        [
            "rsync", "-avhP",
            "--files-from=" + manifest_path,
            f"{REMOTE_HOST}:{REMOTE_BASE}",
            LOCAL_BASE,
        ],
        check=True,
    )
    print("✓ Bulk rsync finished.")
    return len(missing_paths)

if __name__ == "__main__":
    from astropy.table import Table
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fits_download.py <input_fits>")
        print("Example: python fits_download.py reclass.fits")
        sys.exit(1)

    input_fits = sys.argv[1]
    tbl = Table.read(input_fits, hdu=1)
    source_ids = tbl["sourceid"].data
    print(f"Found {len(source_ids)} source IDs in {input_fits}\n")

    # Count existing before
    already_exist = sum(1 for sid in source_ids if cat.source_exists(sid))
    missing = len(source_ids) - already_exist

    downloaded = 0
    if missing:
        downloaded = bulk_rsync_missing(source_ids)

    print("\nSummary:")
    print(f"  Already existed: {already_exist}")
    print(f"  Requested (missing): {missing}")
    print(f"  Bulk rsync attempted: {downloaded}")
    print(f"  Total IDs: {len(source_ids)}")

    # Optional: verify after
    still_missing = sum(1 for sid in source_ids if not cat.source_exists(sid))
    if still_missing:
        print(f"\nWARNING: Still missing after rsync: {still_missing}")
    else:
        print("\n✓ All requested lightcurves appear present locally.")
