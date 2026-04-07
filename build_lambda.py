"""
Rebuild lambda_package.zip by stripping safe-to-remove items from lambda_package/.
Safe to strip: test dirs, dist-info, egg-info, __pycache__, .pyc files.
Never stripped: .so files, .so.* symlinks, or any binary inside pyarrow.
"""
import os
import shutil
import zipfile
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent / "lambda_package"
ZIP_PATH = Path(__file__).parent / "lambda_package.zip"

# Directories whose entire subtree should be deleted
STRIP_DIR_NAMES = {"__pycache__", "tests", "test"}
# Directory suffixes to strip (matched against the full dir name)
STRIP_DIR_SUFFIXES = (".dist-info", ".egg-info")


def should_strip_dir(path: Path) -> bool:
    name = path.name
    if name in STRIP_DIR_NAMES:
        return True
    for suffix in STRIP_DIR_SUFFIXES:
        if name.endswith(suffix):
            return True
    return False


def strip_package(root: Path) -> tuple[int, int]:
    """Remove strip-eligible dirs and .pyc files. Returns (dirs_removed, files_removed)."""
    dirs_removed = 0
    files_removed = 0

    # Walk top-down so we can prune whole subtrees
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        current = Path(dirpath)

        # Remove strip-eligible subdirectories (modify dirnames in-place to skip descent)
        to_remove = [d for d in dirnames if should_strip_dir(current / d)]
        for d in to_remove:
            target = current / d
            print(f"  Removing dir:  {target.relative_to(root)}")
            shutil.rmtree(target)
            dirnames.remove(d)
            dirs_removed += 1

        # Remove .pyc files
        for fname in filenames:
            if fname.endswith(".pyc"):
                fpath = current / fname
                print(f"  Removing file: {fpath.relative_to(root)}")
                fpath.unlink()
                files_removed += 1

    return dirs_removed, files_removed


def zip_package(root: Path, zip_path: Path) -> int:
    """Zip all files in root into zip_path. Returns file count."""
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fpath in sorted(root.rglob("*")):
            if fpath.is_file():
                arcname = fpath.relative_to(root)
                zf.write(fpath, arcname)
                count += 1
    return count


def human_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


if __name__ == "__main__":
    print(f"Package dir: {PACKAGE_DIR}")
    print(f"Size before: {dir_size_mb(PACKAGE_DIR):.1f} MB\n")

    print("Stripping...")
    dirs_removed, files_removed = strip_package(PACKAGE_DIR)
    print(f"\nRemoved {dirs_removed} directories, {files_removed} .pyc files.")

    after_mb = dir_size_mb(PACKAGE_DIR)
    print(f"Size after:  {after_mb:.1f} MB")

    if after_mb > 250:
        print(f"\nWARNING: {after_mb:.1f} MB exceeds the 250 MB Lambda limit!")
    else:
        print(f"OK: under 250 MB limit.")

    # Verify pyarrow .so files are still present
    so_files = list((PACKAGE_DIR / "pyarrow").glob("*.so*"))
    print(f"\npyarrow .so files present: {len(so_files)}")
    if not so_files:
        print("ERROR: no .so files found in pyarrow — something went wrong!")
    else:
        for f in sorted(so_files)[:5]:
            print(f"  {f.name}")
        if len(so_files) > 5:
            print(f"  ... and {len(so_files) - 5} more")

    print(f"\nZipping to {ZIP_PATH} ...")
    count = zip_package(PACKAGE_DIR, ZIP_PATH)
    print(f"Zipped {count} files -> {human_size(ZIP_PATH)}")
