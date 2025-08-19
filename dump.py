#!/usr/bin/env python3
"""
Recursively collects all .py and .jsonc files in a directory tree
and writes their paths and contents to a single dump file.
"""

import sys
from pathlib import Path

# Configuration
ROOT_DIR = Path(".")
OUTPUT_BASE = "codebase_v"
OUTPUT_EXT = ".py"
EXTENSIONS = (".py", ".jsonc")  # Removed .sh from extensions


def get_next_version(output_dir: Path) -> int:
    """
    Find the next available version number for the output file.
    """
    existing_files = list(output_dir.glob(f"{OUTPUT_BASE}*{OUTPUT_EXT}"))
    if not existing_files:
        return 1

    versions = []
    for file in existing_files:
        try:
            version_str = file.stem[len(OUTPUT_BASE) :]
            versions.append(int(version_str))
        except ValueError:
            continue

    return max(versions) + 1 if versions else 1


def dump_files(root_dir: Path, output_path: Path, extensions: tuple[str, ...]) -> None:
    """
    Recursively write every file with a matching extension into the output file.

    Skips symlinks, the output file itself, unreadable paths, the dump script,
    and files named 'readme' or 'dump.py'.
    """
    script_path = Path(__file__).resolve()
    output_abs = output_path.resolve()
    seen_dirs = set()
    excluded_files = {"readme", "dump.py"}  # Files to exclude by name

    with output_path.open("w", encoding="utf-8") as out:
        for dirpath, dirnames, filenames in root_dir.walk():
            # Exclude 'testing' directories
            dirnames[:] = [d for d in dirnames if d != "testing"]

            dir_abs = dirpath.resolve()
            if dir_abs in seen_dirs:
                continue
            seen_dirs.add(dir_abs)

            for name in filenames:
                if (
                    not name.lower().endswith(extensions)
                    or name.lower() in excluded_files
                ):
                    continue
                file_path = (dirpath / name).resolve()
                if (
                    file_path == output_abs
                    or file_path == script_path
                    or file_path.is_symlink()
                ):
                    continue

                out.write(f"===== {file_path} =====\n")
                try:
                    with file_path.open("r", encoding="utf-8", errors="replace") as f:
                        out.write(f.read())
                except Exception as exc:
                    out.write(f"# Could not read file: {type(exc).__name__}: {exc}\n")
                out.write("\n\n")


if __name__ == "__main__":
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT_DIR
    extensions = tuple(sys.argv[2:]) if len(sys.argv) > 2 else EXTENSIONS

    next_version = get_next_version(root_dir)
    output_file = root_dir / f"{OUTPUT_BASE}{next_version}{OUTPUT_EXT}"

    dump_files(root_dir, output_file, extensions)
    print(
        f"Dumped all {', '.join(extensions)} files under '{root_dir}' into '{output_file}'"
    )
