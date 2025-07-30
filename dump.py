#!/usr/bin/env python3
"""
dump_code_and_jsonc.py — Recursively collects all .py and .jsonc files in the local
'src/' directory tree (relative to this script) and writes their paths and contents
to a single dump file.

This version deliberately restricts the search root to 'src/' only.
"""

import os
import sys
from pathlib import Path

# Configuration
SRC_DIR = Path(__file__).resolve().parent / "src"  # Fixed search root
OUTPUT_FILE = "all_code_dump.txt"                  # File to write the dump into
EXTENSIONS = (".py", ".jsonc")                     # File extensions to include


def dump_files(root_dir: str, output_path: str, exts: tuple) -> None:
    """
    Recursively write every file with a matching extension into *output_path*.

    Symlinks, the output file itself, and unreadable paths are skipped
    to prevent infinite recursion and permission errors.
    """
    import os, sys
    output_abs  = os.path.abspath(output_path)
    seen_dirs   = set()

    with open(output_path, "w", encoding="utf-8") as out:
        for dirpath, _, filenames in os.walk(root_dir, followlinks=False):
            dir_abs = os.path.abspath(dirpath)
            if dir_abs in seen_dirs:
                continue
            seen_dirs.add(dir_abs)

            for name in filenames:
                if not any(name.lower().endswith(ext.lower()) for ext in exts):
                    continue
                file_abs = os.path.abspath(os.path.join(dirpath, name))
                # Skip the output file, symlinks, and an old helper name if present
                if file_abs == output_abs or os.path.islink(file_abs) or name == "dump.py":
                    continue

                out.write(f"===== {file_abs} =====\n")
                try:
                    with open(file_abs, "r", encoding="utf-8", errors="replace") as f:
                        out.write(f.read())
                except Exception as exc:
                    out.write(f"# Could not read file: {type(exc).__name__}: {exc}\n")
                out.write("\n\n")


if __name__ == "__main__":
    # Enforce 'src/' as the only search root
    if not SRC_DIR.is_dir():
        sys.exit(f"Error: '{SRC_DIR}' does not exist or is not a directory.")

    # Allow overriding output file and extensions only
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = OUTPUT_FILE

    if len(sys.argv) > 2:
        extensions = tuple(sys.argv[2:])
    else:
        extensions = EXTENSIONS

    root_dir = str(SRC_DIR)
    dump_files(root_dir, output_file, extensions)
    print(f"Dumped all {', '.join(extensions)} files under '{root_dir}' into '{output_file}'")
