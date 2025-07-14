#!/usr/bin/env python3
"""
dump_code_and_jsonc.py — Recursively collects all .py and .jsonc files in a directory tree
and writes their paths and contents to a single dump file.

"""

import os
import sys
from pathlib import Path

# Configuration
ROOT_DIR = "."                   # Directory to start searching from
OUTPUT_FILE = "all_code_dump.txt"  # File to write the dump into
EXTENSIONS = (".py", ".jsonc")   # File extensions to include

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
                if file_abs == output_abs or os.path.islink(file_abs):
                    continue

                out.write(f"===== {file_abs} =====\n")
                try:
                    with open(file_abs, "r", encoding="utf-8", errors="replace") as f:
                        out.write(f.read())
                except Exception as exc:
                    out.write(f"# Could not read file: {type(exc).__name__}: {exc}\n")
                out.write("\n\n")


if __name__ == "__main__":
    # Allow command line arguments for flexibility
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = ROOT_DIR
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = OUTPUT_FILE
    
    if len(sys.argv) > 3:
        extensions = tuple(sys.argv[3:])
    else:
        extensions = EXTENSIONS
    
    dump_files(root_dir, output_file, extensions)
    print(f"Dumped all {', '.join(extensions)} files under '{root_dir}' into '{output_file}'")