#!/usr/bin/env python3
"""
check_invisible_chars.py - Find non-standard / invisible characters in source files.

Usage:
    python check_invisible_chars.py <directory_or_file> [...]

Scans all .py and .json files for characters that can cause mysterious
SyntaxErrors:
  - Non-breaking spaces (U+00A0) -- the #1 culprit
  - Zero-width characters (U+200B, U+200C, U+200D, U+FEFF BOM, etc.)
  - Smart/curly quotes and dashes
  - Any non-ASCII character outside of string literals and comments

For each hit it prints the file, line number, column, the hex codepoint,
a human-readable name, and the line content with the offending character
highlighted.
"""

from __future__ import annotations

import sys
import os
import unicodedata
from pathlib import Path

# Characters that are invisible or look like ASCII but are not.
SUSPECT_CHARS = {
    0x00A0: "NON-BREAKING SPACE",
    0x00AD: "SOFT HYPHEN",
    0x034F: "COMBINING GRAPHEME JOINER",
    0x061C: "ARABIC LETTER MARK",
    0x115F: "HANGUL CHOSEONG FILLER",
    0x1160: "HANGUL JUNGSEONG FILLER",
    0x17B4: "KHMER VOWEL INHERENT AQ",
    0x17B5: "KHMER VOWEL INHERENT AA",
    0x180E: "MONGOLIAN VOWEL SEPARATOR",
    0x2000: "EN QUAD",
    0x2001: "EM QUAD",
    0x2002: "EN SPACE",
    0x2003: "EM SPACE",
    0x2004: "THREE-PER-EM SPACE",
    0x2005: "FOUR-PER-EM SPACE",
    0x2006: "SIX-PER-EM SPACE",
    0x2007: "FIGURE SPACE",
    0x2008: "PUNCTUATION SPACE",
    0x2009: "THIN SPACE",
    0x200A: "HAIR SPACE",
    0x200B: "ZERO WIDTH SPACE",
    0x200C: "ZERO WIDTH NON-JOINER",
    0x200D: "ZERO WIDTH JOINER",
    0x200E: "LEFT-TO-RIGHT MARK",
    0x200F: "RIGHT-TO-LEFT MARK",
    0x2028: "LINE SEPARATOR",
    0x2029: "PARAGRAPH SEPARATOR",
    0x202A: "LEFT-TO-RIGHT EMBEDDING",
    0x202B: "RIGHT-TO-LEFT EMBEDDING",
    0x202C: "POP DIRECTIONAL FORMATTING",
    0x202D: "LEFT-TO-RIGHT OVERRIDE",
    0x202E: "RIGHT-TO-LEFT OVERRIDE",
    0x202F: "NARROW NO-BREAK SPACE",
    0x205F: "MEDIUM MATHEMATICAL SPACE",
    0x2060: "WORD JOINER",
    0x2061: "FUNCTION APPLICATION",
    0x2062: "INVISIBLE TIMES",
    0x2063: "INVISIBLE SEPARATOR",
    0x2064: "INVISIBLE PLUS",
    0x2066: "LEFT-TO-RIGHT ISOLATE",
    0x2067: "RIGHT-TO-LEFT ISOLATE",
    0x2068: "FIRST STRONG ISOLATE",
    0x2069: "POP DIRECTIONAL ISOLATE",
    0x206A: "INHIBIT SYMMETRIC SWAPPING",
    0x206B: "ACTIVATE SYMMETRIC SWAPPING",
    0x206C: "INHIBIT ARABIC FORM SHAPING",
    0x206D: "ACTIVATE ARABIC FORM SHAPING",
    0x206E: "NATIONAL DIGIT SHAPES",
    0x206F: "NOMINAL DIGIT SHAPES",
    0x3000: "IDEOGRAPHIC SPACE",
    0x3164: "HANGUL FILLER",
    0xFE00: "VARIATION SELECTOR-1",
    0xFEFF: "BOM / ZERO WIDTH NO-BREAK SPACE",
    0xFFA0: "HALFWIDTH HANGUL FILLER",
    0xFFF9: "INTERLINEAR ANNOTATION ANCHOR",
    0xFFFA: "INTERLINEAR ANNOTATION SEPARATOR",
    0xFFFB: "INTERLINEAR ANNOTATION TERMINATOR",
    # Smart quotes / dashes (look like ASCII but are not)
    0x2018: "LEFT SINGLE QUOTATION MARK (smart quote)",
    0x2019: "RIGHT SINGLE QUOTATION MARK (smart quote)",
    0x201C: "LEFT DOUBLE QUOTATION MARK (smart quote)",
    0x201D: "RIGHT DOUBLE QUOTATION MARK (smart quote)",
    0x2013: "EN DASH (not ASCII hyphen)",
    0x2014: "EM DASH (not ASCII hyphen)",
    0x2212: "MINUS SIGN (not ASCII hyphen)",
}

EXTENSIONS = {".py", ".json"}


def describe_char(ch):
    cp = ord(ch)
    if cp in SUSPECT_CHARS:
        return SUSPECT_CHARS[cp]
    try:
        return unicodedata.name(ch)
    except ValueError:
        return "UNKNOWN"


def scan_file(filepath):
    """Scan a single file and return a list of findings."""
    findings = []
    try:
        raw = filepath.read_bytes()
    except OSError as e:
        print("  WARNING: cannot read {}: {}".format(filepath, e), file=sys.stderr)
        return findings

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        findings.append({
            "file": str(filepath),
            "line": 0,
            "col": 0,
            "codepoint": "N/A",
            "name": "FILE IS NOT VALID UTF-8",
            "context": "(could not decode file)",
        })
        return findings

    # Check for BOM at start of file
    if raw[:3] == b"\xef\xbb\xbf":
        findings.append({
            "file": str(filepath),
            "line": 1,
            "col": 0,
            "codepoint": "U+FEFF",
            "name": "UTF-8 BOM (can confuse Python)",
            "context": "(byte order mark at start of file)",
        })

    for lineno, line in enumerate(text.splitlines(), start=1):
        for col, ch in enumerate(line, start=1):
            cp = ord(ch)

            # Fast path: normal ASCII printable + tab + standard space
            if 0x20 <= cp <= 0x7E or cp == 0x09:
                continue

            is_suspect = cp in SUSPECT_CHARS
            is_nonascii = cp > 0x7E

            if not is_suspect and not is_nonascii:
                continue

            # For non-ASCII that is not in our curated list,
            # skip if the line is a whole-line comment (non-ASCII in comments is usually fine)
            if not is_suspect and is_nonascii:
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue

            # Build a visual marker showing where the bad char is
            before = line[:col - 1]
            after = line[col:]
            tag = "[U+{:04X}]".format(cp)
            marker_line = before + " >>>" + tag + "<<< " + after

            findings.append({
                "file": str(filepath),
                "line": lineno,
                "col": col,
                "codepoint": "U+{:04X}".format(cp),
                "name": describe_char(ch),
                "context": marker_line.rstrip(),
            })

    return findings


def scan_directory(root):
    all_findings = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden dirs, __pycache__, .git, etc.
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d != "__pycache__" and d != "node_modules"
        ]
        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            if fpath.suffix in EXTENSIONS:
                all_findings.extend(scan_file(fpath))
    return all_findings


def main():
    if len(sys.argv) < 2:
        print("Usage: {} <path> [path ...]".format(sys.argv[0]), file=sys.stderr)
        print("  Scans .py and .json files for invisible/non-standard characters.", file=sys.stderr)
        sys.exit(1)

    all_findings = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_file():
            all_findings.extend(scan_file(p))
        elif p.is_dir():
            all_findings.extend(scan_directory(p))
        else:
            print("WARNING: {} not found, skipping.".format(arg), file=sys.stderr)

    if not all_findings:
        print("No suspicious characters found.")
        sys.exit(0)

    # Print results grouped by file
    current_file = None
    for f in all_findings:
        if f["file"] != current_file:
            current_file = f["file"]
            print("")
            print("=" * 80)
            print("FILE: {}".format(current_file))
            print("=" * 80)

        print("  Line {:>5}, Col {:>3}  |  {}  {}".format(
            f["line"], f["col"], f["codepoint"], f["name"]
        ))
        print("    {}".format(f["context"]))

    print("")
    print("=" * 80)
    print("TOTAL: {} suspicious character(s) found.".format(len(all_findings)))
    print("=" * 80)
    sys.exit(1)


if __name__ == "__main__":
    main()


    