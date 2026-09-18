#!/usr/bin/env python
"""Record and verify the contents of the datasets/ tree.

The snapshot tree is large enough that it may travel separately from the code (as a
release asset, or a copy on a scratch disk), so a copy that is silently truncated or
partially synced is a real failure mode. The manifest makes that detectable: it
records a SHA-256 digest per file, and `--check` reports anything missing, extra or
altered.

    python scripts/dataset_manifest.py --check     # verify against the manifest
    python scripts/dataset_manifest.py --write     # regenerate after an intended change
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = REPO_ROOT / "datasets"
MANIFEST = DATASETS / "MANIFEST.sha256"

#: Files that describe the tree rather than belong to it.
EXCLUDED_NAMES = {"MANIFEST.sha256", "README.md", ".DS_Store"}


def digest(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_datasets():
    """Every data file under datasets/, as paths relative to it, sorted."""
    if not DATASETS.is_dir():
        raise SystemExit(f"No datasets/ directory at {DATASETS}. See datasets/README.md.")
    paths = [
        p for p in DATASETS.rglob("*")
        if p.is_file() and p.name not in EXCLUDED_NAMES
    ]
    return sorted(p.relative_to(DATASETS) for p in paths)


def read_manifest():
    if not MANIFEST.exists():
        raise SystemExit(f"No manifest at {MANIFEST}. Generate one with --write.")
    recorded = {}
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # `sha256  relative/path`, the format sha256sum itself writes.
        sha, _, rel = line.partition("  ")
        recorded[rel] = sha
    return recorded


def write_manifest():
    entries = walk_datasets()
    lines = [
        "# SHA-256 digests of every file under datasets/.",
        "# Regenerate with: python scripts/dataset_manifest.py --write",
        "# Verify with:     python scripts/dataset_manifest.py --check",
    ]
    total = 0
    for rel in entries:
        path = DATASETS / rel
        total += path.stat().st_size
        lines.append(f"{digest(path)}  {rel}")
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"Wrote {MANIFEST.relative_to(REPO_ROOT)}: "
          f"{len(entries)} files, {total / 1e6:.1f} MB.")
    return 0


def check_manifest():
    recorded = read_manifest()
    present = {str(rel) for rel in walk_datasets()}

    missing = sorted(set(recorded) - present)
    extra = sorted(present - set(recorded))
    altered = []
    for rel in sorted(set(recorded) & present):
        if digest(DATASETS / rel) != recorded[rel]:
            altered.append(rel)

    for label, items in (("missing", missing), ("altered", altered), ("unlisted", extra)):
        for rel in items:
            print(f"{label.upper():9s} {rel}")

    if missing or altered:
        print(f"\nFAILED: {len(missing)} missing, {len(altered)} altered, "
              f"{len(extra)} unlisted, out of {len(recorded)} recorded files.")
        return 1
    if extra:
        print(f"\nOK, with {len(extra)} file(s) present but not in the manifest. "
              f"Run --write if they belong there.")
        return 0
    print(f"OK: all {len(recorded)} files match the manifest.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="regenerate the manifest")
    group.add_argument("--check", action="store_true", help="verify the tree against it")
    args = parser.parse_args()
    return write_manifest() if args.write else check_manifest()


if __name__ == "__main__":
    sys.exit(main())
