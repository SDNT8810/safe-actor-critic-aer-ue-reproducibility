#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from _bootstrap import ROOT

MANIFEST = ROOT / "MANIFEST.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return sorted(
        Path(line)
        for line in result.stdout.splitlines()
        if line and line != MANIFEST.name
    )


def write_manifest() -> None:
    lines = [f"{sha256(ROOT / relative)}  {relative.as_posix()}" for relative in release_files()]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"manifest written: {len(lines)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write or verify the release SHA-256 manifest")
    parser.add_argument("--write", action="store_true", help="regenerate MANIFEST.sha256")
    args = parser.parse_args()
    if args.write:
        write_manifest()
        return

    failures: list[str] = []
    entries = 0
    for raw_line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        expected, relative = raw_line.split("  ", 1)
        path = ROOT / Path(relative)
        entries += 1
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"hash mismatch: {relative}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"manifest verified: {entries} files")


if __name__ == "__main__":
    main()
