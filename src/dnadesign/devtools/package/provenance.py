"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/package/provenance.py

Portable build-bundle integrity checks; no network or installation side effects.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from hashlib import file_digest
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def file_record(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_bundle(directory: Path, *, expected_sha256: str | None = None) -> dict:
    """Check recorded artifact bytes; the caller must separately trust the receipt."""
    root = directory.resolve()
    receipt = root / "provenance.json"
    if expected_sha256 is not None and sha256(receipt) != expected_sha256:
        raise ValueError("Build provenance differs from the independently recorded checksum")
    record = json.loads(receipt.read_text())
    if record.get("schema") != "dnadesign-build/v1":
        raise ValueError("Unsupported build provenance schema")
    names = [item["path"] for item in record["files"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate build artifact path")
    required = {"build-requirements.in", "build-requirements.lock", "CITATION.cff"}
    if (
        not required <= set(names)
        or sum(n.endswith(".whl") for n in names) != 1
        or sum(n.endswith(".tar.gz") for n in names) != 1
    ):
        raise ValueError(
            "Incomplete build bundle: requires wheel, source distribution, build requirements and citation"
        )
    for expected in record["files"]:
        relative = PurePosixPath(expected["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise ValueError("Build artifact path must stay within the bundle")
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError("Build artifact symlink escapes the bundle")
        if file_record(path, root) != expected:
            raise ValueError(f"Build artifact differs from provenance: {expected['path']}")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a downloaded build bundle against its provenance")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--receipt-sha256", help="Checksum retained independently from the downloaded bundle")
    args = parser.parse_args(argv)
    try:
        record = verify_bundle(args.directory, expected_sha256=args.receipt_sha256)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Verified {record['name']} {record['version']}: {len(record['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
