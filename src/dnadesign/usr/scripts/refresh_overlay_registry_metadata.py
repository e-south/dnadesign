"""
Refresh compact USR overlay metadata through the registry-validating overlay writer.

This is intentionally narrow maintenance tooling: it rewrites an existing compact
overlay with identical rows and current registry metadata. It does not create
missing overlays, alter base rows, or repair schema drift.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.usr import Dataset
from dnadesign.usr.src.overlays import overlay_metadata


@dataclass(frozen=True)
class RefreshOverlayResult:
    dataset: str
    namespace: str
    overlay_path: str
    rows_refreshed: int
    previous_registry_hash: str | None
    refreshed_registry_hash: str | None


def _default_usr_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets"


def refresh_overlay_registry_metadata(
    *,
    usr_root: Path,
    dataset_name: str,
    namespace: str,
    validate_strict: bool = True,
) -> RefreshOverlayResult:
    dataset = Dataset(usr_root, dataset_name)
    overlay_path = dataset.dir / "_derived" / f"{namespace}.parquet"
    if not overlay_path.exists():
        raise FileNotFoundError(f"Compact overlay not found: {overlay_path}")
    if not overlay_path.is_file():
        raise FileNotFoundError(f"Expected compact overlay file, got: {overlay_path}")

    previous_hash = overlay_metadata(overlay_path).get("registry_hash")
    table = pq.read_table(overlay_path)
    key = overlay_metadata(overlay_path).get("key") or "id"
    with dataset.maintenance(reason=f"refresh_{namespace}_overlay_registry_metadata"):
        rows = dataset.write_overlay(namespace, table, key=key, overwrite=True)
    if validate_strict:
        dataset.validate(strict=True)
    refreshed_hash = overlay_metadata(overlay_path).get("registry_hash")
    return RefreshOverlayResult(
        dataset=dataset.name,
        namespace=namespace,
        overlay_path=str(overlay_path),
        rows_refreshed=int(rows),
        previous_registry_hash=previous_hash,
        refreshed_registry_hash=refreshed_hash,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh compact USR overlay registry metadata.")
    parser.add_argument("dataset")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--usr-root", type=Path, default=_default_usr_root())
    parser.add_argument("--no-strict-validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = refresh_overlay_registry_metadata(
        usr_root=args.usr_root,
        dataset_name=args.dataset,
        namespace=args.namespace,
        validate_strict=not bool(args.no_strict_validate),
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
