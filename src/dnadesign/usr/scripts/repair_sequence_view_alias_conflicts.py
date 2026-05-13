"""
Repair non-unique aliases in a USR sequence-view sidecar.

The repair removes aliases that resolve to more than one view_id. It does not
rewrite records.parquet, view IDs, lineage, bounds, or Infer sidecars.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dnadesign.usr import Dataset
from dnadesign.usr.src.sequence_views.maintenance import repair_sequence_view_alias_conflicts


def _default_usr_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets"


def _actor() -> dict[str, object]:
    return {
        "tool": "usr",
        "run_id": "repair_sequence_view_alias_conflicts",
        "command": "repair_sequence_view_alias_conflicts",
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usr-root", type=Path, default=_default_usr_root())
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--format", choices=["json"], default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset = Dataset(args.usr_root, args.dataset)
    result = repair_sequence_view_alias_conflicts(
        dataset,
        write=bool(args.write),
        example_limit=int(args.example_limit),
        actor=_actor() if args.write else None,
    )
    payload: dict[str, Any] = asdict(result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
