"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/assessment/worker.py

Isolated worker process for one secondary-structure prediction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ..api import load_prediction_request, run_prediction_request


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)
    request, request_path = load_prediction_request(args.request)
    run_prediction_request(
        request,
        output_dir=args.output_dir,
        request_path=request_path,
        raise_on_required_failure=False,
        backend_timeout_seconds=None,
    )
    _normalize_preflight_output_dir(Path(args.output_dir))
    return 0


def _normalize_preflight_output_dir(output_dir: Path) -> None:
    """Remove the ephemeral staging path from portable assessment evidence."""
    root = output_dir.expanduser().resolve()
    path = root / "folding_preflight.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("output_dir") != root.as_posix():
        raise ValueError("Assessment preflight output directory changed before normalization.")
    payload["output_dir"] = "."
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
