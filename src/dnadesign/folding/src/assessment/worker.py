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
from collections.abc import Sequence

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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
