"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/integrations/dense_arrays/__main__.py

Provide the CLI entry point for DenseGen solution-playback publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .publisher import publish_densegen_playback_endpoint


def main() -> None:
    """Publish one configured endpoint from existing persisted records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Endpoint YAML path")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing generated endpoint bundle",
    )
    args = parser.parse_args()
    output = publish_densegen_playback_endpoint(args.config, replace=args.replace)
    print(output)


if __name__ == "__main__":
    main()
