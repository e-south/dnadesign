"""
Materialize RT-lnRNA Reader SPOP condition-structure outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ..reader_spop_plan import build_reader_spop_plan
from .condition_matrix import build_reader_spop_condition_matrix, write_reader_spop_condition_matrix
from .paths import DEFAULT_OUTPUT_DIR, resolve_repo_root
from .render import render_spop_condition_structure_plot
from .structure_manifest import build_retron_structure_thumbnail_manifest, write_retron_structure_thumbnail_manifest


def materialize_spop_condition_structure_outputs(
    *,
    reader_root: Path | None = None,
    repo_root: Path | None = None,
    output_dir: Path | None = None,
    hairpin_output_dir: Path | None = None,
) -> dict[str, object]:
    """Build tables and plots for the RT-lnRNA SPOP condition-structure view."""

    root = resolve_repo_root(repo_root)
    resolved_output_dir = root / (output_dir or DEFAULT_OUTPUT_DIR)
    plan = build_reader_spop_plan(reader_root=reader_root, strict=True)
    matrix = build_reader_spop_condition_matrix(plan)
    matrix_tables = write_reader_spop_condition_matrix(matrix, output_dir=resolved_output_dir / "tables")
    thumbnail_rows = build_retron_structure_thumbnail_manifest(
        repo_root=root,
        assay_subject_keys=tuple(row.assay_subject_key for row in matrix.rows),
        hairpin_output_dir=hairpin_output_dir,
    )
    thumbnail_path = write_retron_structure_thumbnail_manifest(
        thumbnail_rows,
        output_dir=resolved_output_dir / "tables",
    )
    plot_manifest = render_spop_condition_structure_plot(
        condition_matrix=matrix,
        thumbnail_rows=thumbnail_rows,
        output_dir=resolved_output_dir / "plots",
        repo_root=root,
    )
    return {
        "ok": plan.ok,
        "issues": [issue.to_dict() for issue in plan.issues],
        "condition_matrix": matrix_tables.to_dict(),
        "structure_thumbnail_manifest_path": thumbnail_path,
        "plot_manifest": plot_manifest.to_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize RT-lnRNA SPOP condition-structure outputs.")
    parser.add_argument("--reader-root", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hairpin-output-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = materialize_spop_condition_structure_outputs(
        reader_root=args.reader_root,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        hairpin_output_dir=args.hairpin_output_dir,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(payload["plot_manifest"]["plot_png_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
