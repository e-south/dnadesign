"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/outputs/output_guards.py

Fail-fast output-layout guards for Retron MSD compiler bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ..compiler.exceptions import RetronMsdCompilerError
from .layout import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_README_FILENAME,
    CATALOG_FILENAME,
    COMPOSITION_CONFIG_DIRNAME,
    IGNORED_OUTPUT_FILENAMES,
    LEGACY_ASSETS_DIRNAME,
    MANIFEST_BUNDLE_DIRNAME,
    MANIFEST_CATALOG_DIRNAME,
    MANIFEST_CONFIGS_DIRNAME,
    MANIFEST_DIRNAME,
    MANIFEST_INDEXES_DIRNAME,
    REFERENCE_DIRNAME,
    REFERENCE_INDEX_FILENAME,
    SEQUENCE_INDEX_FILENAME,
    SEQUENCE_MANIFEST_FILENAME,
    VARIANT_DIRNAME,
    VARIANT_MANIFEST_COMPOSITION_DIRNAME,
    VARIANT_MANIFEST_CONSTRUCT_DIRNAME,
    VARIANT_MANIFEST_DIRNAME,
    VARIANT_MANIFEST_FOLDING_DIRNAME,
    VARIANT_MANIFEST_PROVENANCE_DIRNAME,
    VARIANT_MANIFEST_REVIEWS_DIRNAME,
    VARIANT_MANIFEST_SECONDARY_STRUCTURE_DIRNAME,
    VARIANT_MANIFEST_VISUAL_DIRNAME,
    VARIANT_PLOTS_DIRNAME,
    VARIANT_RUNTIME_DIRNAME,
    VARIANT_SEQUENCES_DIRNAME,
)


def guard_catalog_output_layout(
    root: Path,
    *,
    expected_reference_filenames: set[str],
    extra_allowed_top_level: set[str],
) -> None:
    if not root.exists():
        return
    if not root.is_dir():
        raise RetronMsdCompilerError(f"MSD compiler output path exists but is not a directory: {root}")

    legacy_assets_dir = root / LEGACY_ASSETS_DIRNAME
    if legacy_assets_dir.exists():
        raise RetronMsdCompilerError(
            f"Legacy MSD compiler output layout exists at {legacy_assets_dir}. "
            "Choose a fresh --out-dir or explicitly archive/remove the old generated assets directory before compiling."
        )

    allowed_top_level = {
        BUNDLE_README_FILENAME,
        BUNDLE_MANIFEST_FILENAME,
        CATALOG_FILENAME,
        REFERENCE_INDEX_FILENAME,
        REFERENCE_DIRNAME,
        *extra_allowed_top_level,
        *IGNORED_OUTPUT_FILENAMES,
    }
    unexpected_top_level = sorted(item.name for item in root.iterdir() if item.name not in allowed_top_level)
    if unexpected_top_level:
        raise RetronMsdCompilerError(
            f"Unexpected MSD compiler output entries at {root}: {', '.join(unexpected_top_level)}. "
            "Choose a fresh --out-dir or explicitly archive/remove unrelated generated entries before compiling."
        )

    references_dir = root / REFERENCE_DIRNAME
    if not references_dir.exists():
        return
    stale_reference_entries = sorted(
        item.name
        for item in references_dir.iterdir()
        if item.name not in IGNORED_OUTPUT_FILENAMES
        and (item.is_dir() or item.name not in expected_reference_filenames)
    )
    if stale_reference_entries:
        raise RetronMsdCompilerError(
            f"Stale MSD design reference output at {references_dir}: {', '.join(stale_reference_entries)}. "
            "Choose a fresh --out-dir or explicitly archive/remove stale generated references before compiling."
        )


def guard_materialize_output_layout(
    root: Path,
    *,
    expected_variant_dirnames: set[str],
    expected_reference_filenames: set[str],
) -> None:
    if not root.exists():
        return
    if not root.is_dir():
        raise RetronMsdCompilerError(f"MSD materialize output path exists but is not a directory: {root}")

    legacy_assets_dir = root / LEGACY_ASSETS_DIRNAME
    if legacy_assets_dir.exists():
        raise RetronMsdCompilerError(
            f"Legacy MSD compiler output layout exists at {legacy_assets_dir}. "
            "Choose a fresh --out-dir or explicitly archive/remove the old generated assets directory before compiling."
        )

    allowed_top_level = {BUNDLE_README_FILENAME, MANIFEST_DIRNAME, VARIANT_DIRNAME, *IGNORED_OUTPUT_FILENAMES}
    unexpected_top_level = sorted(item.name for item in root.iterdir() if item.name not in allowed_top_level)
    if unexpected_top_level:
        raise RetronMsdCompilerError(
            f"Unexpected MSD materialize output entries at {root}: {', '.join(unexpected_top_level)}. "
            "Choose a fresh --out-dir or explicitly archive/remove unrelated generated entries before materializing."
        )

    manifest_dir = root / MANIFEST_DIRNAME
    _guard_materialize_directory_entries(
        manifest_dir,
        allowed_entries={
            MANIFEST_BUNDLE_DIRNAME,
            MANIFEST_CATALOG_DIRNAME,
            MANIFEST_CONFIGS_DIRNAME,
            MANIFEST_INDEXES_DIRNAME,
        },
        output_label="materialize manifest",
        next_step="Choose a fresh --out-dir or archive/remove stale output first.",
    )
    _guard_materialize_directory_entries(
        manifest_dir / MANIFEST_BUNDLE_DIRNAME,
        allowed_entries={BUNDLE_MANIFEST_FILENAME, SEQUENCE_MANIFEST_FILENAME},
        output_label="bundle manifest",
        next_step="Choose a fresh --out-dir or archive/remove stale bundle manifest output first.",
    )
    _guard_materialize_directory_entries(
        manifest_dir / MANIFEST_CATALOG_DIRNAME,
        allowed_entries={CATALOG_FILENAME, REFERENCE_DIRNAME},
        output_label="catalog manifest",
        next_step="Choose a fresh --out-dir or archive/remove stale catalog output first.",
    )
    _guard_materialize_directory_entries(
        manifest_dir / MANIFEST_CONFIGS_DIRNAME,
        allowed_entries={COMPOSITION_CONFIG_DIRNAME},
        output_label="config manifest",
        next_step="Choose a fresh --out-dir or archive/remove stale config output first.",
    )
    _guard_materialize_directory_entries(
        manifest_dir / MANIFEST_INDEXES_DIRNAME,
        allowed_entries={REFERENCE_INDEX_FILENAME, SEQUENCE_INDEX_FILENAME},
        output_label="index manifest",
        next_step="Choose a fresh --out-dir or archive/remove stale index output first.",
    )

    references_dir = manifest_dir / MANIFEST_CATALOG_DIRNAME / REFERENCE_DIRNAME
    if references_dir.exists():
        stale_reference_entries = sorted(
            item.name
            for item in references_dir.iterdir()
            if item.name not in IGNORED_OUTPUT_FILENAMES
            and (item.is_dir() or item.name not in expected_reference_filenames)
        )
        if stale_reference_entries:
            raise RetronMsdCompilerError(
                f"Stale MSD design reference output at {references_dir}: {', '.join(stale_reference_entries)}. "
                "Choose a fresh --out-dir or explicitly archive/remove stale generated references before materializing."
            )

    configs_dir = manifest_dir / MANIFEST_CONFIGS_DIRNAME / COMPOSITION_CONFIG_DIRNAME
    if configs_dir.exists():
        expected_config_names = {
            f"{variant_dirname}.linear_ssdna_composition.yaml" for variant_dirname in expected_variant_dirnames
        } | IGNORED_OUTPUT_FILENAMES
        stale_configs = sorted(item.name for item in configs_dir.iterdir() if item.name not in expected_config_names)
        if stale_configs:
            raise RetronMsdCompilerError(
                f"Stale MSD composition config output at {configs_dir}: {', '.join(stale_configs)}. "
                "Choose a fresh --out-dir or explicitly archive/remove stale generated configs before materializing."
            )

    variants_dir = root / VARIANT_DIRNAME
    if not variants_dir.exists():
        return
    stale_variants = sorted(
        item.name
        for item in variants_dir.iterdir()
        if item.name not in expected_variant_dirnames | IGNORED_OUTPUT_FILENAMES
    )
    if stale_variants:
        raise RetronMsdCompilerError(
            f"Stale MSD sequence output at {variants_dir}: {', '.join(stale_variants)}. "
            "Choose a fresh --out-dir or explicitly archive/remove stale generated variants before materializing."
        )
    allowed_variant_entries = {
        VARIANT_MANIFEST_DIRNAME,
        VARIANT_PLOTS_DIRNAME,
        VARIANT_RUNTIME_DIRNAME,
        VARIANT_SEQUENCES_DIRNAME,
        *IGNORED_OUTPUT_FILENAMES,
    }
    allowed_plot_entries = {
        "composition_overview.png",
        "composition_overview.svg",
        "secondary_structure.native.png",
        *IGNORED_OUTPUT_FILENAMES,
    }
    for variant_dir in variants_dir.iterdir():
        if variant_dir.name in IGNORED_OUTPUT_FILENAMES:
            continue
        if not variant_dir.is_dir():
            raise RetronMsdCompilerError(
                f"Expected MSD variant output directory at {variant_dir}, but found a file. "
                "Choose a fresh --out-dir or archive/remove stale generated variant output first."
            )
        stale_variant_entries = sorted(
            item.name for item in variant_dir.iterdir() if item.name not in allowed_variant_entries
        )
        if stale_variant_entries:
            raise RetronMsdCompilerError(
                f"Unexpected MSD variant output entries at {variant_dir}: {', '.join(stale_variant_entries)}. "
                "Choose a fresh --out-dir or archive/remove stale generated variant output first."
            )
        _guard_materialize_directory_entries(
            variant_dir / VARIANT_SEQUENCES_DIRNAME,
            allowed_entries={
                "features.csv",
                "forward.fa",
                "forward.gb",
                "reverse_complement.fa",
                "reverse_complement.gb",
            },
            output_label="sequence artifact",
            next_step="Choose a fresh --out-dir or archive/remove stale sequence artifacts before materializing.",
        )
        plots_dir = variant_dir / VARIANT_PLOTS_DIRNAME
        _guard_materialize_directory_entries(
            plots_dir,
            allowed_entries=allowed_plot_entries,
            output_label="plot",
            next_step="Choose a fresh --out-dir or archive/remove stale plot artifacts before materializing.",
        )
        _guard_materialize_directory_entries(
            variant_dir / VARIANT_MANIFEST_DIRNAME,
            allowed_entries={
                VARIANT_MANIFEST_COMPOSITION_DIRNAME,
                VARIANT_MANIFEST_CONSTRUCT_DIRNAME,
                VARIANT_MANIFEST_FOLDING_DIRNAME,
                VARIANT_MANIFEST_PROVENANCE_DIRNAME,
                VARIANT_MANIFEST_REVIEWS_DIRNAME,
                VARIANT_MANIFEST_VISUAL_DIRNAME,
            },
            output_label="variant manifest",
            next_step="Choose a fresh --out-dir or archive/remove stale variant manifest output first.",
        )
        variant_manifest_dir = variant_dir / VARIANT_MANIFEST_DIRNAME
        _guard_materialize_directory_entries(
            variant_manifest_dir / VARIANT_MANIFEST_COMPOSITION_DIRNAME,
            allowed_entries={
                "annotation_spans.json",
                "assembled_sequence.json",
                "segment_spans.json",
                "validation_report.json",
            },
            output_label="composition manifest",
            next_step="Choose a fresh --out-dir or archive/remove stale composition manifest output first.",
        )
        _guard_materialize_directory_entries(
            variant_manifest_dir / VARIANT_MANIFEST_CONSTRUCT_DIRNAME,
            allowed_entries={"manifest.json"},
            output_label="construct manifest",
            next_step="Choose a fresh --out-dir or archive/remove stale construct manifest output first.",
        )
        _guard_materialize_directory_entries(
            variant_manifest_dir / VARIANT_MANIFEST_FOLDING_DIRNAME,
            allowed_entries={
                "folding_preflight.json",
                "secondary_structure_prediction_request_v1.yaml",
                "secondary_structure_prediction_v1.json",
            },
            output_label="folding manifest",
            next_step="Choose a fresh --out-dir or archive/remove stale folding manifest output first.",
        )
        _guard_materialize_directory_entries(
            variant_manifest_dir / VARIANT_MANIFEST_PROVENANCE_DIRNAME,
            allowed_entries={"provenance.json"},
            output_label="provenance manifest",
            next_step="Choose a fresh --out-dir or archive/remove stale provenance manifest output first.",
        )
        _guard_materialize_directory_entries(
            variant_manifest_dir / VARIANT_MANIFEST_REVIEWS_DIRNAME,
            allowed_entries={"composition_review_svg_v1.json"},
            output_label="review manifest",
            next_step="Choose a fresh --out-dir or archive/remove stale review manifest output first.",
        )
        _guard_materialize_directory_entries(
            variant_manifest_dir / VARIANT_MANIFEST_VISUAL_DIRNAME,
            allowed_entries={"sequence_evidence_map_v1.json", VARIANT_MANIFEST_SECONDARY_STRUCTURE_DIRNAME},
            output_label="visual manifest",
            next_step="Choose a fresh --out-dir or archive/remove stale visual manifest output first.",
        )
        _guard_materialize_directory_entries(
            variant_manifest_dir / VARIANT_MANIFEST_VISUAL_DIRNAME / VARIANT_MANIFEST_SECONDARY_STRUCTURE_DIRNAME,
            allowed_entries={
                "annotated.svg",
                "annotation_manifest.json",
                "native.svg",
                "viennarna_secondary_structure_svg_v1.json",
            },
            output_label="secondary-structure manifest",
            next_step="Choose a fresh --out-dir or archive/remove stale secondary-structure manifest output first.",
        )


def _guard_materialize_directory_entries(
    directory: Path,
    *,
    allowed_entries: set[str],
    output_label: str,
    next_step: str,
) -> None:
    if not directory.exists():
        return
    if not directory.is_dir():
        raise RetronMsdCompilerError(
            f"Expected MSD {output_label} output directory at {directory}, but found a file. {next_step}"
        )
    stale_entries = sorted(
        item.name for item in directory.iterdir() if item.name not in allowed_entries | IGNORED_OUTPUT_FILENAMES
    )
    if stale_entries:
        raise RetronMsdCompilerError(
            f"Stale MSD {output_label} output at {directory}: {', '.join(stale_entries)}. {next_step}"
        )


__all__ = ["guard_catalog_output_layout", "guard_materialize_output_layout"]
