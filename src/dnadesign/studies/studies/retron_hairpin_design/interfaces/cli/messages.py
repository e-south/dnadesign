"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/interfaces/cli/messages.py

Retron MSD CLI operator next-step messages.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def next_step_for_error(exc: Exception) -> str:
    message = str(exc)
    if "provided profile" in message:
        return "Correct the declared -MWX profile or omit it so the compiler derives S3/S2/S1/S0 from the bases."
    if "S0" in message:
        return (
            "Route the left/right base feasibility question to scar-nick before compiling; the compiler requires S0=M "
            "unless this is a deliberate control rerun with --allow-non-ligatable-s0."
        )
    if "Unknown cap" in message:
        return (
            "Route missing cap or shortening constraints to Snapback, add the validated cap to "
            "compiler/catalog/msd_design_registry.yaml, or provide its explicit 5'->3' sequence in a compiler spec."
        )
    if "Unknown payload" in message:
        return (
            "Add the validated payload to compiler/catalog/msd_design_registry.yaml before compiling "
            "a frozen design reference."
        )
    if "registry" in message:
        return (
            "Open docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry.yaml "
            "and fix the registry before rerunning lint."
        )
    if "Duplicate construct label" in message:
        return "Deduplicate the input labels, then rerun compile with the same explicit --out-dir."
    if "Duplicate MSD design reference filename" in message:
        return "Deduplicate equivalent MSD design IDs before writing a catalog bundle."
    if "Legacy MSD compiler output layout" in message:
        return "Choose a fresh --out-dir or explicitly archive/remove the old generated assets directory."
    if "Unexpected MSD materialize output entries" in message:
        return "Choose a fresh --out-dir or explicitly archive/remove stale flat materialize output first."
    if "Stale MSD plot output" in message:
        return "Choose a fresh --out-dir or explicitly archive/remove stale plot artifacts before materializing."
    if "Unexpected MSD compiler output entries" in message or "Stale MSD design reference output" in message:
        return "Choose a fresh --out-dir or explicitly archive/remove unrelated generated output before compiling."
    if "MSD sequence artifact generation requires concrete sequence subcomponents" in message:
        return (
            "Provide literal subcomponents with --payload-sequence ID=ACGT and --cap-sequence ID=ACGT, "
            "or use a compiler spec with explicit 5'->3' cap sequences or public primitive sources before "
            "generating GenBank/plot artifacts."
        )
    if (
        "ViennaRNA" in message
        or "RNAfold" in message
        or "Folding backend Python module" in message
        or "folding failed" in message
    ):
        return (
            "Install the ViennaRNA Python bindings (importable as RNA) or run from an environment with that "
            "backend available, then rerun materialize into a fresh --out-dir. Retron MSD GenBank/structure/review "
            "deliverables require folding status ok."
        )
    if "compiler spec" in message or "selector" in message or "primitive" in message:
        return (
            "Fix the retron_msd_compiler_spec_v1 file so every design has complete explicit parts, "
            "and use selector mode=rank for each public primitive source."
        )
    if "Stale MSD sequence artifact output" in message:
        return "Choose a fresh --out-dir or explicitly archive/remove stale sequence artifacts before materializing."
    if "Stale MSD sequence output" in message or "Stale MSD composition config output" in message:
        return "Choose a fresh --out-dir or explicitly archive/remove stale generated sequence outputs first."
    return "Run lint on one complete MSD label first; route missing biological constraints before generating a catalog."


def lint_next_step() -> str:
    return "Input is complete; run compile with an explicit --out-dir when a design-reference catalog is needed."


def compile_next_step() -> str:
    return (
        "Catalog bundle emitted with flat references; run materialize with explicit payload/cap sequences "
        "when one GenBank/structure-review sequence bundle per MSD design is needed."
    )


def materialize_warnings(variants: list[dict[str, Any]]) -> list[str]:
    folding_warning_count = sum(1 for variant in variants if variant.get("folding_status") != "ok")
    if folding_warning_count == 0:
        return []
    statuses = sorted(
        {str(variant.get("folding_status")) for variant in variants if variant.get("folding_status") != "ok"}
    )
    return [
        "Folding was attempted for every variant, but "
        f"{folding_warning_count} variant(s) reported {', '.join(statuses)}. "
        "Install ViennaRNA or run where the configured ViennaRNA backend is available; "
        "no substitute prediction was used."
    ]


def materialize_next_step(out_dir: Path, *, warnings: list[str]) -> str:
    if warnings:
        return (
            "Single-unit MSD sequence bundle emitted with GenBank, FASTA/CSV, and plot/status artifacts; "
            f"open {out_dir.as_posix()} or inspect manifest/indexes/sequence_index.tsv for folding status."
        )
    return (
        "Single-unit MSD sequence bundle emitted with GenBank, FASTA/CSV, native structure PNG, review SVG, "
        "and review PNG; "
        f"open {out_dir.as_posix()} or use manifest/indexes/sequence_index.tsv for programmatic handoff."
    )


__all__ = [
    "compile_next_step",
    "lint_next_step",
    "materialize_next_step",
    "materialize_warnings",
    "next_step_for_error",
]
