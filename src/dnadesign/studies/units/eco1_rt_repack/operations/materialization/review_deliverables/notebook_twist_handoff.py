"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_twist_handoff.py

Twist handoff rendering for the Eco1 review notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import yaml


def render_twist_handoff(row: dict[str, Any], *, mo: Any, manifest_path: Path) -> Any:
    """Render order-sequence readiness without implying cloning readiness."""

    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(loaded, dict)
        or loaded.get("schema_id") != "eco1_rt.twist_full_cds_handoff"
        or loaded.get("schema_version") != 2
    ):
        raise ValueError(f"Expected eco1_rt.twist_full_cds_handoff schema version 2 at {manifest_path}")
    assay_host = dict(loaded.get("assay_host") or {})
    codon_policy = dict(loaded.get("codon_policy") or {})
    codon_source = dict(codon_policy.get("codon_table_source") or {})
    status_rows = [
        {"field": "Sequence status", "value": str(loaded.get("sequence_status") or "")},
        {"field": "Cloning status", "value": str(loaded.get("cloning_status") or "")},
        {
            "field": "Assay host",
            "value": f"{assay_host.get('species', '')} {assay_host.get('strain', '')}".strip(),
        },
        {"field": "Codon design", "value": "substitution-only minimal recoding"},
        {
            "field": "Codon-table source",
            "value": f"{codon_source.get('name', '')}; {codon_source.get('reference_organism', '')}".strip("; "),
        },
        {"field": "Vendor codon optimization", "value": "disabled"},
    ]
    sequence_rows = []
    for sequence in loaded.get("sequences") or []:
        if not isinstance(sequence, dict):
            continue
        qc = dict(sequence.get("qc") or {})
        mutation_tokens = [str(value) for value in sequence.get("mutation_tokens") or []]
        sequence_rows.append(
            {
                "sequence_id": str(sequence.get("sequence_id") or ""),
                "candidate_id": str(sequence.get("candidate_id") or ""),
                "selection_rank": sequence.get("selection_rank"),
                "design_group": str(sequence.get("design_group_id") or ""),
                "within_group_rank": sequence.get("within_group_rank"),
                "selection_slot": str(sequence.get("selection_slot") or ""),
                "alpha1_mutations": sequence.get("wang_alpha1_mutation_count"),
                "policy_id": str(sequence.get("policy_id") or ""),
                "mutations": ", ".join(mutation_tokens),
                "mutation_count": len(mutation_tokens),
                "length_bp": sequence.get("length_bp"),
                "gc_fraction": qc.get("gc_fraction"),
                "gc_50bp_span_fraction": qc.get("gc_50bp_span_fraction"),
                "max_homopolymer_run": qc.get("max_homopolymer_run"),
                "forbidden_site_count": qc.get("forbidden_site_count"),
                "genbank_file": str(sequence.get("genbank_file") or ""),
            }
        )
    title = html.escape(str(row.get("title") or "Twist full-CDS handoff"))
    note = (
        "These are exact full-length CDS designs for vendor upload and complexity review. Native WT codons are "
        "retained at unchanged residues; substitutions use the highest-frequency codon in the cited Kazusa "
        "E. coli K-12 table for the MG1655 assay host. This is not whole-gene codon optimization or an expression "
        "claim. Assembly flanks and junctions are not yet part of the sequences."
    )
    return mo.vstack(
        [
            mo.Html(f"<h3 style='margin:0 0 0.35rem 0; font-size:1.08rem;'>{title}</h3>"),
            mo.md(note),
            mo.ui.table(status_rows, page_size=6),
            mo.ui.table(sequence_rows, page_size=10),
        ],
        gap=0.25,
    )


__all__ = ["render_twist_handoff"]
