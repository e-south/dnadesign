"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/twist_handoff/pipeline.py

Materialize the study-owned Eco1 RT full-CDS Twist handoff.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from dnadesign.permuter import default_codon_table_path
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    EXPECTED_SELECTED_POLICY_COUNTS,
)
from dnadesign.thread.foldcheck import sequence_hash

from .codon_design import encode_full_cds, highest_frequency_codons, sequence_qc
from .genbank_export import build_genbank_record
from .models import MaterializedTwistHandoff
from .sequence_contract import (
    FORBIDDEN_SITES,
    read_unique_fasta,
    selected_rows,
    unique_rows,
    validate_mutations,
    validate_panel_shape,
    validate_wild_type,
)

DEFAULT_V3_ROOT = Path(
    "src/dnadesign/studies/units/eco1_rt_repack/workspaces/"
    "eco1_rt_conservative_v1/outputs/thread/generation_policies_v3"
)
DEFAULT_WT_GENBANK = Path(
    "docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/genbank/retron-eco1-rt.gb"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_V3_ROOT / "twist_handoff"


def materialize_twist_handoff(
    *,
    repo_root: Path,
    output_root: Path | None = None,
    candidate_selection_panel_path: Path | None = None,
    candidate_pool_path: Path | None = None,
    foldcheck_fasta_path: Path | None = None,
    generation_policy_positions_path: Path | None = None,
    wild_type_genbank_path: Path | None = None,
) -> MaterializedTwistHandoff:
    """Validate v3 selection closure and emit all eight selected CDS artifacts."""

    root = repo_root.expanduser().resolve()
    codon_table_path = default_codon_table_path("ecoli").expanduser().resolve()
    paths = {
        "candidate_selection_panel": _resolve(
            root, candidate_selection_panel_path or DEFAULT_V3_ROOT / "selection/candidate_selection_panel.parquet"
        ),
        "candidate_pool": _resolve(root, candidate_pool_path or DEFAULT_V3_ROOT / "candidate_pool.parquet"),
        "foldcheck_fasta": _resolve(
            root, foldcheck_fasta_path or DEFAULT_V3_ROOT / "foldcheck_request/input_sequences.fasta"
        ),
        "generation_policy_positions": _resolve(
            root, generation_policy_positions_path or DEFAULT_V3_ROOT / "generation_policy_positions.parquet"
        ),
        "codon_table": codon_table_path,
        "wild_type_cds_genbank": _resolve(root, wild_type_genbank_path or DEFAULT_WT_GENBANK),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    panel_rows = pq.read_table(paths["candidate_selection_panel"]).to_pylist()
    pool_rows = pq.read_table(paths["candidate_pool"]).to_pylist()
    policy_rows = pq.read_table(paths["generation_policy_positions"]).to_pylist()
    fasta_by_id = read_unique_fasta(paths["foldcheck_fasta"])
    wt_record = SeqIO.read(paths["wild_type_cds_genbank"], "genbank")
    wt_dna, wt_protein = validate_wild_type(wt_record)
    validate_panel_shape(panel_rows)
    selected_panel_rows = selected_rows(panel_rows)
    pool_by_id = unique_rows(pool_rows, "candidate_id", "candidate pool")
    codons = highest_frequency_codons(paths["codon_table"])

    sequence_rows: list[dict[str, Any]] = []
    records: list[SeqRecord] = []
    for panel in selected_panel_rows:
        candidate_id = str(panel["candidate_id"])
        if candidate_id not in pool_by_id:
            raise ValueError(f"selected candidate {candidate_id!r} is absent from candidate_pool")
        if candidate_id not in fasta_by_id:
            raise ValueError(f"selected candidate {candidate_id!r} is absent from foldcheck FASTA")
        protein = fasta_by_id[candidate_id]
        if len(protein) != 320:
            raise ValueError(f"candidate {candidate_id} foldcheck translation must contain 320 amino acids")
        pool = pool_by_id[candidate_id]
        raw_candidate_sequence = pool.get("sequence")
        if not isinstance(raw_candidate_sequence, str) or not raw_candidate_sequence:
            raise ValueError(f"candidate {candidate_id} candidate_pool sequence must be a non-empty string")
        observed_hash = sequence_hash(raw_candidate_sequence)
        if panel.get("sequence_hash") != observed_hash or pool.get("sequence_hash") != observed_hash:
            raise ValueError(f"candidate {candidate_id} panel sequence_hash does not match candidate_pool")
        tokens = validate_mutations(candidate_id, pool.get("canonical_mutations"), wt_protein, protein)
        dna = encode_full_cds(wt_dna, wt_protein, protein, codons)
        if str(Seq(dna).translate()) != protein + "*":
            raise ValueError(f"candidate {candidate_id} full-CDS translation does not match foldcheck FASTA")
        forbidden = [site for site in FORBIDDEN_SITES if site in dna]
        if forbidden:
            raise ValueError(f"candidate {candidate_id} contains forbidden internal sites: {', '.join(forbidden)}")
        qc = sequence_qc(dna)
        selection_rank = int(panel["selection_rank"])
        design_group_id = str(panel["design_group_id"])
        sequence_id = f"selected_{selection_rank:02d}_{design_group_id}"
        metadata = {
            "sequence_id": sequence_id,
            "candidate_id": candidate_id,
            "panel_sequence_hash": observed_hash,
            "foldcheck_sequence_hash": sequence_hash(protein),
            "canonical_protein_sha256": _text_sha256(protein),
            "full_cds_sha256": _text_sha256(dna),
            "selection_rank": selection_rank,
            "design_group_id": design_group_id,
            "within_group_rank": int(panel["within_group_rank"]),
            "wang_alpha1_r13_review_status": str(panel["wang_alpha1_r13_review_status"]),
            "wang_alpha1_mutation_count": int(panel["wang_alpha1_mutation_count"]),
            "wang_alpha1_f10_substitution": str(panel["wang_alpha1_f10_substitution"]),
            "wang_alpha1_r13_substitution": str(panel["wang_alpha1_r13_substitution"]),
            "wang_r13a_interface_disruption_evidence_match": bool(
                panel["wang_r13a_interface_disruption_evidence_match"]
            ),
            "rt_msdna_oligomeric_state_review_status": str(panel["rt_msdna_oligomeric_state_review_status"]),
            "selection_slot": str(panel["selection_slot"]),
            "genbank_file": f"genbank/{sequence_id}.gb",
            "policy_id": str(panel["policy_id"]),
            "mutation_tokens": tokens,
            "length_bp": len(dna),
            "translation_length_aa": len(protein),
            "qc": qc,
        }
        sequence_rows.append(metadata)
        records.append(
            build_genbank_record(
                sequence_id=sequence_id,
                dna=dna,
                protein=protein,
                metadata=metadata,
                policy_rows=policy_rows,
            )
        )

    out = _resolve(root, output_root or DEFAULT_OUTPUT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    genbank_root = out / "genbank"
    genbank_root.mkdir(parents=True, exist_ok=True)
    expected_genbank_names = {f"{record.id}.gb" for record in records}
    for stale_path in genbank_root.glob("*.gb"):
        if stale_path.name not in expected_genbank_names:
            stale_path.unlink()
    fasta_path = out / "eco1_rt_twist_full_cds.fasta"
    twist_csv_path = out / "eco1_rt_twist_full_cds.csv"
    manifest_path = out / "twist_handoff_manifest.yaml"
    SeqIO.write(records, fasta_path, "fasta")
    with twist_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "sequence"])
        writer.writeheader()
        writer.writerows({"name": record.id, "sequence": str(record.seq)} for record in records)
    genbank_paths = tuple(genbank_root / f"{record.id}.gb" for record in records)
    for record, path in zip(records, genbank_paths, strict=True):
        SeqIO.write(record, path, "genbank")
    manifest = {
        "schema_id": "eco1_rt.twist_full_cds_handoff",
        "schema_version": 1,
        "sequence_status": "quote_and_upload_ready",
        "cloning_status": "blocked_pending_assembly_flanks_and_vendor_portal_complexity_check",
        "selected_panel_source_count": len(panel_rows),
        "selected_panel_composition": EXPECTED_SELECTED_POLICY_COUNTS,
        "assembly_state_scope": {
            "rt_msdna_oligomeric_state": "not_established",
            "interpretation": (
                "Single-chain fold models do not establish RT-msDNA assembly state. Wang et al. tested R13A as "
                "an interface-disrupting substitution; exact F10/R13 states and R13A matches are reported per "
                "sequence."
            ),
        },
        "codon_policy": {
            "policy_kind": "minimal_variant_aware_recoding",
            "optimization_scope": "substituted_residues_only",
            "global_codon_optimization": False,
            "unchanged_residues": "preserve_authoritative_wt_codon",
            "changed_residues": "highest_frequency_ecoli_codon",
            "codon_table": _portable_path(root, paths["codon_table"]),
            "codon_table_source_provenance": "not_recorded_in_repository",
            "vendor_codon_optimization": False,
        },
        "input_hashes": {name: _sha256_uri(path) for name, path in paths.items()},
        "outputs": {
            "twist_csv": twist_csv_path.name,
            "fasta": fasta_path.name,
            "genbank_directory": genbank_root.name,
        },
        "sequences": sequence_rows,
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return MaterializedTwistHandoff(manifest_path, twist_csv_path, fasta_path, genbank_paths)


def _resolve(repo_root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _sha256_uri(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _text_sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
