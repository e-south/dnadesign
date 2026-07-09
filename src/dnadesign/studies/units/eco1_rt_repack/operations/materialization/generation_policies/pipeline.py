"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/generation_policies/pipeline.py

Materialize Eco1 RT generation-policy manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.aligner.msa import load_fasta_records
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    write_yaml,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.config import (
    build_default_generation_policy_config,
    validate_generation_policy_config,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    ACIDIC_AMINO_ACIDS,
    C_TERMINAL_THUMB_CONTEXT,
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    CONSERVATION_PROFILE_ID,
    CREATED_BY,
    DEFAULT_CREATED_AT,
    DEFAULT_GENERATION_POLICIES_ROOT,
    DEFAULT_SOURCE_OUTPUT_ROOT,
    DIRECT_CONTACT_DISTANCE_ANGSTROM,
    DISTAL_SCAFFOLD_POLICY_ID,
    GENERATION_POLICY_VERSION,
    MOTIF_CONTEXTS,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
    NEAR_REGION_MAX_INCLUSIVE_ANGSTROM,
    NEAR_REGION_MIN_EXCLUSIVE_ANGSTROM,
    PROLINE_GLYCINE_AMINO_ACIDS,
    PROTEINMPNN_ALPHABET,
    STANDARD_AMINO_ACIDS,
    STANDARD_AMINO_ACIDS_NO_CYS,
    TARGET_ALIGNMENT_ROW_ID,
    WANG_THUMB_TRACK_POSITIONS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.models import (
    GenerationPolicyConfig,
    GenerationPolicySpec,
    MaterializedGenerationPolicies,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri


def materialize_generation_policies(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    source_output_root: Path | None = None,
    config: Mapping[str, Any] | None = None,
    created_at: str = DEFAULT_CREATED_AT,
) -> MaterializedGenerationPolicies:
    """Materialize v3 generation-policy manifests without running ProteinMPNN."""

    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = _resolve_path(root, output_root or DEFAULT_GENERATION_POLICIES_ROOT)
    source_root = _resolve_path(root, source_output_root or DEFAULT_SOURCE_OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    validated_config = validate_generation_policy_config(config or build_default_generation_policy_config())
    inputs = _load_inputs(source_root)
    position_rows = _build_position_rows(config=validated_config, inputs=inputs)
    alphabet_rows = _build_alphabet_rows(
        config=validated_config,
        position_rows=position_rows,
        conservation_rows=inputs["conservation_rows"],
        source_root=source_root,
    )

    positions_path = out_root / "generation_policy_positions.parquet"
    alphabets_path = out_root / "generation_policy_alphabets.parquet"
    _write_parquet(positions_path, position_rows)
    _write_parquet(alphabets_path, alphabet_rows)

    manifest_without_hash = _build_manifest(
        config=validated_config,
        created_at=created_at,
        source_root=source_root,
        output_root=out_root,
        input_hashes={
            "residue_map": sha256_uri(source_root / "residue_map.parquet"),
            "contact_geometry_profile": sha256_uri(source_root / "contact_geometry_profile.parquet"),
            "conservation_profile": sha256_uri(source_root / "conservation_profile.parquet"),
        },
        position_rows=position_rows,
        alphabet_rows=alphabet_rows,
        positions_path=positions_path,
        alphabets_path=alphabets_path,
    )
    manifest = {"policy_manifest_hash": _payload_hash(manifest_without_hash), **manifest_without_hash}
    manifest_path = out_root / "generation_policy_manifest.yaml"
    write_yaml(manifest_path, manifest)
    return MaterializedGenerationPolicies(
        manifest_path=manifest_path,
        positions_path=positions_path,
        alphabets_path=alphabets_path,
    )


def _load_inputs(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    paths = {
        "residue_rows": source_root / "residue_map.parquet",
        "contact_rows": source_root / "contact_geometry_profile.parquet",
        "conservation_rows": source_root / "conservation_profile.parquet",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    return {key: pq.read_table(path).to_pylist() for key, path in paths.items()}


def _build_position_rows(
    *,
    config: GenerationPolicyConfig,
    inputs: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    base_rows = _base_position_rows(
        residue_rows=inputs["residue_rows"],
        contact_rows=inputs["contact_rows"],
        conservation_rows=inputs["conservation_rows"],
    )
    rows: list[dict[str, Any]] = []
    for policy in config.enabled_policies:
        open_positions = _open_positions_for_policy(policy=policy, base_rows=base_rows)
        for base_row in base_rows:
            row = {
                "policy_id": policy.policy_id,
                "policy_version": GENERATION_POLICY_VERSION,
                "open_set_id": policy.open_set_id,
                **base_row,
                "is_open_position": int(base_row["eco1_position"]) in open_positions,
            }
            rows.append(row)
    return rows


def _base_position_rows(
    *,
    residue_rows: list[dict[str, Any]],
    contact_rows: list[dict[str, Any]],
    conservation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    contact_by_position = {int(row["canonical_position"]): row for row in contact_rows}
    conserved_positions = {
        int(row["canonical_position"])
        for row in conservation_rows
        if row.get("profile_id") == CONSERVATION_PROFILE_ID and row.get("passes_conservation_mask") is True
    }
    rows: list[dict[str, Any]] = []
    for residue in sorted(residue_rows, key=lambda row: int(row["canonical_position"])):
        position = int(residue["canonical_position"])
        contact = contact_by_position[position]
        distance = _optional_float(contact.get("nearest_context_atom_distance_angstrom"))
        is_mapped = residue.get("mapping_status") == "mapped"
        is_direct_contact = is_mapped and distance is not None and distance <= DIRECT_CONTACT_DISTANCE_ANGSTROM
        is_near_region = (
            is_mapped
            and distance is not None
            and NEAR_REGION_MIN_EXCLUSIVE_ANGSTROM < distance <= NEAR_REGION_MAX_INCLUSIVE_ANGSTROM
        )
        motif_contexts = _motif_context_codes(position)
        is_wang = position in WANG_THUMB_TRACK_POSITIONS
        is_c_terminal = _in_range(position, C_TERMINAL_THUMB_CONTEXT) and is_mapped
        is_conserved = position in conserved_positions
        protected_reasons = _protected_reason_codes(
            motif_contexts=motif_contexts,
            is_direct_contact=is_direct_contact,
            is_wang=is_wang,
            is_c_terminal=is_c_terminal,
            is_conserved=is_conserved,
        )
        structure_residue_id = residue.get("structure_residue_id")
        structure_chain_id = str(residue.get("structure_chain_id") or "")
        rows.append(
            {
                "eco1_position": position,
                "wt_aa": residue["wt_aa"],
                "structure_position": structure_residue_id,
                "chain_position": (
                    "" if structure_residue_id is None else f"{structure_chain_id}:{int(structure_residue_id)}"
                ),
                "is_mapped": is_mapped,
                "is_designable_backbone_position": is_mapped,
                "protected_reason_codes": protected_reasons,
                "distance_to_retained_dna_rna": distance,
                "is_direct_contact_le_5a": is_direct_contact,
                "is_near_region_gt5_le10a": is_near_region,
                "is_wang_thumb_track": is_wang,
                "is_c_terminal_thumb_context": is_c_terminal,
                "is_conserved_core": is_conserved,
                "motif_context_codes": motif_contexts,
            }
        )
    return rows


def _open_positions_for_policy(*, policy: GenerationPolicySpec, base_rows: list[dict[str, Any]]) -> set[int]:
    if policy.policy_id == DISTAL_SCAFFOLD_POLICY_ID:
        return {
            int(row["eco1_position"])
            for row in base_rows
            if row["is_designable_backbone_position"]
            and not row["protected_reason_codes"]
            and not row["is_near_region_gt5_le10a"]
        }
    if policy.policy_id == NEAR_DNA_RNA_ACID_FREE_POLICY_ID:
        return {
            int(row["eco1_position"])
            for row in base_rows
            if row["is_designable_backbone_position"]
            and not row["protected_reason_codes"]
            and row["is_near_region_gt5_le10a"]
        }
    if policy.policy_id == COMBINED_NEAR_PLUS_DISTAL_POLICY_ID:
        distal = _open_positions_for_policy(
            policy=GenerationPolicySpec(
                policy_id=DISTAL_SCAFFOLD_POLICY_ID,
                open_set_id="distal_scaffold",
                alphabet_rule_id="broad_no_new_cysteine",
                requested_variants=policy.requested_variants,
                purpose="",
            ),
            base_rows=base_rows,
        )
        near = _open_positions_for_policy(
            policy=GenerationPolicySpec(
                policy_id=NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
                open_set_id="near_dna_rna_gt5_le10_excluding_protected",
                alphabet_rule_id="msa_observed_acid_free_basic_polar_neutral",
                requested_variants=policy.requested_variants,
                purpose="",
            ),
            base_rows=base_rows,
        )
        return distal | near
    raise ValueError(f"unknown generation policy id {policy.policy_id!r}")


def _build_alphabet_rows(
    *,
    config: GenerationPolicyConfig,
    position_rows: list[dict[str, Any]],
    conservation_rows: list[dict[str, Any]],
    source_root: Path,
) -> list[dict[str, Any]]:
    clade9_counts = _profile_residue_counts(
        profile_id=CONSERVATION_PROFILE_ID,
        conservation_rows=conservation_rows,
        alignment_path=source_root / "conservation_alignments" / f"{CONSERVATION_PROFILE_ID}.aligned.fasta",
    )
    rows: list[dict[str, Any]] = []
    for policy in config.enabled_policies:
        if policy.policy_id in (DISTAL_SCAFFOLD_POLICY_ID, COMBINED_NEAR_PLUS_DISTAL_POLICY_ID):
            rows.append(
                {
                    "policy_id": policy.policy_id,
                    "policy_version": GENERATION_POLICY_VERSION,
                    "alphabet_scope": "distal_scaffold",
                    "alphabet_rule_id": "broad_no_new_cysteine",
                    "alphabet_enforcement_mode": "upstream_omit_AAs_C",
                    "eco1_position": None,
                    "wt_aa": None,
                    "allowed_amino_acids": list(STANDARD_AMINO_ACIDS_NO_CYS),
                    "disallowed_amino_acids": ["C"],
                    "observed_amino_acids": [],
                    "interpretation_limit": "Distal alphabet does not imply a substrate-facing chemistry claim.",
                }
            )
        if policy.policy_id in (NEAR_DNA_RNA_ACID_FREE_POLICY_ID, COMBINED_NEAR_PLUS_DISTAL_POLICY_ID):
            rows.extend(
                _near_region_alphabet_rows(
                    policy_id=policy.policy_id,
                    position_rows=position_rows,
                    residue_counts_by_position=clade9_counts,
                )
            )
    return rows


def _near_region_alphabet_rows(
    *,
    policy_id: str,
    position_rows: list[dict[str, Any]],
    residue_counts_by_position: Mapping[int, Counter[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position_row in position_rows:
        if position_row["policy_id"] != policy_id:
            continue
        if not position_row["is_open_position"] or not position_row["is_near_region_gt5_le10a"]:
            continue
        position = int(position_row["eco1_position"])
        wt_aa = str(position_row["wt_aa"])
        observed = residue_counts_by_position.get(position, Counter())
        allowed = _allowed_near_region_amino_acids(wt_aa=wt_aa, observed=observed)
        rows.append(
            {
                "policy_id": policy_id,
                "policy_version": GENERATION_POLICY_VERSION,
                "alphabet_scope": "near_dna_rna_gt5_le10_excluding_protected",
                "alphabet_rule_id": "msa_observed_acid_free_basic_polar_neutral",
                "alphabet_enforcement_mode": "upstream_omit_AA_jsonl",
                "eco1_position": position,
                "wt_aa": wt_aa,
                "allowed_amino_acids": allowed,
                "disallowed_amino_acids": [aa for aa in PROTEINMPNN_ALPHABET if aa not in allowed],
                "observed_amino_acids": _ordered_amino_acids(aa for aa, count in observed.items() if count > 0),
                "interpretation_limit": (
                    "Near retained DNA/RNA alphabets preserve WT and allow only MSA-observed acid-free "
                    "alternatives; they do not assert that added basic charge improves function."
                ),
            }
        )
    return rows


def _allowed_near_region_amino_acids(*, wt_aa: str, observed: Counter[str]) -> list[str]:
    allowed = {wt_aa} if wt_aa in STANDARD_AMINO_ACIDS_NO_CYS else set()
    for aa, count in observed.items():
        if count <= 0:
            continue
        if aa not in STANDARD_AMINO_ACIDS_NO_CYS:
            continue
        if aa in ACIDIC_AMINO_ACIDS:
            continue
        if aa in PROLINE_GLYCINE_AMINO_ACIDS and aa != wt_aa:
            continue
        allowed.add(aa)
    if not allowed:
        raise ValueError(f"near-region alphabet has no allowed amino acids for WT {wt_aa!r}")
    return _ordered_amino_acids(allowed)


def _ordered_amino_acids(values: Iterable[str]) -> list[str]:
    allowed = set(values)
    return [aa for aa in PROTEINMPNN_ALPHABET if aa in allowed]


def _profile_residue_counts(
    *,
    profile_id: str,
    conservation_rows: list[dict[str, Any]],
    alignment_path: Path,
) -> dict[int, Counter[str]]:
    if not alignment_path.exists():
        raise FileNotFoundError(alignment_path)
    records = load_fasta_records(alignment_path, alphabet="protein", allow_gaps=True)
    source_sequences = [sequence for record_id, sequence in records.items() if record_id != TARGET_ALIGNMENT_ROW_ID]
    if not source_sequences:
        raise ValueError(f"{alignment_path} has no source rows after excluding {TARGET_ALIGNMENT_ROW_ID}")
    counts_by_position: dict[int, Counter[str]] = {}
    for row in conservation_rows:
        if str(row["profile_id"]) != profile_id:
            continue
        position = int(row["canonical_position"])
        column_index = int(row["msa_column"]) - 1
        counts_by_position[position] = Counter(
            sequence[column_index].upper()
            for sequence in source_sequences
            if sequence[column_index] != "-" and sequence[column_index].upper() in STANDARD_AMINO_ACIDS
        )
    if not counts_by_position:
        raise ValueError(f"No conservation rows found for {profile_id}")
    return counts_by_position


def _build_manifest(
    *,
    config: GenerationPolicyConfig,
    created_at: str,
    source_root: Path,
    output_root: Path,
    input_hashes: Mapping[str, str],
    position_rows: list[dict[str, Any]],
    alphabet_rows: list[dict[str, Any]],
    positions_path: Path,
    alphabets_path: Path,
) -> dict[str, Any]:
    position_rows_by_policy = _group_rows_by_policy(position_rows)
    alphabet_rows_by_policy = _group_rows_by_policy(alphabet_rows)
    return {
        "schema_id": "eco1_rt.generation_policy_manifest",
        "schema_version": 1,
        "status": "materialized",
        "created_by": CREATED_BY,
        "created_at": created_at,
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "source_output_root": str(source_root),
        "generation_policies_root": str(output_root),
        "position_manifest_path": str(positions_path),
        "alphabet_manifest_path": str(alphabets_path),
        "upstream_artifact_hashes": dict(input_hashes),
        "generation_total_target_raw": config.generation_total_target_raw,
        "generation_policies": [
            {
                "policy_id": policy.policy_id,
                "policy_version": GENERATION_POLICY_VERSION,
                "open_set_id": policy.open_set_id,
                "alphabet_rule_id": policy.alphabet_rule_id,
                "requested_variants": policy.requested_variants,
                "purpose": policy.purpose,
                "open_position_count": sum(
                    1 for row in position_rows_by_policy[policy.policy_id] if row["is_open_position"]
                ),
                "protected_position_count": sum(
                    1 for row in position_rows_by_policy[policy.policy_id] if row["protected_reason_codes"]
                ),
                "alphabet_enforcement_modes": sorted(
                    {row["alphabet_enforcement_mode"] for row in alphabet_rows_by_policy[policy.policy_id]}
                ),
            }
            for policy in config.enabled_policies
        ],
        "conceptual_boundary": (
            "ProteinMPNN outputs are complete sequences sampled under one policy; do not combine mutations "
            "from separate policy outputs after generation."
        ),
    }


def _group_rows_by_policy(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy_id"]), []).append(row)
    return by_policy


def _protected_reason_codes(
    *,
    motif_contexts: list[str],
    is_direct_contact: bool,
    is_wang: bool,
    is_c_terminal: bool,
    is_conserved: bool,
) -> list[str]:
    reasons = [f"motif_context_{context}" for context in motif_contexts]
    if is_direct_contact:
        reasons.append("direct_retained_dna_rna_contact_le5a")
    if is_wang:
        reasons.append("wang_thumb_contact_track")
    if is_c_terminal:
        reasons.append("c_terminal_thumb_context_255_311")
    if is_conserved:
        reasons.append("conserved_core_clade9_25pct_plurality")
    return reasons


def _motif_context_codes(position: int) -> list[str]:
    return [code for code, span in MOTIF_CONTEXTS.items() if _in_range(position, span)]


def _in_range(position: int, span: tuple[int, int]) -> bool:
    return span[0] <= position <= span[1]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _resolve_path(repo_root: Path, path: Path) -> Path:
    resolved = path.expanduser()
    return resolved if resolved.is_absolute() else (repo_root / resolved).resolve()


def _payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def generation_policy_payload_hash(payload: Mapping[str, Any]) -> str:
    """Return the stable hash used by generation-policy manifests."""

    return _payload_hash(payload)
