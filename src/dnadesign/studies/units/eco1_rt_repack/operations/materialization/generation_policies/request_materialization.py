"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/generation_policies/request_materialization.py

Materialize ProteinMPNN request sidecars from Eco1 RT v3 generation policies.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    load_yaml,
    require_hash,
    require_mapping,
    require_text,
    resolve_source_ref,
    write_yaml,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.structure_io import (
    load_first_model,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    DEFAULT_GENERATION_POLICIES_ROOT,
    DEFAULT_SOURCE_OUTPUT_ROOT,
    GENERATION_POLICY_VERSION,
    POLICY_INPUT_DIR_NAME,
    PROTEINMPNN_ALPHABET,
    PROTEINMPNN_BATCH_SIZE,
    PROTEINMPNN_CHAIN_ID,
    PROTEINMPNN_NAME,
    PROTEINMPNN_SEED_SET,
    PROTEINMPNN_TEMPERATURES,
    REQUEST_CREATED_BY,
    REQUEST_DIR_NAME,
    STRUCTURE_SOURCES,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.manifest_io import (
    load_valid_generation_policy_manifest,
    resolve_recorded_path,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.models import (
    MaterializedGenerationPolicyRequests,
)
from dnadesign.thread.adapters.proteinmpnn import (
    assigned_chains_payload,
    build_request_manifest,
    export_chain_backbone,
    fixed_positions_payload,
    mapped_chain_rows,
    request_hash,
    to_proteinmpnn_positions,
    write_jsonl,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri


def materialize_generation_policy_requests(
    *,
    repo_root: Path | None = None,
    generation_policy_root: Path | None = None,
    source_output_root: Path | None = None,
    seed_set: Sequence[int] = PROTEINMPNN_SEED_SET,
    temperatures: Sequence[float] = PROTEINMPNN_TEMPERATURES,
    batch_size: int = PROTEINMPNN_BATCH_SIZE,
) -> MaterializedGenerationPolicyRequests:
    """Materialize one ProteinMPNN request subtree for each complete v3 policy."""

    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    policy_root = _resolve_path(root, generation_policy_root or DEFAULT_GENERATION_POLICIES_ROOT)
    source_root = _resolve_path(root, source_output_root or DEFAULT_SOURCE_OUTPUT_ROOT)
    policy_manifest_path = policy_root / "generation_policy_manifest.yaml"
    manifest = load_valid_generation_policy_manifest(policy_manifest_path)
    positions_path = resolve_recorded_path(policy_root, manifest["position_manifest_path"])
    alphabets_path = resolve_recorded_path(policy_root, manifest["alphabet_manifest_path"])
    positions = pq.read_table(positions_path).to_pylist()
    alphabets = pq.read_table(alphabets_path).to_pylist()

    residue_map_path = source_root / "residue_map.parquet"
    backbone_bundle_path = source_root / "backbone_bundle.yaml"
    for required_path in (residue_map_path, backbone_bundle_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    structure_sources_path = root / STRUCTURE_SOURCES
    structure_sources = load_yaml(structure_sources_path)
    selected_source = require_mapping(structure_sources.get("selected_source"), "selected_source")
    model_path = resolve_source_ref(root, require_text(selected_source, "ec86kit_model_ref"))
    require_hash(model_path, require_text(selected_source, "ec86kit_model_sha256"))
    model = load_first_model(model_path)
    mapped_rows = mapped_chain_rows(residue_map_path, chain_id=PROTEINMPNN_CHAIN_ID, expected_mapped_count=309)

    request_paths: list[Path] = []
    for policy in manifest["generation_policies"]:
        policy_id = str(policy["policy_id"])
        request_paths.append(
            _materialize_one_policy_request(
                root=policy_root,
                source_root=source_root,
                policy_manifest_path=policy_manifest_path,
                positions_path=positions_path,
                alphabets_path=alphabets_path,
                policy_manifest_hash=str(manifest["policy_manifest_hash"]),
                policy=policy,
                position_rows=[row for row in positions if row["policy_id"] == policy_id],
                alphabet_rows=[row for row in alphabets if row["policy_id"] == policy_id],
                model=model,
                mapped_rows=mapped_rows,
                residue_map_path=residue_map_path,
                backbone_bundle_path=backbone_bundle_path,
                structure_sources_path=structure_sources_path,
                model_path=model_path,
                seed_set=tuple(int(seed) for seed in seed_set),
                temperatures=tuple(float(temperature) for temperature in temperatures),
                batch_size=batch_size,
            )
        )

    return MaterializedGenerationPolicyRequests(
        policy_manifest_path=policy_manifest_path,
        positions_path=positions_path,
        alphabets_path=alphabets_path,
        request_manifest_paths=tuple(request_paths),
    )


def _materialize_one_policy_request(
    *,
    root: Path,
    source_root: Path,
    policy_manifest_path: Path,
    positions_path: Path,
    alphabets_path: Path,
    policy_manifest_hash: str,
    policy: Mapping[str, Any],
    position_rows: list[dict[str, Any]],
    alphabet_rows: list[dict[str, Any]],
    model: Any,
    mapped_rows: list[dict[str, Any]],
    residue_map_path: Path,
    backbone_bundle_path: Path,
    structure_sources_path: Path,
    model_path: Path,
    seed_set: tuple[int, ...],
    temperatures: tuple[float, ...],
    batch_size: int,
) -> Path:
    policy_id = str(policy["policy_id"])
    requested_variants = _require_positive_int(policy.get("requested_variants"), f"{policy_id}.requested_variants")
    num_seq_per_target = _num_seq_per_target(
        requested_variants=requested_variants,
        seed_set=seed_set,
        temperatures=temperatures,
    )
    policy_root = root / policy_id
    request_root = policy_root / REQUEST_DIR_NAME
    request_root.mkdir(parents=True, exist_ok=True)
    _copy_policy_inputs(
        policy_root=policy_root,
        policy_manifest_path=policy_manifest_path,
        positions_path=positions_path,
        alphabets_path=alphabets_path,
    )

    chain_pdb_path = request_root / f"{PROTEINMPNN_NAME}.pdb"
    export = export_chain_backbone(
        model=model,
        mapped_residue_rows=mapped_rows,
        chain_id=PROTEINMPNN_CHAIN_ID,
        output_path=chain_pdb_path,
        target_name=PROTEINMPNN_NAME,
    )
    mapped_positions = sorted(export.canonical_to_proteinmpnn_position)
    open_positions = _canonical_open_positions(position_rows)
    fixed_canonical_positions = sorted(set(mapped_positions) - set(open_positions))
    excluded_positions = _excluded_canonical_positions(position_rows, mapped_positions)
    fixed_positions = to_proteinmpnn_positions(
        fixed_canonical_positions,
        export.canonical_to_proteinmpnn_position,
        "canonical_fixed_positions",
    )
    mutable_positions = to_proteinmpnn_positions(
        open_positions,
        export.canonical_to_proteinmpnn_position,
        "canonical_open_positions",
    )

    parsed_pdbs_path = request_root / "parsed_pdbs.jsonl"
    assigned_chains_path = request_root / "assigned_chains.jsonl"
    fixed_positions_path = request_root / "fixed_positions.jsonl"
    write_jsonl(parsed_pdbs_path, export.parsed_payload)
    write_jsonl(
        assigned_chains_path,
        assigned_chains_payload(target_name=PROTEINMPNN_NAME, chain_id=PROTEINMPNN_CHAIN_ID),
    )
    write_jsonl(
        fixed_positions_path,
        fixed_positions_payload(
            target_name=PROTEINMPNN_NAME,
            chain_id=PROTEINMPNN_CHAIN_ID,
            fixed_positions=fixed_positions,
        ),
    )

    alphabet_modes = sorted({str(row["alphabet_enforcement_mode"]) for row in alphabet_rows})
    sidecar_paths = {
        "chain_a_backbone_pdb": chain_pdb_path,
        "parsed_pdbs_jsonl": parsed_pdbs_path,
        "assigned_chains_jsonl": assigned_chains_path,
        "fixed_positions_jsonl": fixed_positions_path,
    }
    omit_aa_payload = _omit_aa_payload(
        alphabet_rows=alphabet_rows,
        canonical_to_proteinmpnn_position=export.canonical_to_proteinmpnn_position,
    )
    if omit_aa_payload is not None:
        omit_aa_path = request_root / "omit_AA.jsonl"
        write_jsonl(omit_aa_path, omit_aa_payload)
        sidecar_paths["omit_AA_jsonl"] = omit_aa_path

    manifest_without_hash = build_request_manifest(
        artifact_id=f"eco1_rt_generation_policies_v3.{policy_id}.proteinmpnn_request",
        created_by=REQUEST_CREATED_BY,
        profile_id="eco1_rt_v1",
        mask_policy_id=None,
        target_name=PROTEINMPNN_NAME,
        chain_id=PROTEINMPNN_CHAIN_ID,
        sidecar_paths=sidecar_paths,
        upstream_artifact_hashes={
            "generation_policy_manifest": sha256_uri(policy_manifest_path),
            "generation_policy_positions": sha256_uri(positions_path),
            "generation_policy_alphabets": sha256_uri(alphabets_path),
            "residue_map": sha256_uri(residue_map_path),
            "backbone_bundle": sha256_uri(backbone_bundle_path),
            "structure_sources_yaml": sha256_uri(structure_sources_path),
            "ec86kit_model": sha256_uri(model_path),
        },
        source_thread_plan={
            "source_type": "generation_policy_manifest",
            "generation_policy_manifest_path": str(policy_manifest_path),
            "policy_id": policy_id,
            "policy_version": GENERATION_POLICY_VERSION,
            "policy_manifest_hash": policy_manifest_hash,
        },
        canonical_to_mpnn=export.canonical_to_proteinmpnn_position,
        fixed_positions=fixed_positions,
        mutable_positions=mutable_positions,
        excluded_positions=excluded_positions,
        seed_set=list(seed_set),
        temperatures=list(temperatures),
        batch_id=f"eco1_rt_v3_{policy_id}_n{requested_variants}",
        num_seq_per_target=num_seq_per_target,
        batch_size=batch_size,
        expected_sample_count=requested_variants,
    )
    enriched_manifest = {
        **manifest_without_hash,
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "policy_id": policy_id,
        "policy_version": GENERATION_POLICY_VERSION,
        "policy_manifest_hash": policy_manifest_hash,
        "requested_variants": requested_variants,
        "canonical_open_positions": open_positions,
        "canonical_fixed_positions": fixed_canonical_positions,
        "alphabet_enforcement_modes": alphabet_modes,
        "alphabet_enforcement_note": _alphabet_enforcement_note(alphabet_modes),
        "source_generation_policy_manifest": {
            "path": str(policy_manifest_path),
            "hash": sha256_uri(policy_manifest_path),
            "policy_manifest_hash": policy_manifest_hash,
        },
        "source_generation_policy_positions": {"path": str(positions_path), "hash": sha256_uri(positions_path)},
        "source_generation_policy_alphabets": {"path": str(alphabets_path), "hash": sha256_uri(alphabets_path)},
    }
    manifest = {"request_hash": request_hash(enriched_manifest), **enriched_manifest}
    manifest_path = request_root / "request_manifest.yaml"
    write_yaml(manifest_path, manifest)
    return manifest_path


def _copy_policy_inputs(
    *,
    policy_root: Path,
    policy_manifest_path: Path,
    positions_path: Path,
    alphabets_path: Path,
) -> None:
    input_root = policy_root / POLICY_INPUT_DIR_NAME
    input_root.mkdir(parents=True, exist_ok=True)
    for source in (policy_manifest_path, positions_path, alphabets_path):
        shutil.copyfile(source, input_root / source.name)


def _omit_aa_payload(
    *,
    alphabet_rows: list[dict[str, Any]],
    canonical_to_proteinmpnn_position: Mapping[int, int],
) -> dict[str, dict[str, list[list[Any]]]] | None:
    grouped_positions: dict[str, list[int]] = {}
    for row in alphabet_rows:
        if row.get("alphabet_enforcement_mode") != "upstream_omit_AA_jsonl":
            continue
        position = int(row["eco1_position"])
        disallowed = _ordered_disallowed_amino_acids(row.get("disallowed_amino_acids"))
        if not disallowed:
            continue
        proteinmpnn_position = canonical_to_proteinmpnn_position[position]
        grouped_positions.setdefault("".join(disallowed), []).append(proteinmpnn_position)
    if not grouped_positions:
        return None
    groups = [
        [sorted(positions), aa_text]
        for aa_text, positions in sorted(grouped_positions.items(), key=lambda item: min(item[1]))
    ]
    return {PROTEINMPNN_NAME: {PROTEINMPNN_CHAIN_ID: groups}}


def _ordered_disallowed_amino_acids(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        raise ValueError("disallowed_amino_acids must be a list for upstream omit_AA_jsonl rows")
    disallowed = {str(aa) for aa in value}
    return [aa for aa in PROTEINMPNN_ALPHABET if aa in disallowed]


def _canonical_open_positions(position_rows: list[dict[str, Any]]) -> list[int]:
    return sorted(int(row["eco1_position"]) for row in position_rows if row["is_open_position"])


def _excluded_canonical_positions(position_rows: list[dict[str, Any]], mapped_positions: Sequence[int]) -> list[int]:
    mapped = set(mapped_positions)
    return sorted(int(row["eco1_position"]) for row in position_rows if int(row["eco1_position"]) not in mapped)


def _num_seq_per_target(
    *,
    requested_variants: int,
    seed_set: Sequence[int],
    temperatures: Sequence[float],
) -> int:
    denominator = len(seed_set) * len(temperatures)
    if denominator <= 0:
        raise ValueError("seed_set and temperatures must be non-empty")
    if requested_variants % denominator != 0:
        raise ValueError(
            "requested_variants must divide evenly across seeds and temperatures: "
            f"{requested_variants} % {denominator} != 0"
        )
    return requested_variants // denominator


def _alphabet_enforcement_note(modes: Sequence[str]) -> str:
    if "upstream_omit_AA_jsonl" in modes:
        return (
            "This request uses public ProteinMPNN omit_AA_jsonl sidecars for residue-specific near-region "
            "alphabet constraints and omit_AAs for the no-new-cysteine policy."
        )
    return "This request uses the public ProteinMPNN omit_AAs sidecar for the no-new-cysteine policy."


def _resolve_path(repo_root: Path, path: Path) -> Path:
    resolved = path.expanduser()
    return resolved if resolved.is_absolute() else (repo_root / resolved).resolve()


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
