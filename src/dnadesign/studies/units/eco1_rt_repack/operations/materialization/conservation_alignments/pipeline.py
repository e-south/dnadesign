"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/conservation_alignments/pipeline.py

Materialize Eco1 RT conservation aligned FASTA bundles through dnadesign.aligner.msa.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from dnadesign.aligner.msa import MsaBackendSpec, MsaRequest, MsaRunResult, load_fasta_records, run_msa
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation_alignments.manifest import (
    write_alignment_index_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.contracts import (
    load_conservation_source_contract,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.io import (
    load_yaml_mapping,
    resolve_path,
    sha256_file,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.paths import (
    CONSERVATION_SOURCES,
    DEFAULT_OUTPUT_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.sufficiency import (
    validate_source_sequence_bundle_sufficiency,
)

DEFAULT_ALIGNMENT_BUNDLE_ROOT = DEFAULT_OUTPUT_ROOT / "conservation_alignments"
DEFAULT_SOURCE_CACHE_ROOT = DEFAULT_OUTPUT_ROOT / "conservation_source_cache"
DEFAULT_SOURCE_BUNDLE_ROOT = DEFAULT_OUTPUT_ROOT / "conservation_sources"
DEFAULT_CREATED_AT = "2026-06-20T00:00:00Z"
MsaRunner = Callable[[MsaRequest], MsaRunResult]


@dataclass(frozen=True)
class MaterializedConservationAlignmentBundles:
    """Paths and runtime timing emitted by one Eco1 conservation-alignment pass."""

    aligned_fasta_paths: dict[str, Path]
    manifest_paths: dict[str, Path]
    bundle_manifest_path: Path
    total_elapsed_seconds: float


def materialize_conservation_alignment_bundles(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    source_cache_root: Path | None = None,
    source_bundle_root: Path | None = None,
    alignment_root: Path | None = None,
    profile_ids: tuple[str, ...] | None = None,
    created_at: str = DEFAULT_CREATED_AT,
    msa_runner: MsaRunner = run_msa,
) -> MaterializedConservationAlignmentBundles:
    """Run declared MSA alignments for sufficiency-passing Eco1 source FASTA bundles."""

    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = resolve_path(root, output_root or DEFAULT_OUTPUT_ROOT)
    cache_root = (
        resolve_path(root, source_cache_root)
        if source_cache_root is not None
        else resolve_path(root, DEFAULT_SOURCE_CACHE_ROOT)
    )
    source_root = (
        resolve_path(root, source_bundle_root)
        if source_bundle_root is not None
        else resolve_path(root, DEFAULT_SOURCE_BUNDLE_ROOT)
    )
    align_root = (
        resolve_path(root, alignment_root) if alignment_root is not None else out_root / "conservation_alignments"
    )
    align_root.mkdir(parents=True, exist_ok=True)

    _require_source_sufficiency(
        repo_root=root,
        output_root=out_root,
        source_cache_root=cache_root,
        source_bundle_root=source_root,
    )

    source_contract = load_conservation_source_contract(root / CONSERVATION_SOURCES)
    command = _declared_alignment_command(source_contract.sources)
    backend_id, command_args = parse_declared_alignment_command(command)
    target_row_id = source_contract.target_row_id
    target_sequence_hash = source_contract.target_sequence_hash
    selected_profile_ids = _select_profile_ids(
        declared_profile_ids=tuple(source_contract.profile_ids),
        requested_profile_ids=profile_ids,
    )

    base_upstream_hashes = {
        "conservation_sources_yaml": "sha256:" + sha256_file(root / CONSERVATION_SOURCES),
        "source_sequence_bundle_manifest": "sha256:"
        + sha256_file(source_root / "source_sequence_bundle_manifest.yaml"),
    }

    source_fasta_paths: dict[str, Path] = {}
    aligned_fasta_paths: dict[str, Path] = {}
    manifest_paths: dict[str, Path] = {}
    profile_runs: list[dict[str, Any]] = []
    total_started = perf_counter()
    upstream_hashes = dict(base_upstream_hashes)

    for profile_id in selected_profile_ids:
        source_fasta = source_root / f"{profile_id}.source.fasta"
        source_manifest = source_root / f"{profile_id}.source_manifest.yaml"
        _validate_source_target_hash(
            source_fasta,
            target_row_id=target_row_id,
            target_sequence_hash=target_sequence_hash,
        )
        output_fasta = align_root / f"{profile_id}.aligned.fasta"
        output_manifest = align_root / f"{profile_id}.aligned.manifest.yaml"
        request = MsaRequest(
            input_fasta=source_fasta,
            output_fasta=output_fasta,
            manifest_path=output_manifest,
            target_row_id=target_row_id,
            backend=MsaBackendSpec(backend_id=backend_id),
            command_args=command_args,
            run_label=profile_id,
        )

        profile_started = perf_counter()
        result = msa_runner(request)
        elapsed_seconds = result.elapsed_seconds or round(perf_counter() - profile_started, 6)
        _validate_alignment_result(result, expected_fasta=output_fasta, expected_manifest=output_manifest)

        source_fasta_paths[profile_id] = source_fasta
        aligned_fasta_paths[profile_id] = output_fasta
        manifest_paths[profile_id] = output_manifest
        upstream_hashes[f"{profile_id}_source_fasta"] = "sha256:" + sha256_file(source_fasta)
        upstream_hashes[f"{profile_id}_source_manifest"] = "sha256:" + sha256_file(source_manifest)
        upstream_hashes[f"{profile_id}_aligned_fasta"] = "sha256:" + sha256_file(output_fasta)
        upstream_hashes[f"{profile_id}_aligned_manifest"] = "sha256:" + sha256_file(output_manifest)
        profile_runs.append(
            {
                "profile_id": profile_id,
                "backend_id": result.backend_id,
                "backend_version": result.backend_version,
                "input_fasta": str(source_fasta),
                "output_fasta": str(output_fasta),
                "manifest_path": str(output_manifest),
                "input_fasta_sha256": result.input_fasta_sha256,
                "output_fasta_sha256": result.output_fasta_sha256,
                "pixi_lock_sha256": result.pixi_lock_sha256,
                "elapsed_seconds": elapsed_seconds,
                "return_code": result.return_code,
                "stderr_path": str(result.stderr_path) if result.stderr_path is not None else None,
                "run_label": result.run_label,
            }
        )

    total_elapsed_seconds = round(perf_counter() - total_started, 6)
    bundle_manifest_path = align_root / "conservation_alignment_bundle_manifest.yaml"
    write_alignment_index_manifest(
        bundle_manifest_path,
        profile_ids=selected_profile_ids,
        alignment_manifests=manifest_paths,
        aligned_fasta_paths=aligned_fasta_paths,
        source_fasta_paths=source_fasta_paths,
        target_row_id=target_row_id,
        target_sequence_hash=target_sequence_hash,
        command_args=command_args,
        profile_runs=profile_runs,
        upstream_hashes=upstream_hashes,
        total_elapsed_seconds=total_elapsed_seconds,
        created_at=created_at,
    )
    return MaterializedConservationAlignmentBundles(
        aligned_fasta_paths=aligned_fasta_paths,
        manifest_paths=manifest_paths,
        bundle_manifest_path=bundle_manifest_path,
        total_elapsed_seconds=total_elapsed_seconds,
    )


def parse_declared_alignment_command(command: str) -> tuple[str, tuple[str, ...]]:
    """Parse the declared MSA command and return backend id plus explicit arguments."""

    tokens = shlex.split(command)
    if not tokens:
        raise ValueError("alignment_policy.alignment_command must declare an alignment backend")
    if tokens[0] == "mafft":
        if len(tokens) < 4 or tokens[-3:] != ["<input_fasta>", ">", "<output_fasta>"]:
            raise ValueError("MAFFT alignment_command must use '<input_fasta> > <output_fasta>'")
        command_args = tuple(tokens[1:-3])
        if not command_args:
            raise ValueError("MAFFT alignment_command must declare explicit backend arguments")
        return "mafft", command_args
    if tokens[0] == "clustalo":
        if "-i" not in tokens or "-o" not in tokens:
            raise ValueError("Clustal Omega alignment_command must declare '-i <input_fasta> -o <output_fasta>'")
        input_index = tokens.index("-i")
        output_index = tokens.index("-o")
        if tokens[input_index + 1 : input_index + 2] != ["<input_fasta>"]:
            raise ValueError("Clustal Omega alignment_command must use '-i <input_fasta>'")
        if tokens[output_index + 1 : output_index + 2] != ["<output_fasta>"]:
            raise ValueError("Clustal Omega alignment_command must use '-o <output_fasta>'")
        excluded_indexes = {input_index, input_index + 1, output_index, output_index + 1}
        command_args = tuple(token for index, token in enumerate(tokens[1:], start=1) if index not in excluded_indexes)
        if not command_args:
            raise ValueError("Clustal Omega alignment_command must declare explicit backend arguments")
        return "clustalo", command_args
    raise ValueError("alignment_policy.alignment_command must start with 'mafft' or 'clustalo'")


def parse_declared_mafft_args(command: str) -> tuple[str, ...]:
    """Parse a declared MAFFT command and return explicit backend arguments."""

    backend_id, command_args = parse_declared_alignment_command(command)
    if backend_id != "mafft":
        raise ValueError("alignment_policy.alignment_command must start with 'mafft'")
    return command_args


def _select_profile_ids(
    *,
    declared_profile_ids: tuple[str, ...],
    requested_profile_ids: tuple[str, ...] | None,
) -> list[str]:
    if requested_profile_ids is None:
        return list(declared_profile_ids)
    if not requested_profile_ids:
        raise ValueError("At least one conservation alignment profile must be selected")
    declared = set(declared_profile_ids)
    unknown = [profile_id for profile_id in requested_profile_ids if profile_id not in declared]
    if unknown:
        raise ValueError(f"Unknown conservation alignment profile id(s): {', '.join(unknown)}")
    if len(set(requested_profile_ids)) != len(requested_profile_ids):
        raise ValueError("Conservation alignment profile selection contains duplicates")
    return list(requested_profile_ids)


def _require_source_sufficiency(
    *,
    repo_root: Path,
    output_root: Path,
    source_cache_root: Path,
    source_bundle_root: Path,
) -> None:
    report = validate_source_sequence_bundle_sufficiency(
        repo_root=repo_root,
        output_root=output_root,
        source_cache_root=source_cache_root,
        bundle_root=source_bundle_root,
    )
    if report.passed:
        return
    check_ids = ", ".join(sorted({issue.check_id for issue in report.issues}))
    raise ValueError(f"source sequence sufficiency failed before alignment: {check_ids}")


def _declared_alignment_command(sources: Mapping[str, Any]) -> str:
    policy = sources.get("alignment_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("conservation-sources.yaml must declare alignment_policy")
    command = policy.get("alignment_command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("alignment_policy.alignment_command must be a non-empty string")
    return command.strip()


def _validate_source_target_hash(path: Path, *, target_row_id: str, target_sequence_hash: str) -> None:
    records = load_fasta_records(path, alphabet="protein", allow_gaps=False)
    target_sequence = records.get(target_row_id)
    if target_sequence is None:
        raise ValueError(f"source FASTA {path} is missing target row {target_row_id!r}")
    observed_hash = "sha256:" + hashlib.sha256(target_sequence.encode("utf-8")).hexdigest()
    if observed_hash != target_sequence_hash:
        raise ValueError(f"source FASTA {path} target row hash does not match conservation-sources.yaml")


def _validate_alignment_result(result: MsaRunResult, *, expected_fasta: Path, expected_manifest: Path) -> None:
    if result.aligned_fasta != expected_fasta:
        raise ValueError("MSA runner returned an unexpected aligned FASTA path")
    if result.manifest_path != expected_manifest:
        raise ValueError("MSA runner returned an unexpected manifest path")
    if not expected_fasta.exists():
        raise FileNotFoundError(expected_fasta)
    if not expected_manifest.exists():
        raise FileNotFoundError(expected_manifest)
    observed_hash = "sha256:" + sha256_file(expected_fasta)
    if result.output_fasta_sha256 != observed_hash:
        raise ValueError("MSA runner output_fasta_sha256 does not match the aligned FASTA")
    load_yaml_mapping(expected_manifest)


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")
