"""Provider accession policy checks for Eco1 conservation source FASTA bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.contracts import (
    ProviderAccessionPolicy,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.issues import (
    invalid_manifest_record_issue,
)


def validate_record_accessions(
    issues: list[ContractIssue],
    *,
    records: Sequence[Any],
    profile_id: str,
    manifest_path: Path,
    accession_policy: ProviderAccessionPolicy,
    require_exclusion_reason: bool,
) -> None:
    """Validate provider ids, accession shape, and explicit exclusion reasons."""

    declared_provider_ids = set(accession_policy.provider_ids)
    for record in records:
        if not isinstance(record, Mapping):
            issues.append(invalid_manifest_record_issue(profile_id, manifest_path))
            continue
        provider_id = str(record.get("provider_id", ""))
        accession = str(record.get("accession", ""))
        if provider_id not in declared_provider_ids:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.source_sequences.undeclared_provider",
                    message=f"{profile_id} uses undeclared provider_id {provider_id!r}",
                    path=str(manifest_path),
                )
            )
            continue
        if not accession_policy.valid_provider_accession(provider_id, accession):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.source_sequences.invalid_provider_accession",
                    message=f"{profile_id} accession {accession!r} is not a real-looking {provider_id} id",
                    path=str(manifest_path),
                )
            )
        if require_exclusion_reason and not str(record.get("exclusion_reason", "")).strip():
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.source_sequences.exclusion_reason_missing",
                    message=f"{profile_id} excluded records must include exclusion_reason",
                    path=str(manifest_path),
                )
            )
