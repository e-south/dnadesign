"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/records.py

Checked-in synthesis handoff lifecycle record helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .azenta import validate_azenta_workbook
from .contracts import (
    SelectedCandidate,
    optional_nonnegative_integer,
    require_nonnegative_integer,
    require_positive_integer,
)
from .genbank import validate_genbank_record_set

DEFAULT_SYNTHESIS_HANDOFF_RECORD = Path("docs/studies/stress_ethanol_cipro_growth/record/synthesis_handoffs.yaml")
SYNTHESIS_HANDOFF_RECORD_VERSION = 3
SYNTHESIS_HANDOFF_STUDY_ID = "stress_ethanol_cipro_growth"
SYNTHESIS_HANDOFF_RECORD_KIND = "synthesis_handoff_lifecycle"
LIFECYCLE_STATUSES = frozenset(
    {
        "authorized_for_materialization",
        "generated_pending_acceptance",
        "accepted_for_order",
        "ordered",
        "received",
        "assayed",
        "superseded",
    }
)
COMMITTED_LIFECYCLE_STATUSES = frozenset({"accepted_for_order", "ordered", "received", "assayed", "superseded"})


@dataclass(frozen=True)
class ExpectedHandoffArtifact:
    """Expected campaign-scoped generated artifacts for one handoff."""

    campaign_slug: str
    expected_rows: int
    manifest_path: str
    vendor_workbook_path: str
    genbank_dir_path: str
    genbank_feature_table_path: str
    run_id: str | None = None
    manifest_sha256: str | None = None
    vendor_workbook_sha256: str | None = None
    genbank_dir_sha256: str | None = None
    genbank_feature_table_sha256: str | None = None
    workbook_readback_status: str | None = None
    genbank_readback_status: str | None = None

    def __post_init__(self) -> None:
        if not str(self.campaign_slug).strip():
            raise ValueError("expected campaign_slug must be non-empty")
        object.__setattr__(
            self,
            "expected_rows",
            require_positive_integer(self.expected_rows, field="expected_rows"),
        )
        artifact_paths = {
            "manifest_path": self.manifest_path,
            "vendor_workbook_path": self.vendor_workbook_path,
            "genbank_dir_path": self.genbank_dir_path,
            "genbank_feature_table_path": self.genbank_feature_table_path,
        }
        for field, value in artifact_paths.items():
            text = str(value).strip()
            if not text:
                raise ValueError(f"{field} must be non-empty for campaign={self.campaign_slug}")
            path = Path(text)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"{field} must be a repository-relative path without parent traversal for "
                    f"campaign={self.campaign_slug}"
                )

    def to_json(self) -> dict[str, Any]:
        return {
            "campaign_slug": self.campaign_slug,
            "expected_rows": int(self.expected_rows),
            "manifest_path": self.manifest_path,
            "vendor_workbook_path": self.vendor_workbook_path,
            "genbank_dir_path": self.genbank_dir_path,
            "genbank_feature_table_path": self.genbank_feature_table_path,
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "vendor_workbook_sha256": self.vendor_workbook_sha256,
            "genbank_dir_sha256": self.genbank_dir_sha256,
            "genbank_feature_table_sha256": self.genbank_feature_table_sha256,
            "workbook_readback_status": self.workbook_readback_status,
            "genbank_readback_status": self.genbank_readback_status,
        }


@dataclass(frozen=True)
class ExpectedSelectionView:
    """Expected logical membership count for one OPAL selection view."""

    selection_view_id: str
    expected_rows: int

    def __post_init__(self) -> None:
        if not str(self.selection_view_id).strip():
            raise ValueError("expected selection_view_id must be non-empty")
        object.__setattr__(
            self,
            "expected_rows",
            require_positive_integer(self.expected_rows, field="expected_rows"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "selection_view_id": self.selection_view_id,
            "expected_rows": int(self.expected_rows),
        }


def _canonical_study_alias(value: str, *, label: str) -> str:
    alias = str(value).strip()
    match = re.fullmatch(r"SECG-([0-9]{3,})", alias)
    if match is None:
        raise ValueError(f"{label} requires canonical stable study alias syntax SECG-NNN: {alias!r}")
    ordinal = int(match.group(1))
    if ordinal < 1 or alias != f"SECG-{ordinal:03d}":
        raise ValueError(f"{label} requires canonical stable study alias syntax SECG-NNN: {alias!r}")
    return alias


@dataclass(frozen=True)
class MaterializationInputReceipt:
    """Digest-bound repository input used to authorize one materialization."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        raw_path = str(self.path).strip()
        path = Path(raw_path)
        if not raw_path or path.is_absolute() or ".." in path.parts:
            raise ValueError("materialization input path must be repository-relative without parent traversal")
        digest = str(self.sha256).strip()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("materialization input sha256 must contain 64 lowercase hexadecimal characters")
        object.__setattr__(self, "path", path.as_posix())
        object.__setattr__(self, "sha256", digest)

    def to_json(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ExpectedMaterializedCandidate:
    """Exact stable alias, candidate, and promoter-core identity."""

    study_alias: str
    candidate_id: str
    core_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "study_alias",
            _canonical_study_alias(self.study_alias, label="materialization candidate"),
        )
        candidate_id = str(self.candidate_id).strip()
        if not candidate_id:
            raise ValueError("materialization candidate_id must be non-empty")
        digest = str(self.core_sha256).strip()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("materialization candidate core_sha256 must contain 64 lowercase hexadecimal characters")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "core_sha256", digest)

    def to_json(self) -> dict[str, str]:
        return {
            "study_alias": self.study_alias,
            "candidate_id": self.candidate_id,
            "core_sha256": self.core_sha256,
        }


@dataclass(frozen=True)
class MeasuredRoundMaterializationContract:
    """Inputs and exact candidate identities authorized for one measured round."""

    campaign_config: MaterializationInputReceipt
    selection_batch: MaterializationInputReceipt
    candidate_records: MaterializationInputReceipt
    promoter_alias_registry: MaterializationInputReceipt
    cloning_strategy: MaterializationInputReceipt
    expected_candidates: tuple[ExpectedMaterializedCandidate, ...]

    def __post_init__(self) -> None:
        if not self.expected_candidates:
            raise ValueError("measured-round materialization contract requires expected_candidates")
        for field in (
            "study_alias",
            "candidate_id",
            "core_sha256",
        ):
            values = [getattr(row, field) for row in self.expected_candidates]
            if len(values) != len(set(values)):
                raise ValueError(f"measured-round materialization contract has duplicate {field}")

    def to_json(self) -> dict[str, Any]:
        return {
            "campaign_config": self.campaign_config.to_json(),
            "selection_batch": self.selection_batch.to_json(),
            "candidate_records": self.candidate_records.to_json(),
            "promoter_alias_registry": self.promoter_alias_registry.to_json(),
            "cloning_strategy": self.cloning_strategy.to_json(),
            "expected_candidates": [row.to_json() for row in self.expected_candidates],
        }


@dataclass(frozen=True)
class SynthesisHandoffRecord:
    """One lifecycle record from ``synthesis_handoffs.yaml``."""

    handoff_id: str
    lifecycle_status: str
    source_authority: str
    selection_epoch: str
    assay_batch_index: int | None
    model_as_of_round: int | None
    run_id: str | None
    strategy_id: str
    expected_campaigns: tuple[ExpectedHandoffArtifact, ...] = ()
    campaign_slug: str | None = None
    expected_selection_views: tuple[ExpectedSelectionView, ...] = ()
    materialization_contract: MeasuredRoundMaterializationContract | None = None
    expected_artifact: ExpectedHandoffArtifact | None = None

    def __post_init__(self) -> None:
        for field in (
            "handoff_id",
            "lifecycle_status",
            "source_authority",
            "selection_epoch",
            "strategy_id",
        ):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"{field} must be non-empty in synthesis handoff record")
        if self.lifecycle_status not in LIFECYCLE_STATUSES:
            raise ValueError(f"unsupported synthesis handoff lifecycle_status: {self.lifecycle_status!r}")
        if self.assay_batch_index is not None:
            object.__setattr__(
                self,
                "assay_batch_index",
                require_nonnegative_integer(self.assay_batch_index, field="assay_batch_index"),
            )
        if self.model_as_of_round is not None:
            object.__setattr__(
                self,
                "model_as_of_round",
                require_nonnegative_integer(self.model_as_of_round, field="model_as_of_round"),
            )
        if self.source_authority == "study_batch0_selector":
            if self.selection_epoch != "pre_assay_seed":
                raise ValueError(f"batch-0 handoff record {self.handoff_id} requires selection_epoch pre_assay_seed")
            if self.assay_batch_index != 0:
                raise ValueError(f"batch-0 handoff record {self.handoff_id} requires assay_batch_index 0")
            if self.model_as_of_round is not None:
                raise ValueError(f"batch-0 handoff record {self.handoff_id} requires model_as_of_round null")
            if not self.expected_campaigns:
                raise ValueError(f"batch-0 handoff record {self.handoff_id} requires expected_campaigns")
            if self.run_id is None:
                raise ValueError(f"handoff record {self.handoff_id} requires explicit run_id for batch-0 source")
            if (
                self.campaign_slug is not None
                or self.expected_selection_views
                or self.materialization_contract is not None
                or self.expected_artifact is not None
            ):
                raise ValueError(f"batch-0 handoff record {self.handoff_id} cannot declare measured-round fields")
            self._validate_artifact_receipts_if_required()
            return
        if self.source_authority == "opal_selection_batch":
            if self.selection_epoch != "opal_model_round":
                raise ValueError(
                    f"measured-round handoff record {self.handoff_id} requires selection_epoch opal_model_round"
                )
            if self.assay_batch_index is None:
                raise ValueError(
                    f"measured-round handoff record {self.handoff_id} requires non-negative assay_batch_index"
                )
            if self.model_as_of_round is None:
                raise ValueError(
                    f"measured-round handoff record {self.handoff_id} requires non-negative model_as_of_round"
                )
            if self.expected_campaigns:
                raise ValueError(f"measured-round handoff record {self.handoff_id} cannot declare expected_campaigns")
            if self.campaign_slug is None or not str(self.campaign_slug).strip():
                raise ValueError(f"measured-round handoff record {self.handoff_id} requires campaign_slug")
            if self.run_id is None:
                raise ValueError(f"measured-round handoff record {self.handoff_id} requires explicit run_id")
            if not self.expected_selection_views:
                raise ValueError(f"measured-round handoff record {self.handoff_id} requires expected_selection_views")
            if self.expected_artifact is None:
                raise ValueError(f"measured-round handoff record {self.handoff_id} requires expected_artifact")
            if self.materialization_contract is None:
                raise ValueError(f"measured-round handoff record {self.handoff_id} requires materialization_contract")
            candidate_count = len(self.materialization_contract.expected_candidates)
            if candidate_count != int(self.expected_artifact.expected_rows):
                raise ValueError(
                    f"measured-round handoff record {self.handoff_id} materialization candidate count "
                    f"does not match expected rows: candidates={candidate_count} "
                    f"rows={self.expected_artifact.expected_rows}"
                )
            if self.expected_artifact.campaign_slug != self.campaign_slug:
                raise ValueError(
                    f"measured-round handoff record {self.handoff_id} artifact campaign does not match campaign_slug"
                )
            view_ids = [row.selection_view_id for row in self.expected_selection_views]
            if len(view_ids) != len(set(view_ids)):
                raise ValueError(f"measured-round handoff record {self.handoff_id} has duplicate selection views")
            self._validate_artifact_receipts_if_required()
            return
        raise ValueError(f"unsupported synthesis handoff source_authority: {self.source_authority}")

    @property
    def expected_campaign_counts(self) -> dict[str, int]:
        return {row.campaign_slug: int(row.expected_rows) for row in self.expected_campaigns}

    @property
    def expected_run_ids_by_campaign(self) -> dict[str, str]:
        run_ids: dict[str, str] = {}
        for row in self.expected_campaigns:
            run_id = row.run_id or self.run_id
            if run_id is not None:
                run_ids[row.campaign_slug] = run_id
        return run_ids

    @property
    def expected_selection_view_counts(self) -> dict[str, int]:
        return {row.selection_view_id: int(row.expected_rows) for row in self.expected_selection_views}

    @property
    def expected_study_aliases(self) -> tuple[str, ...]:
        if self.materialization_contract is None:
            return ()
        return tuple(row.study_alias for row in self.materialization_contract.expected_candidates)

    @property
    def artifacts(self) -> tuple[ExpectedHandoffArtifact, ...]:
        if self.source_authority == "study_batch0_selector":
            return self.expected_campaigns
        assert self.expected_artifact is not None
        return (self.expected_artifact,)

    def _validate_artifact_receipts_if_required(self) -> None:
        if self.lifecycle_status == "authorized_for_materialization":
            return
        for artifact in self.artifacts:
            digests = {
                "manifest_sha256": artifact.manifest_sha256,
                "vendor_workbook_sha256": artifact.vendor_workbook_sha256,
                "genbank_dir_sha256": artifact.genbank_dir_sha256,
                "genbank_feature_table_sha256": artifact.genbank_feature_table_sha256,
            }
            if (
                any(value is None for value in digests.values())
                or artifact.workbook_readback_status != "pass"
                or (artifact.genbank_readback_status != "pass")
            ):
                raise ValueError(
                    f"handoff record {self.handoff_id} lifecycle_status {self.lifecycle_status} requires complete "
                    f"artifact digests and passing readbacks for campaign {artifact.campaign_slug}"
                )
            malformed = [name for name, value in digests.items() if re.fullmatch(r"[0-9a-f]{64}", str(value)) is None]
            if malformed:
                raise ValueError(
                    f"handoff record {self.handoff_id} artifact digests must contain 64 lowercase hexadecimal "
                    f"characters: {', '.join(malformed)}"
                )


def _optional_int(value: Any, *, field: str) -> int | None:
    return optional_nonnegative_integer(value, field=field)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _record_from_raw(raw: dict[str, Any]) -> SynthesisHandoffRecord:
    def artifact_from(value: Any, *, label: str) -> ExpectedHandoffArtifact:
        item = _require_mapping(value, label=label)
        return ExpectedHandoffArtifact(
            campaign_slug=str(item["campaign_slug"]),
            expected_rows=require_positive_integer(item["expected_rows"], field="expected_rows"),
            manifest_path=str(item["manifest_path"]),
            vendor_workbook_path=str(item["vendor_workbook_path"]),
            genbank_dir_path=str(item["genbank_dir_path"]),
            genbank_feature_table_path=str(item["genbank_feature_table_path"]),
            run_id=_optional_text(item.get("run_id")),
            manifest_sha256=_optional_text(item.get("manifest_sha256")),
            vendor_workbook_sha256=_optional_text(item.get("vendor_workbook_sha256")),
            genbank_dir_sha256=_optional_text(item.get("genbank_dir_sha256")),
            genbank_feature_table_sha256=_optional_text(item.get("genbank_feature_table_sha256")),
            workbook_readback_status=_optional_text(item.get("workbook_readback_status")),
            genbank_readback_status=_optional_text(item.get("genbank_readback_status")),
        )

    source_authority = str(raw["source_authority"])
    expected_campaigns: tuple[ExpectedHandoffArtifact, ...] = ()
    expected_selection_views: tuple[ExpectedSelectionView, ...] = ()
    materialization_contract: MeasuredRoundMaterializationContract | None = None
    expected_artifact: ExpectedHandoffArtifact | None = None
    if source_authority == "study_batch0_selector":
        campaigns_raw = raw.get("expected_campaigns")
        if not isinstance(campaigns_raw, list):
            raise ValueError(f"handoff record {raw.get('handoff_id', '<unknown>')} expected_campaigns must be a list")
        expected_campaigns = tuple(artifact_from(item, label="expected campaign") for item in campaigns_raw)
    elif source_authority == "opal_selection_batch":
        views_raw = raw.get("expected_selection_views")
        if not isinstance(views_raw, list):
            raise ValueError(
                f"handoff record {raw.get('handoff_id', '<unknown>')} expected_selection_views must be a list"
            )
        expected_selection_views = tuple(
            ExpectedSelectionView(
                selection_view_id=str(_require_mapping(item, label="expected selection view")["selection_view_id"]),
                expected_rows=require_positive_integer(item["expected_rows"], field="expected_rows"),
            )
            for item in views_raw
        )
        if "expected_study_aliases" in raw:
            raise ValueError(
                f"handoff record {raw.get('handoff_id', '<unknown>')} uses removed expected_study_aliases; "
                "declare materialization_contract.expected_candidates"
            )
        contract_raw = _require_mapping(raw.get("materialization_contract"), label="materialization contract")

        def receipt_from(field: str) -> MaterializationInputReceipt:
            item = _require_mapping(contract_raw.get(field), label=f"materialization contract {field}")
            return MaterializationInputReceipt(path=str(item["path"]), sha256=str(item["sha256"]))

        candidates_raw = contract_raw.get("expected_candidates")
        if not isinstance(candidates_raw, list):
            raise ValueError(
                f"handoff record {raw.get('handoff_id', '<unknown>')} "
                "materialization_contract.expected_candidates must be a list"
            )
        materialization_contract = MeasuredRoundMaterializationContract(
            campaign_config=receipt_from("campaign_config"),
            selection_batch=receipt_from("selection_batch"),
            candidate_records=receipt_from("candidate_records"),
            promoter_alias_registry=receipt_from("promoter_alias_registry"),
            cloning_strategy=receipt_from("cloning_strategy"),
            expected_candidates=tuple(
                ExpectedMaterializedCandidate(
                    study_alias=str(_require_mapping(item, label="expected candidate")["study_alias"]),
                    candidate_id=str(item["candidate_id"]),
                    core_sha256=str(item["core_sha256"]),
                )
                for item in candidates_raw
            ),
        )
        expected_artifact = artifact_from(raw.get("expected_artifact"), label="expected artifact")
    return SynthesisHandoffRecord(
        handoff_id=str(raw["handoff_id"]),
        lifecycle_status=str(raw["lifecycle_status"]),
        source_authority=source_authority,
        selection_epoch=str(raw["selection_epoch"]),
        assay_batch_index=_optional_int(raw.get("assay_batch_index"), field="assay_batch_index"),
        model_as_of_round=_optional_int(raw.get("model_as_of_round"), field="model_as_of_round"),
        run_id=_optional_text(raw.get("run_id")),
        strategy_id=str(raw["strategy_id"]),
        expected_campaigns=expected_campaigns,
        campaign_slug=_optional_text(raw.get("campaign_slug")),
        expected_selection_views=expected_selection_views,
        materialization_contract=materialization_contract,
        expected_artifact=expected_artifact,
    )


def load_synthesis_handoff_records(path: str | Path) -> dict[str, SynthesisHandoffRecord]:
    """Load all checked-in synthesis handoff lifecycle records."""

    record_path = Path(path)
    if not record_path.exists():
        raise ValueError(f"synthesis handoff record not found: {record_path}")
    with record_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    root = _require_mapping(raw, label="synthesis handoff record")
    observed_identity = (
        root.get("version"),
        root.get("study_id"),
        root.get("record_kind"),
    )
    expected_identity = (
        SYNTHESIS_HANDOFF_RECORD_VERSION,
        SYNTHESIS_HANDOFF_STUDY_ID,
        SYNTHESIS_HANDOFF_RECORD_KIND,
    )
    if observed_identity != expected_identity:
        raise ValueError(
            "synthesis handoff record root identity mismatch: "
            f"expected={expected_identity!r} observed={observed_identity!r}"
        )
    handoffs = root.get("handoffs")
    if not isinstance(handoffs, list):
        raise ValueError(f"synthesis handoff record missing handoffs list: {record_path}")
    records = [_record_from_raw(_require_mapping(item, label="handoff")) for item in handoffs]
    by_id: dict[str, SynthesisHandoffRecord] = {}
    for record in records:
        if record.handoff_id in by_id:
            raise ValueError(f"duplicate handoff_id in synthesis handoff record: {record.handoff_id}")
        by_id[record.handoff_id] = record
    committed_alias_owner: dict[str, str] = {}
    for record in records:
        if record.lifecycle_status not in COMMITTED_LIFECYCLE_STATUSES:
            continue
        if record.source_authority == "study_batch0_selector":
            raise ValueError(
                "legacy batch-0 handoff cannot enter committed lifecycle_status without an exact per-alias "
                f"physical disposition: {record.handoff_id}"
            )
        for alias in record.expected_study_aliases:
            prior_handoff_id = committed_alias_owner.get(alias)
            if prior_handoff_id is not None:
                raise ValueError(
                    f"committed synthesis handoffs reuse study alias {alias}: "
                    f"{prior_handoff_id} and {record.handoff_id}"
                )
            committed_alias_owner[alias] = record.handoff_id
    for record in records:
        if record.lifecycle_status != "authorized_for_materialization":
            continue
        for alias in record.expected_study_aliases:
            prior_handoff_id = committed_alias_owner.get(alias)
            if prior_handoff_id is not None:
                raise ValueError(
                    f"authorized synthesis handoff reuses committed study alias {alias}: "
                    f"{record.handoff_id} conflicts with {prior_handoff_id}"
                )
    authorized_alias_owner: dict[str, str] = {}
    for record in records:
        if record.lifecycle_status != "authorized_for_materialization":
            continue
        for alias in record.expected_study_aliases:
            prior_handoff_id = authorized_alias_owner.get(alias)
            if prior_handoff_id is not None:
                raise ValueError(
                    f"authorized synthesis handoffs reuse study alias {alias}: "
                    f"{prior_handoff_id} and {record.handoff_id}"
                )
            authorized_alias_owner[alias] = record.handoff_id
    return by_id


def get_synthesis_handoff_record(path: str | Path, handoff_id: str) -> SynthesisHandoffRecord:
    """Return one checked-in synthesis handoff lifecycle record."""

    identifier = str(handoff_id).strip()
    if not identifier:
        raise ValueError("handoff_id must be non-empty")
    records = load_synthesis_handoff_records(path)
    try:
        return records[identifier]
    except KeyError as exc:
        available = ", ".join(sorted(records)) or "<none>"
        raise ValueError(f"unknown synthesis handoff id {identifier!r}; available: {available}") from exc


def source_mode_from_handoff_record(record: SynthesisHandoffRecord) -> tuple[str, int | None]:
    """Resolve a lifecycle record to the synthesis CLI source mode."""

    if record.source_authority == "study_batch0_selector" and record.selection_epoch == "pre_assay_seed":
        return "batch0", None
    if record.source_authority == "opal_selection_batch" and record.selection_epoch == "opal_model_round":
        if record.model_as_of_round is None:
            raise ValueError(f"handoff record {record.handoff_id} requires model_as_of_round for OPAL rounds")
        return "opal-round", int(record.model_as_of_round)
    raise ValueError(
        "unsupported synthesis handoff record source "
        f"{record.source_authority!r}/{record.selection_epoch!r} for handoff_id={record.handoff_id}"
    )


def _resolve_repo_path(
    repo_root: str | Path,
    value: str | Path,
    *,
    label: str = "repository path",
) -> Path:
    root = Path(repo_root).resolve()
    raw = Path(value)
    resolved = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside repository root {root}: {resolved}") from exc
    return resolved


def _field_values(manifest: pd.DataFrame, column: str) -> set[Any]:
    values: set[Any] = set()
    for value in manifest[column].tolist():
        if pd.isna(value):
            values.add(None)
        elif isinstance(value, float) and value.is_integer():
            values.add(int(value))
        else:
            values.add(value)
    return values


def _require_manifest_field(manifest: pd.DataFrame, column: str, expected: Any) -> None:
    if column not in manifest.columns:
        raise ValueError(f"synthesis manifest missing lifecycle column required by handoff record: {column}")
    values = _field_values(manifest, column)
    if values != {expected}:
        observed = sorted((repr(value) for value in values))
        raise ValueError(f"handoff record lifecycle mismatch for {column}: expected {expected!r}, observed {observed}")


def _require_batch0_manifest_run_ids(manifest: pd.DataFrame, record: SynthesisHandoffRecord) -> None:
    if "run_id" not in manifest.columns:
        raise ValueError("synthesis manifest missing lifecycle column required by handoff record: run_id")
    if "campaign_slug" not in manifest.columns:
        raise ValueError("synthesis manifest missing campaign_slug required by handoff record")
    expected_by_campaign = record.expected_run_ids_by_campaign
    missing_expected = sorted(set(record.expected_campaign_counts).difference(expected_by_campaign))
    if missing_expected:
        raise ValueError(
            f"handoff record {record.handoff_id} requires explicit run_id for every expected campaign: "
            + ", ".join(missing_expected[:5])
        )
    for campaign_slug, expected_run_id in sorted(expected_by_campaign.items()):
        campaign_manifest = manifest.loc[manifest["campaign_slug"].astype(str) == campaign_slug]
        if campaign_manifest.empty:
            continue
        observed = _field_values(campaign_manifest, "run_id")
        if observed != {expected_run_id}:
            observed_text = sorted((repr(value) for value in observed))
            raise ValueError(
                "handoff record lifecycle mismatch for run_id: "
                f"campaign={campaign_slug} expected {expected_run_id!r}, observed {observed_text}"
            )


def _selection_view_membership_counts(manifest: pd.DataFrame) -> dict[str, int]:
    if "selection_view_ids" not in manifest.columns:
        raise ValueError("synthesis manifest missing selection_view_ids required by measured-round record")
    counts: dict[str, int] = {}
    for raw in manifest["selection_view_ids"].tolist():
        try:
            view_ids = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise ValueError("synthesis manifest contains invalid selection_view_ids JSON") from exc
        if not isinstance(view_ids, list) or not view_ids:
            raise ValueError("synthesis manifest selection_view_ids must be a non-empty JSON list")
        for view_id in view_ids:
            key = str(view_id)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _validate_batch0_manifest(manifest: pd.DataFrame, record: SynthesisHandoffRecord) -> dict[str, Any]:
    observed_counts = manifest.groupby("campaign_slug", sort=True).size().astype(int).to_dict()
    expected_counts = record.expected_campaign_counts
    mismatches = []
    for campaign_slug in sorted(set(expected_counts).union(observed_counts)):
        if expected_counts.get(campaign_slug) != observed_counts.get(campaign_slug):
            mismatches.append(
                f"{campaign_slug} expected={expected_counts.get(campaign_slug)} "
                f"observed={observed_counts.get(campaign_slug)}"
            )
    if mismatches:
        raise ValueError("handoff record campaign row mismatch: " + "; ".join(mismatches[:5]))
    _require_batch0_manifest_run_ids(manifest, record)
    return {"campaign_counts": observed_counts}


def _validate_measured_round_manifest(manifest: pd.DataFrame, record: SynthesisHandoffRecord) -> dict[str, Any]:
    assert record.expected_artifact is not None
    _require_manifest_field(manifest, "campaign_slug", record.campaign_slug)
    _require_manifest_field(manifest, "run_id", record.run_id)
    observed_rows = int(len(manifest))
    if observed_rows != int(record.expected_artifact.expected_rows):
        raise ValueError(
            f"handoff record selection batch row mismatch: expected={record.expected_artifact.expected_rows} "
            f"observed={observed_rows}"
        )
    if "synthesis_name" not in manifest.columns:
        raise ValueError("synthesis manifest missing synthesis_name required by measured-round record")
    observed_aliases = tuple(str(value).strip() for value in manifest["synthesis_name"].tolist())
    if any(not alias for alias in observed_aliases):
        raise ValueError("synthesis manifest contains an empty study alias")
    if len(observed_aliases) != len(set(observed_aliases)):
        raise ValueError("synthesis manifest contains duplicate study aliases")
    expected_aliases = set(record.expected_study_aliases)
    observed_alias_set = set(observed_aliases)
    if observed_alias_set != expected_aliases:
        raise ValueError(
            "handoff record study alias membership mismatch: "
            f"expected={sorted(expected_aliases)} observed={sorted(observed_alias_set)}"
        )
    if record.materialization_contract is not None:
        binding_columns = ("synthesis_name", "id", "core_sha256")
        missing_binding_columns = [column for column in binding_columns if column not in manifest.columns]
        if missing_binding_columns:
            raise ValueError(
                "synthesis manifest missing exact candidate-binding columns required by measured-round record: "
                + ", ".join(missing_binding_columns)
            )
        expected_bindings = {
            (row.study_alias, row.candidate_id, row.core_sha256)
            for row in record.materialization_contract.expected_candidates
        }
        observed_bindings = {
            (
                str(row.synthesis_name).strip(),
                str(row.id).strip(),
                str(row.core_sha256).strip(),
            )
            for row in manifest.loc[:, list(binding_columns)].itertuples(index=False)
        }
        if observed_bindings != expected_bindings:
            raise ValueError(
                "handoff record study alias candidate binding mismatch: "
                f"expected={sorted(expected_bindings)} observed={sorted(observed_bindings)}"
            )
    observed_view_counts = _selection_view_membership_counts(manifest)
    expected_view_counts = record.expected_selection_view_counts
    if observed_view_counts != expected_view_counts:
        raise ValueError(
            "handoff record selection-view membership mismatch: "
            f"expected={expected_view_counts} observed={observed_view_counts}"
        )
    return {
        "campaign_slug": record.campaign_slug,
        "study_aliases": sorted(observed_alias_set),
        "selection_view_counts": observed_view_counts,
        "selection_batch_count": observed_rows,
    }


def validate_manifest_against_handoff_record(
    manifest: pd.DataFrame,
    record: SynthesisHandoffRecord,
    *,
    strategy_id: str,
) -> dict[str, Any]:
    """Validate a manifest against the checked-in lifecycle record."""

    if str(strategy_id) != record.strategy_id:
        raise ValueError(f"handoff record strategy mismatch: expected {record.strategy_id}, observed {strategy_id}")
    if "campaign_slug" not in manifest.columns:
        raise ValueError("synthesis manifest missing campaign_slug required by handoff record")
    if record.source_authority == "study_batch0_selector":
        source_validation = _validate_batch0_manifest(manifest, record)
    elif record.source_authority == "opal_selection_batch":
        source_validation = _validate_measured_round_manifest(manifest, record)
    else:
        raise ValueError(f"unsupported synthesis handoff source_authority: {record.source_authority}")

    _require_manifest_field(manifest, "batch_id", record.handoff_id)
    _require_manifest_field(manifest, "selection_epoch", record.selection_epoch)
    _require_manifest_field(manifest, "assay_batch_index", record.assay_batch_index)
    _require_manifest_field(manifest, "model_as_of_round", record.model_as_of_round)
    return {
        "status": "pass",
        "strategy_id": strategy_id,
        **source_validation,
    }


def apply_handoff_record_lifecycle(
    selected: list[SelectedCandidate],
    record: SynthesisHandoffRecord,
) -> list[SelectedCandidate]:
    """Stamp record-owned lifecycle fields onto selected rows before manifest build."""

    expected_run_ids = record.expected_run_ids_by_campaign
    return [
        SelectedCandidate(
            campaign_slug=row.campaign_slug,
            selection_memberships=row.selection_memberships,
            as_of_round=row.as_of_round,
            run_id=(
                str(record.run_id)
                if record.source_authority == "opal_selection_batch"
                else expected_run_ids.get(row.campaign_slug, row.run_id)
            ),
            selection_rank=row.selection_rank,
            id=row.id,
            sequence=row.sequence,
            synthesis_name=row.synthesis_name,
            selection_source=row.selection_source,
            selection_epoch=record.selection_epoch,
            assay_batch_index=record.assay_batch_index,
            model_as_of_round=record.model_as_of_round,
        )
        for row in selected
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_contained_path(
    repo_root: Path,
    value: str | Path,
    *,
    label: str,
) -> Path:
    """Resolve one path and reject lexical or symlink escape from ``repo_root``."""

    return _resolve_repo_path(repo_root, value, label=label)


def validate_materialization_contract_inputs(
    record: SynthesisHandoffRecord,
    *,
    repo_root: str | Path,
    campaign_config_path: str | Path,
    selection_batch_path: str | Path,
    candidate_records_path: str | Path,
    promoter_alias_registry_path: str | Path,
    cloning_strategy_path: str | Path,
) -> dict[str, Any]:
    """Verify the exact repository inputs authorized for measured-round generation."""

    contract = record.materialization_contract
    if record.source_authority != "opal_selection_batch" or contract is None:
        raise ValueError(f"handoff record {record.handoff_id} has no measured-round materialization contract")
    root = Path(repo_root).resolve()
    actual_by_field = {
        "campaign_config": Path(campaign_config_path),
        "selection_batch": Path(selection_batch_path),
        "candidate_records": Path(candidate_records_path),
        "promoter_alias_registry": Path(promoter_alias_registry_path),
        "cloning_strategy": Path(cloning_strategy_path),
    }
    receipt_by_field = {
        "campaign_config": contract.campaign_config,
        "selection_batch": contract.selection_batch,
        "candidate_records": contract.candidate_records,
        "promoter_alias_registry": contract.promoter_alias_registry,
        "cloning_strategy": contract.cloning_strategy,
    }
    verified: dict[str, dict[str, str]] = {}
    for field, actual_path in actual_by_field.items():
        actual_path = _resolve_repo_contained_path(
            root,
            actual_path,
            label=f"materialization contract {field} input",
        )
        receipt = receipt_by_field[field]
        expected_path = _resolve_repo_contained_path(
            root,
            receipt.path,
            label=f"materialization contract {field} receipt",
        )
        if actual_path != expected_path:
            raise ValueError(
                f"materialization contract {field} path mismatch: expected={expected_path} observed={actual_path}"
            )
        if not actual_path.is_file():
            raise ValueError(f"materialization contract {field} input is missing: {actual_path}")
        observed_sha256 = _sha256_file(actual_path)
        if observed_sha256 != receipt.sha256:
            raise ValueError(
                f"materialization contract {field} sha256 mismatch: "
                f"expected={receipt.sha256} observed={observed_sha256}"
            )
        verified[field] = {
            "path": receipt.path,
            "sha256": observed_sha256,
        }
    return {
        "status": "pass",
        "inputs": verified,
        "expected_candidate_count": len(contract.expected_candidates),
    }


def _sha256_genbank_dir(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(path.glob("*.gb")):
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_artifact_manifest_lifecycle(
    manifest: pd.DataFrame,
    *,
    record: SynthesisHandoffRecord,
    expected: ExpectedHandoffArtifact,
) -> dict[str, Any]:
    _require_manifest_field(manifest, "strategy_id", record.strategy_id)
    if record.source_authority == "opal_selection_batch":
        return validate_manifest_against_handoff_record(
            manifest,
            record,
            strategy_id=record.strategy_id,
        )
    _require_manifest_field(manifest, "campaign_slug", expected.campaign_slug)
    _require_manifest_field(manifest, "batch_id", record.handoff_id)
    _require_manifest_field(manifest, "selection_epoch", record.selection_epoch)
    _require_manifest_field(manifest, "assay_batch_index", record.assay_batch_index)
    _require_manifest_field(manifest, "model_as_of_round", record.model_as_of_round)
    expected_run_id = expected.run_id or record.run_id
    _require_manifest_field(manifest, "run_id", expected_run_id)
    if len(manifest) != int(expected.expected_rows):
        raise ValueError(
            f"handoff record artifact row mismatch for campaign {expected.campaign_slug}: "
            f"expected={expected.expected_rows} observed={len(manifest)}"
        )
    return {
        "status": "pass",
        "campaign_slug": expected.campaign_slug,
        "row_count": int(len(manifest)),
    }


def artifact_status_for_handoff_record(
    record: SynthesisHandoffRecord,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Inspect generated artifacts referenced by a lifecycle record."""

    rows: list[dict[str, Any]] = []
    present_artifacts = 0
    manifest_lifecycle_pass_count = 0
    workbook_readback_pass_count = 0
    genbank_readback_pass_count = 0
    for expected in record.artifacts:
        manifest_path = _resolve_repo_path(
            repo_root,
            expected.manifest_path,
            label="generated manifest_path",
        )
        workbook_path = _resolve_repo_path(
            repo_root,
            expected.vendor_workbook_path,
            label="generated vendor_workbook_path",
        )
        genbank_dir_path = _resolve_repo_path(
            repo_root,
            expected.genbank_dir_path,
            label="generated genbank_dir_path",
        )
        genbank_feature_table_path = _resolve_repo_path(
            repo_root,
            expected.genbank_feature_table_path,
            label="generated genbank_feature_table_path",
        )
        row: dict[str, Any] = {
            "campaign_slug": expected.campaign_slug,
            "expected_rows": int(expected.expected_rows),
            "manifest_path": str(manifest_path),
            "vendor_workbook_path": str(workbook_path),
            "genbank_dir_path": str(genbank_dir_path),
            "genbank_feature_table_path": str(genbank_feature_table_path),
            "manifest_exists": manifest_path.exists(),
            "vendor_workbook_exists": workbook_path.exists(),
            "genbank_dir_exists": genbank_dir_path.is_dir(),
            "genbank_feature_table_exists": genbank_feature_table_path.exists(),
            "manifest_sha256": None,
            "vendor_workbook_sha256": None,
            "genbank_dir_sha256": None,
            "genbank_feature_table_sha256": None,
            "manifest_hash_matches_record": None,
            "vendor_workbook_hash_matches_record": None,
            "genbank_dir_hash_matches_record": None,
            "genbank_feature_table_hash_matches_record": None,
            "manifest_row_count": None,
            "manifest_row_count_matches_record": None,
            "manifest_lifecycle_status": None,
            "workbook_readback_status": None,
            "genbank_readback_status": None,
            "workbook_readback_matches_record": None,
            "genbank_readback_matches_record": None,
        }
        manifest: pd.DataFrame | None = None
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            row["manifest_sha256"] = _sha256_file(manifest_path)
            row["manifest_row_count"] = int(len(manifest))
            row["manifest_row_count_matches_record"] = int(len(manifest)) == int(expected.expected_rows)
            if expected.manifest_sha256 is not None:
                row["manifest_hash_matches_record"] = row["manifest_sha256"] == expected.manifest_sha256
            try:
                row["manifest_lifecycle_validation"] = _validate_artifact_manifest_lifecycle(
                    manifest,
                    record=record,
                    expected=expected,
                )
                row["manifest_lifecycle_status"] = "pass"
            except ValueError as exc:
                row["manifest_lifecycle_status"] = "fail"
                row["manifest_lifecycle_error"] = str(exc)
        if workbook_path.exists():
            row["vendor_workbook_sha256"] = _sha256_file(workbook_path)
            if expected.vendor_workbook_sha256 is not None:
                row["vendor_workbook_hash_matches_record"] = (
                    row["vendor_workbook_sha256"] == expected.vendor_workbook_sha256
                )
        if genbank_dir_path.is_dir():
            row["genbank_dir_sha256"] = _sha256_genbank_dir(genbank_dir_path)
            if expected.genbank_dir_sha256 is not None:
                row["genbank_dir_hash_matches_record"] = row["genbank_dir_sha256"] == expected.genbank_dir_sha256
        if genbank_feature_table_path.exists():
            row["genbank_feature_table_sha256"] = _sha256_file(genbank_feature_table_path)
            if expected.genbank_feature_table_sha256 is not None:
                row["genbank_feature_table_hash_matches_record"] = (
                    row["genbank_feature_table_sha256"] == expected.genbank_feature_table_sha256
                )
        if manifest is not None and workbook_path.exists():
            try:
                readback = validate_azenta_workbook(manifest, workbook_path)
                row["workbook_readback_status"] = readback["status"]
            except ValueError as exc:
                row["workbook_readback_status"] = "fail"
                row["workbook_readback_error"] = str(exc)
        if manifest is not None and genbank_dir_path.is_dir():
            try:
                genbank_readback = validate_genbank_record_set(
                    manifest,
                    genbank_dir_path,
                    feature_table=genbank_feature_table_path,
                )
                row["genbank_readback_status"] = genbank_readback["status"]
            except ValueError as exc:
                row["genbank_readback_status"] = "fail"
                row["genbank_readback_error"] = str(exc)
        if expected.workbook_readback_status is not None:
            row["workbook_readback_matches_record"] = (
                row["workbook_readback_status"] == expected.workbook_readback_status
            )
        if expected.genbank_readback_status is not None:
            row["genbank_readback_matches_record"] = row["genbank_readback_status"] == expected.genbank_readback_status
        if (
            row["manifest_exists"]
            and row["vendor_workbook_exists"]
            and row["genbank_dir_exists"]
            and row["genbank_feature_table_exists"]
        ):
            present_artifacts += 1
        if row["workbook_readback_status"] == "pass":
            workbook_readback_pass_count += 1
        if row["manifest_lifecycle_status"] == "pass":
            manifest_lifecycle_pass_count += 1
        if row["genbank_readback_status"] == "pass":
            genbank_readback_pass_count += 1
        rows.append(row)

    current_contract_ready = bool(rows) and all(
        row["manifest_exists"]
        and row["vendor_workbook_exists"]
        and row["genbank_dir_exists"]
        and row["genbank_feature_table_exists"]
        and row["manifest_hash_matches_record"] is True
        and row["vendor_workbook_hash_matches_record"] is True
        and row["genbank_dir_hash_matches_record"] is True
        and row["genbank_feature_table_hash_matches_record"] is True
        and row["manifest_row_count_matches_record"] is True
        and row["manifest_lifecycle_status"] == "pass"
        and row["workbook_readback_status"] == "pass"
        and row["genbank_readback_status"] == "pass"
        for row in rows
    )
    return {
        "summary": {
            "expected_artifact_count": int(len(record.artifacts)),
            "present_artifact_count": int(present_artifacts),
            "manifest_lifecycle_pass_count": int(manifest_lifecycle_pass_count),
            "workbook_readback_pass_count": int(workbook_readback_pass_count),
            "genbank_readback_pass_count": int(genbank_readback_pass_count),
            "current_contract_ready": current_contract_ready,
        },
        "artifacts": rows,
    }


def handoff_record_payload(
    record: SynthesisHandoffRecord,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Return a JSON-ready record summary for operators and tests."""

    return {
        "handoff_id": record.handoff_id,
        "lifecycle_status": record.lifecycle_status,
        "source_authority": record.source_authority,
        "selection_epoch": record.selection_epoch,
        "assay_batch_index": record.assay_batch_index,
        "model_as_of_round": record.model_as_of_round,
        "run_id": record.run_id,
        "campaign_slug": record.campaign_slug,
        "strategy_id": record.strategy_id,
        "expected_selection_views": [row.to_json() for row in record.expected_selection_views],
        "expected_study_aliases": list(record.expected_study_aliases),
        "materialization_contract": (
            record.materialization_contract.to_json() if record.materialization_contract is not None else None
        ),
        "expected_artifacts": [row.to_json() for row in record.artifacts],
        "artifact_status": artifact_status_for_handoff_record(record, repo_root=repo_root),
    }
