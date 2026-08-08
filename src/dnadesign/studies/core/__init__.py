"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/__init__.py

Package exports for studies core.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .models import (
    StudyOpsContract,
    StudyPhaseContract,
    StudyPreflightContract,
    StudyPreflightNextScopeContract,
    StudyStatusContext,
    StudyStatusService,
)
from .preflight_plan import (
    StudyPreflightPlan,
    build_study_preflight_plan,
    normalize_study_preflight_scope,
)
from .reader_records import (
    READER_CATALOG_SCHEMA_VERSION,
    READER_CLI_SCHEMA,
    READER_RECORD_SCHEMA_VERSION,
    ReaderArtifactFile,
    ReaderDataframeRecordError,
    ReaderDataframeRecordRef,
    ReaderInputArtifactEvidence,
    ReaderRecordError,
    ReaderRecordExpectation,
    ReaderRecordInputEvidence,
    ReaderRecordProducer,
    ReaderRecordRecipeSource,
    ReaderRecordSet,
    ReaderResolvedRecord,
    parse_record_inputs,
    parse_record_producer,
    resolve_digest_verified_dataframe_record,
    resolve_digest_verified_records,
)
from .record_loader import load_study_ops_contract
from .record_locator import (
    ActiveStudySelection,
    discover_active_study_selection,
    discover_study_selection_for_status_kind,
)
from .registry import StudyIndex, StudyIndexEntry, load_study_index
from .workspace import (
    StudyArtifact,
    StudyCatalogProgram,
    StudyEvidenceIndex,
    StudyManifest,
    StudyWorkflow,
    StudyWorkspace,
    load_study_evidence_index,
    load_study_workspace,
)

__all__ = [
    "ActiveStudySelection",
    "READER_CATALOG_SCHEMA_VERSION",
    "READER_CLI_SCHEMA",
    "READER_RECORD_SCHEMA_VERSION",
    "ReaderArtifactFile",
    "ReaderDataframeRecordError",
    "ReaderDataframeRecordRef",
    "ReaderInputArtifactEvidence",
    "ReaderRecordError",
    "ReaderRecordExpectation",
    "ReaderRecordInputEvidence",
    "ReaderRecordProducer",
    "ReaderRecordRecipeSource",
    "ReaderRecordSet",
    "ReaderResolvedRecord",
    "StudyIndex",
    "StudyIndexEntry",
    "StudyArtifact",
    "StudyCatalogProgram",
    "StudyEvidenceIndex",
    "StudyManifest",
    "StudyOpsContract",
    "StudyPhaseContract",
    "StudyPreflightContract",
    "StudyPreflightNextScopeContract",
    "StudyPreflightPlan",
    "StudyStatusService",
    "StudyStatusContext",
    "StudyWorkflow",
    "StudyWorkspace",
    "build_study_preflight_plan",
    "discover_active_study_selection",
    "discover_study_selection_for_status_kind",
    "load_study_index",
    "load_study_evidence_index",
    "load_study_ops_contract",
    "load_study_workspace",
    "normalize_study_preflight_scope",
    "parse_record_inputs",
    "parse_record_producer",
    "resolve_digest_verified_dataframe_record",
    "resolve_digest_verified_records",
]
