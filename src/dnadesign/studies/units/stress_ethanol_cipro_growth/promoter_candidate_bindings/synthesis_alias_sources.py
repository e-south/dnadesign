"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/synthesis_alias_sources.py

Digest-pinned synthesis aliases used as candidate-binding provenance.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd
import yaml

from .contracts import PromoterCandidateBindingsError
from .source_io import file_sha256


def load_synthesis_alias_sources(
    repo_root: Path,
    *,
    record_path: Path,
    handoff_id: str,
) -> pd.DataFrame:
    """Return exact synthesis aliases and candidate identities from the lifecycle record."""

    root = Path(repo_root).expanduser().resolve()
    source_record = root / record_path
    if not source_record.is_file():
        raise PromoterCandidateBindingsError(f"Study synthesis-handoff record not found: {source_record}")
    try:
        payload = yaml.safe_load(source_record.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PromoterCandidateBindingsError(f"Could not parse study synthesis-handoff record: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("handoffs"), list):
        raise PromoterCandidateBindingsError("Study synthesis-handoff record must contain a handoffs list.")
    handoffs = [item for item in payload["handoffs"] if isinstance(item, dict) and item.get("handoff_id") == handoff_id]
    if len(handoffs) != 1:
        raise PromoterCandidateBindingsError(
            f"Expected exactly one checked-in record for {handoff_id}; found {len(handoffs)}."
        )
    handoff = handoffs[0]
    manifest_entries = _handoff_manifest_entries(handoff, handoff_id=handoff_id)

    rows: list[pd.DataFrame] = []
    seen_artifacts: set[str] = set()
    for index, entry in enumerate(manifest_entries):
        rows.append(
            _load_manifest_entry(
                root,
                entry,
                index=index,
                seen_artifacts=seen_artifacts,
                handoff_id=handoff_id,
            )
        )
    out = pd.concat(rows, ignore_index=True)
    duplicates = out.loc[out["synthesis_name"].duplicated(keep=False), "synthesis_name"].astype(str).unique()
    if len(duplicates):
        raise PromoterCandidateBindingsError(
            f"Duplicate synthesis names in handoff {handoff_id!r}: {sorted(duplicates)}"
        )
    return out


def _handoff_manifest_entries(handoff: dict[str, object], *, handoff_id: str) -> list[object]:
    has_collection = "expected_campaigns" in handoff
    has_single = "expected_artifact" in handoff
    if has_collection == has_single:
        raise PromoterCandidateBindingsError(
            f"{handoff_id} must declare exactly one of expected_campaigns or expected_artifact."
        )
    if has_collection:
        entries = handoff.get("expected_campaigns")
        if not isinstance(entries, list) or not entries:
            raise PromoterCandidateBindingsError(f"{handoff_id} expected_campaigns must be a non-empty list.")
        return list(entries)
    entry = handoff.get("expected_artifact")
    if not isinstance(entry, dict):
        raise PromoterCandidateBindingsError(f"{handoff_id} expected_artifact must be a mapping.")
    return [entry]


def _load_manifest_entry(
    root: Path,
    entry: object,
    *,
    index: int,
    seen_artifacts: set[str],
    handoff_id: str,
) -> pd.DataFrame:
    if not isinstance(entry, dict):
        raise PromoterCandidateBindingsError(f"manifest entry {index} must be a mapping.")
    context = f"manifest_entries[{index}]"
    artifact_id = _required_text(entry.get("campaign_slug"), context=f"{context}.campaign_slug")
    raw_source_slug = entry.get("source_campaign_slug", artifact_id)
    source_slug = _required_text(raw_source_slug, context=f"{context}.source_campaign_slug")
    if artifact_id in seen_artifacts:
        raise PromoterCandidateBindingsError(f"Duplicate campaign_slug in {handoff_id}: {artifact_id!r}.")
    seen_artifacts.add(artifact_id)
    expected_rows = entry.get("expected_rows")
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows < 1:
        raise PromoterCandidateBindingsError(f"{context}.expected_rows must be positive.")
    relative_path = _safe_repo_relative(entry.get("manifest_path"), context=f"{context}.manifest_path")
    manifest_path = (root / relative_path).resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as exc:
        raise PromoterCandidateBindingsError(f"Synthesis manifest escapes repository root: {relative_path}") from exc
    if not manifest_path.is_file():
        raise PromoterCandidateBindingsError(f"Synthesis manifest not found: {manifest_path}")
    expected_sha256 = _required_sha256(entry.get("manifest_sha256"), context=f"{context}.manifest_sha256")
    observed_sha256 = file_sha256(manifest_path)
    if observed_sha256 != expected_sha256:
        raise PromoterCandidateBindingsError(
            f"Synthesis manifest digest mismatch for {artifact_id}: "
            f"expected {expected_sha256}, observed {observed_sha256}."
        )
    frame = pd.read_csv(manifest_path)
    _validate_manifest_frame(
        frame,
        manifest_path=manifest_path,
        artifact_id=artifact_id,
        source_slug=source_slug,
        expected_rows=expected_rows,
    )
    frame = frame.copy()
    frame["source_campaign_slug"] = frame["campaign_slug"].astype(str)
    frame["campaign_slug"] = artifact_id
    frame["source_manifest_path"] = relative_path.as_posix()
    frame["source_manifest_sha256"] = observed_sha256
    return frame


def _validate_manifest_frame(
    frame: pd.DataFrame,
    *,
    manifest_path: Path,
    artifact_id: str,
    source_slug: str,
    expected_rows: int,
) -> None:
    required = {"id", "synthesis_name", "core_sequence", "campaign_slug", "validation_status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PromoterCandidateBindingsError(f"Synthesis manifest {manifest_path} missing columns: {missing}")
    if not frame["validation_status"].astype(str).eq("pass").all():
        raise PromoterCandidateBindingsError(f"Synthesis manifest {manifest_path} contains non-pass validation rows.")
    if len(frame) != expected_rows:
        raise PromoterCandidateBindingsError(
            f"Synthesis manifest row count mismatch for {artifact_id}: expected {expected_rows}, observed {len(frame)}."
        )
    observed_source_slugs = set(frame["campaign_slug"].astype(str))
    if observed_source_slugs != {source_slug}:
        raise PromoterCandidateBindingsError(
            f"Synthesis manifest source campaign identity mismatch for {artifact_id}: "
            f"expected {source_slug!r}, observed {sorted(observed_source_slugs)}."
        )


def _required_text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromoterCandidateBindingsError(f"{context} must be a non-empty string.")
    return value.strip()


def _safe_repo_relative(value: object, *, context: str) -> PurePosixPath:
    text = _required_text(value, context=context)
    if "\\" in text:
        raise PromoterCandidateBindingsError(f"{context} must be a relative POSIX path.")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PromoterCandidateBindingsError(f"{context} must be a confined relative POSIX path.")
    return path


def _required_sha256(value: object, *, context: str) -> str:
    text = _required_text(value, context=context).lower().removeprefix("sha256:")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise PromoterCandidateBindingsError(f"{context} must be a SHA-256 digest.")
    return text
