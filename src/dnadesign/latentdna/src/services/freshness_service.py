"""
Freshness evaluation helpers for latentdna artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..contracts.errors import ContractViolationError
from ..sources.provenance import OVERLAY_INVENTORY_DIGEST_MODE, source_provenance_digest
from ..sources.resolver import inspect_source_schema, resolve_source
from ..views.row_contracts import source_backed_view_row_contract
from ..workspaces.loader import WorkspaceContext
from ._artifact_inputs import artifact_kind_for_input_dependency
from ._artifacts import artifact_exists, artifact_manifest_path


@dataclass(slots=True)
class FreshnessCache:
    artifact_results: dict[tuple[str, str], dict[str, object]] = field(default_factory=dict)
    path_digests: dict[str, tuple[bool, str | None]] = field(default_factory=dict)
    overlay_inventory_digests: dict[str, str] = field(default_factory=dict)


def _resolve_path_digest(path: Path, *, cache: FreshnessCache) -> tuple[bool, str | None]:
    key = path.as_posix()
    cached = cache.path_digests.get(key)
    if cached is not None:
        return cached
    if not path.exists():
        result = (False, None)
    else:
        result = (True, source_provenance_digest({"path": path.as_posix()}))
    cache.path_digests[key] = result
    return result


def _resolve_overlay_inventory_digest(path: Path, *, cache: FreshnessCache) -> str:
    key = path.resolve().as_posix()
    cached = cache.overlay_inventory_digests.get(key)
    if cached is not None:
        return cached
    digest = source_provenance_digest({"path": key, "digest_mode": OVERLAY_INVENTORY_DIGEST_MODE})
    cache.overlay_inventory_digests[key] = digest
    return digest


def _view_config_freshness_reasons(
    context: WorkspaceContext,
    *,
    artifact_id: str,
    manifest: dict[str, object],
) -> tuple[list[str], bool]:
    if not all(hasattr(context, attribute) for attribute in ("require_source_view", "require_source", "config")):
        return [], True
    try:
        view = context.require_source_view(artifact_id)
        source = context.require_source(view.source)
        resolved = resolve_source(view.source, source, workspace_dir=context.workspace_dir)
        row_contract = source_backed_view_row_contract(
            context,
            source_id=view.source,
            source=source,
            available_columns=inspect_source_schema(resolved)["columns"],
        )
    except ContractViolationError as exc:
        return [f"stale view config for view:{artifact_id}: {exc}"], True
    except Exception as exc:
        return [f"freshness unknown: could not resolve view config for view:{artifact_id}: {exc}"], False

    params = manifest.get("params")
    if not isinstance(params, dict):
        return [f"freshness unknown: view manifest lacks params for view:{artifact_id}"], False

    reasons: list[str] = []
    expected_pairs = {
        "coordinate_space_id": view.coordinate_space_id,
        "record_key": source.record_key,
        "subject_key": source.subject_key,
        "context_key": source.context_key,
        "source": view.source,
        "role": view.role,
    }
    for key, expected in expected_pairs.items():
        if params.get(key) != expected:
            reasons.append(f"stale view config for view:{artifact_id}: {key}")

    expected_row_columns = row_contract.materialized_row_columns
    recorded_row_columns = [str(value) for value in params.get("row_columns", []) if value is not None]
    missing_row_columns = [column for column in expected_row_columns if column not in recorded_row_columns]
    if missing_row_columns:
        reasons.append(f"stale view config for view:{artifact_id}: missing row columns {missing_row_columns}")
    return reasons, True


def evaluate_artifact_freshness(
    context: WorkspaceContext,
    *,
    artifact_kind: str,
    artifact_id: str,
    _stack: set[tuple[str, str]] | None = None,
    cache: FreshnessCache | None = None,
) -> dict[str, object]:
    if _stack is None:
        _stack = set()
    if cache is None:
        cache = FreshnessCache()
    key = (artifact_kind, artifact_id)
    if key in _stack:
        return {
            "status": "attention",
            "reason": f"freshness unknown: recursive dependency detected for {artifact_kind}:{artifact_id}",
            "known": False,
        }
    cached = cache.artifact_results.get(key)
    if cached is not None:
        return cached
    if not artifact_exists(context, artifact_kind=artifact_kind, artifact_id=artifact_id):
        result = {"status": "missing", "reason": f"artifact is missing: {artifact_kind}:{artifact_id}", "known": True}
        cache.artifact_results[key] = result
        return result

    manifest_path = artifact_manifest_path(context, artifact_kind=artifact_kind, artifact_id=artifact_id)
    manifest = context.read_manifest(manifest_path)
    stack = set(_stack)
    stack.add(key)
    result = evaluate_manifest_freshness(
        context,
        manifest=manifest,
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        _stack=stack,
        cache=cache,
    )
    cache.artifact_results[key] = result
    return result


def evaluate_manifest_freshness(
    context: WorkspaceContext,
    *,
    manifest: dict[str, object],
    artifact_kind: str,
    artifact_id: str,
    _stack: set[tuple[str, str]],
    cache: FreshnessCache,
) -> dict[str, object]:
    if manifest.get("status") == "error":
        return {
            "status": "error",
            "reason": f"artifact manifest is marked error: {artifact_kind}:{artifact_id}",
            "known": True,
        }

    reasons: list[str] = []
    known = True
    checked_any = False

    for entry in manifest.get("source_provenance", []) or []:
        if not isinstance(entry, dict):
            continue
        path_text = str(entry.get("path") or "")
        recorded_digest = str(entry.get("digest") or "")
        if not path_text or not recorded_digest:
            known = False
            reasons.append(f"freshness unknown: incomplete source provenance for {artifact_kind}:{artifact_id}")
            continue
        checked_any = True
        path = Path(path_text)
        digest_mode = str(entry.get("digest_mode") or "")
        if digest_mode == OVERLAY_INVENTORY_DIGEST_MODE:
            if not path.exists():
                known = False
                reasons.append(f"freshness unknown: source path is missing: {path_text}")
                continue
            current_digest = _resolve_overlay_inventory_digest(path, cache=cache)
            if current_digest != recorded_digest:
                namespace = str(entry.get("namespace") or entry.get("id") or path.name)
                reasons.append(
                    f"stale freshness: source overlay inventory for {artifact_kind}:{artifact_id}: {namespace}"
                )
            continue
        path_exists, current_digest = _resolve_path_digest(path, cache=cache)
        if not path_exists:
            known = False
            reasons.append(f"freshness unknown: source path is missing: {path_text}")
            continue
        assert current_digest is not None
        if current_digest != recorded_digest:
            reasons.append(f"stale source freshness for {artifact_kind}:{artifact_id}: {entry.get('id')}")

    for input_entry in manifest.get("inputs", []) or []:
        if not isinstance(input_entry, dict):
            continue
        input_kind = str(input_entry.get("kind") or "")
        input_id = str(input_entry.get("id") or "")
        path_text = input_entry.get("path")
        recorded_digest = str(input_entry.get("digest") or "")
        upstream_kind = artifact_kind_for_input_dependency(input_kind)
        use_recorded_path_digest = (
            path_text is not None
            and recorded_digest
            and (upstream_kind is None or Path(str(path_text)).name == "manifest.json")
        )
        if use_recorded_path_digest:
            checked_any = True
            path = Path(str(path_text))
            path_exists, current_digest = _resolve_path_digest(path, cache=cache)
            if not path_exists:
                known = False
                reasons.append(f"freshness unknown: input path is missing: {path}")
            else:
                assert current_digest is not None
                if current_digest != recorded_digest:
                    reasons.append(f"stale input digest for {input_kind}:{input_id}")

        if upstream_kind is None:
            if not path_text and input_kind not in {"source", "landmark_source"}:
                known = False
                reasons.append(f"freshness unknown for input {input_kind}:{input_id}")
            continue
        checked_any = True
        upstream = evaluate_artifact_freshness(
            context,
            artifact_kind=upstream_kind,
            artifact_id=input_id,
            _stack=_stack,
            cache=cache,
        )
        if upstream["status"] != "ok":
            known = known and bool(upstream.get("known"))
            reason = str(upstream.get("reason") or f"{upstream_kind}:{input_id} is not fresh")
            reasons.append(f"freshness depends on {upstream_kind}:{input_id}: {reason}")

    if reasons:
        return {"status": "attention", "reason": reasons[0], "known": known, "reasons": reasons}
    if artifact_kind == "view":
        view_reasons, view_known = _view_config_freshness_reasons(
            context,
            artifact_id=artifact_id,
            manifest=manifest,
        )
        if view_reasons:
            return {
                "status": "attention",
                "reason": view_reasons[0],
                "known": view_known,
                "reasons": view_reasons,
            }
    if not checked_any:
        return {
            "status": "attention",
            "reason": f"freshness unknown: manifest lacks recorded input provenance for {artifact_kind}:{artifact_id}",
            "known": False,
        }
    return {"status": "ok", "reason": None, "known": True}
