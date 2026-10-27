"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/document_metadata.py

Document metadata for documentation validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping
from pathlib import Path

import yaml

from dnadesign.devtools.docs.check_contracts import (
    _CROSS_TOOL_DOC_ALLOWED_PLANES,
    _CROSS_TOOL_DOC_ALLOWED_TYPES,
    _EXEC_PLAN_REQUIRED_SECTIONS,
    _EXEC_PLAN_STATUSES,
    _METADATA_TOKEN_VALUE_PATTERN,
    _REGISTRY_ID_VALUE_PATTERN,
    _RUNBOOK_CATALOG_METADATA_TYPES,
    CHECKLIST_ITEM_PATTERN,
    CREATED_PATTERN,
    CROSS_TOOL_DOC_METADATA_CONTRACTS,
    ENTRY_ARTIFACT_PATTERN,
    EXECUTION_KIND_METADATA_PATTERN,
    EXIT_ARTIFACT_PATTERN,
    INDEX_MARKDOWN_FILES,
    LINK_PATTERN,
    OWNER_BOUNDARY_PATTERN,
    PLANE_PATTERN,
    PROGRESS_ITEM_TIMESTAMP_PATTERN,
    PROGRESS_SURFACE_GLOSSARY_ROW_PATTERN,
    REGISTRY_ID_METADATA_PATTERN,
    REPO_LOCAL_SKILL_DESCRIPTION_MAX_CHARS,
    REPO_LOCAL_SKILLS_DIR,
    RUNBOOK_CATALOG_DOC_PATH,
    RUNBOOK_MARKDOWN_FILES,
    RUNBOOK_STATUS_GLOSSARY_HEADING,
    SECTION_HEADING_PATTERN,
    SHARED_USR_DATASET_LAYOUT_NUDGE,
    SHARED_USR_DATASETS_ROOT,
    STATUS_KIND_METADATA_PATTERN,
    STATUS_PATTERN,
    SUMMARY_METADATA_PATTERN,
    TYPE_PATTERN,
)
from dnadesign.devtools.docs.freshness import verification_change_issue
from dnadesign.devtools.docs.markdown_inventory import (
    _collect_tool_docs_markdown_files,
)
from dnadesign.devtools.docs.metadata import LAST_VERIFIED_PATTERN, OWNER_PATTERN, SOR_MARKDOWN_FILES
from dnadesign.ops.catalog import (
    CatalogProcedureEntry,
    load_runbook_catalog,
    render_catalog_procedure_section,
    render_catalog_tool_source_section,
    resolve_catalog_doc_path,
    resolve_registry_metadata_path_for_doc_path,
)
from dnadesign.ops.status import list_status_kind_specs_for_repo


def _extract_metadata_field(text: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def _load_markdown_frontmatter(text: str, *, path: Path) -> Mapping[str, object]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError(f"{path}: YAML front matter is missing its closing '---'.")
    try:
        payload = yaml.safe_load(text[4:end])
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML front matter: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: YAML front matter must be a mapping.")
    return payload


def _extract_doc_metadata_field(
    text: str,
    *,
    frontmatter: Mapping[str, object],
    frontmatter_key: str,
    body_pattern: re.Pattern[str],
) -> str | None:
    value = frontmatter.get(frontmatter_key)
    if value is not None:
        if isinstance(value, dt.date):
            return value.isoformat()
        return str(value).strip()
    return _extract_metadata_field(text, body_pattern)


def _markdown_body_without_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end < 0:
        return text
    return text[end + len("\n---\n") :]


def _parse_iso_date(value: str, *, field_name: str, path: Path) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{path}: {field_name} must use YYYY-MM-DD.") from exc


def _find_sor_metadata_issues(
    repo_root: Path,
    *,
    changed_doc_dates: Mapping[str, dt.date] | None = None,
) -> list[str]:
    today = dt.date.today()
    changed_dates = changed_doc_dates or {}
    issues: list[str] = []

    for name in SOR_MARKDOWN_FILES:
        path = repo_root / name
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")
        doc_type = _extract_metadata_field(text, TYPE_PATTERN)
        if doc_type is None:
            issues.append(f"{path}: missing '**Type:**' metadata field.")
            continue
        if doc_type != "system-of-record":
            issues.append(f"{path}: '**Type:**' must be exactly 'system-of-record'.")
            continue

        owner = _extract_metadata_field(text, OWNER_PATTERN)
        if owner is None:
            issues.append(f"{path}: missing '**Owner:**' metadata field.")
            continue
        if not owner:
            issues.append(f"{path}: '**Owner:**' must not be empty.")

        last_verified_raw = _extract_metadata_field(text, LAST_VERIFIED_PATTERN)
        if last_verified_raw is None:
            issues.append(f"{path}: missing '**Last verified:**' metadata field.")
            continue
        if not last_verified_raw:
            issues.append(f"{path}: '**Last verified:**' must not be empty.")
            continue

        try:
            last_verified = _parse_iso_date(last_verified_raw, field_name="Last verified", path=path)
        except ValueError as exc:
            issues.append(str(exc))
            continue

        if last_verified > today:
            issues.append(f"{path}: Last verified date cannot be in the future ({last_verified.isoformat()}).")
            continue

        change_issue = verification_change_issue(
            repo_root=repo_root,
            path=path,
            last_verified=last_verified,
            changed_doc_dates=changed_dates,
        )
        if change_issue is not None:
            issues.append(change_issue)

    return issues


def _find_index_metadata_issues(
    repo_root: Path,
    *,
    changed_doc_dates: Mapping[str, dt.date] | None = None,
) -> list[str]:
    return _find_owner_last_verified_metadata_issues(
        repo_root,
        relative_paths=INDEX_MARKDOWN_FILES,
        changed_doc_dates=changed_doc_dates,
    )


def _find_runbook_metadata_issues(
    repo_root: Path,
    *,
    changed_doc_dates: Mapping[str, dt.date] | None = None,
) -> list[str]:
    return _find_owner_last_verified_metadata_issues(
        repo_root,
        relative_paths=RUNBOOK_MARKDOWN_FILES,
        changed_doc_dates=changed_doc_dates,
    )


def _find_tool_docs_metadata_issues(
    repo_root: Path,
    *,
    changed_doc_dates: Mapping[str, dt.date] | None = None,
) -> list[str]:
    tool_docs = _collect_tool_docs_markdown_files(repo_root)
    return _find_owner_last_verified_metadata_issues_for_files(
        repo_root=repo_root,
        paths=tool_docs,
        changed_doc_dates=changed_doc_dates,
    )


def _find_owner_last_verified_metadata_issues(
    repo_root: Path,
    *,
    relative_paths: tuple[str, ...],
    changed_doc_dates: Mapping[str, dt.date] | None = None,
) -> list[str]:
    files = [repo_root / relative_path for relative_path in relative_paths]
    return _find_owner_last_verified_metadata_issues_for_files(
        repo_root=repo_root,
        paths=files,
        changed_doc_dates=changed_doc_dates,
    )


def _find_owner_last_verified_metadata_issues_for_files(
    *,
    repo_root: Path,
    paths: list[Path],
    changed_doc_dates: Mapping[str, dt.date] | None = None,
) -> list[str]:
    today = dt.date.today()
    changed_dates = changed_doc_dates or {}
    issues: list[str] = []

    for path in paths:
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")
        try:
            frontmatter = _load_markdown_frontmatter(text, path=path)
        except ValueError as exc:
            issues.append(str(exc))
            continue
        owner = _extract_doc_metadata_field(
            text,
            frontmatter=frontmatter,
            frontmatter_key="owner",
            body_pattern=OWNER_PATTERN,
        )
        owner_valid = True
        if owner is None:
            issues.append(f"{path}: missing '**Owner:**' metadata field.")
            owner_valid = False
        elif not owner:
            issues.append(f"{path}: '**Owner:**' must not be empty.")

        last_verified_raw = _extract_doc_metadata_field(
            text,
            frontmatter=frontmatter,
            frontmatter_key="last_verified",
            body_pattern=LAST_VERIFIED_PATTERN,
        )
        last_verified_valid = True
        if last_verified_raw is None:
            issues.append(f"{path}: missing '**Last verified:**' metadata field.")
            last_verified_valid = False
        elif not last_verified_raw:
            issues.append(f"{path}: '**Last verified:**' must not be empty.")
            last_verified_valid = False

        if not owner_valid or not last_verified_valid:
            continue

        try:
            last_verified = _parse_iso_date(last_verified_raw, field_name="Last verified", path=path)
        except ValueError as exc:
            issues.append(str(exc))
            continue

        if last_verified > today:
            issues.append(f"{path}: Last verified date cannot be in the future ({last_verified.isoformat()}).")
            continue

        change_issue = verification_change_issue(
            repo_root=repo_root,
            path=path,
            last_verified=last_verified,
            changed_doc_dates=changed_dates,
        )
        if change_issue is not None:
            issues.append(change_issue)

    return issues


def _find_cross_tool_doc_metadata_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    registry_ids_by_path: dict[str, str] = {}
    for relative_path, contract in CROSS_TOOL_DOC_METADATA_CONTRACTS.items():
        path = repo_root / relative_path
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")

        doc_type = _extract_metadata_field(text, TYPE_PATTERN)
        if doc_type is None:
            issues.append(f"{path}: missing '**Type:**' metadata field.")
        elif doc_type not in _CROSS_TOOL_DOC_ALLOWED_TYPES:
            allowed = ", ".join(sorted(_CROSS_TOOL_DOC_ALLOWED_TYPES))
            issues.append(f"{path}: '**Type:**' must be one of: {allowed}.")
        elif doc_type != contract["type"]:
            issues.append(f"{path}: '**Type:**' must be exactly '{contract['type']}'.")

        plane = _extract_metadata_field(text, PLANE_PATTERN)
        if plane is None:
            issues.append(f"{path}: missing '**Plane:**' metadata field.")
        elif plane not in _CROSS_TOOL_DOC_ALLOWED_PLANES:
            allowed = ", ".join(sorted(_CROSS_TOOL_DOC_ALLOWED_PLANES))
            issues.append(f"{path}: '**Plane:**' must be one of: {allowed}.")
        elif plane != contract["plane"]:
            issues.append(f"{path}: '**Plane:**' must be exactly '{contract['plane']}'.")

        owner_boundary = _extract_metadata_field(text, OWNER_BOUNDARY_PATTERN)
        if owner_boundary is None:
            issues.append(f"{path}: missing '**Owner-boundary:**' metadata field.")
        elif not owner_boundary:
            issues.append(f"{path}: '**Owner-boundary:**' must not be empty.")
        elif owner_boundary != contract["owner_boundary"]:
            issues.append(f"{path}: '**Owner-boundary:**' must be exactly '{contract['owner_boundary']}'.")

        entry_artifact = _extract_metadata_field(text, ENTRY_ARTIFACT_PATTERN)
        if entry_artifact is None:
            issues.append(f"{path}: missing '**Entry artifact:**' metadata field.")
        elif not entry_artifact:
            issues.append(f"{path}: '**Entry artifact:**' must not be empty.")

        exit_artifact = _extract_metadata_field(text, EXIT_ARTIFACT_PATTERN)
        if exit_artifact is None:
            issues.append(f"{path}: missing '**Exit artifact:**' metadata field.")
        elif not exit_artifact:
            issues.append(f"{path}: '**Exit artifact:**' must not be empty.")

        if contract["type"] not in _RUNBOOK_CATALOG_METADATA_TYPES:
            continue

        registry_id = _extract_metadata_field(text, REGISTRY_ID_METADATA_PATTERN)
        if registry_id is None:
            issues.append(f"{path}: missing '**Registry-id:**' metadata field.")
        elif not registry_id:
            issues.append(f"{path}: '**Registry-id:**' must not be empty.")
        elif _REGISTRY_ID_VALUE_PATTERN.fullmatch(registry_id) is None:
            issues.append(f"{path}: '**Registry-id:**' must be dot-qualified lowercase tokens.")
        else:
            registry_ids_by_path[relative_path] = registry_id

        summary = _extract_metadata_field(text, SUMMARY_METADATA_PATTERN)
        if summary is None:
            issues.append(f"{path}: missing '**Summary:**' metadata field.")
        elif not summary:
            issues.append(f"{path}: '**Summary:**' must not be empty.")

        execution_kind = _extract_metadata_field(text, EXECUTION_KIND_METADATA_PATTERN)
        if execution_kind is None:
            issues.append(f"{path}: missing '**Execution-kind:**' metadata field.")
        elif not execution_kind:
            issues.append(f"{path}: '**Execution-kind:**' must not be empty.")
        elif _METADATA_TOKEN_VALUE_PATTERN.fullmatch(execution_kind) is None:
            issues.append(f"{path}: '**Execution-kind:**' must use lowercase slug tokens.")

        status_kind = _extract_metadata_field(text, STATUS_KIND_METADATA_PATTERN)
        if status_kind is None:
            issues.append(f"{path}: missing '**Status-kind:**' metadata field.")
        elif not status_kind:
            issues.append(f"{path}: '**Status-kind:**' must not be empty.")
        elif _METADATA_TOKEN_VALUE_PATTERN.fullmatch(status_kind) is None:
            issues.append(f"{path}: '**Status-kind:**' must use lowercase slug tokens.")

    seen_registry_ids: dict[str, str] = {}
    for relative_path, registry_id in registry_ids_by_path.items():
        existing_path = seen_registry_ids.get(registry_id)
        if existing_path is not None:
            issues.append(
                "cross-tool docs must not reuse registry ids: "
                f"{registry_id} appears in both {existing_path} and {relative_path}."
            )
            continue
        seen_registry_ids[registry_id] = relative_path

    return issues


def _find_runbook_catalog_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    resolved_repo_root = repo_root.resolve()
    catalog_paths = [
        repo_root / relative_path
        for relative_path, contract in CROSS_TOOL_DOC_METADATA_CONTRACTS.items()
        if contract["type"] in _RUNBOOK_CATALOG_METADATA_TYPES and (repo_root / relative_path).exists()
    ]
    if not catalog_paths:
        return issues

    catalog_path = repo_root / RUNBOOK_CATALOG_DOC_PATH
    if not catalog_path.exists():
        return [f"{catalog_path}: missing runbook catalog."]

    catalog_text = catalog_path.read_text(encoding="utf-8")
    if "## Runbook Catalog" not in catalog_text:
        issues.append(f"{catalog_path}: missing '## Runbook Catalog' heading.")

    try:
        catalog = load_runbook_catalog(repo_root=repo_root)
    except ValueError as exc:
        issues.append(f"{catalog_path}: {exc}")
        return issues

    catalog_entries_by_path: dict[str, CatalogProcedureEntry] = {}
    catalog_status_kinds = {entry.status_kind for entry in catalog.procedures}
    registered_status_kinds = {spec.status_kind for spec in list_status_kind_specs_for_repo(repo_root)}
    expected_glossary_status_kinds = registered_status_kinds or catalog_status_kinds
    expected_catalog_paths: set[str] = set()
    for entry in catalog.procedures:
        resolved_path = resolve_catalog_doc_path(catalog_path=catalog.catalog_path, doc_path=entry.doc_path)
        try:
            relative_path = str(resolved_path.relative_to(resolved_repo_root))
        except ValueError:
            issues.append(
                f"{catalog_path}: registry id '{entry.registry_id}' resolves outside the repository: {entry.doc_path}."
            )
            continue
        existing_entry = catalog_entries_by_path.get(relative_path)
        if existing_entry is not None:
            issues.append(
                f"{catalog_path}: duplicate catalog procedure entry for {relative_path} "
                f"({existing_entry.registry_id}, {entry.registry_id})."
            )
            continue
        catalog_entries_by_path[relative_path] = entry

    procedure_section = _extract_markdown_section(catalog_text, heading="### Cross-tool procedures")
    if procedure_section is None:
        issues.append(f"{catalog_path}: missing '### Cross-tool procedures' section.")
    else:
        expected_section = render_catalog_procedure_section(catalog)
        if procedure_section.strip() != expected_section.strip():
            issues.append(
                f"{catalog_path}: cross-tool procedures section is stale; "
                "regenerate it with `uv run python -m dnadesign.devtools.docs.runbook_catalog`."
            )

    tool_source_section = _extract_markdown_section(catalog_text, heading="### Tool docs")
    if tool_source_section is None:
        issues.append(f"{catalog_path}: missing '### Tool docs' section.")
    else:
        expected_tool_source_section = render_catalog_tool_source_section(catalog)
        if tool_source_section.strip() != expected_tool_source_section.strip():
            issues.append(
                f"{catalog_path}: tool docs section is stale; "
                "regenerate it with `uv run python -m dnadesign.devtools.docs.runbook_catalog`."
            )

    for relative_path, contract in CROSS_TOOL_DOC_METADATA_CONTRACTS.items():
        if contract["type"] not in _RUNBOOK_CATALOG_METADATA_TYPES:
            continue
        path = repo_root / relative_path
        if not path.exists():
            continue
        expected_catalog_paths.add(relative_path)
        text = path.read_text(encoding="utf-8")
        registry_id = _extract_metadata_field(text, REGISTRY_ID_METADATA_PATTERN)
        if not registry_id:
            continue
        metadata_relative_path = resolve_registry_metadata_path_for_doc_path(relative_path)
        metadata_path = repo_root / metadata_relative_path
        if not metadata_path.exists():
            issues.append(f"{metadata_path}: missing registry metadata sidecar for {relative_path}.")
            continue
        metadata_payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        if not isinstance(metadata_payload, dict):
            issues.append(f"{metadata_path}: registry metadata must be a mapping.")
            continue

        entry = catalog_entries_by_path.get(relative_path)
        if entry is None:
            issues.append(f"{catalog_path}: missing registry id '{registry_id}' for {relative_path}.")
            continue

        expected_metadata = {
            "Registry-id": registry_id,
            "Type": _extract_metadata_field(text, TYPE_PATTERN),
            "Plane": _extract_metadata_field(text, PLANE_PATTERN),
            "Owner-boundary": _extract_metadata_field(text, OWNER_BOUNDARY_PATTERN),
            "Entry artifact": _extract_metadata_field(text, ENTRY_ARTIFACT_PATTERN),
            "Exit artifact": _extract_metadata_field(text, EXIT_ARTIFACT_PATTERN),
            "Execution-kind": _extract_metadata_field(text, EXECUTION_KIND_METADATA_PATTERN),
            "Status-kind": _extract_metadata_field(text, STATUS_KIND_METADATA_PATTERN),
            "Summary": _extract_metadata_field(text, SUMMARY_METADATA_PATTERN),
        }
        metadata_file_values = {
            "Registry-id": metadata_payload.get("registry_id"),
            "Type": metadata_payload.get("type"),
            "Plane": metadata_payload.get("plane"),
            "Owner-boundary": metadata_payload.get("owner_boundary"),
            "Entry artifact": metadata_payload.get("entry_artifact"),
            "Exit artifact": metadata_payload.get("exit_artifact"),
            "Execution-kind": metadata_payload.get("execution_kind"),
            "Status-kind": metadata_payload.get("status_kind"),
            "Summary": metadata_payload.get("summary"),
        }
        for field_name, expected_value in expected_metadata.items():
            if not expected_value:
                continue
            actual_value = metadata_file_values[field_name]
            if actual_value != expected_value:
                issues.append(
                    f"{metadata_path}: {field_name} for {relative_path} must match owner-local metadata "
                    f"(metadata={actual_value!r}, doc={expected_value!r})."
                )

    for relative_path, entry in sorted(catalog_entries_by_path.items()):
        if relative_path not in expected_catalog_paths:
            issues.append(
                f"{catalog_path}: unexpected catalog procedure '{entry.registry_id}' for {relative_path}; "
                "add a matching cross-tool metadata contract or remove the registry metadata sidecar."
            )

    glossary_text = _extract_markdown_section(catalog_text, heading=RUNBOOK_STATUS_GLOSSARY_HEADING)
    if glossary_text is None:
        issues.append(f"{catalog_path}: missing '{RUNBOOK_STATUS_GLOSSARY_HEADING}' section.")
        return issues

    glossary_status_kinds = {
        match.group("kind").strip()
        for line in glossary_text.splitlines()
        for match in [PROGRESS_SURFACE_GLOSSARY_ROW_PATTERN.match(line.strip())]
        if match is not None
    }
    if not glossary_status_kinds:
        issues.append(f"{catalog_path}: status surface glossary section has no data table.")
        return issues

    for status_kind in sorted(expected_glossary_status_kinds - glossary_status_kinds):
        issues.append(f"{catalog_path}: missing status surface glossary entry for '{status_kind}'.")
    for status_kind in sorted(glossary_status_kinds - expected_glossary_status_kinds):
        issues.append(f"{catalog_path}: unexpected status surface glossary entry for '{status_kind}'.")

    return issues


def _extract_markdown_section(text: str, *, heading: str) -> str | None:
    lines = text.splitlines()
    try:
        heading_index = next(index for index, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return None

    section_lines: list[str] = []
    for line in lines[heading_index + 1 :]:
        stripped = line.strip()
        if stripped.startswith("### "):
            break
        section_lines.append(line)
    return "\n".join(section_lines)


def _find_repo_local_skill_frontmatter_issues(repo_root: Path) -> list[str]:
    skills_root = repo_root / REPO_LOCAL_SKILLS_DIR
    if not skills_root.exists():
        return []

    issues: list[str] = []
    for skill_file in sorted(skills_root.glob("*/SKILL.md")):
        text = skill_file.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            issues.append(f"{skill_file}: missing YAML frontmatter.")
            continue
        try:
            raw_frontmatter = text.split("---", 2)[1]
            payload = yaml.safe_load(raw_frontmatter)
        except (IndexError, yaml.YAMLError) as exc:
            issues.append(f"{skill_file}: unable to parse YAML frontmatter ({exc}).")
            continue
        if not isinstance(payload, dict):
            issues.append(f"{skill_file}: YAML frontmatter must be a mapping.")
            continue

        description = payload.get("description")
        if not isinstance(description, str) or not description.strip():
            issues.append(f"{skill_file}: frontmatter description must be a non-empty string.")
            continue
        description_length = len(description)
        if description_length > REPO_LOCAL_SKILL_DESCRIPTION_MAX_CHARS:
            issues.append(
                f"{skill_file}: frontmatter description length {description_length}/"
                f"{REPO_LOCAL_SKILL_DESCRIPTION_MAX_CHARS}; keep repo-local skill discovery compact."
            )

    return issues


def _collect_markdown_reference_names(text: str) -> set[str]:
    references: set[str] = set()
    for code_span in re.findall(r"`([^`\n]+)`", text):
        normalized = code_span.strip()
        if normalized:
            references.add(normalized)
            references.add(Path(normalized).name)
    for raw in LINK_PATTERN.findall(text):
        link = raw.strip().split()[0]
        target_rel = link.split("#", 1)[0].strip()
        if target_rel:
            references.add(target_rel)
            references.add(Path(target_rel).name)
    return references


def _find_shared_usr_dataset_layout_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    datasets_root = repo_root / SHARED_USR_DATASETS_ROOT
    if not datasets_root.exists():
        return issues

    for records_path in sorted(datasets_root.rglob("records.parquet")):
        dataset_dir = records_path.parent
        relative_dataset = dataset_dir.relative_to(datasets_root).as_posix()
        parts = Path(relative_dataset).parts
        if not parts or parts[0] == "archived":
            continue
        if len(parts) > 1:
            issues.append(
                f"{dataset_dir}: nested shared USR dataset root {relative_dataset!r} is not allowed. "
                f"{SHARED_USR_DATASET_LAYOUT_NUDGE}"
            )

    return issues


def _find_exec_plan_metadata_issues(repo_root: Path) -> list[str]:
    issues: list[str] = []
    exec_root = repo_root / "docs" / "exec-plans"
    if not exec_root.exists():
        return issues

    for lane_name in ("active", "completed"):
        lane_root = exec_root / lane_name
        if not lane_root.exists():
            continue
        for plan_path in sorted(lane_root.rglob("*.md")):
            if plan_path.name == "README.md":
                continue

            text = plan_path.read_text(encoding="utf-8")
            present_sections = {heading.strip() for heading in SECTION_HEADING_PATTERN.findall(text)}
            section_bodies = _extract_section_bodies(text)

            status = _extract_metadata_field(text, STATUS_PATTERN)
            if status is None:
                issues.append(f"{plan_path}: missing '**Status:**' metadata field.")
                continue
            if status not in _EXEC_PLAN_STATUSES:
                allowed_statuses = ", ".join(sorted(_EXEC_PLAN_STATUSES))
                issues.append(f"{plan_path}: invalid status '{status}' (expected one of: {allowed_statuses}).")
                continue
            if lane_name == "completed" and status != "completed":
                issues.append(f"{plan_path}: plans under completed/ must set status to 'completed'.")
            if lane_name == "active" and status == "completed":
                issues.append(f"{plan_path}: plans under active/ cannot set status to 'completed'.")

            owner = _extract_metadata_field(text, OWNER_PATTERN)
            if owner is None:
                issues.append(f"{plan_path}: missing '**Owner:**' metadata field.")
            elif not owner:
                issues.append(f"{plan_path}: '**Owner:**' must not be empty.")

            created_raw = _extract_metadata_field(text, CREATED_PATTERN)
            if created_raw is None:
                issues.append(f"{plan_path}: missing '**Created:**' metadata field.")
            elif not created_raw:
                issues.append(f"{plan_path}: '**Created:**' must not be empty.")
            else:
                try:
                    _parse_iso_date(created_raw, field_name="Created", path=plan_path)
                except ValueError as exc:
                    issues.append(str(exc))

            if not LINK_PATTERN.search(text):
                issues.append(f"{plan_path}: execution plans must include at least one markdown link for traceability.")

            missing_sections = [name for name in _EXEC_PLAN_REQUIRED_SECTIONS if name not in present_sections]
            if missing_sections:
                missing_csv = ", ".join(missing_sections)
                issues.append(f"{plan_path}: missing required execution-plan sections: {missing_csv}.")
                continue

            progress_body = section_bodies.get("Progress", "")
            if not CHECKLIST_ITEM_PATTERN.search(progress_body):
                issues.append(f"{plan_path}: '## Progress' must include checklist items (e.g., '- [ ] ...').")
            progress_items = [line for line in progress_body.splitlines() if CHECKLIST_ITEM_PATTERN.match(line)]
            for progress_item in progress_items:
                if PROGRESS_ITEM_TIMESTAMP_PATTERN.match(progress_item):
                    continue
                issues.append(
                    f"{plan_path}: progress checklist items must include a UTC timestamp "
                    f"in '(YYYY-MM-DD HH:MMZ)' format."
                )
                break

            for section_name, body in section_bodies.items():
                if section_name == "Progress":
                    continue
                if CHECKLIST_ITEM_PATTERN.search(body):
                    issues.append(
                        f"{plan_path}: checklist items are only allowed under "
                        f"'## Progress' (found in '## {section_name}')."
                    )

    return issues


def _extract_section_bodies(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = SECTION_HEADING_PATTERN.match(line)
        if match is not None:
            current = match.group(1).strip()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines) for name, lines in sections.items()}
