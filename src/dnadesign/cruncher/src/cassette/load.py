"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/load.py

Load cassette specs and resolve workspace-relative paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.cassette.errors import CassetteSpecError
from dnadesign.cruncher.cassette.models import HairpinCassetteSpec, HairpinCassetteSpecDocument
from dnadesign.cruncher.cassette.solve_models import HairpinCassetteSolveSpec, HairpinCassetteSolveSpecDocument


def _resolve_workspace_root_for_suffix(spec_path: Path, *, suffix: str, help_message: str) -> Path:
    resolved = spec_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Cassette spec not found: {resolved}")
    if not resolved.name.endswith(suffix):
        raise CassetteSpecError(help_message)
    for parent in resolved.parents:
        if parent.name == "configs":
            return parent.parent.resolve()
    raise CassetteSpecError("--spec must live under a workspace configs/ tree.")


def resolve_workspace_root_for_spec(spec_path: Path) -> Path:
    return _resolve_workspace_root_for_suffix(
        spec_path,
        suffix=".cassette.yaml",
        help_message="--spec must point to a <workspace>/configs/cassettes/<name>.cassette.yaml file.",
    )


def resolve_workspace_root_for_solve_spec(spec_path: Path) -> Path:
    return _resolve_workspace_root_for_suffix(
        spec_path,
        suffix=".cassette.solve.yaml",
        help_message="--spec must point to a <workspace>/configs/cassettes/<name>.cassette.solve.yaml file.",
    )


def resolve_workspace_relative_path(raw_path: Path, *, workspace_root: Path, label: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    if any(part == ".." for part in path.parts):
        raise CassetteSpecError(f"{label} must not traverse outside the workspace: {raw_path}")
    return (workspace_root / path).resolve()


def _expect_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise CassetteSpecError(f"{label} must be a mapping.")
    return dict(value)


def _load_yaml_mapping(spec_path: Path, *, top_level_label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise CassetteSpecError(f"Invalid YAML in {top_level_label} {spec_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CassetteSpecError(f"{top_level_label} {spec_path} must be a YAML mapping.")
    return payload


def _normalize_topology(topology: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(topology)
    mode = normalized.get("stem3p_arm_mode", "derived_reverse_complement")
    if mode in {"derive_reverse_complement", "derived_reverse_complement", "fixed"}:
        normalized["stem3p_arm_mode"] = "derived_reverse_complement"
    return normalized


def _normalize_construct_context(cassette_payload: dict[str, Any]) -> dict[str, Any]:
    has_new = "construct_context" in cassette_payload
    has_old = "duplex_context" in cassette_payload
    if has_new and has_old:
        raise CassetteSpecError("SCHEMA_ALIAS_CONFLICT: use only one of construct_context or duplex_context.")
    if has_new:
        return _expect_mapping(cassette_payload["construct_context"], label="cassette.construct_context")
    if not has_old:
        return {}
    duplex_context = _expect_mapping(cassette_payload["duplex_context"], label="cassette.duplex_context")
    normalized = {
        "left_flank": duplex_context.get("upstream", ""),
        "right_flank": duplex_context.get("downstream", ""),
    }
    if "left_flank" in duplex_context or "right_flank" in duplex_context:
        raise CassetteSpecError("SCHEMA_ALIAS_CONFLICT: duplex_context must use only upstream/downstream.")
    return normalized


def _normalize_target_strand(raw_value: Any) -> str:
    value = str(raw_value or "").strip()
    mapping = {
        "primary": "primary",
        "complement": "complement",
        "primary_strand": "primary",
        "complement_strand": "complement",
    }
    if value not in mapping:
        return value
    return mapping[value]


def _normalize_nicking_and_site_policy(cassette_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    nicking = _expect_mapping(cassette_payload.get("nicking"), label="cassette.nicking")
    has_new_target = "target_strand" in nicking
    has_old_target = "designated_strand" in nicking
    if has_new_target and has_old_target:
        raise CassetteSpecError(
            "SCHEMA_ALIAS_CONFLICT: use only one of nicking.target_strand or nicking.designated_strand."
        )
    if has_new_target:
        nicking["target_strand"] = _normalize_target_strand(nicking["target_strand"])
    elif has_old_target:
        nicking["target_strand"] = _normalize_target_strand(nicking.pop("designated_strand"))
    else:
        nicking["target_strand"] = "primary"

    raw_site_policy = _expect_mapping(cassette_payload.get("site_policy"), label="cassette.site_policy")
    if "forbid_additional_designated_strand_nicks" in nicking and raw_site_policy:
        raise CassetteSpecError(
            "SCHEMA_ALIAS_CONFLICT: use only one of site_policy.forbid_additional_designated_strand_nicks "
            "or nicking.forbid_additional_designated_strand_nicks."
        )
    if "forbid_additional_designated_strand_nicks" in nicking:
        site_policy = {
            "forbid_additional_designated_strand_nicks": nicking.pop("forbid_additional_designated_strand_nicks"),
            "scan_scope": "requested_variants",
        }
    else:
        site_policy = raw_site_policy
    return nicking, site_policy


def _normalize_cassette_document(payload: dict[str, Any]) -> dict[str, Any]:
    if "cassette" not in payload:
        raise CassetteSpecError("Cassette spec must define top-level key 'cassette'.")
    cassette = _expect_mapping(payload["cassette"], label="cassette")
    if "schema_version" not in cassette:
        raise CassetteSpecError("cassette.schema_version is required.")

    normalized = dict(cassette)
    normalized["topology"] = _normalize_topology(_expect_mapping(cassette.get("topology"), label="cassette.topology"))
    normalized["construct_context"] = _normalize_construct_context(cassette)
    normalized["nicking"], normalized["site_policy"] = _normalize_nicking_and_site_policy(cassette)
    normalized["hairpin_validation"] = _expect_mapping(
        cassette.get("hairpin_validation"), label="cassette.hairpin_validation"
    )
    normalized.pop("duplex_context", None)
    return {"cassette": normalized}


def _normalize_solve_document(payload: dict[str, Any]) -> dict[str, Any]:
    if "cassette_solve" not in payload:
        raise CassetteSpecError("Cassette solve spec must define top-level key 'cassette_solve'.")
    solve = _expect_mapping(payload["cassette_solve"], label="cassette_solve")
    if "schema_version" not in solve:
        raise CassetteSpecError("cassette_solve.schema_version is required.")
    normalized = dict(solve)
    normalized["topology"] = _expect_mapping(solve.get("topology"), label="cassette_solve.topology")
    normalized["construct_context"] = _expect_mapping(
        solve.get("construct_context"), label="cassette_solve.construct_context"
    )
    normalized["nick_goal"] = _expect_mapping(solve.get("nick_goal"), label="cassette_solve.nick_goal")
    normalized["assignment_policy"] = _expect_mapping(
        solve.get("assignment_policy"), label="cassette_solve.assignment_policy"
    )
    normalized["site_blacklist"] = _expect_mapping(solve.get("site_blacklist"), label="cassette_solve.site_blacklist")
    normalized["sequence_blacklist"] = _expect_mapping(
        solve.get("sequence_blacklist"), label="cassette_solve.sequence_blacklist"
    )
    normalized["sequence_quality"] = _expect_mapping(
        solve.get("sequence_quality"), label="cassette_solve.sequence_quality"
    )
    normalized["catalog"] = _expect_mapping(solve.get("catalog"), label="cassette_solve.catalog")
    normalized["search"] = _expect_mapping(solve.get("search"), label="cassette_solve.search")
    normalized["output"] = _expect_mapping(solve.get("output"), label="cassette_solve.output")
    return {"cassette_solve": normalized}


def load_cassette_spec(path: str | Path) -> tuple[HairpinCassetteSpec, Path, Path]:
    spec_path = Path(path).expanduser().resolve()
    workspace_root = resolve_workspace_root_for_spec(spec_path)
    payload = _load_yaml_mapping(spec_path, top_level_label="cassette spec")
    try:
        document = HairpinCassetteSpecDocument.model_validate(_normalize_cassette_document(payload))
    except CassetteSpecError:
        raise
    except Exception as exc:
        raise CassetteSpecError(f"Cassette schema validation failed for {spec_path}: {exc}") from exc
    return document.cassette, spec_path, workspace_root


def load_cassette_solve_spec(path: str | Path) -> tuple[HairpinCassetteSolveSpec, Path, Path]:
    spec_path = Path(path).expanduser().resolve()
    workspace_root = resolve_workspace_root_for_solve_spec(spec_path)
    payload = _load_yaml_mapping(spec_path, top_level_label="cassette solve spec")
    try:
        document = HairpinCassetteSolveSpecDocument.model_validate(_normalize_solve_document(payload))
    except CassetteSpecError:
        raise
    except Exception as exc:
        raise CassetteSpecError(f"Cassette solve schema validation failed for {spec_path}: {exc}") from exc
    return document.cassette_solve, spec_path, workspace_root
