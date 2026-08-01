"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_evidence/experiment_routes.py

Consume a bridge-owned Reader experiment route without copying its list.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

READER_EXPERIMENT_ROUTE_SCHEMA = "phd.retron_reader_experiment_routes.v2"

_ROOT_FIELDS = {"schema", "owner", "routes", "experiments", "memberships"}
_ROUTE_FIELDS = {"first_owner", "continue_with", "required_reader_state"}
_EXPERIMENT_FIELDS = {"experiment_id", "reader_config"}
_MEMBERSHIP_FIELDS = {"experiment_id", "route_id", "membership"}
_MEMBERSHIP_VALUES = {"selected", "related"}


class ReaderExperimentRouteError(ValueError):
    """Raised when a bridge experiment route cannot be consumed exactly."""


@dataclass(frozen=True)
class SelectedReaderExperiment:
    """One bridge-selected Reader experiment and its authored config path."""

    experiment_id: str
    reader_config: str


def selected_experiments_for_route(
    registry_path: Path,
    *,
    route_id: str,
) -> tuple[SelectedReaderExperiment, ...]:
    """Return exact selected experiment identities for one bridge route.

    The PhD bridge owns membership and live Reader readiness. This consumer
    validates the complete normalized registry before returning exact authored
    config paths. Related memberships never enter a study evidence selection.
    """

    path = Path(registry_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReaderExperimentRouteError(f"cannot read Reader experiment route registry {path}: {exc}") from exc
    root = _mapping(payload, label="registry")
    _require_exact_fields(root, _ROOT_FIELDS, label="registry")
    if root["schema"] != READER_EXPERIMENT_ROUTE_SCHEMA:
        raise ReaderExperimentRouteError(f"registry.schema must equal {READER_EXPERIMENT_ROUTE_SCHEMA!r}")
    if root["owner"] != "phd-workspace":
        raise ReaderExperimentRouteError("registry.owner must equal 'phd-workspace'")

    route = _text(route_id, label="route_id")
    routes = _mapping(root["routes"], label="registry.routes")
    if route not in routes:
        raise ReaderExperimentRouteError(f"registry does not declare route {route!r}")
    if not routes:
        raise ReaderExperimentRouteError("registry.routes must not be empty")
    for route_key, value in routes.items():
        route_name = _text(route_key, label="registry.routes key")
        route_entry = _mapping(value, label=f"registry.routes.{route_name}")
        _require_exact_fields(route_entry, _ROUTE_FIELDS, label=f"registry.routes.{route_name}")

    experiment_by_id: dict[str, SelectedReaderExperiment] = {}
    seen_configs: set[str] = set()
    for index, value in enumerate(_list(root["experiments"], label="registry.experiments")):
        entry = _mapping(value, label=f"registry.experiments[{index}]")
        _require_exact_fields(entry, _EXPERIMENT_FIELDS, label=f"registry.experiments[{index}]")
        experiment_id = _text(entry["experiment_id"], label=f"registry.experiments[{index}].experiment_id")
        reader_config = _text(entry["reader_config"], label=f"registry.experiments[{index}].reader_config")
        if experiment_id in experiment_by_id:
            raise ReaderExperimentRouteError(f"duplicate experiment_id {experiment_id!r}")
        if reader_config in seen_configs:
            raise ReaderExperimentRouteError(f"duplicate reader_config {reader_config!r}")
        experiment_by_id[experiment_id] = SelectedReaderExperiment(
            experiment_id=experiment_id,
            reader_config=reader_config,
        )
        seen_configs.add(reader_config)

    selected: list[SelectedReaderExperiment] = []
    membership_pairs: set[tuple[str, str]] = set()
    referenced_experiments: set[str] = set()
    for index, value in enumerate(_list(root["memberships"], label="registry.memberships")):
        label = f"registry.memberships[{index}]"
        entry = _mapping(value, label=label)
        _require_exact_fields(entry, _MEMBERSHIP_FIELDS, label=label)
        experiment_id = _text(entry["experiment_id"], label=f"{label}.experiment_id")
        membership_route = _text(entry["route_id"], label=f"{label}.route_id")
        membership = entry["membership"]
        if experiment_id not in experiment_by_id:
            raise ReaderExperimentRouteError(f"{label}.experiment_id references unknown experiment {experiment_id!r}")
        if membership_route not in routes:
            raise ReaderExperimentRouteError(f"{label}.route_id references unknown route {membership_route!r}")
        if membership not in _MEMBERSHIP_VALUES:
            raise ReaderExperimentRouteError(f"{label}.membership must be one of {sorted(_MEMBERSHIP_VALUES)}")
        pair = (experiment_id, membership_route)
        if pair in membership_pairs:
            raise ReaderExperimentRouteError(
                "duplicate experiment-route membership "
                f"for experiment_id={experiment_id!r}, route_id={membership_route!r}"
            )
        membership_pairs.add(pair)
        referenced_experiments.add(experiment_id)
        if membership_route == route and membership == "selected":
            selected.append(experiment_by_id[experiment_id])

    unreferenced = sorted(set(experiment_by_id) - referenced_experiments)
    if unreferenced:
        raise ReaderExperimentRouteError(
            "registry.experiments contains rows without route membership: " + ", ".join(unreferenced)
        )
    if not selected:
        raise ReaderExperimentRouteError(f"route {route!r} selects no Reader experiments")
    return tuple(selected)


def require_route_readiness(
    registry_path: Path,
    *,
    route_id: str,
    reader_root: Path,
) -> Mapping[str, object]:
    """Run the bridge-owned live gate for one exact route and return its receipt."""

    reader = Path(reader_root).expanduser().resolve()
    if reader.name != "reader" or not reader.is_dir():
        raise ReaderExperimentRouteError(f"reader_root must be the sibling Reader repository: {reader}")
    phd_root = reader.parent
    skill_root = (phd_root / ".agents/skills/retron-assay-study-bridge").resolve()
    _require_contained(skill_root, phd_root, label="canonical bridge skill root")
    registry = Path(registry_path).expanduser().resolve()
    canonical_registry = (skill_root / "references/reader-experiment-routes.json").resolve()
    _require_contained(canonical_registry, skill_root, label="canonical bridge registry")
    if registry != canonical_registry:
        raise ReaderExperimentRouteError(
            f"registry must equal canonical bridge registry {canonical_registry}; observed {registry}"
        )
    checker = (skill_root / "scripts/check_reader_experiment_readiness.py").resolve()
    _require_contained(checker, skill_root, label="bridge live-readiness checker")
    if not checker.is_file():
        raise ReaderExperimentRouteError(f"bridge live-readiness checker is missing: {checker}")
    command = [
        sys.executable,
        str(checker),
        "--registry",
        str(registry),
        "--phd-root",
        str(phd_root),
        "--route-id",
        _text(route_id, label="route_id"),
    ]
    repository_reader = reader / ".venv" / "bin" / "reader"
    if repository_reader.is_file():
        command.extend(("--reader-executable", str(repository_reader)))
    environment = os.environ.copy()
    environment.pop("__PYVENV_LAUNCHER__", None)
    completed = subprocess.run(
        command,
        cwd=phd_root,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    raw = completed.stdout.strip() or completed.stderr.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReaderExperimentRouteError(
            f"bridge live-readiness checker returned invalid JSON: {raw or '<empty>'}"
        ) from exc
    report = _mapping(payload, label="readiness receipt")
    if report.get("route_id") != route_id:
        raise ReaderExperimentRouteError("bridge readiness receipt route_id does not match the requested route")
    if completed.returncode != 0 or report.get("ok") is not True:
        blockers = report.get("selected_blockers")
        raise ReaderExperimentRouteError(
            f"Reader route {route_id!r} is not ready: {json.dumps(blockers, sort_keys=True)}"
        )
    return report


def _require_exact_fields(payload: Mapping[str, object], expected: set[str], *, label: str) -> None:
    observed = set(payload)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unknown:
            details.append("unknown=" + ", ".join(unknown))
        raise ReaderExperimentRouteError(f"{label} has invalid fields: {'; '.join(details)}")


def _require_contained(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReaderExperimentRouteError(f"{label} escapes {root}: {path}") from exc


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReaderExperimentRouteError(f"{label} must be an object")
    return value


def _list(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ReaderExperimentRouteError(f"{label} must be an array")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReaderExperimentRouteError(f"{label} must be a non-empty string")
    return value.strip()


__all__ = [
    "READER_EXPERIMENT_ROUTE_SCHEMA",
    "ReaderExperimentRouteError",
    "SelectedReaderExperiment",
    "require_route_readiness",
    "selected_experiments_for_route",
]
