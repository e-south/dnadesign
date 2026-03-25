"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/cassette_workspace_service.py

Scaffold cassette solve workspaces with deterministic runtime profiles.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json

_SCAFFOLD_WORKFLOW = "cassette_workspace_scaffold"
_SCAFFOLD_GENERATOR = "cruncher cassette init-workspace"
_MANIFEST_NAME = "cassette_workspace_manifest.json"
_WORKSPACE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class CassetteWorkspaceScaffoldResult:
    workspace_root: Path
    manifest_path: Path
    readme_path: Path
    runbook_path: Path
    runbook_doc_path: Path
    solve_specs: dict[str, Path]


@dataclass(frozen=True)
class CassetteSolveProfile:
    filename: str
    label: str
    description: str
    search: dict[str, Any]
    selection: dict[str, Any]

    def render_notes(self) -> list[str]:
        selection_bits = [self.selection["policy"], self.selection["distance_metric"]]
        if "diversity_weight" in self.selection:
            selection_bits.append(f"diversity_weight={self.selection['diversity_weight']}")
        return [
            f"{self.label}: {self.description}",
            (
                "  search: "
                f"max_hits={self.search['max_hits']}, "
                f"max_enumerated_candidates={self.search['max_enumerated_candidates']}, "
                f"max_search_nodes={self.search['max_search_nodes']}, "
                f"materialize_top_k={self.search['materialize_top_k']}"
            ),
            (
                "  selection: "
                f"{', '.join(selection_bits)}, "
                f"pool_size={self.selection['pool_size']}, "
                f"min_pairwise_distance={self.selection['min_pairwise_distance']}"
            ),
        ]


def _base_solve_spec() -> dict[str, Any]:
    return {
        "cassette_solve": {
            "schema_version": 1,
            "topology": {
                "stem5p_arm_pattern": "NNNNNCCTCAGC",
                "loop_pattern": "TTT",
            },
            "construct_context": {
                "left_flank": "",
                "right_flank": "",
                "evaluation_scope": "cassette_plus_flanks",
            },
            "nick_goal": {
                "target_strand": "primary",
                "left_nick_window": {"start": 0, "end": 0},
                "right_nick_window": {"start": 24, "end": 24},
                "bounded_segment_length": {"min": 24, "max": 24},
            },
            "assignment_policy": {
                "allowed_left_variant_ids": ["Nt.BbvCI"],
                "allowed_right_variant_ids": ["Nb.BbvCI"],
                "forbidden_intended_variant_ids": [],
                "forbidden_intended_specificity_ids": [],
                "allow_same_variant": True,
                "allow_same_specificity_opposite_variant": True,
            },
            "site_blacklist": {
                "forbidden_any_site_specificity_ids": [],
                "forbidden_unintended_site_specificity_ids": [],
                "forbidden_any_site_variant_ids": [],
                "scope": "evaluation_context",
            },
            "sequence_blacklist": {
                "forbidden_literals": [],
                "forbidden_iupac_motifs": [],
                "forbid_reverse_complements": True,
                "scope": "evaluation_context",
            },
            "sequence_quality": {
                "gc_fraction": {"min": 0.35, "max": 0.65},
                "max_homopolymer_run": 4,
            },
            "catalog": {"preset": "neb_nicking_v1", "additional_paths": []},
            "search": {
                "max_hits": 5,
                "max_enumerated_candidates": 256,
                "max_search_nodes": 50000,
                "bounded_segment_target": 24,
                "gc_target": 0.5,
                "materialize_top_k": 1,
                "selection": {
                    "policy": "greedy_hamming",
                    "pool_size": 24,
                    "distance_metric": "hamming",
                    "min_pairwise_distance": 2,
                },
            },
            "output": {
                "run_dir": "outputs/cassette_solves",
                "emit_visual_contracts": True,
                "emit_baserender_jobs": True,
                "baserender_profiles": [
                    "duplex_qa",
                    "hairpin_qa",
                    "top_hits_duplex_qa",
                    "top_hits_hairpin_qa",
                ],
            },
        }
    }


def _profiles() -> tuple[CassetteSolveProfile, ...]:
    return (
        CassetteSolveProfile(
            filename="demo_hairpin_fast.cassette.solve.yaml",
            label="fast",
            description="Minimal bounded search for quick contract checks and smoke runs.",
            search={
                "max_hits": 5,
                "max_enumerated_candidates": 256,
                "max_search_nodes": 50000,
                "materialize_top_k": 1,
            },
            selection={
                "policy": "greedy_hamming",
                "pool_size": 24,
                "distance_metric": "hamming",
                "min_pairwise_distance": 2,
            },
        ),
        CassetteSolveProfile(
            filename="demo_hairpin_balanced.cassette.solve.yaml",
            label="balanced",
            description="Moderate search budgets for everyday operator use.",
            search={
                "max_hits": 8,
                "max_enumerated_candidates": 1024,
                "max_search_nodes": 100000,
                "materialize_top_k": 2,
            },
            selection={
                "policy": "greedy_hamming",
                "pool_size": 64,
                "distance_metric": "hamming",
                "min_pairwise_distance": 2,
            },
        ),
        CassetteSolveProfile(
            filename="demo_hairpin_deep_mmr.cassette.solve.yaml",
            label="deep_mmr",
            description="Larger bounded pool with opt-in MMR for diversity-first hit selection.",
            search={
                "max_hits": 10,
                "max_enumerated_candidates": 4096,
                "max_search_nodes": 250000,
                "materialize_top_k": 3,
            },
            selection={
                "policy": "mmr",
                "pool_size": 128,
                "distance_metric": "hamming",
                "min_pairwise_distance": 2,
                "diversity_weight": 0.35,
            },
        ),
    )


def _profile_payloads() -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for profile in _profiles():
        payload = _base_solve_spec()
        payload["cassette_solve"]["search"].update(profile.search)
        payload["cassette_solve"]["search"]["selection"] = dict(profile.selection)
        payloads[profile.filename] = payload
    return payloads


def _manifest_path(workspace_root: Path) -> Path:
    return workspace_root / _MANIFEST_NAME


def _repo_root_from(start: Path) -> Path | None:
    try:
        cursor = start.resolve()
    except Exception:
        cursor = start
    for root in [cursor, *cursor.parents]:
        if (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    return None


def default_cassette_workspaces_root() -> Path:
    repo_root = _repo_root_from(Path(__file__).resolve())
    if repo_root is not None:
        return (repo_root / "src" / "dnadesign" / "cruncher" / "workspaces").resolve()
    raise ValueError("Unable to determine the standard Cruncher workspaces root. Pass --root or --output explicitly.")


def _validate_workspace_name(name: str) -> str:
    raw = str(name).strip()
    if raw == "":
        raise ValueError("Cassette workspace name must be non-empty.")
    if "/" in raw or "\\" in raw:
        raise ValueError(
            f"Invalid cassette workspace name: {raw!r}. "
            "Use a simple workspace name with --root, or pass --output for an explicit path."
        )
    if _WORKSPACE_NAME_RE.fullmatch(raw) is None:
        raise ValueError(f"Invalid cassette workspace name: {raw!r}.")
    return raw


def cassette_workspace_path(name: str, *, root: Path | None = None) -> Path:
    workspace_name = _validate_workspace_name(name)
    parent = default_cassette_workspaces_root() if root is None else Path(root).expanduser().resolve()
    return parent / workspace_name


def _assert_no_symlinked_path_segments(path: Path) -> None:
    current = Path(path.root) if path.is_absolute() else Path(".")
    skipped_root_alias_segment = False
    for part in path.parts:
        if part in {"", ".", path.root}:
            continue
        current = current / part
        if path.is_absolute() and not skipped_root_alias_segment:
            skipped_root_alias_segment = True
            continue
        if current.is_symlink():
            raise ValueError(
                "Cassette workspace output path must not traverse a symlinked directory: "
                f"{current} -> {current.resolve()}"
            )


def _is_scaffold_workspace(workspace_root: Path) -> bool:
    manifest_path = _manifest_path(workspace_root)
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("workflow") == _SCAFFOLD_WORKFLOW and payload.get("generator") == _SCAFFOLD_GENERATOR


def _prepare_workspace_root(workspace_root: Path, *, force_overwrite: bool) -> Path:
    expanded_root = workspace_root.expanduser()
    if expanded_root.is_symlink():
        raise ValueError(
            f"Cassette workspace output root must not be a symlink: {expanded_root} -> {expanded_root.resolve()}"
        )
    _assert_no_symlinked_path_segments(expanded_root.parent)
    resolved_root = expanded_root.resolve()
    if resolved_root.exists() and not resolved_root.is_dir():
        raise ValueError(f"Cassette workspace output must be a directory path: {resolved_root}")
    if resolved_root.exists() and any(resolved_root.iterdir()):
        if not force_overwrite:
            raise ValueError(
                f"Cassette workspace output already exists and is not empty: {resolved_root}. "
                "Use --force-overwrite only for a scaffold generated by this command."
            )
        if not _is_scaffold_workspace(resolved_root):
            raise ValueError(
                "Refusing to overwrite a non-empty workspace root that was not generated by "
                "`cruncher cassette init-workspace`."
            )
        shutil.rmtree(resolved_root)
    resolved_root.mkdir(parents=True, exist_ok=True)
    return resolved_root


def _write_readme(workspace_root: Path, *, solve_specs: dict[str, Path]) -> Path:
    lines = [
        "# Cassette Workspace",
        "",
        "This is a cassette-specific scaffold generated by `cruncher cassette init-workspace`.",
        "",
        "It stays cassette-specific while still behaving like a discoverable Cruncher workspace:",
        "",
        "- cassette commands operate on the explicit `configs/cassettes/*.cassette.solve.yaml` paths here",
        "- `configs/runbook.yaml` makes the scaffold appear as `runbook-only` under `cruncher workspaces list`",
        "- there is still no generic `configs/config.yaml` because cassette flows do not use the sampling schema",
        "- overwrite is refused unless this root is reinitialized explicitly with `--force-overwrite`",
        "- symlinked output roots or ancestor directories are rejected so the scaffold stays at the path you named",
        "",
        "## Solve profiles",
        "",
    ]
    profile_lookup = {profile.filename: profile for profile in _profiles()}
    for name, path in solve_specs.items():
        profile = profile_lookup[name]
        lines.extend(
            [
                f"- `{name}`",
                (f"  Run with: `uv run cruncher cassette solve --spec {path.relative_to(workspace_root).as_posix()}`"),
            ]
        )
        for note in profile.render_notes():
            lines.append(f"  {note}")
    lines.extend(
        [
            "",
            "Use the explicit `configs/cassettes/*.cassette.solve.yaml` paths in this root for balanced and deep runs.",
            "The shipped `configs/runbook.yaml` keeps the fast smoke path discoverable through `workspaces run`.",
            "Emitted baserender jobs stay self-contained inside the cassette workspace:",
            (
                "- solve-level jobs read `views/` and write PDFs to sibling `renders/` under "
                "`outputs/cassette_solves/<solve_id>/`"
            ),
            "- per-hit jobs do the same inside each `hits/hit_<rank>_<solution_id>/` bundle",
            "",
            "## Optional local catalog export",
            "",
            "```bash",
            "uv run cruncher cassette catalog init-neb --output inputs/nickases/neb_nicking_v1.yaml",
            "```",
            "",
        ]
    )
    readme_path = workspace_root / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


def _write_runbook_yaml(workspace_root: Path) -> Path:
    runbook_path = workspace_root / "configs" / "runbook.yaml"
    payload = {
        "runbook": {
            "schema_version": 1,
            "name": workspace_root.name,
            "steps": [
                {
                    "id": "cassette_solve_fast",
                    "description": "Run the scaffolded fast cassette solve profile.",
                    "run": ["cassette", "solve", "--spec", "configs/cassettes/demo_hairpin_fast.cassette.solve.yaml"],
                }
            ],
        }
    }
    runbook_path.parent.mkdir(parents=True, exist_ok=True)
    runbook_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return runbook_path


def _write_runbook_markdown(workspace_root: Path, *, runbook_path: Path, solve_specs: dict[str, Path]) -> Path:
    fast_spec = solve_specs["demo_hairpin_fast.cassette.solve.yaml"].relative_to(workspace_root).as_posix()
    balanced_spec = solve_specs["demo_hairpin_balanced.cassette.solve.yaml"].relative_to(workspace_root).as_posix()
    deep_spec = solve_specs["demo_hairpin_deep_mmr.cassette.solve.yaml"].relative_to(workspace_root).as_posix()
    workspace_name = workspace_root.name
    parent_arg = shlex.quote(workspace_root.parent.as_posix())
    runbook_arg = shlex.quote(runbook_path.relative_to(workspace_root).as_posix())
    fast_spec_arg = shlex.quote(fast_spec)
    balanced_spec_arg = shlex.quote(balanced_spec)
    deep_spec_arg = shlex.quote(deep_spec)
    lines = [
        f"## {workspace_name} Cassette Runbook",
        "",
        "**Workspace Path**",
        f"- {workspace_root.as_posix()}/",
        "",
        "**Purpose**",
        "- Scaffolded cassette solve workspace with built-in fast, balanced, and deep MMR profiles.",
        "- Discoverable by `cruncher workspaces list` as `runbook-only`.",
        "",
        "**Run This Fast Smoke Step**",
        "",
        (
            "    uv run cruncher workspaces run "
            f"--workspace {workspace_name} "
            f"--runbook {runbook_arg} "
            "--step cassette_solve_fast"
        ),
        "",
        "**Direct Cassette Commands**",
        "",
        f"    uv run cruncher workspaces list --root {parent_arg}",
        f"    uv run cruncher cassette solve --spec {fast_spec_arg}",
        f"    uv run cruncher cassette solve --spec {balanced_spec_arg}",
        f"    uv run cruncher cassette solve --spec {deep_spec_arg}",
        "",
    ]
    runbook_doc_path = workspace_root / "runbook.md"
    runbook_doc_path.write_text("\n".join(lines), encoding="utf-8")
    return runbook_doc_path


def init_cassette_workspace(path: str | Path, *, force_overwrite: bool = False) -> CassetteWorkspaceScaffoldResult:
    workspace_root = _prepare_workspace_root(Path(path), force_overwrite=force_overwrite)
    spec_root = workspace_root / "configs" / "cassettes"
    inputs_root = workspace_root / "inputs" / "nickases"
    outputs_root = workspace_root / "outputs"
    spec_root.mkdir(parents=True, exist_ok=True)
    inputs_root.mkdir(parents=True, exist_ok=True)
    (outputs_root / "cassettes").mkdir(parents=True, exist_ok=True)
    (outputs_root / "cassette_solves").mkdir(parents=True, exist_ok=True)

    solve_specs: dict[str, Path] = {}
    for filename, payload in _profile_payloads().items():
        spec_path = spec_root / filename
        spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        solve_specs[filename] = spec_path

    (inputs_root / "README.md").write_text(
        "\n".join(
            [
                "# Nickase overlays",
                "",
                "Optional cassette catalog overlays live here.",
                "Use `cruncher cassette catalog init-neb --output inputs/nickases/neb_nicking_v1.yaml`",
                "if you want a local copy of the built-in preset for editing.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    runbook_path = _write_runbook_yaml(workspace_root)
    runbook_doc_path = _write_runbook_markdown(workspace_root, runbook_path=runbook_path, solve_specs=solve_specs)
    readme_path = _write_readme(workspace_root, solve_specs=solve_specs)
    manifest = {
        "schema_version": 1,
        "workflow": _SCAFFOLD_WORKFLOW,
        "generator": _SCAFFOLD_GENERATOR,
        "workspace_root": str(workspace_root),
        "workspace_kind": "runbook-only",
        "runbook_path": str(runbook_path.relative_to(workspace_root)),
        "runbook_doc_path": str(runbook_doc_path.relative_to(workspace_root)),
        "solve_specs": {name: str(path.relative_to(workspace_root)) for name, path in sorted(solve_specs.items())},
        "profiles": [
            {
                "filename": profile.filename,
                "label": profile.label,
                "description": profile.description,
                "search": profile.search,
                "selection": profile.selection,
            }
            for profile in _profiles()
        ],
    }
    manifest_path = _manifest_path(workspace_root)
    atomic_write_json(manifest_path, manifest)
    return CassetteWorkspaceScaffoldResult(
        workspace_root=workspace_root,
        manifest_path=manifest_path,
        readme_path=readme_path,
        runbook_path=runbook_path,
        runbook_doc_path=runbook_doc_path,
        solve_specs=solve_specs,
    )
