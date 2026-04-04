"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workspace_service.py

Scaffold the payload-centric YIU workspace.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import yaml

_WORKSPACE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DEMO_NAME = "example_payload"
_DEMO_SPEC_FILENAME = f"{_DEMO_NAME}.yiu.yaml"
_DEMO_SPEC_RELATIVE_PATH = f"configs/yiu/{_DEMO_SPEC_FILENAME}"
_ADVANCED_SPEC_FILENAME = f"{_DEMO_NAME}.advanced_pwm.example.yaml"
_ADVANCED_SPEC_RELATIVE_PATH = f"configs/yiu/{_ADVANCED_SPEC_FILENAME}"
_PWM_CONTEXT_FILENAME = "example_pwm_context.yaml"
_PWM_CONTEXT_RELATIVE_PATH = f"motifs/{_PWM_CONTEXT_FILENAME}"
_DEMO_BUNDLE_DIR = f"outputs/{_DEMO_NAME}"
_DEMO_PUBLISHED_PLOT = f"outputs/{_DEMO_NAME}__payload_views.pdf"


@dataclass(frozen=True)
class YiuWorkspaceScaffoldResult:
    workspace_root: Path
    runbook_path: Path
    runbook_doc_path: Path
    spec_path: Path


def _workspace_gitignore_text() -> str:
    return "\n".join(
        [
            ".cruncher/",
            "outputs/",
            ".DS_Store",
            "",
        ]
    )


def _repo_root_from(start: Path) -> Path | None:
    cursor = start.resolve()
    for root in [cursor, *cursor.parents]:
        if (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    return None


def default_cruncher_workspaces_root() -> Path:
    repo_root = _repo_root_from(Path(__file__).resolve())
    if repo_root is None:
        raise ValueError(
            "Unable to determine the standard Cruncher workspaces root. Pass --root or --output explicitly."
        )
    return (repo_root / "src" / "dnadesign" / "cruncher" / "workspaces").resolve()


def _validate_workspace_name(name: str) -> str:
    raw = str(name).strip()
    if not raw:
        raise ValueError("YIU workspace name must be non-empty.")
    if "/" in raw or "\\" in raw:
        raise ValueError("YIU workspace name must be a simple directory name or use --output.")
    if _WORKSPACE_NAME_RE.fullmatch(raw) is None:
        raise ValueError(f"Invalid YIU workspace name: {raw!r}.")
    return raw


def yiu_workspace_path(name: str, *, root: Path | None = None) -> Path:
    workspace_name = _validate_workspace_name(name)
    parent = default_cruncher_workspaces_root() if root is None else Path(root).expanduser().resolve()
    return parent / workspace_name


def _canonical_spec_text() -> str:
    return dedent(
        f"""\
        yiu:
          contract: split_yiu_payload_rendering_v4
          schema_version: 1
          name: {_DEMO_NAME}

        input:
          kind: user_sequence
          user_sequence:
            sequence: AAATTTCCCGGGAAATTTCCC

        optimization:
          junction:
            mode: optimize
            overhang_length: 4
            max_payload_body_length: 12
          mismatches:
            count: 1
            candidate_positions: [1, 2]
            allowed_strands: [complement, payload]
            strand_mode: per_position
            default_strand_preference: complement
          pwm:
            mode: none
            source:
              kind: none
            objective:
              primary: maximin
              secondary:
                - total_loss
                - midpoint_proximity
                - body_length_balance
                - terminal_position_avoidance
                - default_strand_preference
                - lexical_stability

        output:
          bundle_dir: {_DEMO_BUNDLE_DIR}
          published_plot_path: {_DEMO_PUBLISHED_PLOT}
          emit_render_jobs_debug: false
        """
    )


def _advanced_pwm_spec_text() -> str:
    return dedent(
        f"""\
        # Advanced example: file-backed PWM context with overlapping motifs on both strands.
        # Copy this file to a `.yiu.yaml` name when you want to run it.
        yiu:
          contract: split_yiu_payload_rendering_v4
          schema_version: 1
          name: example_payload_pwm

        input:
          kind: user_sequence
          user_sequence:
            sequence: AAATTTCCCGGGAAATTTCCC

        optimization:
          junction:
            mode: optimize
            overhang_length: 4
            max_payload_body_length: 12
          mismatches:
            count: 2
            candidate_positions: [1, 2]
            allowed_strands: [complement, payload]
            strand_mode: per_position
            default_strand_preference: complement
          pwm:
            mode: require
            source:
              kind: file
              path: {_PWM_CONTEXT_RELATIVE_PATH}
            objective:
              primary: maximin
              secondary:
                - total_loss
                - midpoint_proximity
                - body_length_balance
                - terminal_position_avoidance
                - default_strand_preference
                - lexical_stability

        output:
          bundle_dir: outputs/example_payload_pwm
          published_plot_path: outputs/example_payload_pwm__payload_views.pdf
          emit_render_jobs_debug: false
        """
    )


def _example_pwm_context_text() -> str:
    return dedent(
        f"""\
        contract: yiu_pwm_context_v1
        schema_version: 1
        name: example_pwm_context
        motifs:
          - motif_instance_id: motif_plus_example
            tf_name: EXAMPLE_TF_PLUS
            motif_name: example_plus
            reference_strand: "+"
            start: 7
            end: 11
            probabilities:
              alphabet: [A, C, G, T]
              rows:
                - [0.70, 0.10, 0.10, 0.10]
                - [0.10, 0.70, 0.10, 0.10]
                - [0.10, 0.10, 0.70, 0.10]
                - [0.10, 0.10, 0.10, 0.70]
            provenance:
              source_kind: file
              source_ref: {_PWM_CONTEXT_RELATIVE_PATH}
          - motif_instance_id: motif_minus_example
            tf_name: EXAMPLE_TF_MINUS
            motif_name: example_minus
            reference_strand: "-"
            start: 8
            end: 12
            probabilities:
              alphabet: [A, C, G, T]
              rows:
                - [0.55, 0.15, 0.15, 0.15]
                - [0.15, 0.55, 0.15, 0.15]
                - [0.15, 0.15, 0.55, 0.15]
                - [0.15, 0.15, 0.15, 0.55]
            provenance:
              source_kind: file
              source_ref: {_PWM_CONTEXT_RELATIVE_PATH}
        """
    )


def _runbook_payload(workspace_name: str) -> dict[str, object]:
    return {
        "runbook": {
            "schema_version": 1,
            "name": workspace_name,
            "steps": [
                {
                    "id": "yiu_validate",
                    "description": "Validate the minimal YIU v4 spec without writing bundle artifacts.",
                    "run": ["yiu", "validate", "--spec", _DEMO_SPEC_RELATIVE_PATH],
                },
                {
                    "id": "yiu_render",
                    "description": "Publish the deterministic YIU v4 bundle and render the canonical views.",
                    "run": ["yiu", "render", "--spec", _DEMO_SPEC_RELATIVE_PATH, "--force-overwrite", "--emit-renders"],
                },
                {
                    "id": "yiu_show",
                    "description": "Inspect the published bundle and run the v4 integrity checks.",
                    "run": ["yiu", "show", "--bundle", _DEMO_BUNDLE_DIR],
                },
            ],
        }
    }


def _runbook_markdown(workspace_root: Path, *, workspace_name: str) -> str:
    return "\n".join(
        [
            f"## {workspace_name} YIU Runbook",
            "",
            "**Workspace Path**",
            f"- {workspace_root.relative_to(_repo_root_from(workspace_root) or workspace_root.parent)}/",
            "",
            "**Purpose**",
            "- YIU workspace for the v4 payload-centric optimization and rendering workflow.",
            "- Covers the validate -> render -> show loop with one minimal no-PWM example.",
            "",
            "**Run This Single Command**",
            "",
            f"    uv run cruncher workspaces run --workspace {workspace_name} --runbook configs/runbook.yaml",
            "",
            "### Step-by-Step Commands",
            "",
            "    set -euo pipefail",
            f"    cd {workspace_root.relative_to(_repo_root_from(workspace_root) or workspace_root.parent)}",
            '    cruncher() { uv run cruncher "$@"; }',
            "",
            "    cruncher yiu validate --spec configs/yiu/example_payload.yiu.yaml",
            "    cruncher yiu render --spec configs/yiu/example_payload.yiu.yaml --force-overwrite --emit-renders",
            "    cruncher yiu show --bundle outputs/example_payload",
            "",
            "### Advanced PWM-Aware Example",
            "",
            f"- Example spec template: `{_ADVANCED_SPEC_RELATIVE_PATH}`",
            f"- Example PWM context: `{_PWM_CONTEXT_RELATIVE_PATH}`",
            "- Copy the advanced example to a `.yiu.yaml` filename when you want to require PWM-aware optimization.",
        ]
    )


def init_yiu_workspace(workspace_root: Path, *, force_overwrite: bool = False) -> YiuWorkspaceScaffoldResult:
    resolved_root = Path(workspace_root).expanduser().resolve()
    workspace_name = resolved_root.name
    if resolved_root.exists():
        if not force_overwrite:
            raise ValueError(f"YIU workspace already exists: {resolved_root}")
        shutil.rmtree(resolved_root)
    (resolved_root / "configs" / "yiu").mkdir(parents=True, exist_ok=True)
    (resolved_root / "outputs").mkdir(parents=True, exist_ok=True)
    (resolved_root / "motifs").mkdir(parents=True, exist_ok=True)

    spec_path = resolved_root / "configs" / "yiu" / _DEMO_SPEC_FILENAME
    spec_path.write_text(_canonical_spec_text(), encoding="utf-8")

    advanced_spec_path = resolved_root / "configs" / "yiu" / _ADVANCED_SPEC_FILENAME
    advanced_spec_path.write_text(_advanced_pwm_spec_text(), encoding="utf-8")

    pwm_context_path = resolved_root / "motifs" / _PWM_CONTEXT_FILENAME
    pwm_context_path.write_text(_example_pwm_context_text(), encoding="utf-8")

    gitignore_path = resolved_root / ".gitignore"
    gitignore_path.write_text(_workspace_gitignore_text(), encoding="utf-8")

    runbook_path = resolved_root / "configs" / "runbook.yaml"
    runbook_path.write_text(yaml.safe_dump(_runbook_payload(workspace_name), sort_keys=False), encoding="utf-8")

    runbook_doc_path = resolved_root / "runbook.md"
    runbook_doc_path.write_text(_runbook_markdown(resolved_root, workspace_name=workspace_name), encoding="utf-8")

    return YiuWorkspaceScaffoldResult(
        workspace_root=resolved_root,
        runbook_path=runbook_path,
        runbook_doc_path=runbook_doc_path,
        spec_path=spec_path,
    )
