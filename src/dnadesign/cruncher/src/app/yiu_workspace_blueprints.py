"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workspace_blueprints.py

Blueprint payloads for scaffolding payload-centric YIU workspaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

DEMO_NAME = "example_payload"
DEMO_SPEC_FILENAME = f"{DEMO_NAME}.yiu.yaml"
DEMO_SPEC_RELATIVE_PATH = f"configs/yiu/{DEMO_SPEC_FILENAME}"
ADVANCED_SPEC_FILENAME = f"{DEMO_NAME}.advanced_pwm.example.yaml"
ADVANCED_SPEC_RELATIVE_PATH = f"configs/yiu/{ADVANCED_SPEC_FILENAME}"
PWM_CONTEXT_FILENAME = "example_pwm_context.yaml"
PWM_CONTEXT_RELATIVE_PATH = f"motifs/{PWM_CONTEXT_FILENAME}"
DEMO_BUNDLE_DIR = f"outputs/{DEMO_NAME}"
DEMO_PUBLISHED_PLOT = f"outputs/{DEMO_NAME}__payload_views.pdf"


def canonical_spec_text() -> str:
    return dedent(
        f"""\
        yiu:
          contract: split_yiu_payload_rendering_v4
          schema_version: 1
          name: {DEMO_NAME}

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
          bundle_dir: {DEMO_BUNDLE_DIR}
          published_plot_path: {DEMO_PUBLISHED_PLOT}
          emit_render_jobs_debug: false
        """
    )


def advanced_pwm_spec_text() -> str:
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
              path: {PWM_CONTEXT_RELATIVE_PATH}
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


def example_pwm_context_text() -> str:
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
              source_ref: {PWM_CONTEXT_RELATIVE_PATH}
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
              source_ref: {PWM_CONTEXT_RELATIVE_PATH}
        """
    )


def runbook_payload(workspace_name: str) -> dict[str, object]:
    return {
        "runbook": {
            "schema_version": 1,
            "name": workspace_name,
            "steps": [
                {
                    "id": "yiu_validate",
                    "description": "Validate the minimal YIU v4 spec without writing bundle artifacts.",
                    "run": ["yiu", "validate", "--spec", DEMO_SPEC_RELATIVE_PATH],
                },
                {
                    "id": "yiu_render",
                    "description": "Publish the deterministic YIU v4 bundle and render the canonical views.",
                    "run": ["yiu", "render", "--spec", DEMO_SPEC_RELATIVE_PATH, "--force-overwrite", "--emit-renders"],
                },
                {
                    "id": "yiu_show",
                    "description": "Inspect the published bundle and run the v4 integrity checks.",
                    "run": ["yiu", "show", "--bundle", DEMO_BUNDLE_DIR],
                },
            ],
        }
    }


def runbook_markdown(*, workspace_name: str, workspace_display_path: Path | str) -> str:
    workspace_text = str(workspace_display_path)
    return "\n".join(
        [
            f"## {workspace_name} YIU Runbook",
            "",
            "**Workspace Path**",
            f"- {workspace_text}/",
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
            f"    cd {workspace_text}",
            '    cruncher() { uv run cruncher "$@"; }',
            "",
            "    cruncher yiu validate --spec configs/yiu/example_payload.yiu.yaml",
            "    cruncher yiu render --spec configs/yiu/example_payload.yiu.yaml --force-overwrite --emit-renders",
            "    cruncher yiu show --bundle outputs/example_payload",
            "",
            "### Advanced PWM-Aware Example",
            "",
            f"- Example spec template: `{ADVANCED_SPEC_RELATIVE_PATH}`",
            f"- Example PWM context: `{PWM_CONTEXT_RELATIVE_PATH}`",
            "- Copy the advanced example to a `.yiu.yaml` filename when you want to require PWM-aware optimization.",
        ]
    )
