"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_yiu_cli.py

CLI contracts for the payload-centric YIU surface.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _payload_spec(
    *,
    sequence: str = "AAATTTCCCGGGAAATTTCCC",
    name: str = "demo_payload",
    candidate_positions: list[int] | None = None,
    mismatch_count: int = 1,
) -> dict[str, object]:
    return {
        "yiu": {
            "schema_version": 1,
            "contract": "split_yiu_payload_rendering_v4",
            "name": name,
        },
        "input": {
            "kind": "user_sequence",
            "user_sequence": {
                "sequence": sequence,
            },
        },
        "optimization": {
            "junction": {
                "mode": "explicit_window",
                "start": 4,
                "end": 8,
                "overhang_length": 4,
                "max_payload_body_length": 12,
            },
            "mismatches": {
                "count": mismatch_count,
                "candidate_positions": candidate_positions or [1, 2],
                "allowed_strands": ["complement", "payload"],
                "strand_mode": "per_position",
                "default_strand_preference": "complement",
            },
            "pwm": {
                "mode": "none",
                "source": {"kind": "none"},
                "objective": {
                    "primary": "maximin",
                    "secondary": [
                        "total_loss",
                        "midpoint_proximity",
                        "body_length_balance",
                        "terminal_position_avoidance",
                        "default_strand_preference",
                        "lexical_stability",
                    ],
                },
            },
        },
        "output": {
            "bundle_dir": f"bundles/{name}",
            "emit_render_jobs_debug": False,
        },
    }


def _legacy_v1_payload_spec() -> dict[str, object]:
    return {
        "yiu": {
            "schema_version": 1,
            "contract": "split_yiu_payload_rendering_v1",
            "name": "legacy_payload",
        },
        "input": {
            "kind": "user_sequence",
            "user_sequence": {"sequence": "AAATTTCCCGGG"},
        },
        "split": {"mode": "derived"},
        "bulge_mask": {"positions": [1]},
        "output": {
            "bundle_dir": "bundles/legacy_payload",
            "emit_render_jobs_debug": False,
        },
    }


def test_root_help_includes_yiu_group() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "yiu" in result.output


def test_yiu_help_lists_payload_commands_only() -> None:
    result = runner.invoke(app, ["yiu", "--help"])

    assert result.exit_code == 0
    assert "init-workspace" in result.output
    assert "validate" in result.output
    assert "render" in result.output
    assert "show" in result.output
    assert "trace" not in result.output
    assert "solve" not in result.output


def test_yiu_show_help_uses_bundle_language() -> None:
    result = runner.invoke(app, ["yiu", "show", "--help"])

    assert result.exit_code == 0
    assert "--bundle" in result.output
    assert "--run" not in result.output


def test_yiu_init_workspace_scaffolds_only_payload_config(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"

    result = runner.invoke(app, ["yiu", "init-workspace", "--output", str(workspace)])

    assert result.exit_code == 0
    spec_path = workspace / "configs" / "yiu" / "example_payload.yiu.yaml"
    assert spec_path.exists()
    assert not (workspace / "configs" / "yiu" / "example_payload.yiu.solve.yaml").exists()
    assert not (workspace / "catalogs").exists()

    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    assert payload["yiu"]["contract"] == "split_yiu_payload_rendering_v4"
    assert payload["input"]["kind"] in {"user_sequence", "sample_hit"}
    assert payload["optimization"]["junction"]["overhang_length"] == 4
    assert payload["optimization"]["mismatches"]["strand_mode"] == "per_position"
    assert payload["output"]["bundle_dir"].startswith("bundles/")
    assert "/yiu/" not in payload["output"]["bundle_dir"]
    assert (workspace / "configs" / "yiu" / "example_payload.advanced_pwm.example.yaml").exists()
    assert (workspace / "motifs" / "example_pwm_context.yaml").exists()
    assert (workspace / ".gitignore").read_text(encoding="utf-8").splitlines() == [
        ".cruncher/",
        "bundles/",
        "outputs/",
        ".DS_Store",
    ]


def test_yiu_validate_fails_fast_on_legacy_v1_spec(tmp_path: Path) -> None:
    spec_path = tmp_path / "demo_yiu_payload" / "configs" / "yiu" / "legacy.yiu.yaml"
    _write_yaml(spec_path, _legacy_v1_payload_spec())

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)])

    assert result.exit_code == 1
    assert "YIU_CONTRACT_UNKNOWN" in result.output


def test_yiu_render_materializes_payload_bundle_from_spec(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _payload_spec())

    result = runner.invoke(app, ["yiu", "render", "--spec", str(spec_path)])

    assert result.exit_code == 0
    bundle_dir = workspace / "bundles" / "demo_payload"
    assert (bundle_dir / "bundle_manifest.json").exists()
    assert (bundle_dir / "normalized_payload.json").exists()
    assert (bundle_dir / "visual_inventory.json").exists()
    assert (bundle_dir / "payload_view.json").exists()
    assert "Bundle manifest ->" in result.output
    assert "Bundle write ->" in result.output


def test_yiu_validate_reports_junction_window_summary(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _payload_spec())

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)])

    assert result.exit_code == 0
    assert "Junction window -> start=4 end=8 mode=explicit_window" in result.output
    assert "Mismatch count -> 1" in result.output
    assert "PWM -> mode=none effective=False" in result.output
    assert "Bundle write -> no" in result.output
    assert "Bulge mask ->" not in result.output
    assert "Watson-Crick pairing valid ->" not in result.output


def test_yiu_show_reports_payload_bundle_summary(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _payload_spec())

    render_result = runner.invoke(app, ["yiu", "render", "--spec", str(spec_path)])
    assert render_result.exit_code == 0

    show_result = runner.invoke(app, ["yiu", "show", "--bundle", str(workspace / "bundles" / "demo_payload")])

    assert show_result.exit_code == 0
    assert "Bundle ->" in show_result.output
    assert "demo_payload" in show_result.output
    assert "Bundle contract -> split_yiu_payload_bundle_v4" in show_result.output
    assert "Input kind -> user_sequence" in show_result.output
    assert "Payload length -> 21" in show_result.output
    assert "Junction window -> start=4 end=8 mode=explicit_window" in show_result.output
    assert "Mismatch count -> 1" in show_result.output
    assert "PWM -> mode=none effective=False" in show_result.output
    assert "Views -> payload, split_payload, assembled_payload" in show_result.output
    assert "Render status -> not_requested" in show_result.output
    assert "Integrity -> ok" in show_result.output
    assert "Composite render ->" in show_result.output
    assert "Selected sticky end" not in show_result.output


def test_yiu_show_verbose_json_exposes_split_row_debug_surface(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "demo_payload_bulged.yiu.yaml"
    _write_yaml(spec_path, _payload_spec(name="demo_payload_bulged", candidate_positions=[2]))

    render_result = runner.invoke(app, ["yiu", "render", "--spec", str(spec_path)])
    assert render_result.exit_code == 0

    default_show = runner.invoke(
        app,
        ["yiu", "show", "--bundle", str(workspace / "bundles" / "demo_payload_bulged"), "--json"],
    )
    verbose_show = runner.invoke(
        app,
        [
            "yiu",
            "show",
            "--bundle",
            str(workspace / "bundles" / "demo_payload_bulged"),
            "--json",
            "--verbose",
        ],
    )

    assert default_show.exit_code == 0
    assert verbose_show.exit_code == 0

    default_payload = json.loads(default_show.output)
    verbose_payload = json.loads(verbose_show.output)

    assert "split_row_debug" not in default_payload
    assert [entry["fragment_side"] for entry in verbose_payload["split_row_debug"]] == ["left", "right"]
    assert len(verbose_payload["split_row_debug"][1]["selected_sticky_end_sequence_5to3"]) == 4
    assert len(verbose_payload["split_row_debug"][1]["canonical_sticky_end_sequence_5to3"]) == 4


def test_yiu_validate_and_show_json_share_payload_summary_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _payload_spec())

    validate_result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"])
    render_result = runner.invoke(app, ["yiu", "render", "--spec", str(spec_path)])
    show_result = runner.invoke(
        app,
        ["yiu", "show", "--bundle", str(workspace / "bundles" / "demo_payload"), "--json"],
    )

    assert validate_result.exit_code == 0
    assert render_result.exit_code == 0
    assert show_result.exit_code == 0

    validate_payload = json.loads(validate_result.output)
    show_payload = json.loads(show_result.output)
    assert show_payload["outputs_root"] is None
    assert show_payload["composite_render_artifact_path"] == str(
        (workspace / "bundles" / "demo_payload" / "payload_views.pdf").resolve()
    )
    for field_name in [
        "input_kind",
        "payload_length",
        "selected_payload_sequence",
        "selected_complement_sequence",
        "junction",
        "mismatches",
        "pwm_mode",
        "pwm_effective",
        "worst_loss",
        "total_loss",
    ]:
        assert show_payload[field_name] == validate_payload[field_name]


def test_yiu_init_workspace_rejects_conflicting_output_and_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"

    result = runner.invoke(app, ["yiu", "init-workspace", "demo_payload", "--output", str(workspace)])

    assert result.exit_code == 2
    assert "Use either WORKSPACE [--root] or --output, not both." in result.output


def test_yiu_render_refuses_to_overwrite_existing_bundle_without_force(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _payload_spec())

    first = runner.invoke(app, ["yiu", "render", "--spec", str(spec_path)])
    second = runner.invoke(app, ["yiu", "render", "--spec", str(spec_path)])

    assert first.exit_code == 0
    assert second.exit_code == 1
    assert "Use --force-overwrite to replace it." in second.output


def test_yiu_show_fails_for_missing_bundle_directory(tmp_path: Path) -> None:
    missing_bundle = tmp_path / "missing_bundle"

    result = runner.invoke(app, ["yiu", "show", "--bundle", str(missing_bundle)])

    assert result.exit_code == 1
    assert "YIU bundle directory not found" in result.output
