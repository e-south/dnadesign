"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_yiu_cli.py

CLI contracts for the payload-centric YIU surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.cli.yiu_presenter import mismatch_summary_text
from dnadesign.cruncher.tests.cli_output import normalized_cli_output

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
    ligation_profile: str = "none",
    ligation_awareness_mode: str = "secondary",
    bad_pattern_heuristics: bool = False,
) -> dict[str, object]:
    junction_start = 4
    junction_end = 8
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
                "start": junction_start,
                "end": junction_end,
                "overhang_length": 4,
                "max_payload_body_length": max(junction_start, len(sequence) - junction_end),
            },
            "mismatches": {
                "count": mismatch_count,
                "candidate_positions": candidate_positions or [1, 2],
                "allowed_strands": ["complement", "payload"],
                "strand_mode": "per_position",
                "default_strand_preference": "complement",
                "ligation_profile": ligation_profile,
                "ligation_awareness_mode": ligation_awareness_mode,
                "bad_pattern_heuristics": bad_pattern_heuristics,
            },
            "pwm": {
                "mode": "none",
                "source": {"kind": "none"},
                "objective": {
                    "primary": "maximin",
                    "secondary": [
                        "total_loss",
                        "ligation_awareness",
                        "midpoint_proximity",
                        "default_strand_preference",
                        "lexical_stability",
                    ],
                },
            },
        },
        "output": {
            "bundle_dir": f"outputs/{name}",
            "published_plot_path": f"outputs/plot__{name}__payload_views.pdf",
            "emit_render_jobs_debug": False,
        },
    }


def _ligation_summary_lines(ligation: dict[str, object]) -> tuple[str, str]:
    chosen_classes = ",".join(str(item) for item in ligation["chosen_mismatch_classes"]) or "-"
    position_classes = ",".join(str(item) for item in ligation["position_classes"]) or "-"
    return (
        "Ligation -> "
        f"profile={ligation['profile']} "
        f"mode={ligation['awareness_mode']} "
        f"applied={ligation['applied']} "
        f"classes={chosen_classes} "
        f"positions={position_classes} "
        f"bad_patterns={ligation['bad_pattern_heuristics']}",
        f"Ligation note -> {ligation['decision_note']}",
    )


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
            "bundle_dir": "outputs/legacy_payload",
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
    normalized_output = normalized_cli_output(result.output)

    assert result.exit_code == 0
    assert "--bundle" in normalized_output
    assert "--run" not in normalized_output


def test_mismatch_summary_text_uses_compact_strand_aware_notation() -> None:
    mismatch_sites = [
        {
            "payload_index": 9,
            "junction_offset": 1,
            "mutated_strand": "complement",
            "native_base": "A",
            "mutated_base": "C",
            "opposing_base": "T",
        },
        {
            "payload_index": 10,
            "junction_offset": 2,
            "mutated_strand": "complement",
            "native_base": "C",
            "mutated_base": "A",
            "opposing_base": "G",
        },
        {
            "payload_index": 12,
            "junction_offset": 3,
            "mutated_strand": "payload",
            "native_base": "T",
            "mutated_base": "G",
            "opposing_base": "A",
        },
    ]

    assert mismatch_summary_text(mismatch_sites) == "AS10A>C,11C>A; PS13T>G"


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
    assert payload["optimization"]["mismatches"]["candidate_positions"] == [0, 1, 2, 3]
    assert payload["optimization"]["mismatches"]["ligation_profile"] == "t4"
    assert payload["optimization"]["mismatches"]["ligation_awareness_mode"] == "secondary"
    assert payload["optimization"]["mismatches"]["bad_pattern_heuristics"] is False
    assert payload["output"]["bundle_dir"].startswith("outputs/")
    assert payload["output"]["published_plot_path"].startswith("outputs/")
    assert (workspace / "configs" / "yiu" / "example_payload.advanced_pwm.example.yaml").exists()
    assert (workspace / "motifs" / "example_pwm_context.yaml").exists()
    assert (workspace / ".gitignore").read_text(encoding="utf-8").splitlines() == [
        ".cruncher/",
        "outputs/",
        ".DS_Store",
    ]


def test_yiu_init_workspace_can_seed_payload_sequence_and_center_locked_mode(tmp_path: Path) -> None:
    workspace = tmp_path / "seeded_yiu_payload"

    result = runner.invoke(
        app,
        [
            "yiu",
            "init-workspace",
            "--output",
            str(workspace),
            "--sequence",
            "AACCGGTTGGTT",
            "--junction-mode",
            "center_locked",
        ],
    )

    assert result.exit_code == 0
    payload = yaml.safe_load((workspace / "configs" / "yiu" / "example_payload.yiu.yaml").read_text(encoding="utf-8"))
    assert payload["input"]["user_sequence"]["sequence"] == "AACCGGTTGGTT"
    assert payload["optimization"]["junction"]["mode"] == "center_locked"


def test_yiu_validate_fails_fast_on_legacy_v1_spec(tmp_path: Path) -> None:
    spec_path = tmp_path / "demo_yiu_payload" / "configs" / "yiu" / "legacy.yiu.yaml"
    _write_yaml(spec_path, _legacy_v1_payload_spec())

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)])

    assert result.exit_code == 1
    assert "YIU_CONTRACT_UNKNOWN" in result.output


def test_yiu_render_materializes_payload_bundle_from_spec(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _payload_spec(candidate_positions=[0, 1, 2, 3], ligation_profile="t4"))

    result = runner.invoke(app, ["yiu", "render", "--spec", str(spec_path), "--emit-renders"])

    assert result.exit_code == 0
    normalized_output = normalized_cli_output(result.output)
    bundle_dir = workspace / "outputs" / "demo_payload"
    bundle_summary = json.loads((bundle_dir / "bundle_summary.json").read_text())
    assert (bundle_dir / "bundle_summary.json").exists()
    assert (bundle_dir / "bundle_manifest.json").exists()
    assert (bundle_dir / "normalized_payload.json").exists()
    assert (bundle_dir / "visual_inventory.json").exists()
    assert (bundle_dir / "payload_view.json").exists()
    assert (workspace / "outputs" / "plot__demo_payload__payload_views.pdf").exists()
    assert "Bundle ->" in result.output
    assert "Composite render ->" in result.output
    assert "Published plot ->" in result.output
    ligation_line, ligation_note_line = _ligation_summary_lines(bundle_summary["ligation"])
    assert normalized_cli_output(ligation_line) in normalized_output
    assert normalized_cli_output(ligation_note_line) in normalized_output
    assert "Bundle summary ->" not in result.output
    assert "Bundle manifest ->" not in result.output
    assert "Normalized payload ->" not in result.output
    assert "Visual inventory ->" not in result.output


def test_yiu_validate_reports_junction_window_summary(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _payload_spec(candidate_positions=[0, 1, 2, 3], ligation_profile="t4"))

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)])
    json_result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"])
    validate_payload = json.loads(json_result.output)

    assert result.exit_code == 0
    assert json_result.exit_code == 0
    normalized_output = normalized_cli_output(result.output)
    assert "Junction window -> start=4 end=8 mode=explicit_window" in result.output
    assert "Mismatch count -> 1" in result.output
    assert (
        "Mismatch edits (PS=payload, AS=complement; 1-based) -> "
        f"{mismatch_summary_text(validate_payload['mismatches'])}"
    ) in result.output
    assert "PWM -> mode=none effective=False" in result.output
    ligation_line, ligation_note_line = _ligation_summary_lines(validate_payload["ligation"])
    assert normalized_cli_output(ligation_line) in normalized_output
    assert normalized_cli_output(ligation_note_line) in normalized_output
    assert "Bundle write -> no" in result.output
    assert "Bulge mask ->" not in result.output
    assert "Watson-Crick pairing valid ->" not in result.output


def test_yiu_validate_rejects_ambiguous_iupac_payload(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "ambiguous_payload.yiu.yaml"
    _write_yaml(spec_path, _payload_spec(sequence="AANNNNTT", name="ambiguous_payload"))

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)])

    assert result.exit_code == 1
    normalized_output = normalized_cli_output(result.output)
    assert "YIU_SEQUENCE_INVALID" in normalized_output
    assert "A/C/G/T" in normalized_output


def test_yiu_show_reports_payload_bundle_summary(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _payload_spec(candidate_positions=[0, 1, 2, 3], ligation_profile="t4"))

    render_result = runner.invoke(app, ["yiu", "render", "--spec", str(spec_path)])
    assert render_result.exit_code == 0
    bundle_summary = json.loads((workspace / "outputs" / "demo_payload" / "bundle_summary.json").read_text())
    views = bundle_summary["sequence_summary"]["views"]

    show_result = runner.invoke(app, ["yiu", "show", "--bundle", str(workspace / "outputs" / "demo_payload")])

    assert show_result.exit_code == 0
    normalized_output = normalized_cli_output(show_result.output)
    assert "Bundle ->" in show_result.output
    assert "demo_payload" in show_result.output
    assert "Input kind -> user_sequence" in show_result.output
    assert "Payload length -> 21" in show_result.output
    assert "Junction window -> start=4 end=8 mode=explicit_window" in show_result.output
    assert "Mismatch count -> 1" in show_result.output
    assert (
        f"Mismatch edits (PS=payload, AS=complement; 1-based) -> {'; '.join(bundle_summary['mismatch_notation'])}"
        in show_result.output
    )
    assert "PWM -> mode=none effective=False" in show_result.output
    ligation_line, ligation_note_line = _ligation_summary_lines(bundle_summary["ligation"])
    assert normalized_cli_output(ligation_line) in normalized_output
    assert normalized_cli_output(ligation_note_line) in normalized_output
    assert (
        f"Junction payload 5' -> 3' -> {bundle_summary['sequence_summary']['junction_payload_sequence_5to3']}"
    ) in show_result.output
    assert (
        "Overhang 5' -> 3' -> "
        f"canonical={bundle_summary['sequence_summary']['overhang_5to3']['canonical_sequence_5to3']} "
        f"mismatch-present={bundle_summary['sequence_summary']['overhang_5to3']['mismatch_present_sequence_5to3']}"
    ) in show_result.output
    for label, key in [
        ("Payload", "payload"),
        ("Split left", "split_left"),
        ("Split right", "split_right"),
        ("Assembled", "assembled"),
    ]:
        assert (
            f"{label} canonical 5' -> 3' -> "
            f"top={views[key]['canonical']['top_strand_5to3']} "
            f"bottom={views[key]['canonical']['bottom_strand_5to3']}"
        ) in show_result.output
        assert (
            f"{label} mismatch-present 5' -> 3' -> "
            f"top={views[key]['mismatch_present']['top_strand_5to3']} "
            f"bottom={views[key]['mismatch_present']['bottom_strand_5to3']}"
        ) in show_result.output
        if views[key]["changed_rows"]:
            assert f"{label} changed rows -> {', '.join(views[key]['changed_rows'])}" in show_result.output
    assert "Views ->" not in show_result.output
    assert "Render status ->" not in show_result.output
    assert "Integrity ->" not in show_result.output
    assert "Composite render ->" not in show_result.output
    assert "Published plot ->" not in show_result.output
    assert "Bundle summary ->" not in show_result.output
    assert "Bundle contract ->" not in show_result.output
    assert "Bundle manifest -> bundle_manifest.json" not in show_result.output
    assert "Normalized payload -> normalized_payload.json" not in show_result.output
    assert "Visual inventory -> visual_inventory.json" not in show_result.output


def test_yiu_show_verbose_text_adds_bundle_detail(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _payload_spec())

    render_result = runner.invoke(app, ["yiu", "render", "--spec", str(spec_path)])
    assert render_result.exit_code == 0

    show_result = runner.invoke(
        app,
        ["yiu", "show", "--bundle", str(workspace / "outputs" / "demo_payload"), "--verbose"],
    )

    assert show_result.exit_code == 0
    assert "Bundle contract -> split_yiu_payload_bundle_v4" in show_result.output
    assert "Views -> payload, split_payload, assembled_payload" in show_result.output
    assert "Render status -> not_requested" in show_result.output
    assert "Integrity -> ok" in show_result.output
    assert "Bundle summary -> bundle_summary.json" in show_result.output
    assert "Bundle manifest -> bundle_manifest.json" in show_result.output
    assert "Normalized payload -> normalized_payload.json" in show_result.output
    assert "Visual inventory -> visual_inventory.json" in show_result.output


def test_yiu_show_verbose_json_exposes_split_row_debug_surface(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "demo_payload_bulged.yiu.yaml"
    _write_yaml(spec_path, _payload_spec(name="demo_payload_bulged", candidate_positions=[2]))

    render_result = runner.invoke(app, ["yiu", "render", "--spec", str(spec_path)])
    assert render_result.exit_code == 0

    default_show = runner.invoke(
        app,
        ["yiu", "show", "--bundle", str(workspace / "outputs" / "demo_payload_bulged"), "--json"],
    )
    verbose_show = runner.invoke(
        app,
        [
            "yiu",
            "show",
            "--bundle",
            str(workspace / "outputs" / "demo_payload_bulged"),
            "--json",
            "--verbose",
        ],
    )

    assert default_show.exit_code == 0
    assert verbose_show.exit_code == 0

    default_payload = json.loads(default_show.output)
    verbose_payload = json.loads(verbose_show.output)

    assert "bundle_summary" in default_payload
    assert "bundle_manifest_path" not in default_payload
    assert "normalized_payload_path" not in default_payload
    assert "visual_inventory_path" not in default_payload
    assert "selected_payload_sequence" not in default_payload
    assert "selected_complement_sequence" not in default_payload
    assert "mismatches" not in default_payload
    assert "provenance" not in default_payload
    assert (
        default_payload["bundle_summary"]["sequence_summary"]["views"]["payload"]["canonical"]["top_strand_5to3"]
        == "AAATTTCCCGGGAAATTTCCC"
    )
    assert (
        default_payload["bundle_summary"]["sequence_summary"]["views"]["payload"]["canonical"]["bottom_strand_5to3"]
        == "GGGAAATTTCCCGGGAAATTT"
    )
    assert (
        default_payload["bundle_summary"]["sequence_summary"]["views"]["assembled"]["mismatch_present"][
            "top_strand_5to3"
        ]
        == "AAATTTCCCGGGAAATTTCCC"
    )
    assert default_payload["bundle_summary"]["mismatch_notation"] == mismatch_summary_text(
        verbose_payload["mismatches"]
    ).split("; ")
    assert "optimization_decision" not in default_payload
    assert "motif_context" not in default_payload
    assert "split_row_debug" not in default_payload
    assert "bundle_manifest_path" in verbose_payload
    assert "normalized_payload_path" in verbose_payload
    assert "visual_inventory_path" in verbose_payload
    assert "selected_payload_sequence" in verbose_payload
    assert "selected_complement_sequence" in verbose_payload
    assert "mismatches" in verbose_payload
    assert "provenance" in verbose_payload
    assert "optimization_decision" in verbose_payload
    assert "motif_context" in verbose_payload
    assert [entry["fragment_side"] for entry in verbose_payload["split_row_debug"]] == ["left", "right"]
    assert len(verbose_payload["split_row_debug"][1]["selected_sticky_end_sequence_5to3"]) == 4
    assert len(verbose_payload["split_row_debug"][1]["canonical_sticky_end_sequence_5to3"]) == 4
    assert verbose_payload["split_row_debug"][0]["payload_body_sequence_5to3"] == "AAAT"


def test_yiu_validate_and_show_json_share_payload_summary_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _payload_spec())

    validate_result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"])
    render_result = runner.invoke(app, ["yiu", "render", "--spec", str(spec_path)])
    show_result = runner.invoke(
        app,
        ["yiu", "show", "--bundle", str(workspace / "outputs" / "demo_payload"), "--json"],
    )

    assert validate_result.exit_code == 0
    assert render_result.exit_code == 0
    assert show_result.exit_code == 0

    validate_payload = json.loads(validate_result.output)
    show_payload = json.loads(show_result.output)
    assert show_payload["outputs_root"] == str((workspace / "outputs").resolve())
    assert show_payload["composite_render_artifact_path"] == str(
        (workspace / "outputs" / "demo_payload" / "payload_views.pdf").resolve()
    )
    assert show_payload["published_plot_artifact_path"] == str(
        (workspace / "outputs" / "plot__demo_payload__payload_views.pdf").resolve()
    )
    assert show_payload["bundle_summary_path"] == str(
        (workspace / "outputs" / "demo_payload" / "bundle_summary.json").resolve()
    )
    assert (
        show_payload["bundle_summary"]["sequence_summary"]["overhang_5to3"]["mismatch_present_sequence_5to3"]
        == (validate_payload["selected_complement_sequence"][4:8][::-1])
    )
    assert (
        show_payload["bundle_summary"]["sequence_summary"]["views"]["payload"]["mismatch_present"]["top_strand_5to3"]
        == validate_payload["selected_payload_sequence"]
    )
    assert (
        show_payload["bundle_summary"]["sequence_summary"]["views"]["payload"]["mismatch_present"]["bottom_strand_5to3"]
        == validate_payload["selected_complement_sequence"][::-1]
    )


def test_yiu_validate_rejects_invalid_pwm_mode_source_combo(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "invalid_pwm.yiu.yaml"
    payload = _payload_spec(name="invalid_pwm")
    payload["optimization"]["pwm"] = {
        "mode": "none",
        "source": {"kind": "file", "path": "motifs/example_pwm_context.yaml"},
        "objective": payload["optimization"]["pwm"]["objective"],
    }
    _write_yaml(spec_path, payload)

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)])

    assert result.exit_code == 1
    normalized_output = normalized_cli_output(result.output)
    assert "optimization.pwm.mode=none requires optimization.pwm.source.kind=none" in normalized_output


def test_yiu_validate_rejects_sample_hit_without_resolution_hints(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "missing_sample_resolution.yiu.yaml"
    _write_yaml(
        spec_path,
        {
            "yiu": {
                "schema_version": 1,
                "contract": "split_yiu_payload_rendering_v4",
                "name": "missing_sample_resolution",
            },
            "input": {
                "kind": "sample_hit",
                "sample_hit": {
                    "hit_id": "elite-1",
                    "sample_name": "demo_sample",
                },
            },
            "optimization": _payload_spec()["optimization"],
            "output": {
                "bundle_dir": "outputs/missing_sample_resolution",
                "emit_render_jobs_debug": False,
            },
        },
    )

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)])

    assert result.exit_code == 1
    normalized_output = normalized_cli_output(result.output)
    assert (
        "sample_hit requires payload_sequence or source_artifact_path "
        "(optionally resolved relative to metadata.source_workspace)."
    ) in normalized_output


def test_yiu_validate_rejects_removed_source_artifact_alias(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "removed_source_alias.yiu.yaml"
    payload = _payload_spec(name="removed_source_alias")
    payload["input"] = {
        "kind": "sample_hit",
        "sample_hit": {
            "hit_id": "elite-1",
            "sample_name": "demo_sample",
            "payload_sequence": "AAATTTCCCGGGAAATTTCCC",
            "source_artifact": "outputs/optimize/tables/elites.parquet",
        },
    }
    _write_yaml(spec_path, payload)

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)])

    assert result.exit_code == 1
    normalized_output = normalized_cli_output(result.output)
    assert "source_artifact" in normalized_output


def test_yiu_validate_rejects_bundle_dir_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "escaped_bundle.yiu.yaml"
    payload = _payload_spec(name="escaped_bundle")
    payload["output"]["bundle_dir"] = "../escaped_bundle"
    _write_yaml(spec_path, payload)

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)])

    assert result.exit_code == 1
    normalized_output = normalized_cli_output(result.output)
    assert "output.bundle_dir must not traverse outside the workspace root" in normalized_output


def test_yiu_validate_rejects_published_plot_path_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"
    spec_path = workspace / "configs" / "yiu" / "escaped_plot.yiu.yaml"
    payload = _payload_spec(name="escaped_plot")
    payload["output"]["published_plot_path"] = "../escaped_plot.pdf"
    _write_yaml(spec_path, payload)

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)])

    assert result.exit_code == 1
    normalized_output = normalized_cli_output(result.output)
    assert "output.published_plot_path must not traverse outside the workspace root" in normalized_output


def test_yiu_init_workspace_rejects_conflicting_output_and_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "demo_yiu_payload"

    result = runner.invoke(app, ["yiu", "init-workspace", "demo_payload", "--output", str(workspace)])
    normalized_output = normalized_cli_output(result.output)

    assert result.exit_code == 2
    assert "Use either WORKSPACE [--root] or --output, not both." in normalized_output


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
