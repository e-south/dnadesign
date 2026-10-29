"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/cli/test_catalog_assets_cli.py

Regression tests for catalog assets CLI Cruncher CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.tests.cli_output import normalized_cli_output

runner = CliRunner()


def _write_motif(path: Path, *, source: str, motif_id: str, tf_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "descriptor": {"source": source, "motif_id": motif_id, "tf_name": tf_name},
        "matrix": [[0.7, 0.1, 0.1, 0.1]],
        "checksums": {"sha256_norm": "good"},
    }
    path.write_text(json.dumps(payload))


def _write_config(tmp_path: Path, *, pwm_source: str = "matrix") -> Path:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA", "cpxR"]]},
            "catalog": {"root": str(catalog_root), "pwm_source": pwm_source},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _write_prob_motif(path: Path, *, source: str, motif_id: str, tf_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "descriptor": {"source": source, "motif_id": motif_id, "tf_name": tf_name, "alphabet": "ACGT"},
        "matrix_semantics": "probabilities",
        "matrix": [[0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]],
        "background": [0.25, 0.25, 0.25, 0.25],
        "checksums": {"sha256_norm": "good"},
    }
    path.write_text(json.dumps(payload))


def _write_densegen_workspace(tmp_path: Path, *, name: str = "demo_densegen") -> Path:
    workspace = tmp_path / "src" / "dnadesign" / "densegen" / "workspaces" / name
    inputs_root = workspace / "inputs"
    inputs_root.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        [
            "densegen:",
            '  schema_version: "2.5"',
            "  run:",
            "    id: demo",
            '    root: "."',
            "",
        ]
    )
    (workspace / "config.yaml").write_text(payload)
    return workspace


def _write_external_densegen_workspace(tmp_path: Path, *, name: str = "external_densegen") -> Path:
    workspace = tmp_path / name
    inputs_root = workspace / "inputs"
    inputs_root.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        [
            "densegen:",
            '  schema_version: "2.5"',
            "  run:",
            "    id: demo",
            '    root: "."',
            "",
        ]
    )
    (workspace / "config.yaml").write_text(payload)
    return workspace


def test_catalog_pwms_defaults_to_regulator_sets(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    entries = {
        "regulondb:RBM1": CatalogEntry(
            source="regulondb",
            motif_id="RBM1",
            tf_name="lexA",
            kind="PFM",
            has_matrix=True,
            matrix_source="alignment",
        ),
        "regulondb:RBM2": CatalogEntry(
            source="regulondb",
            motif_id="RBM2",
            tf_name="cpxR",
            kind="PFM",
            has_matrix=True,
            matrix_source="alignment",
        ),
    }
    CatalogIndex(entries=entries).save(catalog_root)
    _write_motif(
        catalog_root / "normalized" / "motifs" / "regulondb" / "RBM1.json",
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
    )
    _write_motif(
        catalog_root / "normalized" / "motifs" / "regulondb" / "RBM2.json",
        source="regulondb",
        motif_id="RBM2",
        tf_name="cpxR",
    )

    config_path = _write_config(tmp_path)
    result = runner.invoke(app, ["catalog", "pwms", str(config_path)], color=False)
    assert result.exit_code == 0
    assert "PWM summary" in result.output
    assert "lexA" in result.output
    assert "cpxR" in result.output


def test_catalog_logos_writes_pngs(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    entry = CatalogEntry(
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
        kind="PFM",
        has_matrix=True,
        matrix_source="alignment",
    )
    CatalogIndex(entries={entry.key: entry}).save(catalog_root)
    _write_motif(
        catalog_root / "normalized" / "motifs" / "regulondb" / "RBM1.json",
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
    )

    config_path = _write_config(tmp_path)
    out_dir = tmp_path / "logos"
    result = runner.invoke(
        app,
        ["catalog", "logos", "--tf", "lexA", "--out-dir", str(out_dir), str(config_path)],
        color=False,
    )
    assert result.exit_code == 0
    assert "Rendered PWM logos" in result.output
    assert list(out_dir.glob("*_logo.png"))
    manifest = out_dir / "logo_manifest.json"
    assert manifest.exists()

    repeat = runner.invoke(
        app,
        ["catalog", "logos", "--tf", "lexA", "--out-dir", str(out_dir), str(config_path)],
        color=False,
    )
    assert repeat.exit_code == 0
    assert "already rendered" in repeat.output


def test_catalog_logos_default_output_is_flat_plots_dir(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    entry = CatalogEntry(
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
        kind="PFM",
        has_matrix=True,
        matrix_source="alignment",
    )
    CatalogIndex(entries={entry.key: entry}).save(catalog_root)
    _write_motif(
        catalog_root / "normalized" / "motifs" / "regulondb" / "RBM1.json",
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
    )

    config_path = _write_config(tmp_path)
    result = runner.invoke(
        app,
        ["catalog", "logos", "--tf", "lexA", str(config_path)],
        color=False,
    )
    assert result.exit_code == 0, result.output

    plots_root = tmp_path / "runs" / "plots"
    assert plots_root.exists()
    assert list(plots_root.glob("*_logo.png"))
    assert (plots_root / "logo_manifest.json").exists()
    assert not (plots_root / "logos").exists()


def test_export_meme_defaults_to_workspace_artifacts_tree(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    entry = CatalogEntry(
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
        kind="PFM",
        has_matrix=True,
        matrix_source="alignment",
    )
    CatalogIndex(entries={entry.key: entry}).save(catalog_root)
    _write_prob_motif(
        catalog_root / "normalized" / "motifs" / "regulondb" / "RBM1.json",
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
    )

    config_path = _write_config(tmp_path)
    result = runner.invoke(
        app,
        ["catalog", "export-meme", "--tf", "lexA", str(config_path)],
        color=False,
    )

    assert result.exit_code == 0, result.output
    assert "MEME motif exports" in result.output
    out_dir = tmp_path / "runs" / "artifacts" / "meme"
    artifacts = list(out_dir.glob("*.meme"))
    assert len(artifacts) == 1
    text = artifacts[0].read_text()
    assert "MEME version 4" in text
    assert "MOTIF RBM1" in text
    assert (out_dir / "meme_manifest.json").exists()


def test_export_meme_rejects_blank_producer(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    entry = CatalogEntry(
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
        kind="PFM",
        has_matrix=True,
        matrix_source="alignment",
    )
    CatalogIndex(entries={entry.key: entry}).save(catalog_root)
    _write_prob_motif(
        catalog_root / "normalized" / "motifs" / "regulondb" / "RBM1.json",
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
    )

    config_path = _write_config(tmp_path)
    result = runner.invoke(
        app,
        ["catalog", "export-meme", "--tf", "lexA", "--producer", "   ", str(config_path)],
        color=False,
    )
    normalized_output = normalized_cli_output(result.output)

    assert result.exit_code != 0
    assert "--producer must be a non-empty string" in normalized_output


def test_export_meme_reports_config_resolution_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    result = runner.invoke(
        app,
        ["catalog", "export-meme", str(missing)],
        color=False,
    )

    assert result.exit_code == 1
    assert "Config file not found" in result.output


def test_export_meme_requires_overwrite_for_existing_artifact(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    entry = CatalogEntry(
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
        kind="PFM",
        has_matrix=True,
        matrix_source="alignment",
    )
    CatalogIndex(entries={entry.key: entry}).save(catalog_root)
    _write_prob_motif(
        catalog_root / "normalized" / "motifs" / "regulondb" / "RBM1.json",
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
    )

    config_path = _write_config(tmp_path)
    out_dir = tmp_path / "runs" / "artifacts" / "meme"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = out_dir / "lexA__regulondb__RBM1.meme"
    existing.write_text("old\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["catalog", "export-meme", "--tf", "lexA", str(config_path)],
        color=False,
    )

    assert result.exit_code == 1
    assert "Artifact already exists" in result.output
    assert existing.read_text(encoding="utf-8") == "old\n"

    overwrite = runner.invoke(
        app,
        ["catalog", "export-meme", "--tf", "lexA", "--overwrite", str(config_path)],
        color=False,
    )
    assert overwrite.exit_code == 0, overwrite.output
    assert existing.read_text(encoding="utf-8").startswith("MEME version 4")


def test_export_sites_ignores_pwm_source_for_selection(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    entry = CatalogEntry(
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
        kind="sites",
        has_matrix=False,
        has_sites=True,
        site_count=1,
        site_total=1,
        site_kind="curated",
    )
    CatalogIndex(entries={entry.key: entry}).save(catalog_root)
    sites_path = catalog_root / "normalized" / "sites" / "regulondb" / "RBM1.jsonl"
    sites_path.parent.mkdir(parents=True, exist_ok=True)
    sites_path.write_text(json.dumps({"sequence": "ACGT", "site_id": "s1", "motif_ref": entry.key}) + "\n")

    config_path = _write_config(tmp_path, pwm_source="matrix")
    out_path = tmp_path / "densegen_sites.csv"
    result = runner.invoke(
        app,
        ["catalog", "export-sites", "--tf", "lexA", "--out", str(out_path), str(config_path)],
        color=False,
    )
    assert result.exit_code == 0
    assert "Binding-site export" in result.output
    assert out_path.exists()


def test_export_sites_relative_out_resolves_from_config_dir(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    catalog_root = workspace / ".cruncher"
    entry = CatalogEntry(
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
        kind="sites",
        has_matrix=False,
        has_sites=True,
        site_count=1,
        site_total=1,
        site_kind="curated",
    )
    CatalogIndex(entries={entry.key: entry}).save(catalog_root)
    sites_path = catalog_root / "normalized" / "sites" / "regulondb" / "RBM1.jsonl"
    sites_path.parent.mkdir(parents=True, exist_ok=True)
    sites_path.write_text(json.dumps({"sequence": "ACGT", "site_id": "s1", "motif_ref": entry.key}) + "\n")
    config_path = _write_config(workspace, pwm_source="matrix")

    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(invocation_dir)

    rel_out = Path("outputs") / "binding_sites.parquet"
    result = runner.invoke(
        app,
        ["catalog", "export-sites", "--tf", "lexA", "--out", str(rel_out), str(config_path)],
        color=False,
    )

    assert result.exit_code == 0, result.output
    assert (workspace / rel_out).exists()
    assert not (invocation_dir / rel_out).exists()
