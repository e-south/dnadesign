"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_catalog_assets_cli.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex

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
    manifest = out_dir / "run" / "logo_manifest.json"
    assert manifest.exists()

    repeat = runner.invoke(
        app,
        ["catalog", "logos", "--tf", "lexA", "--out-dir", str(out_dir), str(config_path)],
        color=False,
    )
    assert repeat.exit_code == 0
    assert "already rendered" in repeat.output


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
    assert "DenseGen binding-site export" in result.output
    assert out_path.exists()


def test_export_densegen_defaults_to_densegen_workspace(tmp_path: Path) -> None:
    densegen_ws = _write_densegen_workspace(tmp_path)
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
        ["catalog", "export-densegen", "--tf", "lexA", "--densegen-workspace", densegen_ws.name, str(config_path)],
        color=False,
    )
    assert result.exit_code == 0, result.output
    out_dir = densegen_ws / "inputs" / "motif_artifacts"
    assert out_dir.exists()
    assert list(out_dir.glob("*.json"))
    assert (out_dir / "artifact_manifest.json").exists()


def test_export_densegen_cleans_existing_tf_artifacts(tmp_path: Path) -> None:
    densegen_ws = _write_densegen_workspace(tmp_path)
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
    out_dir = densegen_ws / "inputs" / "motif_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    old_lexa = out_dir / "lexA__legacy__OLD1.json"
    old_lexa.write_text("{}")
    keep_cpxr = out_dir / "cpxR__legacy__KEEP1.json"
    keep_cpxr.write_text("{}")

    config_path = _write_config(tmp_path)
    result = runner.invoke(
        app,
        ["catalog", "export-densegen", "--tf", "lexA", "--densegen-workspace", densegen_ws.name, str(config_path)],
        color=False,
    )

    assert result.exit_code == 0, result.output
    assert not old_lexa.exists()
    assert keep_cpxr.exists()
    assert list(out_dir.glob("lexA__*.json"))


def test_export_sites_defaults_to_densegen_workspace(tmp_path: Path) -> None:
    densegen_ws = _write_densegen_workspace(tmp_path)
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
    result = runner.invoke(
        app,
        ["catalog", "export-sites", "--tf", "lexA", "--densegen-workspace", densegen_ws.name, str(config_path)],
        color=False,
    )
    assert result.exit_code == 0, result.output
    out_path = densegen_ws / "inputs" / "densegen_sites.parquet"
    assert out_path.exists()
