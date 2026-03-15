"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/cli/test_seed_command.py

Seed command contracts for construct CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from dnadesign.construct.cli import app
from dnadesign.usr import Dataset

_RUNNER = CliRunner()


def test_seed_promoter_swap_demo_creates_curated_usr_datasets(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    manifest_path = tmp_path / "seed_manifest.yaml"

    result = _RUNNER.invoke(
        app,
        [
            "seed",
            "promoter-swap-demo",
            "--root",
            usr_root.as_posix(),
            "--manifest",
            manifest_path.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.stdout
    anchors = Dataset(usr_root, "mg1655_promoters")
    templates = Dataset(usr_root, "plasmids")
    anchors_frame = anchors.head(n=10)
    templates_frame = templates.head(n=10)
    assert len(anchors_frame) == 4
    assert len(templates_frame) == 1
    assert "usr_label__primary" in anchors_frame.columns
    assert set(anchors_frame["usr_label__primary"]) == {"spyP_MG1655", "sulAp", "soxS", "J23105"}
    assert "construct_seed__label" in anchors_frame.columns
    assert set(anchors_frame["construct_seed__label"]) == {"spyP_MG1655", "sulAp", "soxS", "J23105"}
    assert templates_frame.iloc[0]["construct_seed__label"] == "pDual-10"
    assert "usr_label__primary" in pq.ParquetFile(str(anchors.records_path)).schema_arrow.names
    assert "usr_label__primary" in pq.ParquetFile(str(templates.records_path)).schema_arrow.names
    assert manifest_path.is_file()

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["datasets"]["anchors"] == "mg1655_promoters"
    assert manifest["datasets"]["templates"] == "plasmids"
    assert manifest["slots"]["slot_a"]["start"] == 2300
    assert manifest["slots"]["slot_b"]["start"] == 3621


def test_seed_promoter_swap_demo_normalizes_usr_package_root(tmp_path: Path) -> None:
    usr_pkg_root = tmp_path / "usr"
    usr_pkg_root.mkdir(parents=True, exist_ok=True)
    (usr_pkg_root / "__init__.py").write_text("# stub\n", encoding="utf-8")

    result = _RUNNER.invoke(
        app,
        [
            "seed",
            "promoter-swap-demo",
            "--root",
            usr_pkg_root.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.stdout
    anchors = Dataset(usr_pkg_root / "datasets", "mg1655_promoters")
    templates = Dataset(usr_pkg_root / "datasets", "plasmids")
    assert len(anchors.head(n=10)) == 4
    assert len(templates.head(n=10)) == 1


def test_seed_promoter_swap_demo_warns_about_legacy_dataset_names(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    legacy_controls = usr_root / "construct" / "promoter_swap_controls_demo"
    legacy_templates = usr_root / "construct" / "promoter_swap_templates_demo"
    legacy_controls.mkdir(parents=True, exist_ok=True)
    legacy_templates.mkdir(parents=True, exist_ok=True)

    result = _RUNNER.invoke(
        app,
        [
            "seed",
            "promoter-swap-demo",
            "--root",
            usr_root.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "canonical curated inputs are mg1655_promoters and plasmids" in (result.stdout or "")


def test_seed_import_manifest_creates_generic_usr_datasets(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    manifest_path = tmp_path / "import_manifest.yaml"
    manifest_path.write_text(
        """
manifest_id: custom_construct_inputs
datasets:
  - id: custom_promoters
    notes: Example anchors.
    records:
      - label: sulAp
        role: anchor
        topology: linear
        aliases: [sulA]
        source_ref: canonical local note
        sequence: |
          gttaactacgaaaataggcaacttattcttaaggggcaagattaatttatgttttcccgtcaccaacgacaaaatttgcgaggctctttccgaaaatagggttgatctttgttgtcactggatgtactgtacatccatacagtaactcacaggggctggattgat
  - id: custom_templates
    notes: Example templates.
    records:
      - label: pDual-10
        role: template
        topology: circular
        aliases: [pDual10]
        source_ref: canonical plasmid
        sequence: tttacggctagctcagtcctaggtactatgctagc
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        [
            "seed",
            "import-manifest",
            "--manifest",
            manifest_path.as_posix(),
            "--root",
            usr_root.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.stdout
    anchors = Dataset(usr_root, "custom_promoters")
    templates = Dataset(usr_root, "custom_templates")
    anchors_frame = anchors.head(n=10)
    templates_frame = templates.head(n=10)
    assert list(anchors_frame["usr_label__primary"]) == ["sulAp"]
    assert list(templates_frame["usr_label__primary"]) == ["pDual-10"]
    assert list(anchors_frame["construct_seed__manifest_id"]) == ["custom_construct_inputs"]
    assert list(anchors_frame["construct_seed__source_ref"]) == ["canonical local note"]
    output = result.stdout or ""
    assert "manifest_id: custom_construct_inputs" in output
    assert "dataset: custom_promoters" in output
    assert "dataset: custom_templates" in output


def test_seed_import_manifest_requires_manifest_id(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    manifest_path = tmp_path / "bad_import_manifest.yaml"
    manifest_path.write_text(
        """
datasets:
  - id: custom_promoters
    records:
      - label: sulAp
        role: anchor
        topology: linear
        sequence: ACGT
""",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        [
            "seed",
            "import-manifest",
            "--manifest",
            manifest_path.as_posix(),
            "--root",
            usr_root.as_posix(),
        ],
    )

    assert result.exit_code == 1
    assert "Seed manifest requires a non-empty manifest_id" in (result.stdout or "")


def test_seed_promoter_swap_demo_rejects_non_integer_slot_bounds(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "dnadesign.construct.src.seed._seed_asset_payload",
        lambda: {
            "demo_id": "bad_demo",
            "datasets": {"anchors": "bad_anchors", "templates": "bad_templates"},
            "anchors": [
                {"label": "anchor", "role": "anchor", "topology": "linear", "sequence": "ACGT"},
            ],
            "templates": [
                {"label": "template", "role": "template", "topology": "circular", "sequence": "AAAATTTT"},
            ],
            "slots": [
                {
                    "slot": "slot_a",
                    "template_label": "template",
                    "incumbent_label": "anchor",
                    "start": "not_an_int",
                    "end": 4,
                    "expected_template_sequence": "ACGT",
                }
            ],
        },
    )

    result = _RUNNER.invoke(app, ["seed", "promoter-swap-demo", "--root", (tmp_path / "usr_root").as_posix()])

    assert result.exit_code == 1
    assert "start/end must be integers" in (result.stdout or "")
