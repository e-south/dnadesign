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
import pytest
import yaml
from typer.testing import CliRunner

from dnadesign.construct.src.cli import app
from dnadesign.construct.src.seeding import bootstrap as seed_module
from dnadesign.usr import Dataset

_RUNNER = CliRunner()


def test_seed_promoter_swap_demo_creates_curated_usr_datasets(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    manifest_path = tmp_path / "seed_manifest.yaml"

    result = _RUNNER.invoke(
        app,
        [
            "seed",
            "anchor-template-demo",
            "--root",
            usr_root.as_posix(),
            "--manifest",
            manifest_path.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.stdout
    anchors = Dataset(usr_root, "anchor_parts_demo")
    templates = Dataset(usr_root, "template_parts_demo")
    anchors_frame = anchors.head(n=10)
    templates_frame = templates.head(n=10)
    assert len(anchors_frame) == 4
    assert len(templates_frame) == 1
    assert "usr_label__primary" in anchors_frame.columns
    assert set(anchors_frame["usr_label__primary"]) == {
        "anchor_part_alpha",
        "anchor_part_beta",
        "anchor_part_gamma",
        "anchor_part_short_ref",
    }
    assert "construct_seed__label" in anchors_frame.columns
    assert set(anchors_frame["construct_seed__label"]) == {
        "anchor_part_alpha",
        "anchor_part_beta",
        "anchor_part_gamma",
        "anchor_part_short_ref",
    }
    assert templates_frame.iloc[0]["construct_seed__label"] == "template_backbone_dual_slot"
    assert "usr_label__primary" in pq.ParquetFile(str(anchors.records_path)).schema_arrow.names
    assert "usr_label__primary" in pq.ParquetFile(str(templates.records_path)).schema_arrow.names
    assert manifest_path.is_file()

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["datasets"]["anchors"] == "anchor_parts_demo"
    assert manifest["datasets"]["templates"] == "template_parts_demo"
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
            "anchor-template-demo",
            "--root",
            usr_pkg_root.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.stdout
    anchors = Dataset(usr_pkg_root / "datasets", "anchor_parts_demo")
    templates = Dataset(usr_pkg_root / "datasets", "template_parts_demo")
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
            "anchor-template-demo",
            "--root",
            usr_root.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "canonical packaged demo inputs are anchor_parts_demo and template_parts_demo" in (result.stdout or "")


def test_seed_promoter_swap_demo_is_idempotent(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"

    first = _RUNNER.invoke(
        app,
        [
            "seed",
            "anchor-template-demo",
            "--root",
            usr_root.as_posix(),
        ],
    )
    second = _RUNNER.invoke(
        app,
        [
            "seed",
            "anchor-template-demo",
            "--root",
            usr_root.as_posix(),
        ],
    )

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout

    anchors = Dataset(usr_root, "anchor_parts_demo")
    templates = Dataset(usr_root, "template_parts_demo")
    anchors_frame = anchors.head(n=10)
    templates_frame = templates.head(n=10)

    assert len(anchors_frame) == 4
    assert len(templates_frame) == 1
    assert set(anchors_frame["construct_seed__label"]) == {
        "anchor_part_alpha",
        "anchor_part_beta",
        "anchor_part_gamma",
        "anchor_part_short_ref",
    }
    assert list(templates_frame["construct_seed__label"]) == ["template_backbone_dual_slot"]


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
      - label: anchor_part_beta
        topology: linear
        aliases: [sulA]
        source_ref: canonical local note
        sequence: |
          gttaactacgaaaataggcaacttattcttaaggggcaagattaatttatgttttcccgtcaccaacgacaaaatttgcgaggctctttccgaaaatagggttgatctttgttgtcactggatgtactgtacatccatacagtaactcacaggggctggattgat
  - id: custom_templates
    notes: Example templates.
    records:
      - label: template_backbone_dual_slot
        intended_role: template
        topology: circular
        aliases: [dual_slot_template]
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
    assert list(anchors_frame["usr_label__primary"]) == ["anchor_part_beta"]
    assert list(templates_frame["usr_label__primary"]) == ["template_backbone_dual_slot"]
    assert list(anchors_frame["construct_seed__manifest_id"]) == ["custom_construct_inputs"]
    assert list(anchors_frame["construct_seed__role"]) == [""]
    assert list(templates_frame["construct_seed__role"]) == ["template"]
    assert list(anchors_frame["construct_seed__source_ref"]) == ["canonical local note"]
    output = result.stdout or ""
    assert "manifest_id: custom_construct_inputs" in output
    assert "dataset: custom_promoters" in output
    assert "dataset: custom_templates" in output


def test_seed_import_manifest_uses_env_root_when_root_is_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usr_root = tmp_path / "usr_root"
    manifest_path = tmp_path / "import_manifest.yaml"
    manifest_path.write_text(
        """
manifest_id: custom_construct_inputs
datasets:
  - id: custom_promoters
    records:
      - label: anchor_part_beta
        topology: linear
        sequence: ACGT
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("DNADESIGN_USR_ROOT", str(usr_root))

    result = _RUNNER.invoke(
        app,
        [
            "seed",
            "import-manifest",
            "--manifest",
            manifest_path.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.stdout
    anchors = Dataset(usr_root, "custom_promoters")
    assert len(anchors.head(n=10)) == 1


def test_seed_import_manifest_requires_root_outside_checkout_without_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "import_manifest.yaml"
    manifest_path.write_text(
        """
manifest_id: custom_construct_inputs
datasets:
  - id: custom_promoters
    records:
      - label: anchor_part_beta
        topology: linear
        sequence: ACGT
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("DNADESIGN_USR_ROOT", raising=False)
    monkeypatch.setattr(seed_module, "project_root_or_none", lambda: None)

    with pytest.raises(
        seed_module.ConfigError,
        match="construct seed requires --root outside a dnadesign checkout unless DNADESIGN_USR_ROOT is set",
    ):
        seed_module.import_seed_manifest(root=None, manifest=manifest_path)


def test_seed_import_manifest_requires_manifest_id(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    manifest_path = tmp_path / "bad_import_manifest.yaml"
    manifest_path.write_text(
        """
datasets:
  - id: custom_promoters
    records:
      - label: anchor_part_beta
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
        "dnadesign.construct.src.seeding.bootstrap._seed_asset_payload",
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

    result = _RUNNER.invoke(app, ["seed", "anchor-template-demo", "--root", (tmp_path / "usr_root").as_posix()])

    assert result.exit_code == 1
    assert "start/end must be integers" in (result.stdout or "")


def test_seed_promoter_swap_demo_rejects_reversed_slot_bounds(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "dnadesign.construct.src.seeding.bootstrap._seed_asset_payload",
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
                    "start": 5,
                    "end": 5,
                    "expected_template_sequence": "ACGT",
                }
            ],
        },
    )

    result = _RUNNER.invoke(app, ["seed", "anchor-template-demo", "--root", (tmp_path / "usr_root").as_posix()])

    assert result.exit_code == 1
    assert "end must be greater than start" in (result.stdout or "")


def test_seed_import_manifest_rejects_non_string_alias_values(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    manifest_path = tmp_path / "bad_alias_manifest.yaml"
    manifest_path.write_text(
        """
manifest_id: custom_construct_inputs
datasets:
  - id: custom_promoters
    notes: Example anchors.
    records:
      - label: anchor_part_beta
        topology: linear
        aliases: [ok_alias, 42]
        source_ref: canonical local note
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
    assert "aliases must contain only strings" in (result.stdout or "")
