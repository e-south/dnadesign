"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_sources.py

Source-loading contract tests for Construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from dnadesign.construct.src.contracts.config import JobConfig, NormalizeTemplateConfig
from dnadesign.construct.src.contracts.errors import ValidationError
from dnadesign.construct.src.sources.input_rows import (
    classic_input_scan_fields,
    input_usr_labels,
    require_distinct_input_output_or_opt_in,
    scan_usr_rows,
)
from dnadesign.construct.src.sources.templates import load_normalize_template, load_template_sequence
from dnadesign.construct.tests.runtime.run_construct_helpers import write_registry
from dnadesign.usr import Dataset


@dataclass(frozen=True)
class _Batch:
    rows: list[dict[str, object]]

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    def to_pydict(self) -> dict[str, list[object]]:
        keys = sorted({key for row in self.rows for key in row})
        return {key: [row.get(key) for row in self.rows] for key in keys}


@dataclass(frozen=True)
class _Schema:
    names: list[str]


@dataclass
class _Dataset:
    rows: list[dict[str, object]]
    schema_names: list[str]
    scan_columns: list[str] | None = None
    include_overlays: bool | None = None

    def scan(self, *, columns: list[str], include_overlays: bool) -> list[_Batch]:
        self.scan_columns = columns
        self.include_overlays = include_overlays
        return [_Batch(self.rows)]

    def schema(self) -> _Schema:
        return _Schema(self.schema_names)


def _classic_cfg(*, template_source: dict[str, object], usr_root: Path) -> JobConfig:
    return JobConfig.model_validate(
        {
            "job": {
                "id": "source_fixture",
                "input": {
                    "source": {
                        "kind": "usr",
                        "dataset": "input_refs",
                        "root": usr_root.as_posix(),
                    },
                    "field": "sequence",
                },
                "template": {
                    "id": "template_fixture",
                    "source": template_source,
                },
                "parts": [
                    {
                        "name": "anchor",
                        "role": "anchor",
                        "sequence": {
                            "source": "input_field",
                            "field": "sequence",
                        },
                        "placement": {
                            "kind": "replace",
                            "orientation": "forward",
                            "locator": {
                                "kind": "coordinates",
                                "start": 0,
                                "end": 4,
                            },
                        },
                    }
                ],
                "realize": {
                    "mode": "full_construct",
                },
                "output": {
                    "target": {
                        "kind": "usr",
                        "dataset": "output_refs",
                        "root": usr_root.as_posix(),
                    }
                },
            }
        }
    )


def test_load_template_sequence_rejects_multi_record_fasta(tmp_path: Path) -> None:
    fasta = tmp_path / "template.fa"
    fasta.write_text(">one\nAAAA\n>two\nCCCC\n", encoding="utf-8")
    cfg = _classic_cfg(template_source={"kind": "path", "path": "template.fa"}, usr_root=tmp_path / "usr")

    with pytest.raises(ValidationError, match="exactly one record"):
        load_template_sequence(tmp_path, cfg)


def test_load_normalize_template_reads_single_record_fasta(tmp_path: Path) -> None:
    fasta = tmp_path / "normalize.fa"
    fasta.write_text(">template\nAAAA\nCCCC\n", encoding="utf-8")
    cfg = NormalizeTemplateConfig.model_validate(
        {
            "id": "normalize_fixture",
            "source": {
                "kind": "path",
                "path": "normalize.fa",
            },
        }
    )

    template = load_normalize_template(base_dir=tmp_path, cfg=cfg)

    assert template.id == "normalize_fixture"
    assert template.kind == "path"
    assert template.sequence == "AAAACCCC"
    assert template.source == str(fasta)


def test_load_template_sequence_uses_exact_explicit_usr_root(tmp_path: Path) -> None:
    usr_root = tmp_path / "operator_usr"
    usr_root.mkdir()
    (usr_root / "__init__.py").write_text("", encoding="utf-8")
    write_registry(usr_root)
    dataset = Dataset(usr_root, "templates")
    dataset.init(source="test")
    result = dataset.add_sequences(["AAAACCCC"], bio_type="dna", alphabet="dna_4", source="test")
    record_id = result.ids[0]
    cfg = _classic_cfg(
        template_source={
            "kind": "usr",
            "dataset": "templates",
            "field": "sequence",
            "record_id": record_id,
        },
        usr_root=usr_root,
    )

    template = load_template_sequence(tmp_path, cfg, usr_root=usr_root)

    assert template.sequence == "AAAACCCC"
    assert template.record_id == record_id


def test_scan_usr_rows_preserves_requested_id_order_and_overlays() -> None:
    dataset = _Dataset(
        rows=[
            {"id": "row_b", "sequence": "CCCC"},
            {"id": "row_a", "sequence": "AAAA"},
        ],
        schema_names=["id", "sequence"],
    )

    rows = scan_usr_rows(dataset, columns=["id", "sequence"], ids=["row_a", "row_b"])

    assert [row["id"] for row in rows] == ["row_a", "row_b"]
    assert dataset.scan_columns == ["id", "sequence"]
    assert dataset.include_overlays is True


def test_scan_usr_rows_fails_fast_for_missing_requested_id() -> None:
    dataset = _Dataset(rows=[{"id": "row_a", "sequence": "AAAA"}], schema_names=["id", "sequence"])

    with pytest.raises(ValidationError, match="requested input id"):
        scan_usr_rows(dataset, columns=["id", "sequence"], ids=["row_a", "missing"])


def test_classic_input_scan_fields_adds_optional_label_overlays(tmp_path: Path) -> None:
    dataset = _Dataset(
        rows=[],
        schema_names=["id", "sequence", "usr_label__primary", "usr_label__aliases", "ignored_overlay"],
    )
    cfg = _classic_cfg(template_source={"kind": "literal", "sequence": "AAAACCCC"}, usr_root=tmp_path / "usr")

    assert classic_input_scan_fields(dataset, cfg) == [
        "id",
        "sequence",
        "usr_label__aliases",
        "usr_label__primary",
    ]


def test_input_usr_labels_deduplicates_primary_and_blank_aliases() -> None:
    primary, aliases = input_usr_labels(
        {
            "usr_label__primary": "anchor_a",
            "usr_label__aliases": ["anchor_a", "", "alias_one", "alias_one", "alias_two"],
        }
    )

    assert primary == "anchor_a"
    assert aliases == ["alias_one", "alias_two"]


def test_require_distinct_input_output_or_opt_in_rejects_recursive_dataset(tmp_path: Path) -> None:
    cfg = _classic_cfg(template_source={"kind": "literal", "sequence": "AAAACCCC"}, usr_root=tmp_path / "usr")
    cfg.job.output.target.dataset = cfg.job.input.source.dataset

    with pytest.raises(ValidationError, match="same root/dataset as input"):
        require_distinct_input_output_or_opt_in(
            cfg=cfg,
            input_root=tmp_path / "usr",
            output_root=tmp_path / "usr",
        )
