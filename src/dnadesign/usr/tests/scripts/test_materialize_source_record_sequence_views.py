from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from dnadesign.usr import Dataset, load_sequence_views, load_view_semantics
from dnadesign.usr.scripts.materialize_source_record_sequence_views import (
    materialize_source_record_sequence_views,
)


def _usr_root(tmp_path: Path) -> Path:
    usr_root = tmp_path / "usr_datasets"
    usr_root.mkdir()
    shutil.copy(Path("src/dnadesign/usr/datasets/registry.yaml"), usr_root / "registry.yaml")
    return usr_root


def _write_rows(dataset: Dataset, rows: list[dict[str, object]]) -> None:
    with dataset.write_session() as session:
        session.init(source="fixture", notes=f"{dataset.name} fixture")
        session.import_rows(rows)


def test_materialize_source_record_sequence_views_writes_missing_sidecars(tmp_path: Path) -> None:
    usr_root = _usr_root(tmp_path)
    dataset = Dataset(usr_root, "densegen_demo_sampling_baseline")
    _write_rows(
        dataset,
        [
            {
                "sequence": "ACGT" * 15,
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "plan_pool__demo",
            },
            {
                "sequence": "TGCA" * 15,
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "plan_pool__demo",
            },
        ],
    )

    dry_run = materialize_source_record_sequence_views(
        usr_root=usr_root,
        dataset_names=[dataset.name],
        write=False,
    )

    assert dry_run.views_planned == 2
    assert dry_run.view_semantics_planned == 2
    assert dry_run.views_written == 0
    assert dry_run.by_dataset[dataset.name]["state"] == "planned"

    result = materialize_source_record_sequence_views(
        usr_root=usr_root,
        dataset_names=[dataset.name],
        write=True,
    )

    assert result.views_written == 2
    assert result.view_semantics_written == 2
    views = load_sequence_views(dataset)
    semantics = load_view_semantics(dataset)
    assert len(views) == 2
    assert len(semantics) == 2
    assert {view.product_kind for view in views} == {"source_record"}
    assert {view.orientation for view in views} == {"unknown"}
    assert {view.recommended_pooling for view in views} == {"seq_mean"}
    assert {row.source_family for row in semantics} == {"densegen_demo"}
    assert {row.selection_basis for row in semantics} == {"densegen_source_record"}

    rerun = materialize_source_record_sequence_views(
        usr_root=usr_root,
        dataset_names=[dataset.name],
        write=True,
    )

    assert rerun.views_written == 0
    assert rerun.view_semantics_written == 0
    assert len(load_sequence_views(dataset)) == 2
    assert len(load_view_semantics(dataset)) == 2


def test_materialize_source_record_sequence_views_uses_label_aliases_and_template_semantics(
    tmp_path: Path,
) -> None:
    usr_root = _usr_root(tmp_path)
    dataset = Dataset(usr_root, "usr_pdual10_plasmid_template")
    _write_rows(
        dataset,
        [
            {
                "sequence": "ACGT" * 25,
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "construct seed promoter-swap-demo",
                "usr_label__primary": "pDual-10",
                "usr_label__aliases": ["pDual10"],
            }
        ],
    )
    dataset.write_overlay(
        "usr_label",
        pd.DataFrame(
            [
                {
                    "id": str(dataset.head(1, include_derived=False)["id"].tolist()[0]),
                    "usr_label__primary": "pDual-10",
                    "usr_label__aliases": ["pDual10"],
                }
            ]
        ),
    )

    materialize_source_record_sequence_views(usr_root=usr_root, dataset_names=[dataset.name], write=True)

    view = load_sequence_views(dataset)[0]
    semantics = load_view_semantics(dataset)[0]
    assert view.view_name == "pDual-10_source_record"
    assert view.aliases == ["pDual-10", "pDual10"]
    assert semantics.source_family == "construct_template"
    assert semantics.selection_basis == "template_source_record"
    assert semantics.role_tags == ["source_record", "template_seed"]


def test_materialize_source_record_sequence_views_drops_non_unique_aliases(tmp_path: Path) -> None:
    usr_root = _usr_root(tmp_path)
    dataset = Dataset(usr_root, "usr_sfxi_pdual10_densegen_promoters")
    _write_rows(
        dataset,
        [
            {
                "sequence": "ACGT" * 15,
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "archived/60bp_dual_promoter_cpxR_LexA;reader_sfxi_pdual",
            },
            {
                "sequence": "TGCA" * 15,
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "archived/60bp_dual_promoter_cpxR_LexA;reader_sfxi_pdual",
            },
        ],
    )
    ids = dataset.head(2, include_derived=False)["id"].tolist()
    dataset.write_overlay(
        "usr_label",
        pd.DataFrame(
            [
                {
                    "id": str(ids[0]),
                    "usr_label__primary": "pDual-10-ES1p",
                    "usr_label__aliases": ["ES1p", "reader_experiment:shared"],
                },
                {
                    "id": str(ids[1]),
                    "usr_label__primary": "pDual-10-ES2p",
                    "usr_label__aliases": ["ES2p", "reader_experiment:shared"],
                },
            ]
        ),
    )

    materialize_source_record_sequence_views(usr_root=usr_root, dataset_names=[dataset.name], write=True)

    aliases_by_name = {view.view_name: view.aliases for view in load_sequence_views(dataset)}
    assert aliases_by_name["pDual-10-ES1p_source_record"] == ["pDual-10-ES1p", "ES1p"]
    assert aliases_by_name["pDual-10-ES2p_source_record"] == ["pDual-10-ES2p", "ES2p"]


def test_materialize_source_record_sequence_views_refuses_archived_dataset(tmp_path: Path) -> None:
    usr_root = _usr_root(tmp_path)

    with pytest.raises(ValueError, match="Archived datasets are excluded"):
        materialize_source_record_sequence_views(
            usr_root=usr_root,
            dataset_names=["archived/60bp_dual_promoter_cpxR_LexA"],
            write=False,
        )
