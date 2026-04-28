from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from dnadesign.usr import Dataset, load_sequence_views
from dnadesign.usr.scripts.materialize_promoter_anchor_sequence_views import (
    materialize_promoter_anchor_sequence_views,
)
from dnadesign.usr.src.contracts import compute_id


def _usr_root(tmp_path: Path) -> Path:
    usr_root = tmp_path / "usr_datasets"
    usr_root.mkdir()
    shutil.copy(Path("src/dnadesign/usr/datasets/registry.yaml"), usr_root / "registry.yaml")
    return usr_root


def test_materialize_promoter_anchor_sequence_views_uses_construct_insert_for_every_anchor_row(
    tmp_path: Path,
) -> None:
    usr_root = _usr_root(tmp_path)
    dataset = Dataset(usr_root, "usr_prom_eth_cip_anchor")
    native_60 = "ACGT" * 15
    native_long = "A" * 71
    derived_core = "T" * 60
    core_id = compute_id("dna", derived_core)
    with dataset.write_session() as session:
        session.init(source="fixture", notes="merged anchor fixture")
        session.import_rows(
            [
                {
                    "id": compute_id("dna", native_60),
                    "bio_type": "dna",
                    "sequence": native_60,
                    "alphabet": "dna_4",
                    "length": len(native_60),
                    "source": "densegen",
                },
                {
                    "id": compute_id("dna", native_long),
                    "bio_type": "dna",
                    "sequence": native_long,
                    "alphabet": "dna_4",
                    "length": len(native_long),
                    "source": "usr_promoter_references",
                },
                {
                    "id": core_id,
                    "bio_type": "dna",
                    "sequence": derived_core,
                    "alphabet": "dna_4",
                    "length": len(derived_core),
                    "source": "construct_prom_eth_cip_reference_core60",
                },
            ],
            source="fixture",
        )
        session.write_overlay(
            "usr_label",
            pd.DataFrame(
                [
                    {"id": core_id, "usr_label__primary": "spyP_core60", "usr_label__aliases": ["spyP_core"]},
                ]
            ),
        )
        session.write_overlay(
            "derived",
            pd.DataFrame(
                [
                    {
                        "id": core_id,
                        "derived__product_kind": "analysis_window",
                        "derived__operation": "normalize_anchor",
                        "derived__analysis_only": True,
                        "derived__spec_id": "construct_prom_eth_cip_reference_core60",
                    }
                ]
            ),
        )

    dry_run = materialize_promoter_anchor_sequence_views(usr_root=usr_root, write=False)

    assert dry_run.views_planned == 3
    assert dry_run.views_written == 0
    assert dry_run.analysis_only_views == 1
    assert dry_run.analysis_window_source_rows == 1

    result = materialize_promoter_anchor_sequence_views(usr_root=usr_root, write=True)

    assert result.views_written == 3
    views = sorted(load_sequence_views(dataset), key=lambda view: view.sequence_id)
    assert {view.product_kind for view in views} == {"construct_insert"}
    assert {view.context_kind for view in views} == {"anchor_only"}
    assert {view.recommended_pooling for view in views} == {"seq_mean"}
    assert sum(1 for view in views if view.analysis_only) == 1
    assert next(view for view in views if view.sequence_id == core_id).source_label == "spyP_core60"

    rerun = materialize_promoter_anchor_sequence_views(usr_root=usr_root, write=True)

    assert rerun.views_written == 0
    assert len(pq.read_table(dataset.dir / "_views" / "sequence_views.parquet")) == 3
