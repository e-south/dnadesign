from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from dnadesign.usr.scripts import build_reader_sfxi_reference_overlay as overlay
from dnadesign.usr.src.contracts import SchemaError

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _base_records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": "usr-es1", "sequence": "AACCGGTT"},
            {"id": "usr-es2", "sequence": "GGCCAATT"},
        ]
    )


def _scored_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "design_id": "pDual-10-ES1p",
                "sequence": "aaccggtt",
                "sequence_source_id": "reader-seq-1",
                "experiment_id": "20260117_sfxi_ref-pDual10",
                "experiment_date": 20260117,
                "time_selected_h": 19.2,
                "reference_design_id": "pDual-10",
                "r_logic": 2.1,
                "flat_logic": False,
                "setpoint_name": "and",
                "objective_name": "sfxi_v1",
                "api_version": "1",
                "state_order": ["00", "10", "01", "11"],
                "setpoint_vector": [0.0, 0.0, 0.0, 1.0],
                "denom_percentile": 95,
                "denom_used": 8.5,
                "logic_fidelity": 0.4,
                "effect_raw": 1.2,
                "effect_scaled": 0.14,
                "sfxi": 0.056,
                "clip_lo_mask": False,
                "clip_hi_mask": False,
                "intensity_disabled": False,
            },
            {
                "design_id": "pDual-10-ES2p",
                "sequence": "ggccaatt",
                "sequence_source_id": "reader-seq-2",
                "experiment_id": "20251103_sfxi_pES1-8_ref-pDual10",
                "experiment_date": 20251103,
                "time_selected_h": 17.5,
                "reference_design_id": "pDual-10",
                "r_logic": 3.1,
                "flat_logic": False,
                "setpoint_name": "and",
                "objective_name": "sfxi_v1",
                "api_version": "1",
                "state_order": ["00", "10", "01", "11"],
                "setpoint_vector": [0.0, 0.0, 0.0, 1.0],
                "denom_percentile": 95,
                "denom_used": 8.5,
                "logic_fidelity": 0.7,
                "effect_raw": 2.3,
                "effect_scaled": 0.27,
                "sfxi": 0.189,
                "clip_lo_mask": False,
                "clip_hi_mask": False,
                "intensity_disabled": False,
            },
        ]
    )


def test_reader_sfxi_overlay_joins_to_usr_ids_by_case_insensitive_sequence() -> None:
    frame = overlay.build_sfxi_reference_overlay_frame(
        base_records=_base_records(),
        scored_rows=_scored_rows(),
        collection_id="reader_sfxi_pdual10_latest",
        campaign_id="20260501_sfxi_promoter_setpoint_scatter",
        source_ref="../reader/experiments/2026/fixture/vec8.parquet",
    )

    assert frame["id"].tolist() == ["usr-es1", "usr-es2"]
    assert frame["sfxi_ref__reference_instance_id"].tolist() == ["pDual-10-ES1p", "pDual-10-ES2p"]
    assert frame["sfxi_ref__metric_id"].tolist() == ["sfxi_v1/and/sfxi", "sfxi_v1/and/sfxi"]
    assert frame["sfxi_ref__metric_value"].tolist() == [0.056, 0.189]
    assert frame["sfxi_ref__metric_provenance"].str.contains("reader.vec8").all()
    assert frame["sfxi_ref__batch_id"].tolist() == [
        "20260117_sfxi_ref-pDual10",
        "20251103_sfxi_pES1-8_ref-pDual10",
    ]
    assert frame["sfxi_ref__setpoint_vector"].tolist() == [[0.0, 0.0, 0.0, 1.0]] * 2


def test_reader_sfxi_overlay_rejects_duplicate_metric_instances() -> None:
    duplicate = pd.concat([_scored_rows(), _scored_rows().iloc[[0]]], ignore_index=True)

    with pytest.raises(SchemaError, match="duplicate normalized sequence"):
        overlay.build_sfxi_reference_overlay_frame(
            base_records=_base_records(),
            scored_rows=duplicate,
            collection_id="reader_sfxi_pdual10_latest",
            campaign_id="20260501_sfxi_promoter_setpoint_scatter",
            source_ref="fixture",
        )


def test_reader_sfxi_overlay_rejects_nonfinite_metric_values() -> None:
    scored = _scored_rows()
    scored.loc[0, "sfxi"] = math.nan

    with pytest.raises(SchemaError, match="non-finite"):
        overlay.build_sfxi_reference_overlay_frame(
            base_records=_base_records(),
            scored_rows=scored,
            collection_id="reader_sfxi_pdual10_latest",
            campaign_id="20260501_sfxi_promoter_setpoint_scatter",
            source_ref="fixture",
        )


def test_reader_sfxi_overlay_table_uses_registered_contract_types() -> None:
    frame = overlay.build_sfxi_reference_overlay_frame(
        base_records=_base_records(),
        scored_rows=_scored_rows(),
        collection_id="reader_sfxi_pdual10_latest",
        campaign_id="20260501_sfxi_promoter_setpoint_scatter",
        source_ref="fixture",
    )

    table = overlay.sfxi_reference_overlay_table(frame)

    assert table.schema.field("sfxi_ref__metric_value").type == pa.float64()
    assert table.schema.field("sfxi_ref__setpoint_vector").type == pa.list_(pa.float64())
    assert table.schema.field("sfxi_ref__clip_lo_mask").type == pa.bool_()


def test_reader_sfxi_overlay_records_canonical_vec8_source_ref(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, Path] = {}

    def fake_read_usr_base_records(usr_root: Path, dataset_name: str) -> pd.DataFrame:
        del usr_root, dataset_name
        return _base_records()

    def fake_read_reader_vec8(vec8_path: Path) -> pd.DataFrame:
        captured["vec8_path"] = vec8_path
        return pd.DataFrame({"placeholder": [1]})

    def fake_score_reader_vec8(
        vec8: pd.DataFrame,
        *,
        reader_root: Path,
        setpoint_name: str,
        setpoint_vector,
    ) -> pd.DataFrame:
        del vec8, reader_root, setpoint_name, setpoint_vector
        return _scored_rows()

    monkeypatch.setattr(overlay, "read_usr_base_records", fake_read_usr_base_records)
    monkeypatch.setattr(overlay, "read_reader_vec8", fake_read_reader_vec8)
    monkeypatch.setattr(overlay, "score_reader_vec8", fake_score_reader_vec8)

    frame = overlay.build_overlay_from_reader(
        usr_root=tmp_path / "usr",
        dataset_name="usr_prom_eth_cip_anchor",
        reader_root=Path("reader"),
        vec8_path=Path("reader_outputs/vec8.parquet"),
        collection_id="reader_sfxi_pdual10_latest",
        campaign_id="20260501_sfxi_promoter_setpoint_scatter",
    )

    expected_vec8_path = (tmp_path / "reader_outputs/vec8.parquet").resolve()
    assert captured["vec8_path"] == expected_vec8_path
    assert frame["sfxi_ref__source_ref"].unique().tolist() == [str(expected_vec8_path)]


def test_reader_sfxi_overlay_validates_against_packaged_usr_registry() -> None:
    frame = overlay.build_sfxi_reference_overlay_frame(
        base_records=_base_records(),
        scored_rows=_scored_rows(),
        collection_id="reader_sfxi_pdual10_latest",
        campaign_id="20260501_sfxi_promoter_setpoint_scatter",
        source_ref="fixture",
    )

    table = overlay.validate_sfxi_reference_overlay_contract(
        frame,
        usr_root=_REPO_ROOT / "src/dnadesign/usr/datasets",
    )

    assert table.num_rows == 2
    assert "sfxi_ref__metric_value" in table.column_names
