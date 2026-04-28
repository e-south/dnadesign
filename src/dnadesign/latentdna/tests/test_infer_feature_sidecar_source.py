"""Contracts for LatentDNA Infer feature sidecar sources."""

from __future__ import annotations

import pyarrow as pa

from dnadesign.latentdna.src.sources.infer_feature_sidecar_source import _stable_batch_schema


def test_stable_batch_schema_uses_later_non_null_metadata_values() -> None:
    schema = _stable_batch_schema(
        ["alias_id", "usr_label__primary", "value"],
        {
            "fv_a": {"alias_id": "alias_a", "usr_label__primary": None},
            "fv_b": {"alias_id": "alias_b", "usr_label__primary": "spyP"},
        },
    )

    assert schema.field("usr_label__primary").type == pa.string()
    assert schema.field("value").type == pa.list_(pa.float64())
