"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/promoter_candidate_bindings/test_resolution.py

Exact identity and render-projection tests for promoter candidate bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    BindingSourceArtifact,
    PromoterCandidateBindingsError,
    preview_promoter_candidate_bindings,
    resolve_promoter_candidate_bindings,
)

SEQUENCE = "ACGTACGT" + "CTGACA" + "AAAA" + "TATAAT"


def densegen_candidate(*, candidate_id: str = "candidate-1", sequence: str = SEQUENCE) -> dict[str, object]:
    return {
        "id": candidate_id,
        "sequence": sequence,
        "usr_label__primary": None,
        "opal_candidate__source_class": "densegen",
        "opal_candidate__design_family": "ethanol",
        "densegen__plan": "ethanol__sig35=f",
        "densegen__run_id": "run-1",
        "densegen__sampling_library_hash": "library-sha",
        "densegen__used_tfbs_detail": [
            {
                "part_kind": "tfbs",
                "sequence": "ACGT",
                "regulator": "baeR",
                "offset": 0,
                "offset_raw": 0,
                "length": 4,
                "end": 4,
                "orientation": "fwd",
            },
            {
                "part_kind": "fixed_element",
                "role": "upstream",
                "constraint_name": "sigma70_core",
                "sequence": "CTGACA",
                "offset": 8,
                "offset_raw": 8,
                "length": 6,
                "end": 14,
                "spacer_length": 4,
                "placement_index": 0,
            },
            {
                "part_kind": "fixed_element",
                "role": "downstream",
                "constraint_name": "sigma70_core",
                "sequence": "TATAAT",
                "offset": 18,
                "offset_raw": 18,
                "length": 6,
                "end": 24,
                "spacer_length": 4,
                "placement_index": 0,
            },
        ],
        "densegen__required_regulators": ["baeR"],
    }


def aliases(*, sequence: str = SEQUENCE) -> pd.DataFrame:
    return pd.DataFrame(
        [
            alias_row("reader.design_id", "pDual-10-A", sequence=sequence, authority="reader-source"),
            alias_row("synthesis.name", "A", sequence=sequence, authority="synthesis-source"),
        ]
    )


def alias_row(
    namespace: str,
    alias: str,
    *,
    sequence: str = SEQUENCE,
    authority: str = "source",
) -> dict[str, str]:
    return {
        "alias_namespace": namespace,
        "alias": alias,
        "display_label": alias,
        "candidate_id": "candidate-1",
        "authority_sequence": sequence,
        "sequence_authority_dataset_id": authority,
        "sequence_authority_id": f"{authority}-row",
        "sequence_authority_sha256": hashlib.sha256(authority.encode()).hexdigest(),
    }


def resolve(alias_rows: pd.DataFrame | None = None, candidates: pd.DataFrame | None = None) -> pd.DataFrame:
    return resolve_promoter_candidate_bindings(
        alias_rows=aliases() if alias_rows is None else alias_rows,
        candidate_records=pd.DataFrame([densegen_candidate()]) if candidates is None else candidates,
        genbank_annotations=pd.DataFrame(),
        candidate_table_id="candidate-table",
        candidate_selection_sha256="b" * 64,
    )


def test_typed_aliases_share_candidate_with_source_specific_authority() -> None:
    resolved = resolve()

    assert resolved[["alias_namespace", "alias"]].to_records(index=False).tolist() == [
        ("reader.design_id", "pDual-10-A"),
        ("synthesis.name", "A"),
    ]
    assert resolved["candidate_id"].tolist() == ["candidate-1", "candidate-1"]
    assert resolved["sequence_sha256"].unique().tolist() == [hashlib.sha256(SEQUENCE.encode()).hexdigest()]
    assert resolved["binding_method"].unique().tolist() == ["exact_alias"]
    assert not any("latentdna" in column.lower() or column.lower().startswith("x_") for column in resolved)


def test_same_alias_text_is_valid_in_distinct_namespaces() -> None:
    rows = pd.DataFrame(
        [
            alias_row("reader.design_id", "A", authority="reader"),
            alias_row("synthesis.name", "A", authority="synthesis"),
        ]
    )
    assert len(resolve(rows)) == 2


@pytest.mark.parametrize(
    ("alias_rows", "candidates", "message"),
    [
        (
            pd.DataFrame([alias_row("reader.design_id", "A"), alias_row("reader.design_id", "A")]),
            None,
            "unique within each namespace",
        ),
        (aliases().assign(candidate_id="missing"), None, "has no exact candidate"),
        (aliases().assign(authority_sequence="TTTT"), None, "sequence does not match"),
    ],
)
def test_resolution_rejects_non_exact_identity(
    alias_rows: pd.DataFrame,
    candidates: pd.DataFrame | None,
    message: str,
) -> None:
    with pytest.raises(PromoterCandidateBindingsError, match=message):
        resolve(alias_rows, candidates)


@pytest.mark.parametrize("mutation", ["missing_tfbs", "unpaired_fixed_element"])
def test_public_baserender_contract_rejects_invalid_densegen_projection(mutation: str) -> None:
    candidate = densegen_candidate()
    annotations = list(candidate["densegen__used_tfbs_detail"])
    if mutation == "missing_tfbs":
        annotations[0] = {**annotations[0], "sequence": "GGGG"}
    else:
        annotations = annotations[:-1]
    candidate["densegen__used_tfbs_detail"] = annotations

    with pytest.raises(PromoterCandidateBindingsError, match="incompatible with BaseRender"):
        preview_promoter_candidate_bindings(
            alias_rows=aliases(),
            candidate_records=pd.DataFrame([candidate]),
            genbank_annotations=pd.DataFrame(),
            candidate_table_id="candidate-table",
            candidate_selection_sha256="b" * 64,
            source_artifacts=(BindingSourceArtifact("source", "inputs/source.parquet", "a" * 64),),
        )


def test_construct_candidate_uses_genbank_adapter() -> None:
    candidate = {
        **densegen_candidate(),
        "usr_label__primary": "spyp",
        "opal_candidate__source_class": "construct_derived",
        "opal_candidate__design_family": "control",
        "densegen__plan": None,
        "densegen__run_id": None,
        "densegen__sampling_library_hash": None,
        "densegen__used_tfbs_detail": None,
        "densegen__required_regulators": None,
    }
    annotations = pd.DataFrame(
        [
            {
                "id": "candidate-1",
                "seq_annot__features": [
                    {
                        "feature_id": "feature-1",
                        "feature_type": "promoter",
                        "label": "spyp",
                        "start_0": 0,
                        "end_0": 6,
                        "strand": 1,
                    }
                ],
                "seq_annot__source_artifact_uri": "artifacts/genbank/spyp.gb",
            }
        ]
    )

    resolved = resolve_promoter_candidate_bindings(
        alias_rows=aliases().iloc[:1],
        candidate_records=pd.DataFrame([candidate]),
        genbank_annotations=annotations,
        candidate_table_id="candidate-table",
        candidate_selection_sha256="b" * 64,
    )

    assert resolved.loc[0, "baserender_adapter_kind"] == "usr_genbank_annotations_v1"
    assert resolved.loc[0, "seq_annot__source_file"] == "artifacts/genbank/spyp.gb"


def test_genbank_source_reference_must_be_confined() -> None:
    candidate = {
        **densegen_candidate(),
        "usr_label__primary": "spyp",
        "opal_candidate__source_class": "construct_derived",
        "opal_candidate__design_family": "control",
    }
    annotations = pd.DataFrame(
        [
            {
                "id": "candidate-1",
                "seq_annot__features": [
                    {
                        "feature_id": "f",
                        "feature_type": "promoter",
                        "label": "spyp",
                        "start_0": 0,
                        "end_0": 4,
                        "strand": 1,
                    }
                ],
                "seq_annot__source_artifact_uri": "/private/spyp.gb",
            }
        ]
    )
    with pytest.raises(PromoterCandidateBindingsError, match="relative confined path"):
        resolve_promoter_candidate_bindings(
            alias_rows=aliases().iloc[:1],
            candidate_records=pd.DataFrame([candidate]),
            genbank_annotations=annotations,
            candidate_table_id="candidate-table",
            candidate_selection_sha256="b" * 64,
        )
