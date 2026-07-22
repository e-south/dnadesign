"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/promoter_candidate_bindings/test_sources_and_cli.py

Study source-registry and CLI tests for promoter candidate bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    PromoterCandidateBindingsError,
    preview_promoter_candidate_bindings_from_repo,
    verify_promoter_candidate_bindings,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings.cli import main

from .test_resolution import SEQUENCE, densegen_candidate


def test_registry_builds_namespace_qualified_alias_universe(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    write_repository_sources(repo_root)

    preview = preview_promoter_candidate_bindings_from_repo(repo_root)

    assert len(preview.bindings) == 7
    assert preview.bindings["candidate_id"].nunique() == 4
    assert preview.bindings["alias_namespace"].value_counts().to_dict() == {
        "reader.design_id": 4,
        "source.alias": 2,
        "synthesis.name": 1,
    }
    assert set(preview.bindings["baserender_adapter_kind"]) == {
        "densegen_tfbs",
        "usr_genbank_annotations_v1",
    }
    assert "promoter-candidate-binding-source-registry" in {
        artifact.artifact_id for artifact in preview.source_artifacts
    }


def test_cli_previews_materializes_and_verifies(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = tmp_path / "dnadesign"
    write_repository_sources(repo_root)
    allowed_root = tmp_path / "artifacts"
    bundle = allowed_root / "latest"

    assert main(["preview", "--repo-root", str(repo_root)]) == 0
    preview_payload = json.loads(capsys.readouterr().out)
    assert preview_payload["schema_id"] == "dnadesign.study.promoter_candidate_bindings.v1"
    assert preview_payload["binding_count"] == 7

    assert (
        main(
            [
                "materialize",
                "--repo-root",
                str(repo_root),
                "--out-dir",
                str(bundle),
                "--allowed-output-root",
                str(allowed_root),
            ]
        )
        == 0
    )
    materialized = json.loads(capsys.readouterr().out)
    assert materialized["bindings_parquet"].endswith("bindings.parquet")
    assert main(["verify", "--bundle-dir", str(bundle), "--allowed-root", str(allowed_root)]) == 0
    assert json.loads(capsys.readouterr().out)["binding_count"] == 7
    assert verify_promoter_candidate_bindings(bundle, allowed_root=allowed_root).candidate_count == 4


def test_registry_rejects_malformed_alias_collection(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    write_repository_sources(repo_root)
    registry_path = registry_file(repo_root)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    payload["alias_sources"][1]["config"]["aliases"] = "spyp"
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PromoterCandidateBindingsError, match="must be a non-empty list"):
        preview_promoter_candidate_bindings_from_repo(repo_root)


def test_registry_uses_exact_reference_labels(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    write_repository_sources(repo_root)
    path = repo_root / "data/references/records.parquet"
    records = pd.read_parquet(path)
    records.loc[records["usr_label__primary"].eq("spyp"), "usr_label__primary"] = "spyp-extra"
    records.to_parquet(path, index=False)

    with pytest.raises(PromoterCandidateBindingsError, match="must resolve exactly once; found 0"):
        preview_promoter_candidate_bindings_from_repo(repo_root)


def test_synthesis_manifest_digest_is_enforced(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    write_repository_sources(repo_root)
    record_path = repo_root / "record/synthesis_handoffs.yaml"
    payload = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    payload["handoffs"][0]["expected_campaigns"][0]["manifest_sha256"] = "0" * 64
    record_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PromoterCandidateBindingsError, match="digest mismatch"):
        preview_promoter_candidate_bindings_from_repo(repo_root)


def test_synthesis_alias_source_accepts_one_selection_batch_artifact(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    write_repository_sources(repo_root)
    record_path = repo_root / "record/synthesis_handoffs.yaml"
    payload = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    handoff = payload["handoffs"][0]
    source_entry = handoff.pop("expected_campaigns")[0]
    handoff["source_authority"] = "opal_selection_batch"
    source_entry.pop("source_campaign_slug")
    source_entry["campaign_slug"] = "secg_msrb_greedy"
    manifest_path = repo_root / source_entry["manifest_path"]
    manifest = pd.read_csv(manifest_path)
    manifest["campaign_slug"] = "secg_msrb_greedy"
    manifest.to_csv(manifest_path, index=False)
    source_entry["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    handoff["expected_artifact"] = source_entry
    record_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    preview = preview_promoter_candidate_bindings_from_repo(repo_root)

    assert "SYN-01" in set(preview.bindings["alias"])


def test_synthesis_alias_source_rejects_ambiguous_artifact_shapes(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    write_repository_sources(repo_root)
    record_path = repo_root / "record/synthesis_handoffs.yaml"
    payload = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    payload["handoffs"][0]["expected_artifact"] = payload["handoffs"][0]["expected_campaigns"][0]
    record_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PromoterCandidateBindingsError, match="exactly one"):
        preview_promoter_candidate_bindings_from_repo(repo_root)


def write_repository_sources(repo_root: Path) -> None:
    write_sequence_views(repo_root)
    write_reference_records(repo_root)
    write_synthesis_handoff(repo_root)
    write_candidate_table(repo_root)
    write_registry(repo_root)


def write_sequence_views(repo_root: Path) -> None:
    root = repo_root / "data/reader-source"
    root.mkdir(parents=True)
    pd.DataFrame([{"id": "candidate-source", "sequence": SEQUENCE}]).to_parquet(root / "records.parquet", index=False)
    pd.DataFrame([{"sequence_id": "candidate-source", "source_label": "pDual-10-A"}]).to_parquet(
        root / "views.parquet", index=False
    )


def write_reference_records(repo_root: Path) -> None:
    root = repo_root / "data/references"
    root.mkdir(parents=True)
    rows = [
        {"id": "candidate-spyp", "sequence": SEQUENCE, "usr_label__primary": "spyp"},
        {"id": "candidate-sulap", "sequence": SEQUENCE, "usr_label__primary": "sulAp"},
    ]
    pd.DataFrame(rows).to_parquet(root / "records.parquet", index=False)
    pd.DataFrame(
        [
            {
                "id": row["id"],
                "seq_annot__source_artifact_uri": f"artifacts/{row['usr_label__primary']}.gb",
                "seq_annot__features": [
                    {
                        "feature_id": f"{row['id']}-feature",
                        "feature_type": "promoter",
                        "label": row["usr_label__primary"],
                        "start_0": 0,
                        "end_0": 6,
                        "strand": 1,
                    }
                ],
            }
            for row in rows
        ]
    ).to_parquet(root / "annotations.parquet", index=False)


def write_synthesis_handoff(repo_root: Path) -> None:
    manifest_path = repo_root / "outputs/pre-assay/manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "id": "candidate-synthesis",
                "synthesis_name": "SYN-01",
                "core_sequence": SEQUENCE,
                "campaign_slug": "source-selection",
                "validation_status": "pass",
            }
        ]
    ).to_csv(manifest_path, index=False)
    record_path = repo_root / "record/synthesis_handoffs.yaml"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        yaml.safe_dump(
            {
                "handoffs": [
                    {
                        "handoff_id": "pre-assay-1",
                        "source_authority": "study_batch0_selector",
                        "expected_campaigns": [
                            {
                                "campaign_slug": "sfxi-source-selection",
                                "source_campaign_slug": "source-selection",
                                "expected_rows": 1,
                                "manifest_path": manifest_path.relative_to(repo_root).as_posix(),
                                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                            }
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def write_candidate_table(repo_root: Path) -> None:
    root = repo_root / "data/candidates"
    root.mkdir(parents=True)
    candidates = [
        densegen_candidate(candidate_id="candidate-source"),
        densegen_candidate(candidate_id="candidate-synthesis"),
        control_candidate("candidate-spyp", "spyp"),
        control_candidate("candidate-sulap", "sulAp"),
    ]
    pd.DataFrame(candidates).to_parquet(root / "records.parquet", index=False)


def control_candidate(candidate_id: str, label: str) -> dict[str, object]:
    return {
        **densegen_candidate(candidate_id=candidate_id),
        "usr_label__primary": label,
        "opal_candidate__source_class": "construct_derived",
        "opal_candidate__design_family": "control",
        "densegen__plan": None,
        "densegen__run_id": None,
        "densegen__sampling_library_hash": None,
        "densegen__used_tfbs_detail": None,
        "densegen__required_regulators": None,
    }


def write_registry(repo_root: Path) -> None:
    path = registry_file(repo_root)
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "dnadesign.study.promoter_candidate_binding_sources.v1",
                "schema_version": "1",
                "study_id": "stress_ethanol_cipro_growth",
                "candidate_table": {
                    "dataset_id": "candidate-table",
                    "records_path": "data/candidates/records.parquet",
                },
                "alias_sources": [
                    {
                        "source_id": "reader-labels",
                        "adapter": "sequence_view_source_label.v1",
                        "config": {
                            "records_path": "data/reader-source/records.parquet",
                            "views_path": "data/reader-source/views.parquet",
                            "alias_namespace": "reader.design_id",
                            "authority_dataset_id": "reader-source",
                        },
                    },
                    {
                        "source_id": "references",
                        "adapter": "reference_label_aliases.v1",
                        "config": {
                            "records_path": "data/references/records.parquet",
                            "annotations_path": "data/references/annotations.parquet",
                            "authority_dataset_id": "references",
                            "aliases": [
                                reference_alias("spyp", "pDual-10-spyp"),
                                reference_alias("sulAp", "pDual-10-sulAp"),
                            ],
                        },
                    },
                    {
                        "source_id": "synthesis",
                        "adapter": "synthesis_handoff.v1",
                        "config": {
                            "record_path": "record/synthesis_handoffs.yaml",
                            "handoff_id": "pre-assay-1",
                            "authority_dataset_id": "pre-assay-1",
                            "aliases": [
                                {"namespace": "synthesis.name", "template": "{synthesis_name}"},
                                {
                                    "namespace": "reader.design_id",
                                    "template": "pDual-10-{synthesis_name}",
                                },
                            ],
                        },
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def reference_alias(label: str, reader_alias: str) -> dict[str, object]:
    return {
        "source_label": label,
        "display_label": label,
        "names": [
            {"namespace": "reader.design_id", "alias": reader_alias},
            {"namespace": "source.alias", "alias": label},
        ],
    }


def registry_file(repo_root: Path) -> Path:
    return repo_root / "docs/studies/stress_ethanol_cipro_growth/record/promoter_candidate_binding_sources.yaml"
