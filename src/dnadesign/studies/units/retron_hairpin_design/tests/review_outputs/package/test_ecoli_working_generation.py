"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/package/test_ecoli_working_generation.py

Tests for Eco1 tetO retained-span review-package generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from dnadesign.studies.units.retron_hairpin_design.review_outputs.service import (
    generate_retron_hairpin_review_outputs,
)

from ...support.paths import repo_root_from
from ...support.review_outputs import fake_video_writer, write_fake_materialized_bundle
from ...support.review_plans import write_review_plan_fixture


def test_review_outputs_service_exposes_plan_driven_api_only() -> None:
    from dnadesign.studies.units.retron_hairpin_design import review_outputs
    from dnadesign.studies.units.retron_hairpin_design.review_outputs import service

    assert hasattr(review_outputs, "generate_retron_hairpin_review_outputs")
    assert not hasattr(review_outputs, "generate_teto_retained_span_trim_tetr_pwm_elite_review_outputs")
    assert not hasattr(service, "generate_teto_retained_span_trim_tetr_pwm_elite_review_outputs")


def test_teto_ecoli_working_review_outputs_generate_review_package(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    deliverable_plan_path = write_review_plan_fixture(
        tmp_path / "plan",
        repo_root=repo_root,
        deliverable_plan_id="teto_retained_span_trim_ecoli_working_v1",
    )
    materialized_root = write_fake_materialized_bundle(
        tmp_path / "materialized",
        repo_root=repo_root,
        design_set_id="teto_retained_span_trim_ecoli_working_v1",
    )
    out_dir = tmp_path / "workbench" / "outputs" / "teto_retained_span_trim_ecoli_working_v1"

    result = generate_retron_hairpin_review_outputs(
        deliverable_plan_path=deliverable_plan_path,
        materialized_root=materialized_root,
        out_dir=out_dir,
        repo_root=repo_root,
        video_writer=fake_video_writer,
    )

    assert result.sequence_row_count == 6
    assert result.handoff_verified_count == 6
    assert (
        result.pwm_triptych_svg
        == out_dir / "reviews" / "pwm" / "teto_retained_span_trim_ecoli_working_v1.pwm_trim_triptych.svg"
    )
    assert (
        result.handoff_tsv == out_dir / "reviews" / "handoff" / "teto_retained_span_trim_ecoli_working_v1.handoff.tsv"
    )
    assert result.benchling_genbank_count == 6
    assert sorted(path.name for path in result.benchling_genbank_dir.iterdir() if not path.name.startswith(".")) == [
        "msd-retron-201.gb",
        "msd-retron-202.gb",
        "msd-retron-203.gb",
        "msd-retron-204.gb",
        "msd-retron-205.gb",
        "msd-retron-206.gb",
    ]

    review_manifest = json.loads(result.review_manifest_path.read_text(encoding="utf-8"))
    assert review_manifest["deliverable_plan_id"] == "teto_retained_span_trim_ecoli_working_v1"
    assert review_manifest["pwm_triptych"]["payload_trim_ids"] == [
        "tetO_ecoli_working_w00_19",
        "tetO_ecoli_working_w02_17",
        "tetO_ecoli_working_w03_16",
    ]
    assert review_manifest["pwm_triptych"]["review_only_payload_trim_ids"] == ["tetO_ecoli_working_w00_19"]
    assert review_manifest["pwm_triptych"]["materialized_payload_trim_ids"] == [
        "tetO_ecoli_working_w02_17",
        "tetO_ecoli_working_w03_16",
    ]
    assert review_manifest["pwm_triptych"]["motif_occurrences"] == [
        {
            "motif_instance_id": "tetR:1:18:+:1",
            "occurrence_rank": 1,
            "span_0": {"end": 18, "start": 1},
            "strand": "+",
        },
        {
            "motif_instance_id": "tetR:1:18:-:2",
            "occurrence_rank": 2,
            "span_0": {"end": 18, "start": 1},
            "strand": "-",
        },
    ]
    pwm_svg = result.pwm_triptych_svg.read_text(encoding="utf-8")
    assert 'data-motif-layer-count="2"' in pwm_svg
    assert 'data-payload-trim-id="tetO_ecoli_working_w00_19"' in pwm_svg
    assert 'data-requires-materialized-sequence="false"' in pwm_svg
    assert review_manifest["benchling_genbank_import"]["assigned_retron_ids"] == {
        "r26-w02-17": "pES-retron-201",
        "r26-w03-16": "pES-retron-202",
        "r43-w02-17": "pES-retron-205",
        "r43-w03-16": "pES-retron-206",
        "r180-w02-17": "pES-retron-203",
        "r180-w03-16": "pES-retron-204",
    }
    assert review_manifest["benchling_genbank_import"]["record_ids"] == {
        "r26-w02-17": "msd-retron-201",
        "r26-w03-16": "msd-retron-202",
        "r43-w02-17": "msd-retron-205",
        "r43-w03-16": "msd-retron-206",
        "r180-w02-17": "msd-retron-203",
        "r180-w03-16": "msd-retron-204",
    }
    assert review_manifest["benchling_genbank_import"]["descriptions"]["r180-w03-16"] == (
        "pES-retron-180 P4 scaffold; 13 nt [3,16) retained span from the tetO payload used by pES-retron-26"
    )
    assert review_manifest["benchling_genbank_import"]["files"] == [
        "benchling_genbank/msd-retron-201.gb",
        "benchling_genbank/msd-retron-202.gb",
        "benchling_genbank/msd-retron-205.gb",
        "benchling_genbank/msd-retron-206.gb",
        "benchling_genbank/msd-retron-203.gb",
        "benchling_genbank/msd-retron-204.gb",
    ]
    benchling_rows = list(
        csv.DictReader(result.benchling_genbank_index.read_text(encoding="utf-8").splitlines(), delimiter="\t")
    )
    assert benchling_rows[0]["record_id"] == "msd-retron-201"
    assert benchling_rows[0]["description"] == (
        "pES-retron-26 P4 scaffold; 15 nt [2,17) retained span from the tetO payload used by pES-retron-26"
    )
    genbank = (result.benchling_genbank_dir / "msd-retron-201.gb").read_text(encoding="utf-8")
    assert (
        "DEFINITION  msd-retron-201; pES-retron-26 P4 scaffold; 15 nt [2,17) retained span "
        "from the tetO payload used by pES-retron-26; reverse-complement MSD handoff from pES-teto-r26-w02-17."
    ) in genbank
    assert _feature_block(genbank, "3' Flanking").startswith("     misc_feature    1..17")
    assert '/strand="1"' in _feature_block(genbank, "3' Flanking")
    assert _feature_block(genbank, "Right Base").startswith("     misc_feature    14..17")
    assert '/strand="1"' in _feature_block(genbank, "Right Base")
    assert _feature_block(genbank, "msd[tetO_ecoli_working_w02_17] complement").startswith(
        "     misc_feature    18..32"
    )
    assert '/strand="1"' in _feature_block(genbank, "msd[tetO_ecoli_working_w02_17] complement")
    assert _feature_block(genbank, "Foldback").startswith("     misc_feature    33..36")
    assert "/strand=" not in _feature_block(genbank, "Foldback")
    assert _feature_block(genbank, "Cap").startswith("     misc_feature    33..36")
    assert "/strand=" not in _feature_block(genbank, "Cap")
    assert _feature_block(genbank, "msd[tetO_ecoli_working_w02_17]").startswith(
        "     misc_feature    complement(37..51)"
    )
    assert '/strand="-1"' in _feature_block(genbank, "msd[tetO_ecoli_working_w02_17]")
    assert _feature_block(genbank, "Left Base").startswith("     misc_feature    complement(52..55)")
    assert '/strand="-1"' in _feature_block(genbank, "Left Base")
    assert _feature_block(genbank, "5' Flanking").startswith("     misc_feature    complement(52..66)")
    assert '/strand="-1"' in _feature_block(genbank, "5' Flanking")
    video_manifest = json.loads(result.sequence_montage_manifest.read_text(encoding="utf-8"))
    assert video_manifest["frames"][0]["variant_id"] == "r26-w02-17"
    assert video_manifest["frames"][0]["evidence_label"] == ("pES-retron-201 | tetO PWM [2,17) | r26 scaffold | 15 nt")


def _feature_block(genbank: str, label: str) -> str:
    marker = f'/label="{label}"'
    lines = genbank.splitlines()
    label_index = next(index for index, line in enumerate(lines) if marker in line)
    start = max(index for index in range(label_index + 1) if lines[index].startswith("     misc_feature"))
    end = next(
        (
            index
            for index in range(label_index + 1, len(lines))
            if lines[index].startswith("     misc_feature") or lines[index].startswith("ORIGIN")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])
