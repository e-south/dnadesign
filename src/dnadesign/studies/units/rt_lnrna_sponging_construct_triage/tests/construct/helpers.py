"""Shared fixtures for RT-lnRNA Construct materialization tests."""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.contracts.sequence.msd_design_reference_v1 import compute_scar_nick_profile_s3s2s1s0
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.construct_materialization import (
    ControlConstructMaterializationReport,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.source_promotions import ConstructWindowPolicy
from dnadesign.usr import Dataset

_CONSTRUCT_SUBJECT_SEQUENCE_FIELDS = ("construct_subject__lnrna_sequence", "construct_subject__rt_cds_sequence")
_WT_RT_CDS_SEQUENCE = "ATG" * 320 + "TAA"
_TETO_PAYLOAD = "tccctatcagtgatagaga"
_SNAPBACK_FOLDBACK = "GAGAGACTC"
_ALL_SNAPBACK_CAP_RANKS = (1, 2, 3, 4, 5)
_ALL_SCAR_NICK_STEM_BASE_RANKS = tuple(range(1, 17))
_BASES = "ACGT"
_SCAR_NICK_STEM_BASE_PRIMITIVE_PAIRS = (
    ("AGTG", "CAAT"),
    ("AGTG", "CATG"),
    ("AGTG", "CTTT"),
    ("AGTG", "CGAT"),
    ("AATG", "CGTG"),
    ("AGTG", "CATT"),
    ("AATG", "CGTT"),
    ("AGTG", "CGTT"),
    ("AGTG", "CAAG"),
    ("AATG", "CTTG"),
    ("AATG", "CAGT"),
    ("AGTG", "CAGT"),
    ("AATG", "CAAT"),
    ("AATG", "CACT"),
    ("CTCT", "AGTG"),
    ("CTCA", "TGTG"),
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _source_window_policy() -> ConstructWindowPolicy:
    return ConstructWindowPolicy(
        context_id="dual_cassette_2000bp_context_v1",
        target_start_0=56,
        target_length_nt=2000,
        template_length_nt=4956,
        lnrna_template_span_0=(186, 359),
        rt_cds_template_span_0=(524, 1487),
    )


def _assert_construct_subject_envelope_inputs(report: ControlConstructMaterializationReport) -> None:
    inputs = Dataset(report.usr_root, report.input_dataset).head(n=len(report.input_ids_by_subject_id) + 5)

    assert set(inputs["construct_subject__record_kind"]) == {"construct_subject_envelope"}
    assert set(inputs["construct_subject__sequence_authority"]) == {"overlay_only"}
    assert set(inputs["construct_subject__envelope_carrier_policy"]) == {"synthetic_unique_dna4_v1"}
    assert inputs["id"].is_unique
    assert {tuple(fields) for fields in inputs["construct_subject__biological_sequence_fields"]} == {
        _CONSTRUCT_SUBJECT_SEQUENCE_FIELDS
    }


def _assert_construct_output_subject_bridge(report: ControlConstructMaterializationReport) -> None:
    output = Dataset(report.usr_root, report.output_dataset).head(n=len(report.input_ids_by_subject_id) * 4 + 20)

    assert set(output["construct_subject__record_kind"]) == {"construct_output"}
    assert set(output["construct_subject__sequence_authority"]) == {"realized_construct_sequence"}
    assert {tuple(fields) for fields in output["construct_subject__biological_sequence_fields"]} == {
        _CONSTRUCT_SUBJECT_SEQUENCE_FIELDS
    }
    for construct_subject_id, input_id in report.input_ids_by_subject_id.items():
        subject_output = output[output["construct__input_id"] == input_id]
        assert subject_output.shape[0] == 2
        assert set(subject_output["construct_subject__id"]) == {construct_subject_id}
        assert subject_output["construct_subject__lnrna_sequence"].nunique() == 1
        assert subject_output["construct_subject__rt_cds_sequence"].nunique() == 1


def _assert_usr_contracts_strictly_validate(report: ControlConstructMaterializationReport) -> None:
    Dataset(report.usr_root, report.input_dataset).validate(strict=True)
    Dataset(report.usr_root, report.output_dataset).validate(strict=True)


def _write_msd_compiler_pool_spec(
    path: Path,
    *,
    pool_id: str = "test_compiler_msd_pool_v1",
    construct_id_prefix: str = "rt-lnrna-yiu-compatible-compiler",
    cap_ids: tuple[str, ...] = ("C172",),
    use_primitive_sources: bool = True,
    cap_ranks: tuple[int, ...] = _ALL_SNAPBACK_CAP_RANKS,
    stem_base_ranks: tuple[int, ...] = _ALL_SCAR_NICK_STEM_BASE_RANKS,
    cap_id_prefix: str = "CDE033R",
    stem_base_id_prefix: str = "scar_nick_teto_rank",
    expected_5p_flank: str = "TCCTGCATTGAA",
    expected_3p_flank: str = "GTAAGGGTGCGC",
    max_variant_count: int = 80,
    expected_variant_count: int | None = 80,
    extra_cap_sequences: dict[str, str] | None = None,
    extra_stem_base_fields: str = "",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cap_sequences = {"C26": "AGGC", **(extra_cap_sequences or {})}
    if not use_primitive_sources:
        cap_sequences = {
            "C26": "AGGC",
            "C172": _SNAPBACK_FOLDBACK,
            **(extra_cap_sequences or {}),
        }
    expected_line = "" if expected_variant_count is None else f"expected_variant_count: {expected_variant_count}\n"
    cap_sequence_lines = "\n".join(
        f"    {cap_id}:\n      sequence: {sequence}" for cap_id, sequence in cap_sequences.items()
    )
    cap_id_lines = "\n".join(f"    - {cap_id}" for cap_id in cap_ids)
    cap_rank_lines = "\n".join(f"        - {rank}" for rank in cap_ranks)
    stem_base_rank_lines = "\n".join(f"        - {rank}" for rank in stem_base_ranks)
    literal_source_ref = (
        "docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_cap_sources_spec.yaml#pES-retron-179"
    )
    if use_primitive_sources:
        primitive_sources = _write_msd_primitive_source_fixtures(
            root=path.parent / f"{path.stem}_primitive_sources",
            max_cap_rank=max(cap_ranks),
            max_stem_base_rank=max(stem_base_ranks),
        )
        design_space_block = f"""design_space:
  construct_id_prefix: {construct_id_prefix}
  payload_ids:
    - TetR
  cap_primitives:
    - source_id: de033_released_snapback_cap_primitives_v1
      kind: snapback_released_solve_cap
      run_dir: {primitive_sources["snapback_run_dir"]}
      cap_id_prefix: {cap_id_prefix}
      ranks:
{cap_rank_lines}
      expected_primitive_count: {len(cap_ranks)}
  stem_base_primitives:
    - source_id: scar_nick_teto_bbsI_hf_stem_base_primitives_v1
      kind: scar_nick_stem_bases
      run_dir: {primitive_sources["scar_nick_run_dir"]}
      stem_base_id_prefix: {stem_base_id_prefix}
      ranks:
{stem_base_rank_lines}
      expected_primitive_count: {len(stem_base_ranks)}
source_refs:
  - {primitive_sources["snapback_run_dir"]}
  - {primitive_sources["scar_nick_run_dir"]}
"""
    else:
        design_space_block = f"""design_space:
  construct_id_prefix: {construct_id_prefix}
  payload_ids:
    - TetR
  cap_ids:
{cap_id_lines}
  stem_bases:
    - stem_base_id: lagtg_rcaat
      left_base: AGTG
      right_base: CAAT
      profile_s3s2s1s0: MXMM
      source_ref: {literal_source_ref}
      nick_orientation: bottom
      nickase: Nb.BtsI
{extra_stem_base_fields}source_refs:
  - {literal_source_ref}
"""
    path.write_text(
        f"""contract: rt_lnrna_msd_variant_pool_spec_v1
schema_version: 1
pool_id: {pool_id}
study_id: rt_lnrna_sponging_construct_triage
payload_program_id: tetO_sponging_v1
max_variant_count: {max_variant_count}
{expected_line}dedupe_policy: fail
template_lnrna:
  sequence_ref: genbank:pes-retron-26-a1-a2.gb#a1-a2
  genbank_path: docs/studies/rt_lnrna_sponging_construct_triage/workbench/provenance/genbank/pes-retron-26-a1-a2.gb
  sequence_span_0: [0, 173]
template_msd_design:
  construct_id: rt-lnrna-template-retron26
  payload_id: TetR
  cap_id: C26
  left_base: CGGG
  right_base: ACAG
  profile_s3s2s1s0: MXMX
placement:
  expected_5p_flank: {expected_5p_flank}
  expected_3p_flank: {expected_3p_flank}
compiler_inputs:
  payload_sequences:
    TetR:
      sequence: {_TETO_PAYLOAD.upper()}
  cap_sequences:
{cap_sequence_lines}
{design_space_block}
""",
        encoding="utf-8",
    )
    return path


def _write_msd_primitive_source_fixtures(*, root: Path, max_cap_rank: int, max_stem_base_rank: int) -> dict[str, str]:
    snapback_run = root / "snapback_run"
    scar_nick_run = root / "scar_nick_run"
    _write_snapback_cap_primitive_fixture(snapback_run, max_rank=max_cap_rank)
    _write_scar_nick_stem_base_primitive_fixture(scar_nick_run, max_rank=max_stem_base_rank)
    return {
        "snapback_run_dir": snapback_run.resolve().as_posix(),
        "scar_nick_run_dir": scar_nick_run.resolve().as_posix(),
    }


def _write_snapback_cap_primitive_fixture(run_dir: Path, *, max_rank: int) -> None:
    report_path = run_dir / "analysis" / "solve_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    hits = [
        {
            "rank": rank,
            "hit_kind": "exact",
            "nickase_variant_id": "Nb.BtsI",
            "release_variant_id": "BspQI",
            "target_search_hit": {
                "final_candidate": {
                    "designed_sequence": _snapback_cap_sequence_for_rank(rank),
                    "paired_bp": 3,
                    "cap_nt": 3,
                }
            },
        }
        for rank in range(1, max_rank + 1)
    ]
    report_path.write_text(
        json.dumps({"workflow": "snapback_released_solve", "hits": hits}, indent=2) + "\n",
        encoding="utf-8",
    )


def _snapback_cap_sequence_for_rank(rank: int) -> str:
    if rank == 1:
        return _SNAPBACK_FOLDBACK
    value = rank - 1
    encoded = []
    for _ in range(9):
        encoded.append(_BASES[value % len(_BASES)])
        value //= len(_BASES)
    return "".join(encoded)


def _write_scar_nick_stem_base_primitive_fixture(run_dir: Path, *, max_rank: int) -> None:
    table_path = run_dir / "export" / "table__scar_nick_candidates.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, (left_base, right_base) in enumerate(_stem_base_pairs(max_rank), start=1):
        rows.append(
            ",".join(
                [
                    str(rank),
                    f"scar-rank-{rank:02d}",
                    left_base,
                    right_base,
                    compute_scar_nick_profile_s3s2s1s0(left_base=left_base, right_base=right_base),
                    "Nb.BtsI",
                    "bottom",
                    "top",
                ]
            )
        )
    table_path.write_text(
        "\n".join(
            [
                "rank,candidate_id,left_base,right_base,profile_s3s2s1s0,nickase_variant_id,nicked_strand,surviving_strand",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _stem_base_pairs(count: int) -> tuple[tuple[str, str], ...]:
    if count <= len(_SCAR_NICK_STEM_BASE_PRIMITIVE_PAIRS):
        return _SCAR_NICK_STEM_BASE_PRIMITIVE_PAIRS[:count]
    raise AssertionError(f"Scar-nick primitive fixture only defines {len(_SCAR_NICK_STEM_BASE_PRIMITIVE_PAIRS)} ranks.")


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1].upper()
