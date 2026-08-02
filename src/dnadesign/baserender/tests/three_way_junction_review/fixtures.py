"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/three_way_junction_review/fixtures.py

Neutral review-contract fixtures for BaseRender Junction tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import dnadesign.baserender as baserender
from dnadesign.baserender.src.core import Record


def _adapt_payload(payload: dict[str, object]) -> Record:
    return baserender.adapt_records([payload], adapter_kind="three_way_junction_review_v1")[0]


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _rename_target_geometry(payload: dict[str, object], *, target_id: str) -> None:
    """Give one synthetic review row producer-shaped plan-scoped identities."""

    payload["target"]["target_id"] = target_id  # type: ignore[index]
    fragment_id_map = {
        fragment["fragment_id"]: f"{target_id}:fragment-{index + 1:04d}"
        for index, fragment in enumerate(payload["geometry"]["fragments"])  # type: ignore[index]
    }
    junction_id_map = {
        junction["junction_id"]: f"{target_id}:junction-{index + 1:04d}"
        for index, junction in enumerate(payload["geometry"]["junctions"])  # type: ignore[index]
    }
    for fragment in payload["geometry"]["fragments"]:  # type: ignore[index]
        fragment["fragment_id"] = fragment_id_map[fragment["fragment_id"]]
    for junction in payload["geometry"]["junctions"]:  # type: ignore[index]
        junction["junction_id"] = junction_id_map[junction["junction_id"]]
        junction["left_fragment_id"] = fragment_id_map[junction["left_fragment_id"]]
        junction["right_fragment_id"] = fragment_id_map[junction["right_fragment_id"]]
    for strand in payload["strands"]:  # type: ignore[index]
        strand["fragment_id"] = fragment_id_map[strand["fragment_id"]]
        if strand["incoming_junction_id"] is not None:
            strand["incoming_junction_id"] = junction_id_map[strand["incoming_junction_id"]]
        if strand["outgoing_junction_id"] is not None:
            strand["outgoing_junction_id"] = junction_id_map[strand["outgoing_junction_id"]]
    payload["recovery"]["first_fragment_id"] = fragment_id_map[payload["recovery"]["first_fragment_id"]]  # type: ignore[index]
    payload["recovery"]["last_fragment_id"] = fragment_id_map[payload["recovery"]["last_fragment_id"]]  # type: ignore[index]
    for check in payload["checks"]:  # type: ignore[index]
        if check["subject"]["kind"] == "target":
            check["subject"]["id"] = target_id


def _payload() -> dict[str, object]:
    target = "AAAACCCCGGGGTTTTAAAACCCC"
    toehold = target[10:14]
    reverse_binding = _reverse_complement(target[-4:])
    return {
        "contract_kind": "three_way_junction_review_v1",
        "source": {
            "plan_schema": "dnadesign.junction.plan.v1",
            "plan_id": f"sha256:{'a' * 64}",
            "request_sha256": f"sha256:{'b' * 64}",
            "algorithm": "junction.v1",
        },
        "target": {
            "target_id": "target-01",
            "assembly_group_id": "assembly-01",
            "sequence_5to3": target,
            "sequence_sha256": f"sha256:{hashlib.sha256(target.encode()).hexdigest()}",
        },
        "geometry": {
            "fragments": [
                {
                    "fragment_id": "fragment-01",
                    "index": 0,
                    "role": "first",
                    "domain_span": {"start": 0, "end": 10},
                },
                {
                    "fragment_id": "fragment-02",
                    "index": 1,
                    "role": "last",
                    "domain_span": {"start": 14, "end": len(target)},
                },
            ],
            "junctions": [
                {
                    "junction_id": "junction-01",
                    "toehold_span": {"start": 10, "end": 14},
                    "left_fragment_id": "fragment-01",
                    "right_fragment_id": "fragment-02",
                    "toehold": toehold,
                    "toehold_complement": _reverse_complement(toehold),
                    "barcode": "AACCGGTT",
                    "barcode_complement": "AACCGGTT",
                    "complement_nick_sequence_layout_valid": True,
                    "complement_end_preparation": "vendor_5_prime_phosphate",
                }
            ],
        },
        "strands": [
            {
                "fragment_id": "fragment-01",
                "role": "first",
                "incoming_junction_id": None,
                "outgoing_junction_id": "junction-01",
                "barcode_bearing_sequence_5to3": target[:14] + "AACCGGTT",
                "complement_sequence_5to3": _reverse_complement(target[:10]),
            },
            {
                "fragment_id": "fragment-02",
                "role": "last",
                "incoming_junction_id": "junction-01",
                "outgoing_junction_id": None,
                "barcode_bearing_sequence_5to3": _reverse_complement("AACCGGTT") + target[14:],
                "complement_sequence_5to3": _reverse_complement(target[14:]) + _reverse_complement(toehold),
            },
        ],
        "recovery": {
            "mode": "universal",
            "forward": {
                "direction": "forward",
                "binding_sequence_5to3": target[:4],
                "five_prime_extension_5to3": "",
                "order_sequence_5to3": target[:4],
                "target_binding_span": {"start": 0, "end": 4},
            },
            "reverse": {
                "direction": "reverse",
                "binding_sequence_5to3": reverse_binding,
                "five_prime_extension_5to3": "",
                "order_sequence_5to3": reverse_binding,
                "target_binding_span": {"start": 20, "end": 24},
            },
            "first_fragment_id": "fragment-01",
            "last_fragment_id": "fragment-02",
            "expected_target_sequence_5to3": target,
            "extended_top_sequence_5to3": target,
            "extended_bottom_sequence_5to3": _reverse_complement(target),
        },
        "search": {
            "assembly_group_id": "assembly-01",
            "toehold_seed": 11,
            "barcode_generation_seed": 12,
            "barcode_subset_seed": 13,
            "matching_seed": 14,
            "locus_count": 1,
            "toehold_paths_evaluated": 20,
            "toehold_min_distance": 0.0,
            "toehold_mean_distance": 0.0,
            "toehold_rank_score": 1.5,
            "barcode_candidates_generated": 25,
            "barcode_forbidden_toehold_k": 3,
            "barcode_forbidden_barcode_k": 4,
            "barcode_subsets_evaluated": 20,
            "barcode_min_distance": 0.0,
            "barcode_mean_distance": 0.0,
            "barcode_rank_score": 1.5,
            "matchings_evaluated": 1,
            "matching_max_pairwise_lcs": 0,
            "thermodynamic_screening": "not_run",
        },
        "checks": [
            {
                "subject": {"kind": "target", "id": "target-01"},
                "check": "exact_target_reconstruction",
                "status": "passed",
                "detail": "exact",
            },
            {
                "subject": {"kind": "assembly_group", "id": "assembly-01"},
                "check": "thermodynamic_screening",
                "status": "not_run",
                "detail": "not performed",
            },
        ],
    }


def _review_job(
    source: Path,
    *,
    contract_kind: str = "three_way_junction_review_render_v1",
    input_narrowing: dict[str, object] | None = None,
) -> dict[str, object]:
    input_config: dict[str, object] = {
        "kind": "json",
        "path": source.name,
        "adapter": {"kind": "three_way_junction_review_v1"},
        "alphabet": "DNA",
    }
    if input_narrowing is not None:
        input_config.update(input_narrowing)
    return {
        "version": 4,
        "contract": {"kind": contract_kind},
        "bundle": {"path": "review-render"},
        "input": input_config,
        "render": {
            "renderer": "three_way_junction_review",
            "style": {"preset": None, "overrides": {}},
        },
        "outputs": [{"kind": "images", "dir": "images", "fmt": "svg"}],
        "run": {"strict": True, "fail_on_skips": True},
    }


def _payload_with_long_recovery_primers() -> dict[str, object]:
    payload = _payload()
    target = ("ACGATTCGGTACCTGATGCACTGA" * 10)[:240]
    toehold_start = 118
    toehold_end = 122
    toehold = target[toehold_start:toehold_end]
    forward_binding = target[:96]
    reverse_binding = _reverse_complement(target[-96:])
    forward_extension = ("GATTACA" * 15)[:100]
    reverse_extension = ("CCGTTA" * 17)[:100]
    barcode = "AGTCCTGA"

    payload["target"] = {
        "target_id": "target-01",
        "assembly_group_id": "assembly-01",
        "sequence_5to3": target,
        "sequence_sha256": f"sha256:{hashlib.sha256(target.encode()).hexdigest()}",
    }
    payload["geometry"]["fragments"] = [
        {
            "fragment_id": "fragment-01",
            "index": 0,
            "role": "first",
            "domain_span": {"start": 0, "end": toehold_start},
        },
        {
            "fragment_id": "fragment-02",
            "index": 1,
            "role": "last",
            "domain_span": {"start": toehold_end, "end": len(target)},
        },
    ]
    payload["geometry"]["junctions"][0].update(
        {
            "toehold_span": {"start": toehold_start, "end": toehold_end},
            "toehold": toehold,
            "toehold_complement": _reverse_complement(toehold),
            "barcode": barcode,
            "barcode_complement": _reverse_complement(barcode),
        }
    )
    payload["strands"] = [
        {
            "fragment_id": "fragment-01",
            "role": "first",
            "incoming_junction_id": None,
            "outgoing_junction_id": "junction-01",
            "barcode_bearing_sequence_5to3": target[:toehold_start] + toehold + barcode,
            "complement_sequence_5to3": _reverse_complement(target[:toehold_start]),
        },
        {
            "fragment_id": "fragment-02",
            "role": "last",
            "incoming_junction_id": "junction-01",
            "outgoing_junction_id": None,
            "barcode_bearing_sequence_5to3": _reverse_complement(barcode) + target[toehold_end:],
            "complement_sequence_5to3": _reverse_complement(target[toehold_end:]) + _reverse_complement(toehold),
        },
    ]
    payload["recovery"] = {
        "mode": "target_specific",
        "forward": {
            "direction": "forward",
            "binding_sequence_5to3": forward_binding,
            "five_prime_extension_5to3": forward_extension,
            "order_sequence_5to3": forward_extension + forward_binding,
            "target_binding_span": {"start": 0, "end": len(forward_binding)},
        },
        "reverse": {
            "direction": "reverse",
            "binding_sequence_5to3": reverse_binding,
            "five_prime_extension_5to3": reverse_extension,
            "order_sequence_5to3": reverse_extension + reverse_binding,
            "target_binding_span": {"start": len(target) - len(reverse_binding), "end": len(target)},
        },
        "first_fragment_id": "fragment-01",
        "last_fragment_id": "fragment-02",
        "expected_target_sequence_5to3": target,
        "extended_top_sequence_5to3": forward_extension + target + _reverse_complement(reverse_extension),
        "extended_bottom_sequence_5to3": reverse_extension
        + _reverse_complement(target)
        + _reverse_complement(forward_extension),
    }
    return payload


def _payload_with_long_junction_sequences() -> dict[str, object]:
    payload = _payload_with_long_recovery_primers()
    target = payload["target"]["sequence_5to3"]
    toehold_start = 70
    toehold_end = 170
    toehold = target[toehold_start:toehold_end]
    barcode = ("AACCGGTT" * 13)[:100]
    payload["geometry"]["fragments"] = [
        {
            "fragment_id": "fragment-01",
            "index": 0,
            "role": "first",
            "domain_span": {"start": 0, "end": toehold_start},
        },
        {
            "fragment_id": "fragment-02",
            "index": 1,
            "role": "last",
            "domain_span": {"start": toehold_end, "end": len(target)},
        },
    ]
    payload["geometry"]["junctions"][0].update(
        {
            "toehold_span": {"start": toehold_start, "end": toehold_end},
            "toehold": toehold,
            "toehold_complement": _reverse_complement(toehold),
            "barcode": barcode,
            "barcode_complement": _reverse_complement(barcode),
        }
    )
    payload["strands"] = [
        {
            "fragment_id": "fragment-01",
            "role": "first",
            "incoming_junction_id": None,
            "outgoing_junction_id": "junction-01",
            "barcode_bearing_sequence_5to3": target[:toehold_start] + toehold + barcode,
            "complement_sequence_5to3": _reverse_complement(target[:toehold_start]),
        },
        {
            "fragment_id": "fragment-02",
            "role": "last",
            "incoming_junction_id": "junction-01",
            "outgoing_junction_id": None,
            "barcode_bearing_sequence_5to3": _reverse_complement(barcode) + target[toehold_end:],
            "complement_sequence_5to3": _reverse_complement(target[toehold_end:]) + _reverse_complement(toehold),
        },
    ]
    return payload


def _payload_with_large_display_scalars() -> dict[str, object]:
    payload = _payload()
    target_id = "target-" + ("x" * 10_000)
    assembly_group_id = "assembly-" + ("y" * 10_000)
    payload["target"]["target_id"] = target_id
    payload["target"]["assembly_group_id"] = assembly_group_id
    payload["search"]["assembly_group_id"] = assembly_group_id
    payload["search"]["toehold_paths_evaluated"] = 10**10_000
    payload["checks"][0]["subject"]["id"] = target_id
    payload["checks"][1]["subject"]["id"] = assembly_group_id
    return payload


def _payload_with_many_junctions(junction_count: int = 20) -> dict[str, object]:
    payload = _payload()
    target = ("ACGT" * (junction_count + 1))[: (2 * junction_count) + 1]

    def _indexed_barcode(index: int) -> str:
        value = index
        encoded: list[str] = []
        for _ in range(4):
            value, digit = divmod(value, 4)
            encoded.append("ACGT"[digit])
        return "AA" + "".join(reversed(encoded)) + "AA"

    fragments = []
    junctions = []
    strands = []
    for index in range(junction_count + 1):
        domain_start = index * 2
        domain_end = domain_start + 1
        fragment_id = f"fragment-{index + 1:02d}"
        fragments.append(
            {
                "fragment_id": fragment_id,
                "index": index,
                "role": "first" if index == 0 else "last" if index == junction_count else "internal",
                "domain_span": {"start": domain_start, "end": domain_end},
            }
        )
        previous = None if index == 0 else junctions[index - 1]
        following_id = None if index == junction_count else f"junction-{index + 1:02d}"
        following_barcode = None if following_id is None else _indexed_barcode(index)
        domain = target[domain_start:domain_end]
        if previous is None:
            assert following_barcode is not None
            following_toehold = target[domain_end : domain_end + 1]
            barcode_bearing = domain + following_toehold + following_barcode
            complement = _reverse_complement(domain)
        elif following_id is None:
            barcode_bearing = _reverse_complement(previous["barcode"]) + domain
            complement = _reverse_complement(domain) + previous["toehold_complement"]
        else:
            assert following_barcode is not None
            following_toehold = target[domain_end : domain_end + 1]
            barcode_bearing = _reverse_complement(previous["barcode"]) + domain + following_toehold + following_barcode
            complement = _reverse_complement(domain) + previous["toehold_complement"]
        strands.append(
            {
                "fragment_id": fragment_id,
                "role": fragments[-1]["role"],
                "incoming_junction_id": None if previous is None else previous["junction_id"],
                "outgoing_junction_id": following_id,
                "barcode_bearing_sequence_5to3": barcode_bearing,
                "complement_sequence_5to3": complement,
            }
        )
        if following_id is not None:
            assert following_barcode is not None
            toehold = target[domain_end : domain_end + 1]
            junctions.append(
                {
                    "junction_id": following_id,
                    "toehold_span": {"start": domain_end, "end": domain_end + 1},
                    "left_fragment_id": fragment_id,
                    "right_fragment_id": f"fragment-{index + 2:02d}",
                    "toehold": toehold,
                    "toehold_complement": _reverse_complement(toehold),
                    "barcode": following_barcode,
                    "barcode_complement": _reverse_complement(following_barcode),
                    "complement_nick_sequence_layout_valid": True,
                    "complement_end_preparation": "vendor_5_prime_phosphate",
                }
            )
    payload["target"]["sequence_5to3"] = target
    payload["target"]["sequence_sha256"] = f"sha256:{hashlib.sha256(target.encode()).hexdigest()}"
    payload["geometry"] = {"fragments": fragments, "junctions": junctions}
    payload["strands"] = strands
    payload["recovery"] = {
        "mode": "universal",
        "forward": {
            "direction": "forward",
            "binding_sequence_5to3": target[:4],
            "five_prime_extension_5to3": "",
            "order_sequence_5to3": target[:4],
            "target_binding_span": {"start": 0, "end": 4},
        },
        "reverse": {
            "direction": "reverse",
            "binding_sequence_5to3": _reverse_complement(target[-4:]),
            "five_prime_extension_5to3": "",
            "order_sequence_5to3": _reverse_complement(target[-4:]),
            "target_binding_span": {"start": len(target) - 4, "end": len(target)},
        },
        "first_fragment_id": fragments[0]["fragment_id"],
        "last_fragment_id": fragments[-1]["fragment_id"],
        "expected_target_sequence_5to3": target,
        "extended_top_sequence_5to3": target,
        "extended_bottom_sequence_5to3": _reverse_complement(target),
    }
    payload["search"]["locus_count"] = junction_count
    payload["search"].update(
        {
            "toehold_min_distance": 2.0,
            "toehold_mean_distance": 2.0,
            "barcode_candidates_generated": 5 * junction_count,
            "barcode_forbidden_toehold_k": 1,
            "barcode_forbidden_barcode_k": 4,
        }
    )
    return payload
