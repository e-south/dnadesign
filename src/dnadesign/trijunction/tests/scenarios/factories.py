"""Compact factories for exact-DNA scale scenarios without checked-in sequence blobs."""

from __future__ import annotations

from hashlib import shake_256
from typing import Any, Literal

from dnadesign.trijunction.sequence import reverse_complement

_DNA = "ACGT"


def deterministic_dna(label: str, length: int) -> str:
    """Expand a stable label into balanced exact DNA without search-aware tuning."""

    if length < 1:
        raise ValueError("deterministic DNA length must be positive")
    byte_count = (length + 3) // 4
    for retry in range(10_000):
        digest = shake_256(
            b"dnadesign.trijunction.dogfood.target.v1\0"
            + label.encode()
            + b"\0"
            + str(length).encode()
            + b"\0"
            + str(retry).encode()
        ).digest(byte_count)
        bases: list[str] = []
        for byte in digest:
            for shift in (0, 2, 4, 6):
                bases.append(_DNA[(byte >> shift) & 0b11])
        sequence = "".join(bases[:length])
        gc_fraction = sum(base in {"G", "C"} for base in sequence) / length
        longest_run = 1
        current_run = 1
        for previous, current in zip(sequence, sequence[1:], strict=False):
            current_run = current_run + 1 if current == previous else 1
            longest_run = max(longest_run, current_run)
        if 0.45 <= gc_fraction <= 0.55 and longest_run <= 8:
            return sequence
    raise AssertionError("unable to generate a deterministic DNA fixture within its quality envelope")


def scale_request_mapping(
    *,
    target_count: int,
    target_length: int,
    topology: Literal["shared", "independent"],
    oligo_length: int,
    search_range: int,
    barcode_generation_attempts: int,
) -> dict[str, Any]:
    """Return one bounded deterministic request for topology and load dogfood."""

    if target_count < 1:
        raise ValueError("target_count must be positive")
    targets: list[dict[str, Any]] = []
    used_sequences: dict[str, set[str]] = {}
    used_recovery_pairs: dict[str, set[tuple[str, str]]] = {}
    for index in range(target_count):
        target_id = f"target-{index:04d}"
        pool_id = "shared-pool" if topology == "shared" else f"pool-{index:04d}"
        pool_sequences = used_sequences.setdefault(pool_id, set())
        pool_recovery_pairs = used_recovery_pairs.setdefault(pool_id, set())
        for collision_retry in range(10_000):
            sequence = deterministic_dna(
                f"{topology}:{target_count}:{target_length}:{target_id}:{collision_retry}",
                target_length,
            )
            recovery_pair = (sequence[:20], reverse_complement(sequence[-20:]))
            if sequence not in pool_sequences and recovery_pair not in pool_recovery_pairs:
                pool_sequences.add(sequence)
                pool_recovery_pairs.add(recovery_pair)
                break
        else:
            raise AssertionError(f"unable to generate unique recovery geometry for {target_id}")
        targets.append(
            {
                "id": target_id,
                "pool_id": pool_id,
                "sequence": sequence,
                "recovery_primers": {
                    "mode": "target_specific",
                    "forward": {
                        "binding_sequence": recovery_pair[0],
                        "five_prime_extension": "GGTCTCA",
                    },
                    "reverse": {
                        "binding_sequence": recovery_pair[1],
                        "five_prime_extension": "CGTCTCA",
                    },
                },
            }
        )
    return {
        "schema": "dnadesign.trijunction.request.v1",
        "seed": 20_260_801,
        "planning": {
            "oligo_length": oligo_length,
            "barcode_length": 22,
            "toehold_length": 10,
            "search_range": search_range,
            "toehold_search_iterations": 2 if search_range > 1 else 1,
            "barcode_pool_factor": 5,
            "barcode_generation_attempts": barcode_generation_attempts,
            "barcode_toehold_k": 9,
            "barcode_pair_k": 13,
            "barcode_subset_iterations": 2 if target_count == 1 else 1,
            "matching_iterations": 2 if target_count == 1 else 1,
            "barcode_gc_min": 0.3,
            "barcode_gc_max": 0.7,
            "barcode_max_homopolymer": 3,
        },
        "targets": targets,
        "order_policy": {
            "synthesis_scale": "declared-test-scale",
            "barcode_bearing_purification": "declared-test-purification",
            "complement_purification": "declared-test-purification",
            "primer_purification": "declared-test-purification",
            "complement_end_preparation": "vendor_5_prime_phosphate",
            "max_oligo_length": oligo_length + search_range - 1,
        },
    }
