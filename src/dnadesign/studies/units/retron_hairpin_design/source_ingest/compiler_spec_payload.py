"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/compiler_spec_payload.py

Build explicit compiler specs from normalized MSD-region records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence

from .genbank_utils import reverse_complement, variant_number, variant_sort_key
from .models import MsdRegionIngestError, NormalizedMsdRegionRecord


def compiler_spec_payload_from_records(records: Sequence[NormalizedMsdRegionRecord]) -> dict[str, object]:
    """Build an explicit compiler-spec payload from normalized MSD records."""

    designs: list[dict[str, object]] = []
    payload_sequences: dict[str, dict[str, object]] = {}
    payload_complement_sequences: dict[str, dict[str, object]] = {}
    cap_sequences: dict[str, dict[str, object]] = {}
    for record in sorted(records, key=lambda item: variant_sort_key(item.variant_id)):
        number = variant_number(record.variant_id)
        payload_id = f"MSDRegion{number}_payload"
        cap_id = f"C{number}_msd_region"
        payload, cap, payload_complement = compiler_interval_sequences(record)
        payload_sequences[payload_id] = {
            "sequence": payload,
            "display_name": f"{record.display_id} source primary arm",
            "selection_basis": "msd_region_canonical_interval",
        }
        if payload_complement != reverse_complement(payload):
            payload_complement_sequences[payload_id] = {
                "sequence": payload_complement,
                "display_name": f"{record.display_id} source complement arm",
                "selection_basis": "msd_region_canonical_interval",
            }
        cap_sequences[cap_id] = {"sequence": cap}
        designs.append(
            {
                "construct_id": record.display_id,
                "payload_id": payload_id,
                "cap_id": cap_id,
                "left_base": record.primitive("stem_base_left").sequence_5to3.upper(),
                "right_base": record.primitive("stem_base_right").sequence_5to3.upper(),
                "source_notes": (
                    f"Generated from decomposed retron-hairpin MSD-region source record {record.file_stem}.yaml."
                ),
            }
        )
    return {
        "contract": "retron_msd_compiler_spec_v1",
        "schema_version": 1,
        "allow_non_ligatable_s0": True,
        "designs": designs,
        "payload_sequences": payload_sequences,
        "payload_complement_sequences": payload_complement_sequences,
        "cap_sequences": cap_sequences,
    }


def compiler_interval_sequences(record: NormalizedMsdRegionRecord) -> tuple[str, str, str]:
    left = record.primitive("stem_base_left")
    right = record.primitive("stem_base_right")
    payload = record.primitive("payload_primary")
    complement = record.primitive("payload_complement")
    if not (left.display_end_0 <= payload.display_end_0 <= complement.display_start_0 <= right.display_start_0):
        raise MsdRegionIngestError(f"{record.variant_id}: source features cannot be ordered as an MSD unit.")
    primary = record.msd_sequence_5to3[left.display_end_0 : payload.display_end_0].upper()
    cap = record.msd_sequence_5to3[payload.display_end_0 : complement.display_start_0].upper()
    payload_complement = record.msd_sequence_5to3[complement.display_start_0 : right.display_start_0].upper()
    if not primary or not cap or not payload_complement:
        raise MsdRegionIngestError(f"{record.variant_id}: source features produced an empty compiler interval.")
    return primary, cap, payload_complement


__all__ = ["compiler_interval_sequences", "compiler_spec_payload_from_records"]
