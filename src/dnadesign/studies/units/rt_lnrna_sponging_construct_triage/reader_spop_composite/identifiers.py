"""
Reader SPOP composite identifier helpers.
"""

from __future__ import annotations


def assay_subject_key_for_display_id(display_id: str) -> str:
    prefix = "pES-retron-"
    if not display_id.startswith(prefix):
        raise ValueError(f"unsupported retron display id: {display_id!r}")
    number = display_id[len(prefix) :]
    if not number.isdigit():
        raise ValueError(f"unsupported retron display id: {display_id!r}")
    return f"retron{number}"


def display_id_for_assay_subject(assay_subject_key: str) -> str:
    prefix = "retron"
    if assay_subject_key.startswith(prefix) and assay_subject_key[len(prefix) :].isdigit():
        return f"pES-retron-{assay_subject_key[len(prefix) :]}"
    return assay_subject_key


def variant_sort_key(value: str) -> tuple[int, str]:
    prefix = "retron"
    if value.startswith(prefix) and value[len(prefix) :].isdigit():
        return int(value[len(prefix) :]), value
    return 10**9, value
