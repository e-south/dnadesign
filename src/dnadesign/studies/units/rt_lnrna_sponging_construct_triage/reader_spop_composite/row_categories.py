"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/row_categories.py

Study-owned row categories for the Reader SPOP composite plot.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class RetronRowCategory:
    category_id: str
    label: str
    display_label: str
    description: str
    color: str
    text_color: str = "#1f2937"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetronRowCategorySpan:
    category_id: str
    label: str
    display_label: str
    description: str
    color: str
    text_color: str
    start_index: int
    stop_index: int
    assay_subject_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "category_id": self.category_id,
            "label": self.label,
            "display_label": self.display_label,
            "description": self.description,
            "color": self.color,
            "text_color": self.text_color,
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "assay_subject_keys": list(self.assay_subject_keys),
        }


UNKNOWN_ROW_CATEGORY = RetronRowCategory(
    category_id="unclassified",
    label="Unclassified",
    display_label="Unclassified",
    description="Variant lacks a curated Reader SPOP row category.",
    color="#e5e7eb",
)

ROW_CATEGORIES: dict[str, RetronRowCategory] = {
    "guu_reference": RetronRowCategory(
        category_id="guu_reference",
        label="GUU reference",
        display_label="GUU\nreference",
        description="Eco1 retron reference carrying the GUU RT-recognition edit.",
        color="#dbeafe",
    ),
    "teto_hop_reference": RetronRowCategory(
        category_id="teto_hop_reference",
        label="tetO HOP",
        display_label="tetO\nHOP",
        description="msd-HOP tetO insertion reference derived from the GUU scaffold.",
        color="#e0f2fe",
    ),
    "stem_base_context": RetronRowCategory(
        category_id="stem_base_context",
        label="Stem-base context",
        display_label="Stem-base\ncontext",
        description="WT stem-base context edits around the tetO-containing MSD stem.",
        color="#dcfce7",
    ),
    "sso7d_rt_fusion": RetronRowCategory(
        category_id="sso7d_rt_fusion",
        label="Sso7d-RT fusions",
        display_label="Sso7d-RT\nfusions",
        description="N- or C-terminal Sso7d fusions to the Eco1 reverse transcriptase.",
        color="#fef3c7",
    ),
    "evo2_rt_mutants": RetronRowCategory(
        category_id="evo2_rt_mutants",
        label="Evo2 RT mutants",
        display_label="Evo2 RT\nmutants",
        description="Single amino-acid point mutations applied to the Eco1 RT.",
        color="#ffedd5",
    ),
    "teto_site_swaps": RetronRowCategory(
        category_id="teto_site_swaps",
        label="tetO site swaps",
        display_label="tetO site\nswaps",
        description="TetR binding-site replacements in the tetO payload context.",
        color="#fae8ff",
    ),
    "foldback_cores": RetronRowCategory(
        category_id="foldback_cores",
        label="Foldback cores",
        display_label="Foldback\ncores",
        description="Compact foldback-core variants in the TetR payload context.",
        color="#ede9fe",
    ),
    "stem_cap_wobbles": RetronRowCategory(
        category_id="stem_cap_wobbles",
        label="Stem/cap wobbles",
        display_label="Stem/cap\nwobbles",
        description="Stem-base, cap, and wobble scans around the tetO MSD payload.",
        color="#fce7f3",
    ),
    "teto_truncations": RetronRowCategory(
        category_id="teto_truncations",
        label="tetO truncations",
        display_label="tetO\ntruncations",
        description="Truncated tetO PWM windows placed into 26-, 43-, or 180-like scaffolds.",
        color="#ccfbf1",
    ),
}

ASSAY_SUBJECT_CATEGORY_IDS: dict[str, str] = {
    "retron26": "guu_reference",
    "retron43": "teto_hop_reference",
    "retron45": "stem_base_context",
    "retron46": "stem_base_context",
    "retron47": "sso7d_rt_fusion",
    "retron48": "sso7d_rt_fusion",
    "retron49": "evo2_rt_mutants",
    "retron50": "evo2_rt_mutants",
    "retron51": "evo2_rt_mutants",
    "retron52": "evo2_rt_mutants",
    "retron53": "evo2_rt_mutants",
    "retron54": "evo2_rt_mutants",
    "retron55": "evo2_rt_mutants",
    "retron56": "evo2_rt_mutants",
    "retron170": "teto_site_swaps",
    "retron171": "teto_site_swaps",
    "retron172": "foldback_cores",
    "retron173": "foldback_cores",
    "retron174": "foldback_cores",
    "retron175": "foldback_cores",
    "retron176": "foldback_cores",
    "retron177": "stem_cap_wobbles",
    "retron178": "stem_cap_wobbles",
    "retron179": "stem_cap_wobbles",
    "retron180": "stem_cap_wobbles",
    "retron181": "stem_cap_wobbles",
    "retron182": "stem_cap_wobbles",
    "retron183": "stem_cap_wobbles",
    "retron184": "stem_cap_wobbles",
    "retron185": "stem_cap_wobbles",
    "retron186": "stem_cap_wobbles",
    "retron195": "teto_truncations",
    "retron196": "teto_truncations",
    "retron197": "teto_truncations",
    "retron198": "teto_truncations",
    "retron199": "teto_truncations",
    "retron200": "teto_truncations",
}


def category_for_assay_subject(assay_subject_key: str) -> RetronRowCategory:
    """Return the curated row category for a Reader assay-subject key."""

    category_id = ASSAY_SUBJECT_CATEGORY_IDS.get(assay_subject_key)
    if category_id is None:
        return UNKNOWN_ROW_CATEGORY
    return ROW_CATEGORIES[category_id]


def category_spans_for_variants(variants: Sequence[str]) -> tuple[RetronRowCategorySpan, ...]:
    """Return contiguous category spans for variants in plotted row order."""

    if not variants:
        return ()
    spans: list[RetronRowCategorySpan] = []
    start_index = 0
    current_category = category_for_assay_subject(variants[0])
    current_variants: list[str] = [variants[0]]
    for index, variant in enumerate(variants[1:], start=1):
        category = category_for_assay_subject(variant)
        if category.category_id == current_category.category_id:
            current_variants.append(variant)
            continue
        spans.append(_span_for_category(current_category, start_index, index, tuple(current_variants)))
        start_index = index
        current_category = category
        current_variants = [variant]
    spans.append(_span_for_category(current_category, start_index, len(variants), tuple(current_variants)))
    return tuple(spans)


def _span_for_category(
    category: RetronRowCategory,
    start_index: int,
    stop_index: int,
    assay_subject_keys: tuple[str, ...],
) -> RetronRowCategorySpan:
    return RetronRowCategorySpan(
        category_id=category.category_id,
        label=category.label,
        display_label=category.display_label,
        description=category.description,
        color=category.color,
        text_color=category.text_color,
        start_index=start_index,
        stop_index=stop_index,
        assay_subject_keys=assay_subject_keys,
    )
