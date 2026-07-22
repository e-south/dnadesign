"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/views/promoter_metadata_stress.py

Stress-promoter study metadata derivations for LatentDNA rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

from ..contracts.errors import ContractViolationError
from .promoter_metadata_common import canonical_regulator_name, normalize_text, normalized_regulators

SIG35_PATTERN = re.compile(r"__sig35[=_]([A-Za-z0-9]+)")
CONTROL_LABELS = {"spyp", "sulap", "soxsp", "j23105", "spy_p", "sul_ap", "sox_sp"}


def construct_template_id(row: dict[str, object]) -> str | None:
    return (
        normalize_text(row.get("construct_template_id"))
        or normalize_text(row.get("template_id"))
        or normalize_text(row.get("construct__template_id"))
    )


def is_control_row(row: dict[str, object]) -> bool:
    label = (normalize_text(row.get("usr_label__primary")) or "").lower()
    template_id = (construct_template_id(row) or "").lower()
    source_family = (normalize_text(row.get("source_family")) or "").lower()
    has_densegen_plan = normalize_text(row.get("densegen__plan")) is not None
    if template_id in {"wt", "wildtype", "manual"}:
        return True
    if label in CONTROL_LABELS:
        return True
    if has_densegen_plan:
        return False
    if normalize_text(row.get("promoter_standard__collection_id")) is not None:
        return True
    return "reference" in source_family or "genbank" in source_family or "standard" in source_family


def design_family(row: dict[str, object]) -> str:
    plan = normalize_text(row.get("densegen__plan"))
    if plan is not None:
        if plan.startswith("background_only"):
            return "background_only"
        if plan.startswith("ethanol_ciprofloxacin"):
            return "ethanol_ciprofloxacin"
        if plan.startswith("ethanol"):
            return "ethanol"
        if plan.startswith("ciprofloxacin"):
            return "ciprofloxacin"
        raise ContractViolationError(f"design_family does not support densegen__plan value: {plan!r}")
    if is_control_row(row):
        return "control"
    raise ContractViolationError(
        "design_family could not be derived; expected densegen__plan or explicit control/reference provenance"
    )


def design_regulator_composition(row: dict[str, object]) -> str:
    if is_control_row(row):
        return "control"
    family = design_family(row)
    regulators = normalized_regulators(row.get("densegen__required_regulators"))
    if family == "background_only" and not regulators:
        return "background"
    if regulators:
        return regulators[0] if len(regulators) == 1 else "+".join(regulators)

    plan = normalize_text(row.get("densegen__plan")) or ""
    tokens = [token for token in plan.split("__") if token]
    if len(tokens) >= 2 and not tokens[1].startswith("sigma70_"):
        composition_parts = [canonical_regulator_name(token) for token in tokens[1].replace("_", "+").split("+")]
        composition_parts = sorted(
            {
                part
                for part in composition_parts
                if part not in {None, "control"} and not str(part).startswith("sig35=")
            },
            key=str.casefold,
        )
        if composition_parts:
            return composition_parts[0] if len(composition_parts) == 1 else "+".join(composition_parts)
    if family == "background_only":
        return "background"
    return "unknown"


def campaign_prior(row: dict[str, object]) -> str:
    family = design_family(row)
    return {
        "background_only": "background",
        "ethanol": "ethanol",
        "ciprofloxacin": "cipro",
        "ethanol_ciprofloxacin": "and",
        "control": "control",
    }[family]


def source_class(row: dict[str, object]) -> str:
    if normalize_text(row.get("densegen__plan")) is not None:
        return "densegen"
    if (
        normalize_text(row.get("regulondb__primary_promoter_name")) is not None
        or normalize_text(row.get("derived__parent_dataset")) == "usr_regulondb_native_promoters"
    ):
        return "native_regulondb"
    source_family = normalize_text(row.get("source_family"))
    if source_family is not None:
        normalized = source_family.lower()
        if "densegen" in normalized:
            return "densegen"
        if "reference" in normalized or "genbank" in normalized or "standard" in normalized:
            return "reference_control"
        return normalized
    if normalize_text(row.get("promoter_standard__collection_id")) is not None:
        return "synthetic_reference_standard"
    if is_control_row(row):
        return "manual_or_wildtype"
    raise ContractViolationError(
        "source_class could not be derived; expected densegen, RegulonDB, reference, standard, "
        "or explicit control provenance"
    )
