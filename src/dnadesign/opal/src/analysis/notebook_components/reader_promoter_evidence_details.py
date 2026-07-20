"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/reader_promoter_evidence_details.py

Render a compact notebook disclosure for verified Reader promoter evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

from .reader_evidence_media import reader_reduction_display_label
from .reader_promoter_evidence_contract import verify_reader_promoter_evidence_context

_ADAPTER_LABELS = {
    "densegen_tfbs": "DenseGen TFBS",
    "usr_genbank_annotations_v1": "GenBank source",
}


def render_notebook_reader_promoter_evidence_details(row: Mapping[str, Any], *, mo: Any) -> Any:
    """Render one concise accordion from already verified v5 evidence fields."""

    context = verify_reader_promoter_evidence_context(row)
    response = context["response_window"]
    bindings = context["candidate_bindings"]
    baserender = context["baserender"]
    selected_binding = context["selected_binding"]
    claim_label = "Objective-neutral" if context["claim_status"] == "objective_neutral" else "Screen only"
    adapter_label = _ADAPTER_LABELS.get(str(baserender["adapter_kind"]), str(baserender["adapter_kind"]))
    rows = [
        ("Claim", claim_label),
        ("Reader experiment", str(row.get("reader_experiment_id") or "")),
        ("Design", str(row.get("design_id") or "")),
        ("Candidate", str(row.get("candidate_id") or "")),
        ("Response summary", reader_reduction_display_label(row.get("reduction_id"))),
        (
            "Trajectory marks",
            "line: across-well median; band: central across-well interval",
        ),
        (
            "Phenotype marks",
            "white points: observed response replicates; colored bar: joint-bootstrap interval; "
            "gray bar: reduction range when the event is placed at either recorded timing bound",
        ),
        ("Response source", f"{response['schema_version']} · {_short_digest(response['manifest_sha256'])}"),
        (
            "Sequence annotation",
            f"{adapter_label} · {baserender['sequence_length_bp']} bp · {baserender['feature_count']} features",
        ),
        (
            "Candidate binding",
            f"{selected_binding.get('binding_method', 'not recorded')} · "
            f"{bindings['candidate_table_id']} · {_short_digest(bindings['manifest_sha256'])}",
        ),
    ]
    overlay = context["objective_overlay"]
    if overlay is not None:
        component_text = "; ".join(
            f"{item['label']}: {float(item['value']):.3g} {item['unit']}" for item in overlay["components"]
        )
        rows.append(("Objective overlay", f"{overlay['objective_display_label']} · screen only · {component_text}"))
    markdown = [
        "| Evidence field | Verified value |",
        "|---|---|",
        *[f"| {_escape(label)} | {_escape(value)} |" for label, value in rows],
        "",
        f"> {_escape(context['non_claim_boundary'])}",
    ]
    return mo.accordion(
        {"Evidence details": mo.md("\n".join(markdown))},
        multiple=False,
    )


def _short_digest(value: object) -> str:
    text = str(value or "")
    if text.startswith("sha256:") and len(text) == 71:
        return f"sha256:{text[7:17]}…{text[-8:]}"
    return text


def _escape(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


__all__ = ["render_notebook_reader_promoter_evidence_details"]
