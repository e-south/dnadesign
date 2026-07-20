"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/visual_panel_baserender.py

BaseRender-backed panels for OPAL notebook visual review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Mapping

from ._support import display_name
from .baserender_record_sources import compact_record_id


def build_notebook_baserender_panel_title(candidate_evidence: Mapping[str, Any]) -> str:
    """Build an in-canvas title from one authoritative candidate-evidence row."""

    record_id = str(candidate_evidence.get("record_id") or "").strip()
    if not record_id:
        raise ValueError("BaseRender panel title requires a non-empty record_id.")
    memberships = _selection_memberships(candidate_evidence)
    active_view_id = str(candidate_evidence.get("active_selection_view_id") or "").strip()
    active_membership = [row for row in memberships if row["selection_view_id"] == active_view_id]
    active_rank = candidate_evidence.get("active_view_rank")
    if active_rank is not None:
        if len(active_membership) != 1 or int(active_rank) != int(active_membership[0]["view_rank"]):
            raise ValueError("BaseRender candidate active-view rank does not match its selection membership.")
    if active_membership:
        return _selection_title(record_id=record_id, membership=active_membership[0])
    if len(memberships) == 1:
        return _selection_title(record_id=record_id, membership=memberships[0])
    if memberships:
        membership_text = ", ".join(
            f"{_view_label(row['selection_view_id'])} rank {row['view_rank']}" for row in memberships
        )
        return f"Selected candidate · {membership_text} · {compact_record_id(record_id)}"
    observed_rounds = _observed_rounds(candidate_evidence)
    if observed_rounds:
        round_text = ", ".join(f"{value}" for value in observed_rounds)
        return f"Observed candidate · round {round_text} · {compact_record_id(record_id)}"
    raise ValueError("BaseRender panel title requires selected or observed campaign evidence.")


def render_notebook_baserender_panel(
    *,
    baserender_campaign_model: Mapping[str, Any] | None,
    baserender_record_id: Any,
    baserender_record_row: Mapping[str, Any] | None,
    baserender_candidate_evidence: Mapping[str, Any] | None,
    build_notebook_baserender_contract_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]] | None,
    build_notebook_baserender_label_rows: Callable[..., list[dict[str, Any]]] | None,
    controls: Any | None,
    mo: Any,
    opal_table: Callable[..., Any],
    pl: Any,
    render_notebook_baserender_record: Callable[..., Mapping[str, Any]] | None,
    selected_baserender_round: int | None,
    selected_baserender_status_rows: Iterable[Mapping[str, Any]],
    selected_campaign_baserender_contract: Mapping[str, Any] | None,
    selected_campaign_labels_df: Any,
) -> Any:
    """Render one campaign candidate with binding, label, and selection evidence."""

    contract = selected_campaign_baserender_contract or {}
    label_rows = _require_callable(
        build_notebook_baserender_label_rows,
        "build_notebook_baserender_label_rows",
    )(
        selected_campaign_labels_df,
        record_id="" if baserender_record_id is None else str(baserender_record_id),
        round_value=None,
    )
    label_panel = (
        opal_table(pl.DataFrame(label_rows), page_size=5)
        if label_rows
        else mo.md("No observed label is available for this campaign candidate.")
    )
    if baserender_record_row is None:
        visual = mo.md("No campaign candidate is available under the declared BaseRender adapter.")
    else:
        visual = _render_candidate_sequence(
            baserender_campaign_model=baserender_campaign_model,
            baserender_record_id=baserender_record_id,
            baserender_record_row=baserender_record_row,
            baserender_candidate_evidence=baserender_candidate_evidence,
            contract=contract,
            mo=mo,
            render_notebook_baserender_record=render_notebook_baserender_record,
            selected_baserender_round=selected_baserender_round,
        )
    evidence_detail_rows = list(selected_baserender_status_rows)
    if baserender_candidate_evidence:
        evidence_detail_rows.extend(_candidate_evidence_rows(baserender_candidate_evidence))
    detail_items = [
        opal_table(pl.DataFrame(evidence_detail_rows), page_size=10),
        label_panel,
        opal_table(
            pl.DataFrame(
                _require_callable(
                    build_notebook_baserender_contract_rows,
                    "build_notebook_baserender_contract_rows",
                )(contract)
            ),
            page_size=8,
        ),
    ]
    return _render_panel_stack(
        mo=mo,
        items=[
            controls,
            visual,
            mo.accordion({"Candidate and campaign evidence": mo.vstack(detail_items, gap=0.35)}, multiple=True),
        ],
    )


def _render_candidate_sequence(
    *,
    baserender_campaign_model: Mapping[str, Any] | None,
    baserender_record_id: Any,
    baserender_record_row: Mapping[str, Any],
    baserender_candidate_evidence: Mapping[str, Any] | None,
    contract: Mapping[str, Any],
    mo: Any,
    render_notebook_baserender_record: Callable[..., Mapping[str, Any]] | None,
    selected_baserender_round: int | None,
) -> Any:
    candidate_evidence = dict(baserender_candidate_evidence or {})
    title = build_notebook_baserender_panel_title(candidate_evidence)
    expected_record_id = _validate_candidate_record_identity(
        candidate_evidence=candidate_evidence,
        candidate_record_id=baserender_record_id,
        record_row=baserender_record_row,
    )
    payload = _require_callable(
        render_notebook_baserender_record,
        "render_notebook_baserender_record",
    )(baserender_record_row, contract, title=title)
    _validate_rendered_record_identity(payload=payload, expected_record_id=expected_record_id)
    slug = str((baserender_campaign_model or {}).get("campaign", {}).get("slug") or "unknown campaign")
    return _render_sequence_image(
        payload=payload,
        mo=mo,
        alt_suffix=_candidate_alt_suffix(
            candidate_evidence,
            campaign_slug=slug,
            selected_round=selected_baserender_round,
        ),
    )


def _candidate_alt_suffix(
    candidate_evidence: Mapping[str, Any],
    *,
    campaign_slug: str,
    selected_round: int | None,
) -> str:
    statements: list[str] = []
    if _selection_memberships(candidate_evidence):
        round_suffix = f", round {selected_round}" if selected_round is not None else ""
        statements.append(f"Selected in campaign {campaign_slug}{round_suffix}.")
    observed_rounds = _observed_rounds(candidate_evidence)
    if observed_rounds:
        noun = "round" if len(observed_rounds) == 1 else "rounds"
        values = ", ".join(str(value) for value in observed_rounds)
        statements.append(f"Observed in campaign {campaign_slug}, {noun} {values}.")
    if not statements:
        raise ValueError("BaseRender candidate alt text requires selected or observed campaign evidence.")
    return " " + " ".join(statements)


def _validate_candidate_record_identity(
    *,
    candidate_evidence: Mapping[str, Any],
    candidate_record_id: Any,
    record_row: Mapping[str, Any],
) -> str:
    expected_record_id = str(candidate_evidence.get("record_id") or "").strip()
    row_record_id = str(record_row.get("id") or "").strip()
    if row_record_id != expected_record_id:
        raise ValueError(
            "BaseRender record does not match its authoritative campaign evidence: "
            f"expected `{expected_record_id}`, received `{row_record_id or 'missing'}`."
        )
    control_record_id = str(candidate_record_id or "").strip()
    if control_record_id != expected_record_id:
        raise ValueError(
            "BaseRender lookup does not match its authoritative campaign evidence: "
            f"expected `{expected_record_id}`, received `{control_record_id or 'missing'}`."
        )
    return expected_record_id


def _validate_rendered_record_identity(*, payload: Mapping[str, Any], expected_record_id: str) -> None:
    payload_record_id = str(payload.get("record_id") or "").strip()
    if payload_record_id != expected_record_id:
        raise ValueError(
            "BaseRender payload does not match its authoritative campaign evidence: "
            f"expected `{expected_record_id}`, received `{payload_record_id or 'missing'}`."
        )


def _render_sequence_image(
    *,
    payload: Mapping[str, Any],
    mo: Any,
    alt_suffix: str = "",
) -> Any:
    return mo.image(
        payload["image_bytes"],
        alt=f"{payload['alt_text']}{alt_suffix}",
        caption=str(payload["caption"]),
        rounded=True,
        style={
            "width": "100%",
            "max-width": "100%",
            "height": "auto",
            "object-fit": "contain",
            "margin": "0",
            "display": "block",
            "background-color": "#FFFFFF",
        },
    )


def _render_panel_stack(*, mo: Any, items: Iterable[Any | None]) -> Any:
    return mo.vstack([item for item in items if item is not None], gap=0.45)


def _require_callable(value: Callable[..., Any] | None, name: str) -> Callable[..., Any]:
    if value is None:
        raise ValueError(f"{name} is required for this OPAL notebook visual surface.")
    return value


def _selection_memberships(candidate_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    memberships: list[dict[str, Any]] = []
    for raw in candidate_evidence.get("selection_memberships") or ():
        if not isinstance(raw, Mapping):
            raise ValueError("BaseRender candidate selection membership must be an object.")
        view_id = str(raw.get("selection_view_id") or "").strip()
        rank_value = raw.get("view_rank")
        if not view_id or isinstance(rank_value, bool):
            raise ValueError("BaseRender candidate selection membership requires a view and positive rank.")
        try:
            rank = int(rank_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("BaseRender candidate selection membership requires a view and positive rank.") from exc
        if rank <= 0 or float(rank_value) != float(rank):
            raise ValueError("BaseRender candidate selection membership requires a view and positive rank.")
        memberships.append({"selection_view_id": view_id, "view_rank": rank})
    return memberships


def _observed_rounds(candidate_evidence: Mapping[str, Any]) -> list[int]:
    rounds: list[int] = []
    for value in candidate_evidence.get("observed_rounds") or ():
        if isinstance(value, bool):
            raise ValueError("BaseRender observed round must be a non-negative integer.")
        try:
            round_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("BaseRender observed round must be a non-negative integer.") from exc
        if round_value < 0 or float(value) != float(round_value):
            raise ValueError("BaseRender observed round must be a non-negative integer.")
        rounds.append(round_value)
    return sorted(set(rounds))


def _selection_title(*, record_id: str, membership: Mapping[str, Any]) -> str:
    return (
        f"{_view_label(str(membership['selection_view_id']))} selection · "
        f"competition rank {int(membership['view_rank'])} · candidate {compact_record_id(record_id)}"
    )


def _view_label(view_id: str) -> str:
    return "AND" if view_id.lower() == "and" else display_name(view_id)


def _candidate_evidence_rows(candidate_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [{"field": "candidate", "value": str(candidate_evidence["record_id"])}]
    memberships = _selection_memberships(candidate_evidence)
    if memberships:
        rows.append(
            {
                "field": "selected views",
                "value": ", ".join(
                    f"{_view_label(row['selection_view_id'])} rank {row['view_rank']}" for row in memberships
                ),
            }
        )
    observed_rounds = _observed_rounds(candidate_evidence)
    if observed_rounds:
        rows.append({"field": "observed rounds", "value": ", ".join(map(str, observed_rounds))})
    return rows


__all__ = [
    "build_notebook_baserender_panel_title",
    "render_notebook_baserender_panel",
]
