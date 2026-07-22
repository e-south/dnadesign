"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_sequences.py

Protein-sequence display helpers for Eco1 review-deliverable notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html


def handoff_sequence_list_html(rows: list[dict[str, str]]) -> str:
    """Render selected protein sequences as wrapped, searchable notebook HTML."""

    cards = "\n".join(_handoff_sequence_card_html(row) for row in rows)
    return f"""
    <section class="eco1-handoff-sequence-list" aria-label="Selected RT protein sequence list">
      <style>
        .eco1-handoff-sequence-list {{
          display:grid;
          grid-template-columns:1fr;
          gap:0.55rem;
          width:100%;
          max-width:100%;
          margin:0 0 0.25rem 0;
        }}
        .eco1-handoff-sequence-card {{
          border:1px solid #d8dee4;
          border-radius:6px;
          padding:0.55rem 0.65rem;
          background:#ffffff;
          color:#24292f;
        }}
        .eco1-handoff-sequence-card header {{
          display:flex;
          flex-wrap:wrap;
          gap:0.45rem 0.8rem;
          align-items:baseline;
          margin:0 0 0.35rem 0;
          font-size:0.88rem;
          color:#57606a;
        }}
        .eco1-handoff-sequence-card strong {{
          color:#24292f;
          font-weight:650;
        }}
        .eco1-handoff-sequence-card code {{
          display:block;
          font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
          font-size:0.78rem;
          line-height:1.35;
          white-space:normal;
          overflow-wrap:anywhere;
          word-break:break-word;
        }}
      </style>
      {cards}
    </section>
    """


def _handoff_sequence_card_html(row: dict[str, str]) -> str:
    variant_id = html.escape(str(row.get("variant_id") or ""))
    candidate_id = html.escape(str(row.get("candidate_id") or ""))
    selection_slot = html.escape(str(row.get("selection_slot") or ""))
    dna_status = html.escape(str(row.get("dna_design_status") or ""))
    restriction_status = html.escape(str(row.get("restriction_site_screen_status") or ""))
    sequence = html.escape(str(row.get("protein_sequence") or ""))
    return f"""
      <article class="eco1-handoff-sequence-card" data-candidate-id="{candidate_id}">
        <header>
          <strong>{variant_id or candidate_id}</strong>
          <span>{candidate_id}</span>
          <span>{selection_slot}</span>
          <span>DNA: {dna_status}</span>
          <span>restriction screen: {restriction_status}</span>
        </header>
        <code>{sequence}</code>
      </article>
    """
