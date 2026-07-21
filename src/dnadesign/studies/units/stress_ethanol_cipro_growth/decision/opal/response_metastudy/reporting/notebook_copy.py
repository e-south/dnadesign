"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/notebook_copy.py

Explanatory copy embedded in the generated metastudy review notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

STUDY_CONTEXT_MARKDOWN = """
        **Scope:** {review_summary.scope}

        **Observed labels:** {review_summary.label_state}

        **Predictor support:** {review_summary.predictor_support}

        **Basis:** {review_summary.basis}

        **Primary assay summary:** {review_summary.primary_assay_summary}

        **Evidence base:** {review_summary.evidence_base}

        **Prospective hill climb:** {review_summary.prospective_hill_climb}
        """

COMPARATOR_GUIDE_MARKDOWN = """
Reader publishes the same eight measured values for every target view:

- `r_i` is the median across design wells of the declared primary-window mean
  `log2[(YFP / CFP)_design,i(t)]` shown in the review header.
- `b_i` is the design-well median over that same primary window
  `log2[(YFP / OD600)_design,i(t)]` minus the same-state pDual-10 well median.

The study target mask changes only which states are treated as ON and OFF:

| Target | ON conditions | OFF conditions |
| --- | --- | --- |
| Ethanol | ethanol; ethanol + ciprofloxacin | no stress; ciprofloxacin |
| Ciprofloxacin | ciprofloxacin; ethanol + ciprofloxacin | no stress; ethanol |
| AND | ethanol + ciprofloxacin | no stress; ethanol; ciprofloxacin |
| OR (screen only) | ethanol; ciprofloxacin; ethanol + ciprofloxacin | no stress |

`m_response = min_ON(r) - max_OFF(r)`
`b_on = min_ON(b)`
`b_off = max_OFF(b)`
`S_RMF = min(q_response, q_on, q_off)`

RMF asks whether prespecified requirements are jointly met. Positive values clear its configured
requirement boundaries, and a strong component cannot compensate for a failed one. SFXI evaluates
a separate setpoint-fidelity and intensity phenotype. Both remain retrospective comparators here;
neither is the active campaign selector.
"""

__all__ = ["COMPARATOR_GUIDE_MARKDOWN", "STUDY_CONTEXT_MARKDOWN"]
