"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/evaluators/evo2_llr.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from dnadesign.infer import run_extract

from .base import Evaluator


class Evo2LogLikelihoodRatioEvaluator(Evaluator):
    """
    Evo2 log-likelihood ratio (LLR) vs reference.

    LLR(seq) = LL(seq) - LL(reference)
    (Uses same reduction for both; prefer reduction="mean" for length-normalized LLR.)

    Params (constructor kwargs):
      - model_id: str = "evo2_7b"
      - device: str = "cuda:0"
      - precision: str = "bf16"
      - alphabet: str = "dna"
      - method: str = "native"
      - reduction: str = "mean"          # "sum" or "mean"
      - batch_size: int | None = None
    """

    def __init__(
        self,
        *,
        model_id: str = "evo2_7b",
        device: str = "cuda:0",
        precision: str = "bf16",
        alphabet: str = "dna",
        method: str = "native",
        reduction: str = "mean",
        batch_size: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.precision = precision
        self.alphabet = alphabet
        self.method = method
        self.reduction = reduction
        self.batch_size = batch_size

    def _ll(self, seqs: List[str]) -> List[float]:
        outputs = [
            {
                "id": "ll",
                "fn": "evo2.log_likelihood",
                "params": {"method": self.method, "reduction": self.reduction},
                "format": "float",
            }
        ]
        res = run_extract(
            seqs,
            model_id=self.model_id,
            outputs=outputs,
            device=self.device,
            precision=self.precision,
            alphabet=self.alphabet,
            batch_size=self.batch_size,
        )
        return [float(x) for x in res["ll"]]

    # Evaluator interface
    def score(
        self,
        sequences: List[str],
        *,
        metric: str,
        ref_sequence: str | None = None,
        ref_embedding=None,
    ) -> List[float]:
        if not ref_sequence:
            raise ValueError(
                "Evo2LogLikelihoodRatioEvaluator requires ref_sequence (got None)."
            )

        # Compute LL for variants (batched) and LL for reference (once), then subtract.
        ll_variants = self._ll(sequences)
        ll_ref = self._ll([ref_sequence])[0]
        return [v - ll_ref for v in ll_variants]
