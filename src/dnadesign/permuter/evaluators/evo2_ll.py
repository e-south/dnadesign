"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/evaluators/evo2_ll.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from dnadesign.infer import run_extract

from .base import Evaluator


class Evo2LogLikelihoodEvaluator(Evaluator):
    """
    Evo2 log-likelihood (LL) via dnadesign.infer.

    Params (constructor kwargs):
      - model_id: str = "evo2_7b"
      - device: str = "cuda:0"
      - precision: str = "bf16"          # fp32|fp16|bf16
      - alphabet: str = "dna"
      - method: str = "native"           # fixed for Evo2
      - reduction: str = "mean"          # "sum" or "mean"
      - batch_size: int | None = None    # optional, passed through to infer
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

    # Evaluator interface
    def score(
        self,
        sequences: List[str],
        *,
        metric: str,
        ref_sequence: str | None = None,
        ref_embedding=None,
    ) -> List[float]:
        # One call to infer; returns a column named by our 'id'
        outputs = [
            {
                "id": "ll",
                "fn": "evo2.log_likelihood",
                "params": {"method": self.method, "reduction": self.reduction},
                "format": "float",
            }
        ]
        res = run_extract(
            sequences,
            model_id=self.model_id,
            outputs=outputs,
            device=self.device,
            precision=self.precision,
            alphabet=self.alphabet,
            batch_size=self.batch_size,
        )
        return [float(x) for x in res["ll"]]
