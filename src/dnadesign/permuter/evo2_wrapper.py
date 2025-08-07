"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/evo2_wrapper.py

Facade for evoinference: compute LL and LLR via API.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from evoinference.model_invocation import initialize_model, run_model

# initialize once
_MODEL = None


def _get_model(name: str):
    global _MODEL
    if _MODEL is None:
        _MODEL = initialize_model(name)
    return _MODEL


def score_log_likelihood(sequence: str, evaluator: str) -> float:
    model = _get_model(evaluator)
    input_tensor = model.tokenizer.tokenize(sequence)
    return run_model(
        model, input_tensor, [{"type": "log_likelihood", "reduction": "sum"}]
    )["evo2_log_likelihood"]


def compute_llr(ref_seq: str, var_seq: str, evaluator: str) -> float:
    ll_ref = score_log_likelihood(ref_seq, evaluator)
    ll_var = score_log_likelihood(var_seq, evaluator)
    return ll_var - ll_ref
