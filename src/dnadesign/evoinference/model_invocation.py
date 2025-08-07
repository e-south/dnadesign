"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/evoinference/model_invocation.py

Provides initialization, tokenization, and inference adapters for Evo2 models,
including support for logits, embeddings, and log-likelihood output types.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import torch
from evo2 import Evo2  # Assumes Evo2 is installed
from logger import get_logger

logger = get_logger(__name__)


def initialize_model(model_version: str) -> Evo2:
    """
    Instantiate and prepare an Evo2 model by version name.
    """
    model = Evo2(model_version)
    if hasattr(model, "eval"):
        model.eval()
    logger.info(f"Initialized Evo2 model '{model_version}'")
    return model


def tokenize_sequence(model: Evo2, sequence: str) -> torch.Tensor:
    """
    Tokenize a nucleotide sequence and move to GPU as int64 tensor [1, seq_len].
    Raises if CUDA is unavailable.
    """
    tokens = model.tokenizer.tokenize(sequence)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. GPU is required.")
    return torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).cuda()


def score_log_likelihood(
    model: Evo2, sequences: list[str], reduction: str = "sum"
) -> list[float]:
    """
    Batch compute log-likelihoods via Evo2's built-in score_sequences method.
    Returns a list of floats matching input order.
    If reduction is 'mean', divides each score by its sequence length.
    """
    # Use Evo2's optimized batch scorer
    ll_values = model.score_sequences(sequences)
    if reduction == "mean":
        return [ll / len(seq) for ll, seq in zip(ll_values, sequences)]
    return ll_values


def run_model(
    model: Evo2, input_tensor: torch.Tensor, output_types: list[dict], raw_sequence: str
) -> dict:
    """
    Execute inference and extract requested outputs.
    Supports:
      - logits: returns evo2_logits tensor
      - embeddings: returns evo2_embeddings_<layer>
      - log_likelihood: returns evo2_log_likelihood float

    raw_sequence is required if type 'log_likelihood' is requested.
    """
    # Determine requested outputs
    want_logits = any(conf.get("type") == "logits" for conf in output_types)
    ll_conf = next(
        (conf for conf in output_types if conf.get("type") == "log_likelihood"), None
    )
    want_ll = ll_conf is not None
    reduction = ll_conf.get("reduction", "sum") if ll_conf else None

    # Embedding layers
    embed_layers = []
    for conf in output_types:
        if conf.get("type") == "embeddings":
            embed_layers += [lyr.replace("_", ".") for lyr in conf.get("layers", [])]
    want_embeds = bool(embed_layers)

    results = {}

    # Compute log-likelihood first if requested (uses batch API)
    if want_ll:
        ll_value = score_log_likelihood(model, [raw_sequence], reduction=reduction)[0]
        results["evo2_log_likelihood"] = ll_value

    # Only run forward pass if logits or embeddings are requested
    if want_logits or want_embeds:
        outputs, embeddings = model(
            input_tensor, return_embeddings=want_embeds, layer_names=embed_layers
        )
        if want_logits:
            results["evo2_logits"] = outputs[0]
        if want_embeds:
            for conf in output_types:
                if conf.get("type") == "embeddings":
                    for lyr in conf.get("layers", []):
                        key = lyr.replace(".", "_")
                        dot = lyr.replace("_", ".")
                        if dot not in embeddings:
                            raise ValueError(f"Layer '{dot}' not found in embeddings")
                        results[f"evo2_embeddings_{key}"] = embeddings[dot]

    return results
