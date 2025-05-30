"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/evoinference/model_invocation.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import torch
from evo2 import Evo2  # Assumes Evo2 is installed and available
from logger import get_logger

logger = get_logger(__name__)


def initialize_model(model_version: str):
    """
    Initialize and return the Evo2 model for the given version.
    Note: Evo2 does not support the .to() method.
    """
    try:
        model = Evo2(model_version)
        # If the model has an eval() method, call it.
        if hasattr(model, "eval"):
            model.eval()
        logger.info(f"Model {model_version} initialized")
        return model
    except Exception as e:
        logger.error(f"Error initializing model {model_version}: {str(e)}")
        raise e


def tokenize_sequence(model, sequence: str):
    """
    Tokenize the sequence using the model's tokenizer and return a tensor
    with the batch dimension, moved to the GPU. Raises an error if a GPU is not available.
    """
    try:
        tokens = model.tokenizer.tokenize(sequence)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device not available. GPU is required.")
        device = "cuda:0"
        input_tensor = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).to(device)
        return input_tensor
    except Exception as e:
        logger.error(f"Error tokenizing sequence: {str(e)}")
        raise e


def run_model(model, input_tensor, output_types: list):
    """
    Run the model on the input tensor.

    Parameters:
      output_types: list of dicts, for example:
          [
              {"type": "logits"},
              {"type": "embeddings", "layers": ["blocks_28_mlp_l3", "blocks_10_mlp_l3"]}
          ]

    Returns a dictionary with keys like 'evo2_logits' and 'evo2_embeddings_<layer>'.
    """
    logits_requested = any(item.get("type") == "logits" for item in output_types)
    embedding_layers = []
    for item in output_types:
        if item.get("type") == "embeddings" and "layers" in item:
            for layer in item["layers"]:
                # Convert underscore notation to dot notation for model input
                embedding_layers.append(layer.replace("_", "."))
    return_embeddings = len(embedding_layers) > 0

    try:
        outputs, embeddings = model(input_tensor, return_embeddings=return_embeddings, layer_names=embedding_layers)
    except Exception as e:
        logger.error(f"Error during model forward pass: {str(e)}")
        raise e

    results = {}
    if logits_requested:
        results["evo2_logits"] = outputs[0]
    if return_embeddings:
        for item in output_types:
            if item.get("type") == "embeddings" and "layers" in item:
                for layer in item["layers"]:
                    dot_layer = layer.replace("_", ".")
                    if dot_layer not in embeddings:
                        logger.error(f"Requested layer {dot_layer} not found in model outputs.")
                        raise ValueError(f"Requested layer {dot_layer} not found in model outputs.")
                    results[f"evo2_embeddings_{layer}"] = embeddings[dot_layer]
    return results
