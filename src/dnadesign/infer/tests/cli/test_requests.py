"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/tests/cli/test_requests.py

Characterization tests for infer CLI request assembly helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.infer.src.cli.requests import build_extract_request, build_generate_request


def test_build_extract_request_from_single_output_flags() -> None:
    request = build_extract_request(
        model_id="evo2_7b",
        device="cpu",
        precision="fp32",
        alphabet="dna",
        batch_size=4,
        preset=None,
        fn="evo2.logits",
        format="float",
        out_id="llr",
        pool_method="mean",
        pool_dim=1,
        layer=None,
        write_back=True,
        overwrite=False,
    )

    assert request.job.operation == "extract"
    assert request.job.io.write_back is True
    assert request.job.io.overwrite is False
    assert request.job.outputs is not None
    assert request.job.outputs[0].id == "llr"
    assert request.job.outputs[0].params == {"pool": {"method": "mean", "dim": 1}}


def test_build_generate_request_default_params_when_no_preset() -> None:
    request = build_generate_request(
        model_id=None,
        device=None,
        precision=None,
        alphabet=None,
        batch_size=None,
        preset=None,
        max_new_tokens=None,
        temperature=None,
        top_k=None,
        top_p=None,
        seed=None,
    )

    assert request.job.operation == "generate"
    assert request.job.params == {"max_new_tokens": 64, "temperature": 1.0}
