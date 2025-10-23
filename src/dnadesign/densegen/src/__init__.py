"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/densegen/src/__init__.py

densegen — Dense Array Generator

DenseGen builds densely packed promoter sequences using the `dense-arrays` ILP
optimizer. Inputs are local CSVs or USR datasets. Outputs can be saved to USR,
to local newline-delimited JSON (JSONL), or both.

Public modules:
- main: CLI entrypoint / orchestration
- data_ingestor: CSV + USR dataset loaders
- sampler: TF/TFBS sampling and subsampling
- optimizer_wrapper: thin wrapper around dense-arrays Optimizer
- progress_tracker: YAML progress state
- usr_adapter: USRWriter buffer/flush utility
- logging_utils: opinionated logging setup
- outputs: pluggable output sinks (USR, JSONL)

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

__all__ = [
    "data_ingestor",
    "sampler",
    "optimizer_wrapper",
    "progress_tracker",
    "usr_adapter",
    "logging_utils",
    "outputs",
    "plotting",
]
