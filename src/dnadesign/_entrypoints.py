"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/_entrypoints.py

Lazy installed command launchers with optional-runtime diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from importlib import import_module

# Only diagnose imports supplied by the recommended installation profile.
# Unknown failures, including command/plugin defects, retain their traceback.
_CLI_IMPORTS = frozenset({"pandas", "pyarrow", "pydantic", "yaml", "rich", "typer"})
_PROFILE_IMPORTS = {
    "seqledger": _CLI_IMPORTS | {"Bio", "duckdb", "psutil"},
    "permuter": _CLI_IMPORTS | {"matplotlib", "seaborn", "psutil"},
    "opal": _CLI_IMPORTS | {"polars", "sklearn", "scipy", "joblib", "typing_extensions", "matplotlib", "openpyxl"},
    "full": _CLI_IMPORTS
    | {
        "anyio",
        "arviz",
        "Bio",
        "click",
        "igraph",
        "leidenalg",
        "logomaker",
        "markdown_it",
        "matplotlib",
        "openpyxl",
        "PIL",
        "pygments",
        "Levenshtein",
        "scanpy",
        "sklearn",
        "scipy",
        "seaborn",
        "tqdm",
        "umap",
        "duckdb",
        "xlrd",
        "torch",
        "polars",
        "altair",
        "vegafusion",
        "vl_convert",
        "pysam",
        "netCDF4",
        "certifi",
        "keyring",
        "fastparquet",
        "filelock",
        "marimo",
        "RNA",
        "zstd",
        "py3Dmol",
        "plotly",
        "psutil",
        "typing_extensions",  # Required by the profile's Pydantic dependency.
    },
    "infer-evo2": {"evo2", "flash_attn", "transformer_engine"},
}
_PROFILE_IMPORTS["infer"] = _PROFILE_IMPORTS["seqledger"] | {"tqdm"}

_PROFILE_IMPORTS["ops"] = {"pydantic", "yaml", "typer", "pyarrow", "click"}
_PROFILE_IMPORTS["notify"] = {"typer", "click", "keyring", "certifi"}
_PROFILE_IMPORTS["folding"] = {"pydantic", "yaml", "typer", "psutil", "RNA"}
_PROFILE_IMPORTS["construct"] = _PROFILE_IMPORTS["seqledger"] | _PROFILE_IMPORTS["folding"] | {"vl_convert"}
_PROFILE_IMPORTS["baserender"] = {"pydantic", "yaml", "typer", "pyarrow", "matplotlib", "PIL", "psutil"}
_PROFILE_IMPORTS["junction"] = {"pydantic", "yaml", "typer", "matplotlib", "psutil"}
_PROFILE_IMPORTS["cluster"] = _PROFILE_IMPORTS["seqledger"] | {
    "scanpy",
    "igraph",
    "leidenalg",
    "sklearn",
    "scipy",
    "seaborn",
    "matplotlib",
    "umap",
    "Levenshtein",
    "click",
}
_PROFILE_IMPORTS["latentdna"] = _PROFILE_IMPORTS["seqledger"] | {
    "anndata",
    "igraph",
    "leidenalg",
    "pynndescent",
    "scipy",
    "sklearn",
    "matplotlib",
    "umap",
}
_PROFILE_IMPORTS["cruncher"] = _PROFILE_IMPORTS["baserender"] | {
    "Bio",
    "Levenshtein",
    "arviz",
    "certifi",
    "click",
    "logomaker",
    "numba",
    "pandas",
    "pysam",
    "rich",
    "tqdm",
}


for _notebook_tool in ("opal", "latentdna", "cruncher"):
    _PROFILE_IMPORTS[f"{_notebook_tool}-notebook"] = _PROFILE_IMPORTS[_notebook_tool] | {"marimo"}


def _run(target: str, extra: str):
    try:
        module, attribute = target.split(":")
        return getattr(import_module(module), attribute)()
    except ModuleNotFoundError as exc:
        if not exc.name or exc.name.startswith("dnadesign") or "." in exc.name:
            raise
        if exc.name == "marimo" and extra in {"opal", "latentdna", "cruncher"}:
            extra = f"{extra}-notebook"
        if extra in {"infer", "permuter", "full"}:
            if exc.name in {"evo2", "flash_attn", "transformer_engine"}:
                extra = "infer-evo2"
            elif exc.name == "torch":
                extra = "full"
        if exc.name.split(".")[0] not in _PROFILE_IMPORTS.get(extra, ()):
            raise
        raise SystemExit(
            f"This command requires dnadesign-tools[{extra}] (missing {exc.name}). "
            f"Install with pip install 'dnadesign-tools[{extra}]', or in a checkout "
            f"run uv sync --locked --no-default-groups --extra {extra}."
        ) from None


def seqledger():
    return _run("dnadesign.seqledger:main", "seqledger")


def seqledger_sync_audit_drill():
    return _run("dnadesign.seqledger.ops.sync_audit_drill:main", "seqledger")


def opal():
    return _run("dnadesign.opal:main", "opal")


def notify():
    return _run("dnadesign.notify.cli:app", "notify")


def infer():
    return _run("dnadesign.infer.cli:app", "infer")


def construct():
    return _run("dnadesign.construct:main", "construct")


def folding():
    return _run("dnadesign.folding.cli:main", "folding")


def cluster():
    return _run("dnadesign.cluster.cli:main", "cluster")


def latentdna():
    return _run("dnadesign.latentdna.src:main", "latentdna")


def permuter():
    return _run("dnadesign.permuter:main", "permuter")


def baserender():
    return _run("dnadesign.baserender:app", "baserender")


def cruncher():
    return _run("dnadesign.cruncher.cli.app:app", "cruncher")


def junction():
    return _run("dnadesign.junction.cli:main", "junction")


def ops():
    return _run("dnadesign.ops.cli:main", "ops")
