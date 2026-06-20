# Multiple Sequence Alignment

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-20

Use `dnadesign.aligner.msa` when a workflow needs a generic aligned FASTA
bundle. This package owns FASTA validation, MAFFT preflight/execution, and
aligned-bundle manifests. It does not own study-specific roster curation,
provider fetching, conservation scoring, or mask algebra.

## Public API

```python
from pathlib import Path

from dnadesign.aligner.msa import MsaBackendSpec, MsaRequest, run_msa

request = MsaRequest(
    input_fasta=Path("source.fasta"),
    output_fasta=Path("source.aligned.fasta"),
    manifest_path=Path("source.aligned.manifest.yaml"),
    target_row_id="target",
    backend=MsaBackendSpec(backend_id="mafft"),
    command_args=("--globalpair", "--maxiterate", "1000", "--reorder"),
)

result = run_msa(request)
```

## Dependency Contract

MAFFT is a native bioinformatics tool and is installed through Pixi, not `uv`.
Run MAFFT-backed workflows through Pixi:

```bash
pixi run mafft --version
pixi run uv run pytest src/dnadesign/aligner/tests/msa -q
```

The MAFFT wrapper fails fast when the executable is unavailable. There is no
implicit fallback backend.

## Bundle Manifest

Every run writes a YAML manifest with:

- backend id and version
- executable path
- full command
- input and output FASTA paths
- input and output SHA256 hashes
- target row id, when declared
- environment and Pixi lock hash, when discoverable
- explicit failure policy

## Boundary

`aligner.msa` emits aligned FASTA evidence. Downstream tools decide how to
interpret that evidence. For Eco1 RT repack, the study owns Mestre roster
curation, provider policy, target-sequence hash policy, and the T301/A301
source-authority mismatch.
