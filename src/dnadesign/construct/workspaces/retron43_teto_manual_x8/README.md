## Retron-43 TetO manual x8 composition

This workspace dogfoods the generic `linear_ssdna_composition_v1` Construct
workflow for the Retron Hairpin study.

The config is intentionally literal and Retron-specific only as data. Construct
assembles ordered ssDNA segments, expands eight copies, validates the declared
TetO reverse-complement arm, and writes local artifacts under `outputs/`.

Commands:

```bash
uv run construct compose validate --config config.composition.yaml
uv run construct compose run --config config.composition.yaml
uv run folding preflight --request outputs/retron43_teto_manual_x8/folding/secondary_structure_prediction_request_v1.yaml
uv run folding run --request outputs/retron43_teto_manual_x8/folding/secondary_structure_prediction_request_v1.yaml
uv run baserender job run outputs/retron43_teto_manual_x8/baserender_jobs/component_span_qa_svg.yaml
```

ViennaRNA is the folding backend package. This workspace uses its uv-managed
Python interface (`RNA`); `RNAfold` is the optional ViennaRNA CLI program for
requests that explicitly select `backend.interface: cli`. If a selected
interface is unavailable and folding is advisory, the generated prediction
records `warning_optional_missing`.

Generated `outputs/` artifacts are not hand-edited.
