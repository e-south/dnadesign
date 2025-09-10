# OPAL Registries

```bash
src/registries/
  __init__.py                # optional; can re-export convenience getters
  ingest_transforms.py       # CSV→Y
  rep_transforms.py          # X-column→matrix
  objectives.py              # Ŷ→score
  selections.py              # score→ranks
  models.py                  # model factory (you may already have this under models/registry.py)
```