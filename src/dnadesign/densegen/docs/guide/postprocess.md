## Postprocess (pad)

This page explains what happens when solver output is shorter than your target sequence length.

Postprocess runs after solving and records what it did in metadata.

### Contents
- [Modes](#modes)
- [GC feasibility](#gc-feasibility)

---

### Modes

- `off`: fail if sequence is short
- `strict`: pad while enforcing GC targets; fail if infeasible
- `adaptive`: relax GC constraints when strict targets are impossible; record final target/achieved GC

```yaml
postprocess:
  pad:
    mode: adaptive
    end: 5prime
    gc:
      mode: range
      min: 0.40
      max: 0.60
      target: 0.50
      tolerance: 0.10
      min_pad_length: 0
    max_tries: 2000
```

---

### GC feasibility

Short pads can make some GC targets impossible (for example, a 1-nt pad cannot hit middle GC ranges).

- `strict` fails fast on impossible targets
- `adaptive` relaxes bounds and records the relaxed values

---

@e-south
