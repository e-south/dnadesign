## Postprocess (pad)

If dense-arrays returns a sequence shorter than `sequence_length`, DenseGen can pad with
random bases. Postprocess runs after optimization and is recorded in metadata.

### Contents
- [Modes](#modes) - off, strict, or adaptive pad.
- [GC feasibility](#gc-feasibility) - why some targets are impossible.

---

### Modes

- `off` - fail if the sequence is short.
- `strict` - pad while enforcing GC targets; raise on infeasible targets.
- `adaptive` - relax GC targets when infeasible and record the relaxation.

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

Very short pads (for example, a 1 nt pad) cannot hit mid-range GC targets. `strict` fails fast
in these cases; `adaptive` relaxes bounds (via `gc.min_pad_length`) and records the final target and achieved GC in
metadata.

---

@e-south
