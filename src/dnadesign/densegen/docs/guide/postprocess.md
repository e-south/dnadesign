## Postprocess (gap fill)

If dense-arrays returns a sequence shorter than `sequence_length`, DenseGen can gap-fill with
random bases. Postprocess runs after optimization and is recorded in metadata.

### Contents
- [Modes](#modes) - off, strict, or adaptive gap fill.
- [GC feasibility](#gc-feasibility) - why some targets are impossible.

---

### Modes

- `off` - fail if the sequence is short.
- `strict` - fill while enforcing the GC window; raise on infeasible targets.
- `adaptive` - relax the GC window when infeasible and record the relaxation.

```yaml
postprocess:
  gap_fill:
    mode: adaptive
    end: 5prime
    gc_min: 0.40
    gc_max: 0.60
    max_tries: 2000
```

---

### GC feasibility

Very short gaps (for example, a 1 nt gap) cannot hit mid-range GC targets. `strict` fails fast
in these cases; `adaptive` relaxes bounds and records the final target and achieved GC in
metadata.

---

@e-south
