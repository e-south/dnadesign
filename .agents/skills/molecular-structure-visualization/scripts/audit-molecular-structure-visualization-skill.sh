#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SKILL_DIR/../../.." && pwd)"
failures=0

pass() { printf 'PASS: %s\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1" >&2; failures=$((failures + 1)); }

for path in \
  "$SKILL_DIR/SKILL.md" \
  "$SKILL_DIR/agents/openai.yaml" \
  "$SKILL_DIR/references/renderer-router.md" \
  "$SKILL_DIR/references/py3dmol-rendering-contract.md" \
  "$SKILL_DIR/references/cross-renderer-contract.md" \
  "$SKILL_DIR/references/external-sources.md" \
  "$SKILL_DIR/references/test-matrix.md" \
  "$SCRIPT_DIR/verify-molecular-scene-contract.py" \
  "$SCRIPT_DIR/verify-py3dmol-webgl.py"; do
  if [[ -f "$path" ]]; then pass "found ${path#$REPO_ROOT/}"; else fail "missing ${path#$REPO_ROOT/}"; fi
done

if grep -q '^name: molecular-structure-visualization$' "$SKILL_DIR/SKILL.md" \
  && grep -q 'Do not use for structure prediction' "$SKILL_DIR/SKILL.md"; then
  pass "frontmatter defines a narrow renderer-routing contract"
else
  fail "frontmatter routing contract is incomplete"
fi

if python3 -m py_compile "$SCRIPT_DIR/verify-molecular-scene-contract.py"; then
  pass "molecular scene verifier parses"
else
  fail "molecular scene verifier does not parse"
fi
if [[ -x "$SCRIPT_DIR/verify-molecular-scene-contract.py" ]]; then
  pass "molecular scene verifier is executable"
else
  fail "molecular scene verifier is not executable"
fi
if python3 -m py_compile "$SCRIPT_DIR/verify-py3dmol-webgl.py"; then
  pass "py3Dmol WebGL verifier parses"
else
  fail "py3Dmol WebGL verifier does not parse"
fi

tmp_dir="$(mktemp -d)"
cat >"$tmp_dir/browser.yaml" <<'YAML'
visual_contract:
  protein_surface_scope: protein_only
  protein_surface_alpha: 0.65
  dna_color: '#B97700'
  rna_color: '#C84C5A'
  py3dmol_nucleic_display: backbone_ribbon_with_base_spokes
  py3dmol_nucleic_ribbon_width_angstrom: 1.35
  py3dmol_nucleic_ribbon_thickness_angstrom: 0.28
  chimerax_nucleic_display: ladder
  chimerax_surface_transparency_percent: 35
  chimerax_nucleotide_color_target: acf
protein_surface_default: false
structures:
  - candidate_id: fixture
    molecule_styles:
      - {molecule_class: protein, style: surface, opacity: 0.65}
      - {molecule_class: dna, style: backbone_ribbon_with_base_spokes, color: '#B97700', width: 1.35, thickness: 0.28}
      - {molecule_class: rna, style: backbone_ribbon_with_base_spokes, color: '#C84C5A', width: 1.35, thickness: 0.28}
YAML
cat >"$tmp_dir/scene.cxc" <<'CXC'
nucleotides #1/D,E,F ladder
color #1/D #B97700 target acf
color #1/E,F #C84C5A target acf
surface #1/A
transparency #1/A 35 target s
CXC
if uv run python "$SCRIPT_DIR/verify-molecular-scene-contract.py" \
  --browser-manifest "$tmp_dir/browser.yaml" \
  --chimerax-script "$tmp_dir/scene.cxc" >/dev/null; then
  pass "molecular scene verifier accepts the canonical contract"
else
  fail "molecular scene verifier rejected the canonical contract"
fi
sed 's/transparency #1\/A 35/transparency #1\/A 0/' "$tmp_dir/scene.cxc" >"$tmp_dir/invalid.cxc"
if uv run python "$SCRIPT_DIR/verify-molecular-scene-contract.py" \
  --browser-manifest "$tmp_dir/browser.yaml" \
  --chimerax-script "$tmp_dir/invalid.cxc" >/dev/null 2>&1; then
  fail "molecular scene verifier accepted an opaque surface"
else
  pass "molecular scene verifier rejects an opaque surface"
fi
cat >"$tmp_dir/review.yaml" <<'YAML'
deliverables:
  - deliverable_id: browser
    artifact_kind: structure_browser_manifest
    status: rendered
    path: browser.yaml
  - deliverable_id: desktop
    artifact_kind: chimerax_script
    status: rendered
    path: scene.cxc
YAML
if uv run python "$SCRIPT_DIR/verify-molecular-scene-contract.py" \
  --review-manifest "$tmp_dir/review.yaml" >/dev/null; then
  pass "molecular scene verifier audits a complete review manifest"
else
  fail "molecular scene verifier rejected a valid review manifest"
fi
rm -rf "$tmp_dir"

if grep -q 'dnadesign.thread.structure_views' "$SKILL_DIR/references/py3dmol-rendering-contract.md" \
  && grep -q 'backbone_ribbon_with_base_spokes' "$SKILL_DIR/references/py3dmol-rendering-contract.md" \
  && grep -q "C4-prime" "$SKILL_DIR/references/py3dmol-rendering-contract.md" \
  && grep -q 'addCustom' "$SKILL_DIR/references/py3dmol-rendering-contract.md" \
  && grep -q 'WebGL' "$SKILL_DIR/references/py3dmol-rendering-contract.md"; then
  pass "py3Dmol contract exposes public API, coordinate geometry, and runtime checks"
else
  fail "py3Dmol contract is incomplete"
fi

if grep -q 'ladder' "$SKILL_DIR/references/cross-renderer-contract.md" \
  && grep -q 'ribbon-with-spokes' "$SKILL_DIR/references/cross-renderer-contract.md"; then
  pass "cross-renderer contract maps distinct backend primitives"
else
  fail "cross-renderer backend mapping is incomplete"
fi

if grep -q 'Retrieved: 2026-07-12' "$SKILL_DIR/references/external-sources.md" \
  && grep -q '3Dmol.js AtomStyleSpec' "$SKILL_DIR/references/external-sources.md" \
  && grep -q 'UCSF ChimeraX `nucleotides`' "$SKILL_DIR/references/external-sources.md"; then
  pass "official renderer sources are recorded"
else
  fail "external-source table is incomplete"
fi

if [[ -f "$REPO_ROOT/src/dnadesign/thread/structure_views/__init__.py" ]] \
  && grep -q 'render_structure_view_html' "$REPO_ROOT/src/dnadesign/thread/structure_views/__init__.py"; then
  pass "documented public browser-view facade exists"
else
  fail "documented public browser-view facade is missing"
fi

if grep -q 'PROTEIN_SURFACE_OPACITY = 0.65' "$REPO_ROOT/src/dnadesign/thread/structure_views/styles.py" \
  && grep -q 'protein_surface_opacity_by_model_id' "$REPO_ROOT/src/dnadesign/thread/structure_views/backends/py3dmol.py"; then
  pass "browser backend propagates the shared surface alpha"
else
  fail "browser surface alpha is not enforced by the backend"
fi

if [[ -d "$REPO_ROOT/outputs" ]]; then fail "repo-root outputs/ must not be created"; else pass "repo-root outputs/ absent"; fi

printf 'Audit finished with %s failures.\n' "$failures"
exit "$failures"
