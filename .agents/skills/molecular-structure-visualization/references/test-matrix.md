# Test Matrix

| Scenario | Prompt | Expected Behavior | Pass/Fail |
| --- | --- | --- | --- |
| Trigger positive | "Render this protein-DNA-RNA complex in py3Dmol." | Use this skill and the public browser-view API. | Pass if molecule roles, source format, and browser validation are explicit. |
| Trigger positive | "Keep the browser and ChimeraX views consistent." | Use renderer-specific mappings from the cross-renderer contract. | Pass if ChimeraX ladder and py3Dmol ribbon-with-spokes views are not conflated. |
| Trigger negative | "Capture this manually oriented ChimeraX pose." | Route to `chimerax-structure-review`. | Pass if this skill does not own the GUI session. |
| Functional core | DNA and RNA browser scene. | Use ordered C4-prime ribbon meshes and attached base spokes. | Pass if the mesh is wider than thick, every nucleotide has one spoke, and full-ring or phosphate sticks are absent. |
| Functional edge | Converted structure drops `C1'` atoms. | Fail before rendering. | Pass if the response routes to structured coordinate conversion. |
| Browser runtime | Interactive toggles are in scope. | Exercise the real WebGL view. | Pass if canvas, controls, role colors, connectivity, and console are checked. |
| Camera framing | A structure view opens or its default framing changes. | Fit the whole visible complex, apply bounded zoom, and version the camera-memory key. | Pass if the complex is prominent, unclipped, and stale local storage cannot restore an obsolete view. |
| Trackpad interaction | A browser structure can be panned. | Use reduced pointer-pan sensitivity. | Pass if a normal two-finger gesture produces a small, controllable translation. |
| Model-state matrix | Reference, DNA, RNA, surface, mutation, side-chain, or residue-highlight controls are available. | Render every row and highlight across representative control states. | Pass if every selection style targets a loaded model and each molecule control remains independent. |
| Saved browser artifact | A py3Dmol HTML file is ready for review. | Run `verify-py3dmol-webgl.py`. | Pass if the scene audit has one spoke per nucleotide and the screenshot contains nonblank DNA and RNA role pixels. |
| ChimeraX mapping | Default nucleotide view. | Route to native cartoons plus ladder display. | Pass if connected atom sticks are not described as the default. |
| Cross-renderer colors | DNA and RNA appear in both backends. | Color each nucleotide representation and its backbone from the same role color. | Pass if DNA is gold, RNA is salmon, and ChimeraX uses `target acf`. |
| Cross-renderer surface | A protein surface is enabled in both backends. | Keep the interactive default off and apply the surface to protein only at 0.65 alpha when shown. | Pass if py3Dmol opacity is `0.65` and ChimeraX transparency is `35`. |
| Artifact verification | Browser manifest and ChimeraX script are materialized. | Run `verify-molecular-scene-contract.py`. | Pass if its JSON status is `pass` with no failures. |
| Bundle verification | A review manifest links several structure manifests and scripts. | Audit the top-level review manifest. | Pass if every non-skipped molecular artifact is checked and listed. |
| Repeatability | Run the skill audit twice. | Both runs pass without generated repo-root output. | Pass if the audit exits zero twice. |
