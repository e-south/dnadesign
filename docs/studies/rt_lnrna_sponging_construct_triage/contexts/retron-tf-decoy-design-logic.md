---
doc_id: study-rt-lnrna-sponging-construct-triage-retron-tf-decoy-design-logic
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-28
related_studies:
  - rt_lnrna_sponging_construct_triage
  - retron_hairpin_design
---

## Retron TF Decoy Design Logic

Use this page when manuscript, figure, or talk work needs the shared semantics
between the RT-lnRNA construct triage study and the Retron hairpin design study.
This is a progressively disclosed orientation document: start at the center of
gravity, then open only the downstream layer needed for the current writing or
review task.

### First-Hop Summary

The center of gravity is not a design platform and not a library workflow.

> Retron TF decoys are attractive because they can genetically encode abundant
> intracellular decoy DNA, but useful decoy design requires sequence exploration
> inside a retron hairpin whose structure also controls whether the decoy is
> produced.

That framing keeps the paper scientific rather than hype-driven. It names two
real tensions:

1. **Network / payload uncertainty:** the relevant TF, binding site, valency,
   spacing, orientation, or combination may be unknown in the target context.
2. **Retron biogenesis constraint:** the decoy sequence is not an isolated
   oligo; it is embedded in a structured RT-DNA product, and changes to that
   structure can collapse abundance.

The synthesis and msd-HOP story is a practical consequence of that tension, not
a third coequal pillar. Empirical search requires many variants, and many
variants require a construction route compatible with inverted-repeat retron
hairpins.

### Reader-Value Argument

Treat the manuscript as a reader-value argument, not a list of capabilities.
The paper should not say "we built a workflow for retron decoy design." It
should say:

> Readers who want to use retrons as intracellular TF decoys face a coupled
> uncertainty: the decoy sequence may need to be discovered empirically, but
> every candidate must remain compatible with retron RT-DNA production.

That is the reader's problem. The work creates value by making that problem
visible, testable, and partially actionable.

Use this as the reader-centered premise:

> Retron-encoded TF decoys are useful only when two conditions hold at once:
> the payload perturbs the intended regulator, and the modified retron hairpin
> still produces the decoy at functional abundance.

Use this more manuscript-like version when the Eco1/TetR model anchors the
paper:

> This work uses Eco1 TetR sponging to expose a coupled design constraint for
> retron TF decoys: payload function and RT-DNA biogenesis cannot be optimized
> independently.

Use this forward-looking version for the full-paper spine:

> Because useful TF decoys may require empirical sequence search, retron decoy
> design must make that search compatible with the structural constraints of
> RT-DNA production.

The sharpest value claim is diagnostic:

> This work gives readers a way to interpret retron-decoy success and failure:
> sponging is abundance-gated, so payload design must be evaluated together
> with retron hairpin processability.

Before this paper, the natural reader question is:

> Can I put a binding site in Eco1 and get a decoy?

After this paper, the better question is:

> Can I find a binding-site architecture that perturbs the regulator while
> preserving RT-DNA production?

That is the value transformation. The reader should leave with a decision rule:

> A retron decoy must bind the regulator, but first the retron must make the
> decoy.

### Reader-Value Map

Use this map to keep the manuscript centered on the reader's instability,
rather than on the existence of a toolchain.

| Reader community | Accepted view | What is unstable | Cost of the instability | Intervention | Payoff |
| --- | --- | --- | --- | --- | --- |
| Synthetic biologists | TF decoys can tune gene circuits, and retrons can express protein-binding DNAs inside cells. | Demonstrations are mostly with well-characterized TFs and chosen operators; the next use case is finding decoys that perturb a target state. | A design that works for TetR may not generalize to uncertain, multivalent, or phenotype-driven perturbations. | Frame retron decoys as an empirical design problem, not a motif-placement problem. | Readers get a sober path from demonstration to testing: start with known operators, then expand through constrained variant search. |
| Retron engineers / retron biologists | Retrons can make abundant RT-DNA, and Eco1 can be modified for biotechnology. | RT-DNA output is brittle; Eco1 ncRNA changes can alter reverse-transcription efficiency and abundance. | Failed sponging can be misread as failed binding when the real problem is failed production. | Use TetR/tetO as a clean readout to test scaffold changes while holding payload recognition mostly constant. | Readers learn which failures should be interpreted as scaffold or processability failures rather than payload failures. |
| Gene-regulatory-network readers | Perturbing TF activity can alter cellular state, but the relevant TF or TF combination may not be known. | Phenotype-first perturbation may not begin with a known TF, site, spacing, valency, or architecture. | Rational single-design decoys are unlikely to be enough for complex states. | Explain why empirical search is needed, and why retron search has unique constraints. | Readers see retron decoys as a possible perturbation modality with a clear screening and validation burden. |
| Genome-model readers | Genome models can encode biological sequence information and can represent regulatory features. | A model embedding is not a functional label; abundance geometry is not sponging. | Overclaiming makes the work vulnerable because "Evo2 predicts sponging" is not justified without functional labels. | Use Evo2/LatentDNA as an abundance prior or audit, not as the endpoint. | Readers get a disciplined example of model use: triage before screening, not replacement of screening. |
| Therapeutic / translational readers | TF decoys have therapeutic precedent, but delivery, persistence, and potency are limiting. | Retron decoys are attractive but are not yet clinically or broadly biologically validated. | Hype invites the wrong evaluation standard. | Present retrons as a modality for intracellular decoy presentation, not as a therapeutic claim. | Readers see a design bottleneck being addressed, not a premature translational promise. |

The highest-value reader is the retron engineer / synthetic biologist hybrid:
someone who believes retrons can make useful DNA, but does not yet know how to
modify the hairpin without losing function.

### Route By Reader Job

| Reader job | Open next | Use this claim posture |
| --- | --- | --- |
| Manuscript premise or abstract | this page, then `../workbench/figure_mocks/retron_tf_decoy_manuscript_figures.html` | Conservative design-logic claim |
| Paired RT + lnRNA construct semantics | `construct-overview.md` and `construct-contract.md` | Construct subject, not RT-only catalog |
| Literature abundance priors | `source-overlays.md` and `../workbench/provenance/source-handoff-ledger.md` | Abundance prior, not sponging label |
| Evo2 / LatentDNA triage | `latentdna/trait-axis-projection.md` | Abundance plausibility prior, not functional prediction |
| Retron MSD product design | `../../retron_hairpin_design/routes/compiler/msd-design-references.md` | Traceable product/reference compiler |
| Snapback cap or scar-nick primitive meaning | `../../retron_hairpin_design/routes/product/released-product-snapback.md` and `../../retron_hairpin_design/routes/product/scar-nick-base-junction.md` | Primitive owner, not whole-paper thesis |
| Persistent hairpin study rationale | `../../retron_hairpin_design/workbench/README.md` | Durable design-set meaning and effect tags |

### One-Sentence Premises

Use the broad premise when the manuscript has functional evidence beyond a
single model context:

> Retron-encoded TF decoys could enable intracellular transcription-factor
> sponging, but moving from single known operators to useful network
> perturbations requires balancing empirical payload search with the structural
> constraints of retron RT-DNA production.

Use the methods-forward premise when the paper should emphasize the design
problem rather than the eventual biological application:

> This work asks how to design retron-encoded TF decoys when the perturbing
> sequence is uncertain but the retron scaffold is structurally constrained.

Use the most conservative premise when the final data are mostly TetR,
construction, and abundance triage:

> Using Eco1 TetR sponging as a tractable model, this work defines the coupled
> design constraints that govern retron decoys: the payload must bind the
> intended regulator, and the modified retron hairpin must still support
> productive RT-DNA biogenesis.

### Situation - Complication - Resolution

#### Situation

DNA decoys are a general way to titrate DNA-binding proteins, and retrons
provide a genetically encoded way to make decoy DNA inside cells.

Retrons make TF decoys interesting because they turn decoy DNA from an externally
delivered or plasmid-encoded object into an intracellular reverse-transcribed
molecule whose copy number and structure can, in principle, be engineered. That
is a modality claim, not a platform claim.

Use this literature posture:

- Natural and engineered DNA decoys establish the regulatory perturbation idea.
- Lee/Kim-style pretroDNA establishes the closest intracellular retron decoy
  precedent, including TetR/LuxR-responsive systems and feedback circuits.
- Khan-style retron census data motivate retron diversity and RT-DNA abundance
  variability.

#### Complication

The hard part is not inserting a motif. The hard part is making a useful
perturbing decoy without breaking RT-DNA production.

For many phenotypes, the relevant regulator, binding architecture, and effective
combination of decoy sites are not known a priori.

| Design situation | What is uncertain | Experiment needed |
| --- | --- | --- |
| Known TF, known operator | Whether the operator still works on the retron scaffold, at the needed abundance and valency | Low-N validation, such as TetR/tetO |
| Known TF, uncertain site or architecture | Which affinity, spacing, orientation, or valency gives the desired effect | Curated variant panel or small library |
| Phenotype-first perturbation | Which TFs or TF combinations should be sponged | Functional screen against a reporter, stress phenotype, or cell state |

A retron decoy is only functional if the modified ncRNA still matures into
abundant RT-DNA. Crawford-style Eco1 variant-library data support the claim that
reverse-transcription efficiency is a limiting step for retron technologies and
that Eco1 ncRNA variants produce different RT-DNA levels. Lee/Kim-style stem
length results support the claim that fully matched long stems can sharply
reduce retron-DNA productivity. Wang-style structural data support the reason
the molecule cannot be treated as a free oligo: the msDNA stem-loop is part of a
structured RT-msDNA complex.

Construction enters as a consequence:

> Because the active decoy sequence may need to be discovered empirically, we
> need libraries; because the molecule is a retron hairpin, those libraries must
> obey synthesis and processing constraints; and because abundance is required
> for sponging, every design is also an RT-DNA biogenesis experiment.

#### Resolution

The humble resolution is:

> We do not claim to solve retron decoy design generally; we define and test the
> constraints that any useful retron-decoy design strategy must satisfy.

The preferred resolution sentence is:

> We use Eco1 TetR sponging as a controlled readout to ask which retron-decoy
> modifications remain compatible with function, then develop construction and
> computational triage routes that make broader empirical search feasible.

### Layer Hierarchy

| Layer | What it answers | Emphasis |
| --- | --- | --- |
| Functional anchor | Can Eco1 carrying a known operator sponge a TF in vivo? | High; this is the biological anchor |
| Scaffold/processability analysis | Which hairpin changes preserve or destroy sponging or likely RT-DNA output? | Highest; this is the mechanistic heart |
| Construction route | Can degenerate or semi-degenerate payloads be converted into retron-compatible constructs? | Medium; useful method, not the whole paper |
| Evo2 / LatentDNA abundance triage | Can representation geometry prioritize abundance-compatible variants before functional screening? | Conditional; valuable only if axes are stable |

### Refined Problem Statement

Retron decoy design has a built-in tension. On one side,
transcriptional-network perturbation often requires empirical search because the
relevant TFs, binding sites, affinities, valencies, or combinations may be
uncertain in the target context. On the other side, retron-derived DNA is
produced through a structured ncRNA and RT-DNA biogenesis pathway, so payload
changes that are attractive from a binding perspective can reduce or abolish
production. The design problem is therefore not simply to choose a TF-binding
sequence, but to find decoy architectures that both perturb the regulator and
remain compatible with retron maturation.

### Paper Or Talk Structure

| Section | Premise as a complete sentence | Minimal essential data |
| --- | --- | --- |
| 1. Decoy DNA is a general regulatory perturbation strategy | DNA decoys can perturb gene-regulatory networks by titrating DNA-binding proteins away from their native targets. | TF bound to genomic site versus TF titrated by decoy; short prior-art panel across ODNs, plasmids, nanoparticles, and retrons |
| 2. Retrons create a distinct intracellular decoy modality | Retrons are attractive decoy carriers because they genetically encode abundant, non-genomic DNA products inside cells. | Retron-Eco1 schematic; Lee/Kim pretroDNA precedent; Khan diversity and abundance variability; definitions for msr, msd, and RT-DNA |
| 3. The design problem is coupled, not modular | A retron decoy must satisfy two constraints at once: it must bind the right regulator and it must still be produced by the retron. | Two-axis home diagram: payload uncertainty and RT-DNA processability |
| 4. A known TetR/tetO decoy provides a tractable anchor | A known operator in Eco1 gives a controlled system for separating payload binding from scaffold effects. | Retron-26 or equivalent TetO construct; reporter derepression; RFP/OD or fluorescence readout; negative controls |
| 5. Hairpin stabilization can destroy the functional readout | Increasing or over-stabilizing the hairpin can reduce sponging, consistent with decoy function being gated by RT-DNA production. | Retron-26 versus Retron-43/45/170/171; long fully complementary P4 stem failure; mismatch/bulge attempts; growth/OD controls |
| 6. Prior retron data explain why this failure is plausible | Retron literature shows that RT-DNA production is structurally constrained and cannot be assumed after payload insertion. | Crawford variant-library summary; Lee/Kim matched-stem productivity; Wang Ec86 structure; Khan diversity |
| 7. Empirical search is needed when payloads are uncertain | Once the target site, valency, or TF combination is uncertain, retron decoy design becomes an empirical search problem. | Known-TF, unknown-site, and phenotype-first examples; TetR as low-N proof; optional second reporter |
| 8. msd-HOP addresses the construction bottleneck | Degenerate decoy search requires a way to turn memoryless ssDNA oligo pools into retron-compatible hairpins. | msd-HOP schematic; Sanger/NGS validation; recovery of expected TetO or designed variants; diversity retention if available |
| 9. Evo2/LatentDNA is a pre-screen for abundance plausibility | Because abundance is a hard gate, representation geometry may prioritize variants likely to remain RT-DNA-compatible before functional screening. | Crawford/Khan abundance axes; endpoint separation; axis concordance; candidate projections; explicit non-sponging-label statement |
| 10. Functional endpoint | The biological endpoint is whether retron decoys perturb TF activity in context, first for TetR and then for additional biosensor-accessible TFs. | TetR reporter; second TF if available; low-N selected variants |
| 11. Conclusion | The contribution is a constrained design logic for retron TF decoys: useful payloads must be found empirically, but every payload must pass the retron-production constraint. | Revisit payload uncertainty, biogenesis fragility, construction route, and abundance triage |

### Component Positioning

#### TetR/tetO

TetR/tetO is the model system, not the destination. Its value is that payload
binding is already known, so the study can ask:

> When payload binding is not the limiting uncertainty, what scaffold changes
> break or preserve retron-decoy function?

#### msd-HOP

msd-HOP is an enabling method, not the thesis. Use:

> To make such searches practical, we developed a hairpin-oligo processing route
> that converts designed or degenerate ssDNA oligos into Eco1-compatible decoy
> constructs.

Avoid:

> We built a platform for retron decoy discovery.

#### Evo2 / LatentDNA

Evo2 is a triage hypothesis, not a predictive claim. Frame it as:

> If RT-DNA abundance is a hard gate, can a genome model provide a weak but
> useful prior for which modified constructs remain abundance-compatible?

Use this internal rule:

> Crawford/Khan abundance != sponging. Evo2 abundance geometry != sponging
> prediction. TetR reporter response = sponging assay in one model context.

If the axes are strong, the safe claim is:

> Evo2 representations preserve an abundance-associated geometry that can help
> triage retron-decoy candidates before functional testing.

If the axes are weak, the useful claim is:

> The failure of abundance axes to generalize shows that retron-decoy triage
> cannot currently be outsourced to a genome-model embedding and still requires
> direct abundance or functional measurement.

#### Compiler

The compiler should almost disappear from the narrative surface. It is a method
that makes designs precise and reproducible:

> Designs were specified as explicit combinations of payload, cap, stem-base,
> and scar/nick choices so that each tested construct could be traced to its
> intended sequence and construction constraints.

Avoid:

> We built a genetic compiler for retron design.

### Master Figure Logic

Build the figure set around one recurring home slide:

> Retron decoy design sits at the intersection of payload search and RT-DNA
> processability.

The home slide has three boxes:

1. **What should the decoy bind?** TF, site, affinity, valency, combination,
   phenotype context.
2. **Will the retron still produce it?** Hairpin length, stem stability, bulges,
   cap geometry, RT/ncRNA processing, RT-DNA abundance.
3. **Can we build enough variants to learn?** Curated designs, degenerate
   oligos, msd-HOP, low-N screens, possible pooled screens.

Every data episode should point back to one of these boxes.

Use the companion HTML mock set at
`../workbench/figure_mocks/retron_tf_decoy_manuscript_figures.html` to inspect
five multi-panel manuscript arrangements.

### Figure Narrative

| Figure | Premise | Show |
| --- | --- | --- |
| Figure 1 | Retron decoys are a plausible intracellular modality for TF sponging because they encode decoy DNA as a reverse-transcribed cellular product. | TF decoy concept, delivery/presentation modes, Retron-Eco1 schematic, Lee/Kim precedent |
| Figure 2 | Retron decoys are hard because payload uncertainty and retron biogenesis constraints are coupled in the same molecule. | Payload body inside the hairpin; arrows to TF binding and RT-DNA production; known-TF/unknown-site and phenotype-first cases |
| Figure 3 | A known TetO payload gives a controlled assay for testing retron-decoy function. | Reporter repression/derepression logic; Retron-26 or equivalent; RFP/GFP output; OD control |
| Figure 4 | Retron decoy function can be lost by scaffold changes that likely interfere with RT-DNA production. | Retron-26 versus Retron-43/45/170/171; long fully complementary stem failure; mismatch/bulge attempts; Sso7d if included |
| Figure 5 | Observed scaffold sensitivity is expected from known retron biology. | Crawford Eco1 tolerance/intolerance map; Lee/Kim stem-length productivity; Wang Ec86 structure; Khan diversity |
| Figure 6 | Effective decoy payloads may need empirical discovery, so the build route must support sequence variation inside retron-compatible hairpins. | Curated versus degenerate designs; memoryless oligo synthesis problem; msd-HOP schematic; Sanger/NGS confirmation |
| Figure 7 | Abundance-associated representation geometry may prioritize designs before functional screening. | Crawford Eco1 axis; Khan retron axis; candidate projections; TetR constructs if available |
| Figure 8 | The final biological test is whether selected retron decoys perturb TF activity in the intended reporter or cellular context. | TetR validation; second TF if available; low-N selected variants; future pooled screen only if not done |

### Evidence Boundaries

The phrase "what it does not prove" is not defensive. It builds reader trust by
making the evaluation boundary explicit.

| Evidence | What it lets the reader conclude | What it does not prove |
| --- | --- | --- |
| TetR/tetO reporter sponging | A retron-borne known operator can produce a functional TF-sponging phenotype in a controlled assay. | Broad endogenous TF sponging. |
| Long or stabilized hairpin failure | Some scaffold changes can destroy functional output even when the payload concept is known. | The exact molecular failure mode unless RT-DNA abundance or processing is measured directly. |
| Crawford-style Eco1 variant data | Eco1 ncRNA has tolerant and intolerant regions, and RT-DNA production is a limiting variable. | Direct labels for the TetR sponging constructs unless they are measured or mapped carefully. |
| Lee/Kim-style pretroDNA circuits | Retron-derived protein-binding DNAs can regulate TF circuits and feedback behavior. | Payload search, or a guarantee that long or multivalent designs will work. |
| Wang-style Ec86 structure | Eco1 msDNA is structurally integrated with RT, supporting the idea that the hairpin is not an arbitrary free oligo. | Productive or unproductive status for every payload sequence. |
| msd-HOP construction | Degenerate or designed oligos can be converted into retron-compatible constructs. | Functional discovery unless paired with a screen. |
| Evo2 / LatentDNA axes | Model geometry may provide a weak abundance prior or audit surface. | Sponging, TF binding, or biological mechanism. |
| Second TF reporter, if obtained | The logic extends beyond TetR in at least one additional context. | General endogenous TF programmability. |

### Reader Outcomes

After reading the paper, the reader should be able to do five things more
clearly than before:

1. Interpret a failed retron decoy experiment more carefully. Failure may mean
   poor payload binding, poor RT-DNA production, toxicity, expression burden, or
   assay mismatch.
2. Choose TetR/tetO as a model for the right reason. It is not the destination;
   it is a controlled system for isolating scaffold constraints.
3. Treat payload search and scaffold design as coupled. The binding site cannot
   be optimized as if it were a free oligo.
4. Use construction methods in the right place. msd-HOP belongs after the need
   for empirical search is established.
5. Use Evo2 cautiously. Representation geometry may prioritize candidates, but
   only functional assays define sponging.

### Transition Sentences

Use reader-problem transitions instead of tool-list transitions.

Prior art to problem:

> Prior work shows that retron-derived DNAs can bind TFs and regulate synthetic
> circuits. The unresolved question is not whether a known operator can ever be
> embedded in retron DNA; it is how to design retron decoys when the desired
> payload is uncertain and the scaffold itself controls production.

TetR to scaffold experiments:

> We first removed payload uncertainty by using TetR/tetO, where the binding
> interaction is well defined. This allowed us to ask a narrower question: when
> the payload should work, which scaffold changes still break retron-decoy
> function?

Scaffold constraints to msd-HOP:

> Once scaffold sensitivity is visible, broader decoy discovery becomes a
> construction problem as well as a biological problem. Variant search is only
> useful if the variants can be built in a retron-compatible form.

msd-HOP to Evo2/LatentDNA:

> Even buildable candidates are not equally plausible. Because sponging depends
> on sufficient RT-DNA, we asked whether existing abundance measurements and
> genome-model representations could provide a pre-screening prior before
> functional testing.

Conclusion:

> Together, these results do not eliminate empirical screening. They clarify
> what the screen must respect: every payload is also a retron-production
> perturbation.

### Draft Prose

#### Opening Paragraph

DNA decoys can perturb transcriptional regulatory networks by titrating
DNA-binding proteins away from their genomic targets. Retrons offer an
attractive way to present such decoys inside cells because they genetically
encode short reverse-transcribed DNA products from structured noncoding RNAs.
Recent work has shown that retron-derived DNA can carry protein-binding
sequences and regulate synthetic TF circuits, suggesting a route to
intracellular TF sponging. However, retron decoy design is not simply a matter
of inserting a binding motif. The sequence or combination needed to perturb a
target regulatory state may be uncertain, and the same sequence must be embedded
in a retron hairpin whose structure controls RT-DNA biogenesis. This work uses
Eco1 TetR sponging as a tractable model to examine that coupled design problem,
asking how decoy payloads, hairpin processability, construction constraints, and
abundance priors can be handled together without assuming that every designed
hairpin will be produced.

#### Abstract-Style Paragraph

Transcription-factor decoys provide a direct way to perturb regulatory networks,
but their intracellular presentation and effective sequence design remain
challenging. Engineered retrons offer a genetically encoded route to produce
decoy DNA in cells, and recent demonstrations show that operator-bearing
retron-DNAs can sponge well-characterized TFs. The design problem becomes harder
when the desired perturbation is phenotype-driven or multivalent, because the
effective decoy architecture may not be known in advance. At the same time, the
decoy sequence is embedded in a structured retron hairpin, so changes that
improve binding or valency can disrupt RT-DNA production. Here we use
Eco1-based TetR sponging as a controlled model to study this coupled constraint.
We test how hairpin architecture affects decoy function, develop a construction
route for retron-compatible variant libraries, and evaluate whether
abundance-associated sequence representations can prioritize candidates before
functional sponging screens. Together, these results define a conservative
design logic for retron TF decoys: empirical payload search is useful, but only
within scaffolds that preserve retron biogenesis.

### Community-Specific Value

| Community | Value statement |
| --- | --- |
| Synthetic biologists | This work clarifies what has to be true for retron decoys to become reliable regulatory perturbation tools: they must be screenable, but they must also remain compatible with RT-DNA production. |
| Retron biologists / engineers | This work treats TF decoys as a stress test for retron programmability, asking how far the Eco1 msd hairpin can be altered before function is lost. |
| Gene-regulatory-network readers | This work offers a way to perturb TF activity without editing genomic binding sites, while acknowledging that the relevant target and architecture may need empirical discovery. |
| Genome-model readers | This work uses model representations as abundance priors, not as replacements for functional assays. |
| Therapeutic readers | This work is not a therapeutic claim; it addresses a design bottleneck for intracellular decoy DNA that could matter for future therapeutic or cell-engineering applications. |

### Claim Discipline

Use:

- coupled payload-scaffold design problem
- retron-compatible decoy architecture
- RT-DNA-compatible payload search
- abundance-gated sponging
- functional decoy readout
- construction-compatible library design
- abundance prior
- pre-screening for RT-DNA plausibility
- phenotype-driven decoy search
- multivalent decoy hypothesis

Avoid or heavily qualify:

- platform
- universal retron decoy design
- Evo2 predicts sponging
- compiler discovers design rules
- source-audited workflow
- TF decoy discovery engine
- solves payload uncertainty
- generalizable to endogenous TFs, unless shown
- high-throughput screen, unless actually performed
- abundance label, when the intended meaning is abundance prior

### Safe Claims By Data Outcome

| Data outcome | Safe claim |
| --- | --- |
| TetR works, extended hairpins fail, msd-HOP works, Evo2 unclear | Eco1 TF decoy design is constrained by hairpin processability, and degenerate construction methods enable future empirical search. |
| TetR works, second TF works | The design logic extends beyond TetR to at least one additional TF context, supporting broader retron-decoy testing. |
| Evo2 abundance axes correlate with Crawford/Khan and TetR function | Representation-derived abundance priors can help rank retron-decoy candidates before functional testing. |
| Evo2 axes correlate with abundance but not TetR sponging | Abundance-like geometry is useful for scaffold triage but insufficient for predicting decoy function. |
| Pooled screen works | Retron-compatible degenerate libraries can empirically recover functional decoy architectures. |
| Pooled screen does not work | The construction route is feasible, but functional pooled screening remains bottlenecked by assay/readout constraints. |

### Final Synthesis

Retron decoys are promising not because any binding site can be dropped into
Eco1, but because retrons create a powerful intracellular decoy modality whose
useful design space can be explored only when payload function and RT-DNA
biogenesis are considered together.

The shorter final takeaway:

> The central design lesson is that retron TF sponging is abundance-gated: a
> decoy must bind the regulator, but first the retron must make the decoy.
