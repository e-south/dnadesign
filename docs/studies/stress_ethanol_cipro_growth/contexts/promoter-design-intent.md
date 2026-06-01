---
doc_id: study-stress-ethanol-cipro-growth-promoter-design-intent
surface: study-context
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-05-28
entrypoint: ../routes/README.md
---

## Promoter Design Intent And Semantics

This page orients the biological motivation and manuscript semantics for
`stress_ethanol_cipro_growth`. It is not a status record, OPAL runbook,
LatentDNA report, or assay result. Use `../record/status.md` for current state
and `../routes/README.md` for owner-surface routing.

### Center Of Gravity

The study should not be framed primarily as an ethanol/ciprofloxacin promoter
design study. Ethanol and ciprofloxacin are the case study. The broader problem
is:

> We still cannot reliably design promoters to produce a desired expression
> program in a chosen cellular context.

Everything else in this study - DenseGen arrays, TFBS curation, Evo2/LatentDNA
representations, SFXI, OPAL, ethanol stress, SOS-like ciprofloxacin response,
and the pDual10 context - serves that problem.

The strongest one-sentence premise is:

> Because promoter activity depends on incompletely understood regulatory
> grammar, cellular context, and user-defined expression goals, this work
> develops a specification-driven design loop that generates controlled
> promoter candidates, represents them with learned sequence features, measures
> multi-condition behavior, and iteratively selects sequences that move toward
> a desired expression program.

Shorter talk version:

> Promoter design is becoming less limited by our ability to generate sequences
> and more limited by our ability to specify, choose, measure, and learn which
> sequences produce the desired expression program in context.

Manuscript-like version:

> This study treats promoter design as a specification-driven search problem:
> given a desired expression profile across cellular states, we generate an
> interpretable regulatory sequence space and use learned representations,
> multi-condition measurements, and active learning to navigate it
> experimentally.

The shift is from:

> We design stress-responsive promoters.

to:

> We demonstrate a general strategy for designing promoters toward
> user-specified expression behavior, using two-input stress response as a
> tractable test case.

### What The Study Is Trying To Achieve

The study is trying to make promoter design operational under context
dependence.

The grand challenge is not merely that promoters are useful. The challenge is
that promoter function is a many-body problem: DNA sequence, TF availability,
TF activity state, binding-site affinity, spacing, orientation, promoter
architecture, plasmid or genomic context, cell physiology, and the desired
output all interact.

The practical design question remains unstable:

> Given a gene, a host context, and a desired expression program across
> conditions, which promoter sequence should I build next?

The contribution is not "another promoter library." It is:

> A route from a desired regulatory behavior to a structured candidate
> universe, an assayable objective, and an iterative learning strategy.

### Prior-Art Positioning

The manuscript version should make clear which pieces of the field are already
strong and which design gap remains. This page is a semantic guide, not a
citation ledger, so source-backed manuscript drafts should attach exact
references for each lane.

| Prior-art lane | What it establishes | Remaining design gap |
| --- | --- | --- |
| Combinatorial promoter logic | Promoter architecture, including binding-site identity and arrangement, can encode different regulatory functions. | Architecture-to-function rules are still not general enough to compile arbitrary multi-state specifications. |
| Environment-specific promoter behavior | The same regulatory architecture can behave differently across environmental or cellular states. | A useful design must be evaluated in the intended context, not only by motif presence or single-condition strength. |
| MPRA-scale sequence-function maps | Very large regulatory sequence spaces can be measured and modeled. | Scale does not by itself tell a user which sequence to build next for a chosen response program. |
| Machine-learning regulatory DNA design | Learned models can generate or rank regulatory DNA in some settings. | Model scores are priors unless tied to task-specific measurements and a declared objective. |
| Active-learning regulatory design | Iterative selection can reduce experimental burden when labels are costly. | Active learning needs a candidate universe, feature space, assay label, and objective that match the biological specification. |

The reader-value claim is therefore not that this study invents promoter
logic, high-throughput assay, learned sequence features, or active learning.
The claim is that these elements are organized around a user-specified
multi-state expression goal.

Source-anchor notes for manuscript drafting, to be verified against the final
reference set:

- Cox-style combinatorial promoter work supports the claim that promoter
  architecture can encode diverse expression functions and regulatory logic.
- Gertz and Cohen-style environment-specific promoter work supports the claim
  that combinatorial behavior can change across conditions.
- Large random-promoter, MPRA, and CRE-design work, including de Boer/Gosai-
  and Yu/Chen-style studies, supports the claim that scale and ML have expanded
  sequence-to-expression modeling without eliminating the specification gap.
- Dense-array SPP work supports the computational premise that compact TFBS
  arrangements can be generated efficiently under promoter constraints.
- Genomic foundation model work, including Evo2-style representations, supports
  the use of learned nucleotide features as candidate priors, not as functional
  labels.

### High-Throughput Mapping Versus Active Search

The tension between high-throughput mapping and active learning is real, but
it is not a contradiction. The clean way to square the two narratives is:

> High-throughput mapping is a data-generation regime. Active learning is a
> decision policy.

They answer different questions. High-throughput mapping asks:

> How do we generate enough labeled data to learn a useful design-to-function
> map?

The stress-study active-learning framing asks:

> Given a large candidate universe and a desired behavior, which designs should
> we actually spend experiments on next?

The deeper comparison is therefore not high-throughput versus active learning.
It is:

> Exhaustive or broad design-space mapping versus specification-driven
> experimental search.

Those can be combined. A strong practical strategy is often hybrid: use
high-throughput data to establish a base map or transferable prior, then use
active learning to navigate toward a specific design goal under real
experimental constraints.

#### Rai / CLASSIC Positioning

Use the Rai-style ML-driven DBTL framing as a bounded prior-art claim, not as a
rebuttal to active learning. The Figure 1a-style narrative is strongest when:

1. The design space is poorly understood and broad libraries can reveal
   nonlinear effects that a designer would not sample manually.
2. The measurement cost per additional variant is low once pooled cloning,
   barcoding, sequencing, or sorting pipelines are running.
3. The goal includes learning design rules or composability, not only finding
   one working construct.

This supports the field-level premise that genetic parts interact
nonlinearly with one another and with host-cell machinery, making naive
modularity and low-dimensional hand models unreliable. Machine learning is
valuable because it can learn patterns from labeled design spaces that were not
specified a priori.

CLASSIC is a strong example of the wide-map strategy. Its value is not merely
"more constructs." It solves a practical bottleneck: assaying long, multipart
genetic constructs at high scale using pooled assembly, long-read indexing,
short-read phenotyping, and barcode mapping. It converts a combinatorial design
space into a predictive and analyzable object.

Source-anchor details for manuscript drafting, to verify against the final
reference set before citation:

- CLASSIC reports expression profiling for more than `10^5` gene-circuit
  designs, with 5-20 kb designs measured in one experiment in human cells.
- In the single-input circuit example, the designed space included `165,888`
  compositions; `121,292` compositions, about `73%`, were mapped.
- CLASSIC-derived fold-change values agreed with clonal isolates with reported
  `MAE = 0.15`; predicted high-fold-change designs validated with reported
  `MAE = 0.18`.
- MLP and random-forest models outperformed biophysical and linear models, and
  the MLP was used to impute basal and induced expression for the full
  `165,888`-composition space.
- The reported data-efficiency result - roughly `9%` of the data reaching
  about `95%` of maximum model performance, plus smaller fine-tuning libraries
  around `3,500` members with about `1,000` instances - is important because it
  weakens a naive "more data is always the answer" reading.

That last point matters for this study. Large-scale mapping can reveal that a
strategically chosen subset may be enough once the model, design grammar, and
objective are good enough.

#### Map-Making Versus Target-Seeking

Use this comparison when explaining why the stress study is not opposed to
CLASSIC-like scale.

| Dimension | High-throughput mapping / CLASSIC-like strategy | Active-learning / specification-driven strategy |
| --- | --- | --- |
| Primary goal | Learn a broad design-to-function map. | Reach a desired behavior efficiently. |
| Unit of success | Coverage, predictive accuracy, and design rules. | Hit rate, regret reduction, target satisfaction, and useful failure modes. |
| Best when | Priors are weak, interactions are unknown, and rule discovery matters. | Assays are costly, the objective is specific, and prior structure exists. |
| Data philosophy | Measure broadly, then model. | Model uncertainty, then measure selectively. |
| Main strength | Finds unexpected global structure and nonlinear composability rules. | Avoids spending experiments on regions irrelevant to the specification. |
| Main weakness | Can make maps broader than the practical objective. | Can get trapped by bad priors, poor seed data, or over-exploitation. |

When a review critiques low-throughput DBTL for incomplete design-space
mapping, that critique is aimed at uninformed low-throughput iteration. It does
not refute informed, model-guided, actively selected iteration.

#### Active Learning Is Adaptive Data

Do not frame active learning as simply a "small data" alternative. That
undersells it and creates an unnecessary contrast with high-throughput work.

Active learning is adaptive data acquisition. Sometimes the adaptive batch
should be 24 designs. Sometimes it should be 384 designs. Sometimes it should
be 10,000 designs. The active part is not the batch size; it is that the batch
is chosen to maximize expected learning or expected progress toward a
specification.

Use these synthesis sentences:

> High-throughput methods increase the number of designs we can test per
> cycle; active learning decides which designs deserve those tests.

> CLASSIC expands the experimental bandwidth of DBTL; active learning improves
> the allocation of that bandwidth.

For this study, the central claim is:

> The field's bottleneck is shifting from "Can we generate data?" to "Can we
> spend experimental measurements on the designs that matter for a
> user-specified behavior?"

This aligns with the study's center of gravity: promoter design is becoming
less limited by the ability to generate sequences and more limited by the
ability to specify, choose, measure, and learn which sequences produce the
desired expression program in context.

#### What Is More Powerful

The most powerful general frame is neither high-throughput nor active learning
alone. It is:

> Closed-loop, specification-conditioned design under uncertainty.

That requires four pieces:

1. A design space expressive enough to contain the target.
2. A measurement system that returns relevant labels.
3. A model that can generalize from measured to unmeasured designs.
4. A selection policy that chooses the next experiments based on the goal.

CLASSIC is especially strong on pieces 1-3 for composition-to-function
mapping. This stress-study active-learning framing is strongest on piece 4:
given a large promoter candidate universe and a multi-state expression
specification, decide what to build next.

The underlying core is:

> Learn a model of design-to-behavior uncertainty, then use experiments to
> reduce the uncertainty that matters for the desired function.

For this paper, the sharper version is:

> Promoter design does not require mapping all promoter space. It requires a
> disciplined way to move from a desired expression program to the next most
> informative sequences to build.

#### Where Each Strategy Is Stronger

CLASSIC-like high-throughput mapping is stronger when the goal is global
understanding: learning composability rules across many part categories,
identifying multiple solution families, discovering unexpected interactions,
and building reusable training datasets. It is also stronger when the marginal
cost of measuring more variants is low after the pooled pipeline is running, or
when rare design regimes would be missed by a narrow seed model.

The stress-study active-learning framing is stronger when the goal is a
specified promoter behavior, especially across multiple conditions. A promoter
can be bright in the wrong state, responsive but too weak, specific but
unusable, or correct in shape but too low in dynamic range. Multi-state
promoter behavior therefore needs a setpoint-conditioned objective. SFXI should
be treated as a selection objective, not a mechanism score.

#### Foot Guns To Avoid

High-throughput foot guns:

- Confusing coverage with understanding. A large library covers only the design
  space that was chosen.
- Confusing more labels with better labels. Noisy high-throughput labels may
  still need smaller high-quality validation sets.
- Optimizing the wrong phenotype. A fluorescence label under limited
  conditions does not automatically imply robustness across media, growth
  phase, burden, plasmid copy, integration site, strain background, or long-term
  stability.
- Over-reading post hoc interpretability. SHAP values, mutual information, and
  clusters can be confounded by how the library was built.
- Treating the number of designs tested as the value metric. The better metric
  is uncertainty reduction about designs that satisfy the objective.

Active-learning foot guns:

- Pretending that smart batches replace the need for a meaningful candidate
  universe, useful representation, aligned score, and informative seed data.
- Getting trapped by bad priors, poor seed labels, over-exploitation, or a
  poorly managed exploration-exploitation tradeoff.
- Becoming too target-myopic. A narrow campaign may find hits without teaching
  broadly reusable promoter grammar.
- Hiding negative evidence. A convincing study still needs controls, diversity
  samples, and interpretable failure modes.
- Collapsing biology into a scalar. SFXI and related objectives must keep
  shape, amplitude, noise, burden, and condition-specific behavior visible.
- Treating embeddings as function. Evo2/LatentDNA features can structure the
  search before labels close the loop; they are not promoter-function evidence.

#### Positioning Against CLASSIC

Avoid:

> CLASSIC is brute force; our approach is smarter.

That is wrong and easy to attack. CLASSIC is not brute force; it uses pooled
assembly, indexing, model imputation, validation, and fine-tuning.

Use:

> CLASSIC shows how high-throughput experiments can convert large genetic
> design spaces into predictive maps. This study addresses the complementary
> problem: when the design space is too large to map exhaustively and the goal
> is a user-specified expression program, how should the next promoter
> sequences be chosen?

Or:

> CLASSIC expands DBTL throughput; active learning makes DBTL selective.

Strongest version:

> The central question is not whether more data helps. It does. The question is
> which data are worth buying next for the design specification at hand.

### Reader-Value Frame

The primary reader community is people who need regulatory DNA to control gene
expression in living cells.

| Reader-value element | Study stance |
| --- | --- |
| Accepted view | Promoters are central regulatory elements; promoter architecture matters; high-throughput assays and ML can map parts of sequence-function space. |
| Instability | Those advances still do not let a user reliably say, "I want this expression profile across these cell states," and compile that specification into promoter DNA. |
| Cost | Researchers reuse imperfect natural promoters, build ad hoc libraries, screen too broadly, optimize single-condition strength, or trust model scores that may not transfer to the desired context. |
| Intervention | This study defines promoter design as a specification-driven search loop: curate regulatory priors, generate controlled candidate architectures, represent sequences, measure multi-state behavior, score against a setpoint, and actively select next designs. |
| Payoff | Readers gain a way to make promoter-design decisions under uncertainty, rather than only cataloging variants or treating a model as an oracle. |

### Situation, Complication, Resolution

#### Situation

Cells constantly change state, and transcriptional regulatory networks convert
signals into gene expression programs. Promoters are compact DNA elements where
those signals are integrated. If promoters could be designed reliably, a user
could decide when, where, and under what cellular state a gene should be
expressed.

The field can map TF binding, build combinatorial promoters, assay large
libraries, train sequence-to-expression models, and use ML to propose
regulatory DNA. Those capabilities are real, but they do not make design
automatic.

#### Complication

The ability to generate or assay many sequences does not solve the design
problem by itself.

Users often want a response program, not simply high expression: low in one
state, high in another, moderate in a third, higher under a combined
perturbation, or tuned to a downstream phenotype. A promoter that is bright in
the wrong state is not a good design. A promoter with the right response shape
but too little usable expression may also fail.

Existing assays and models often compress promoter behavior into narrower
labels such as strength, fold-change, cell-type specificity, accessibility, or
one inducible response. Those labels are useful, but they do not fully answer
arbitrary multi-state expression design.

Regulatory maps are partly descriptive. Knowing that a TF can bind a site, or
that a promoter responds in a genome-scale perturbation assay, does not
automatically tell us how to arrange multiple sites into a new promoter that
will produce a desired expression profile in this host, assay, and construct
context.

#### Resolution

Treat promoter design as specification-driven experimental search, not
one-shot prediction:

```text
desired expression program
  -> relevant regulatory priors
  -> controlled promoter candidate universe
  -> learned sequence representation
  -> multi-condition assay
  -> setpoint-conditioned objective
  -> active-learning selection
  -> refined promoter designs
```

Dense-array design matters because it creates a structured, auditable candidate
universe rather than an opaque random library. Evo2/LatentDNA representations
matter because they provide candidate feature spaces, but they are priors or
features, not proof of function. SFXI matters because multi-condition promoter
design needs a setpoint-conditioned objective. OPAL matters because the space is
too large and the labels too costly for exhaustive measurement.

The study-specific design loop has this responsibility split:

| Loop element | Study role | Boundary |
| --- | --- | --- |
| Regulatory priors | Choose TFBS families and promoter constraints that plausibly connect to the ethanol/ciprofloxacin testbed. | TFBS presence is not a mechanism claim by itself. |
| DenseGen candidate universe | Generate controlled promoter architectures with traceable part and design-family metadata. | Dense-array generation does not guarantee regulatory logic. |
| Evo2/LatentDNA features | Provide a fixed-length candidate representation for pre-assay selection. | Embeddings are not phenotype labels or proof of condition-dependent activity. |
| SFXI objective | Score measured or predicted four-state response vectors against a setpoint while preserving logic/effect diagnostics. | SFXI is a selection objective, not a biological mechanism score. |
| OPAL campaign loop | Train on observed labels, rank/select next candidates, and preserve campaign-local ledgers. | OPAL owns the generic active-learning loop, not DenseGen-specific biological interpretation. |
| Assay labels | Close the loop with measured multi-condition promoter responses. | Pre-assay analyses prepare the search; they do not establish promoter function. |

### Case-Study Semantics

The E. coli ethanol/ciprofloxacin system is the minimal demonstration problem,
not the conceptual boundary.

> We instantiate the broader promoter-design problem in a two-input, four-state
> bacterial system because it is simple enough to measure, but rich enough to
> expose the central challenge: a promoter must integrate regulatory state and
> produce a specified expression profile.

Use ethanol stress, ciprofloxacin/SOS-like response, LexA, CpxR, BaeR,
sigma-70 promoter architecture, and pDual10 as the testbed. Do not let those
details become the thesis.

General formulation:

```text
cell state 00: baseline
cell state 10: condition A
cell state 01: condition B
cell state 11: combined condition

desired promoter behavior = [y00, y10, y01, y11]
```

Study instance:

```text
00 = no stress
10 = ethanol-associated state
01 = ciprofloxacin-associated state
11 = combined ethanol/ciprofloxacin state
```

Current OPAL campaign setpoints follow the same state order:

| Campaign intent | Setpoint vector |
| --- | --- |
| Ethanol factor | `[0, 1, 0, 1]` |
| Ciprofloxacin factor | `[0, 0, 1, 1]` |
| Combined-state AND | `[0, 0, 0, 1]` |

The stress system is therefore a concrete example of a general multi-state
promoter specification problem.

### Narrative Spine

Use these three questions to keep study prose from becoming a tool tour:

1. What should the promoter do?
2. What sequence space might contain promoters that do it?
3. How do we choose what to build when the space is too large to assay
   exhaustively?

Those questions map to the study components:

| Question | Study component | Reader-value wording |
| --- | --- | --- |
| What should the promoter do? | Four-state setpoints and SFXI | A useful promoter objective is a desired expression profile across states, not a single scalar brightness value. |
| What sequence space might contain promoters that do it? | TFBS curation and DenseGen dense arrays | Regulatory priors define a controlled candidate universe whose motif composition, spacing, and promoter constraints remain auditable. |
| How do we choose what to build? | Evo2/LatentDNA features and OPAL | Learned sequence representations and active learning help prioritize candidates before exhaustive measurement is possible. |

### Section-Level Premises

These premises should guide talks, manuscripts, and study-facing summaries:

| Section | Complete-sentence premise | Minimal evidence or visual |
| --- | --- | --- |
| Opening problem | Promoter design is a specification problem because useful gene expression depends on what a gene should do across cellular states. | Sequence, regulatory context, and desired expression program. |
| Biological motivation | Promoters are sequence-level interfaces through which transcriptional regulatory networks convert cellular signals into gene expression. | Signal -> TF activity -> promoter architecture -> expression. |
| Field gap | Even though we can map, mutate, assay, and model many regulatory sequences, we still cannot reliably compile arbitrary expression specifications into promoter DNA. | Capability-vs-gap figure. |
| Prior-art synthesis | Prior work shows that promoter architecture can encode logic and context dependence, but existing approaches often optimize narrower labels such as strength, fold-change, or cell-type specificity. | Literature map with axes: library scale and behavioral richness. |
| Design specification | A useful promoter objective is a desired expression profile across states, not a single scalar brightness value. | Four-state response grid `[00, 10, 01, 11]`. |
| Candidate-space construction | Regulatory priors can be converted into an interpretable candidate universe whose parts and constraints are explicit. | TF evidence -> motif pools -> promoter grammar -> candidate population. |
| Dense-array generation | Dense-array design systematically explores compact regulatory syntax while preserving architecture constraints. | Example promoter architectures and design-family distributions. |
| Representation choice | Learned sequence representations are useful only if they preserve design-relevant structure before guiding experiments. | Feature health, family separation, sigma-core gradients, and context robustness. |
| Multi-condition objective | Multi-state promoter design requires an objective that rewards both the right response shape and enough usable expression. | SFXI decomposition: response vector, target vector, logic fidelity, effect scaling. |
| Active learning | Once labels exist, promoter design becomes iterative search rather than exhaustive screening. | Candidate universe -> assay -> response vector -> SFXI -> model -> next batch. |
| Experimental results | Measured promoter responses determine whether the designed grammar yields the requested expression programs. | Hit rate versus baseline, response-profile clusters, example promoters, and failure modes. |
| Interpretation | The biological value comes from learning which regulatory architectures produce which expression programs, not merely from identifying top-ranked sequences. | Feature enrichment among successful and failed promoters; architecture-to-response examples. |
| Conclusion | The contribution is a general design discipline for moving from regulatory priors to specified promoter behavior under context dependence. | Return to the opening specification loop. |

For the current pre-assay posture, soften experimental-result language to:

> The current study prepares the experimental search by generating,
> representing, auditing, and scoring a candidate universe before measured
> round-0 labels are available.

After measured labels exist, the result section should foreground response
profiles, hit enrichment, architecture-to-function interpretation, and failure
modes.

### Minimal Figure Logic

Use this figure sequence when turning the study into a paper, talk, or browser
artifact:

| Figure | Premise | Visual focus |
| --- | --- | --- |
| 1 | Promoter design requires matching sequence to desired expression behavior in a cellular regulatory context. | Regulatory context, promoter sequence, expression program. |
| 2 | Prior work solves pieces of the problem but leaves specification-driven multi-state design unresolved. | Literature landscape: sequence scale versus behavioral richness. |
| 3 | Regulatory priors define an interpretable candidate universe. | TF evidence -> motif pools -> promoter grammar -> candidate population. |
| 4 | Dense-array design samples regulatory syntax under promoter constraints. | Example promoter architectures and design-family summaries. |
| 5 | Representation learning is a pre-assay decision problem. | Representation audits, not a lone UMAP. |
| 6 | Multi-state promoter behavior needs a setpoint-conditioned objective. | SFXI decomposition across `[00, 10, 01, 11]`. |
| 7 | Active learning makes specification-driven promoter search experimentally tractable. | Round-based selection loop. |
| 8 | Measured rounds reveal both design successes and grammar failures. | Hit enrichment, response clusters, architecture interpretation. |

Figure 1 should not show DenseGen, Evo2, OPAL, or implementation details. It
should establish sequence, regulatory context, and desired expression program.
Figure 2 should place this work in the gap between sequence scale and
behavioral richness: combinatorial promoter logic, environment-specific
synthetic promoters, MPRA-scale sequence-function maps, cell-state regulatory
element design, and active-learning regulatory DNA design should be shown as
pieces of the field that do not fully solve specified multi-state promoter
behavior.

Figures 3 and 4 should say "we know what we varied." Figure 5 should say "we
audited the representation before trusting it." Figure 6 should say "the scalar
score decomposes into visible response-shape and effect terms." Figure 8 should
eventually say "the measured rounds reveal both useful designs and grammar
failures."

### Planned Response-Shape And Metadata Analyses

Flag this as a future OPAL/study-analysis deliverable after measured
multi-condition labels exist. The direct CLASSIC-like analog is not a UMAP of
sequence embeddings and not a plot computed from SFXI alone. It is a
response-space map computed from the underlying four-condition expression
vector:

```text
y_i = [baseline, ethanol, ciprofloxacin, ethanol + ciprofloxacin]
p_i = (y_i + epsilon) / sum(y_i + epsilon)
```

Then compare each promoter response distribution to response archetypes such
as:

```text
qAND   = [0.01, 0.01, 0.01, 0.97]
qOR    = [0.01, 0.33, 0.33, 0.33]
qEtOH  = [0.01, 0.495, 0.01, 0.495]
qCIP   = [0.01, 0.01, 0.495, 0.495]
qXOR   = [0.01, 0.495, 0.495, 0.01]
qCONST = [0.25, 0.25, 0.25, 0.25]
```

The first OPAL-owned plot should be a round-aware response-archetype divergence
map:

```text
x-axis: KL divergence from AND-like combined-stress target
y-axis: KL divergence from OR-like general-stress target
point size: SFXI score or effect-scaled utility
point color: OPAL round
optional encodings: replicate uncertainty, campaign target, or validation set
```

Use Jensen-Shannon divergence as a robustness check when symmetry and bounded
scale are useful. Low divergence is interpretable as closeness to an archetype;
high divergence can mean many different failure modes and should not be
overinterpreted.

SFXI remains an overlay and selection objective:

> KL divergence maps response shape; SFXI ranks design utility.

This distinction matters because KL over normalized expression discards
magnitude. A weak promoter can look shape-compatible but be useless; a bright
promoter can be useful in magnitude but wrong in logic. Keep shape, amplitude,
noise, burden, and ON/OFF thresholds visible instead of collapsing all biology
into one scalar.

After a campaign has measured labels, add mutual-information and enrichment
analyses over DenseGen metadata. Candidate features can include TFBS identity,
TFBS family or regulon, motif count, motif density, motif order, motif spacing,
motif orientation, distance to -35 / -10 / TSS, core promoter variant,
sigma-factor motif strength, GC content, predicted DNA shape, and
Evo2/LatentDNA cluster or margin features.

The intended questions are:

- Which DenseGen architecture features are enriched among AND-like,
  OR-like, ethanol-like, ciprofloxacin-like, or constitutive-like promoters?
- Which features have high mutual information with behavior class?
- Which feature pairs show conditional coupling within a behavior class?
- Do successful designs require co-localized ethanol-associated and SOS-like
  motifs, separated independent modules, or another architecture?
- Do the study-owned DenseGen probes recover the same feature-behavior
  relationships under synthetic or semi-synthetic labels?

Guardrails:

- Define behavior classes in measured response space using KL, Jensen-Shannon,
  SFXI decomposition, or explicit response-vector thresholds.
- Use UMAP only as a visualization layer for architecture clusters after
  behavior classes are defined.
- Account for OPAL sampling bias. Actively selected records are the campaign's
  acquisition history, not an unbiased sample of the design universe; keep
  random, diversity, or probe controls where possible.
- Treat mutual information, SHAP, enrichment, and clusters as hypotheses about
  architecture-response relationships until validated experimentally.
- Keep this analysis study-owned unless a generic OPAL plot primitive emerges;
  OPAL owns the round-aware campaign plot and ledger integration.

### Foreground And Demote

Foreground these concepts:

```text
specified expression program
regulatory context
promoter grammar
controlled candidate universe
multi-state response
setpoint-conditioned selection
data-efficient search
measured feedback
architecture-to-function interpretation
```

Demote these to methods unless a specific route or runbook requires them:

```text
DenseGen
Evo2
LatentDNA
OPAL
pDual10
LexA/CpxR/BaeR
157,160 candidates
8192-dimensional X
UMAP
random forest
```

Preferred wording:

| Tool-first wording | Reader-value wording |
| --- | --- |
| We used DenseGen to create 157,160 promoters. | We generated a controlled candidate universe large enough to search but structured enough to interpret. |
| We embedded promoters with Evo2. | We used a learned genomic representation as a candidate feature space, then audited whether it preserved design-relevant structure. |
| We used OPAL for active learning. | We used measured promoter responses to decide which candidates should be built next. |
| We introduce SFXI. | We needed a selection objective that rewards the right response shape and sufficient output across multiple states. |

### Claim Boundaries

Use these claims:

- Promoter design should be framed as matching a desired expression program to
  a regulatory context.
- High-throughput generation and measurement create a selection problem, not
  just a data opportunity.
- Regulatory priors make the candidate universe interpretable, but measured
  expression is still required to establish function.
- Learned sequence representations are useful as design priors only after they
  are audited against known structure.
- A multi-condition promoter assay needs a setpoint-conditioned objective
  because brightness alone is not the design goal.
- The stress-response case study demonstrates a tractable version of a broader
  promoter-design problem.

Avoid these claims:

- We solved promoter design.
- We decoded the cis-regulatory code.
- Evo2 predicts condition-dependent promoter activity.
- Dense arrays guarantee regulatory logic.
- TFBS presence implies functional regulation.
- SFXI is a biological mechanism score.
- The E. coli stress system is the entire contribution.

Safest boundary sentence:

> This work does not claim that sequence priors or embeddings are sufficient to
> predict promoter function; it claims that they can structure the search before
> measured labels close the loop.

### Current And Future Claim State

Keep the claim state tied to the available evidence.

| Evidence state | Safe claim |
| --- | --- |
| Pre-assay candidate table materialized, X selected, no measured round-0 labels | The study has prepared a controlled candidate universe, selected a representation, and defined setpoint-conditioned objectives for the first measured search round. |
| Round-0 assay labels ingested | The study can evaluate whether measured four-state promoter responses match the declared SFXI setpoints better than baseline or diversity-only choices. |
| Multiple active-learning rounds completed | The study can ask whether iterative selection improves hit recovery, exposes architecture-response relationships, and identifies failure modes. |
| Strong hits validated across repeats or secondary contexts | The study can claim specific useful promoter designs within the tested host, construct, assay, and stress conditions. |

Do not let pre-assay representation audits sound like phenotype results. Do not
let a successful OPAL round sound like a general solution to promoter design.
Do not let a strong stress-study hit become evidence that the same grammar will
transfer to unrelated organisms, constructs, or cellular states.

### Abstract-Style Narrative

Designing promoters remains difficult because the desired output is rarely
simple expression strength. In practice, a promoter must produce an expression
program in a cellular context: low, high, graded, or condition-specific
depending on the regulatory state of the cell. Although high-throughput
reporter assays, regulatory maps, and machine-learning models have expanded
our ability to characterize and generate regulatory DNA, they do not yet
provide a general way to compile arbitrary multi-state expression
specifications into promoter sequence. Here we frame promoter design as a
specification-driven search problem. We curate regulatory priors, generate a
controlled promoter candidate universe with auditable architecture, represent
sequences using learned genomic features, and define multi-condition response
objectives that can drive active learning once measured labels are available.
We instantiate this framework in a two-input bacterial stress-response setting,
using promoter designs intended to distinguish baseline, ethanol-associated,
ciprofloxacin-associated, and combined-stress states. This case study
demonstrates how regulatory grammar, learned sequence representation,
multi-condition measurement, and active learning can be integrated into a
practical design loop for promoters with specified expression behavior.

### Title Directions

Broad titles that preserve the study's intended center:

- Specification-driven promoter design across regulatory contexts
- Designing promoters as multi-state expression programs
- From regulatory grammar to expression programs: a design loop for
  context-dependent promoters
- Learning to choose promoters for specified gene expression programs
- Promoter design beyond strength: searching regulatory DNA for specified
  expression behavior

The cleanest title direction is:

> Designing promoters as multi-state expression programs

### Final Takeaway

The central problem in promoter design is no longer only whether we can make or
measure many sequences; it is whether we can specify the expression behavior we
want, construct an interpretable search space, and learn efficiently from
measured context-dependent outcomes.

The stress-response promoter study is one tractable demonstration of that
broader design principle.
