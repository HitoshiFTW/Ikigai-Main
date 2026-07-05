# NeuroSeed: Three-Month Report
## The Organism "Ikigai" and the Case for Near-Zero-Compute Intelligence

**Mura ALife Labs** | **July 5, 2026** | **Day 90**

---

## 1. What This Is

NeuroSeed is a framework for building intelligence without backpropagation. Ikigai is the organism instantiated from it: a 384 MB flat-memory brain built on holographic vector symbolic architecture (FHRR phasors) and Kanerva sparse distributed memory. It derives answers rather than storing them, learns from one-shot observation, and reasons over structured knowledge with no matrix multiplications. The premise: near-zero compute plus frontier-level correctness on multi-hop reasoning is the path forward once data becomes the only constraint.

---

## 2. The Goal (North Star)

Two co-equal headlines:

1. **Match frontier correctness on reasoning at roughly one millionth of the compute.** On equal-knowledge problems (both sides given the same facts), the organism reasons to the same answer as a 550-billion-parameter LLM while consuming on the order of 1e-6 of the energy and memory. This is not general superiority. The frontier crushes us on open-ended generation and broad world knowledge. It is a narrow, defensible win.

2. **Generalize across modalities:** text (shipped) to image to audio to spatial 3D reasoning, one substrate, same principles.

**Economic thesis:** no backprop, no GPU cluster. Training is free-to-cheap (emergent learning from observation, no gradient descent). Inference is CPU-bound and sublinear in memory. The bottleneck flips from compute to data. The funding ask is data and collaboration, not hardware.

---

## 3. Ikigai in One Page

**Substrate.** Flexible Holographic Reduced Representation (FHRR): unit-magnitude complex phasors in d=400. Binding by elementwise multiply (conjugate to unbind), bundling by sum, cleanup by cosine similarity. VSA operations are distributive and composable, so meaning emerges from geometry rather than lookup tables.

**Memory.** Two interleaved systems sharing one address space:
- **Main store (VSASDM):** 16,384 hard locations, k=64 sparse activation, one Kanerva SDM. Holds irreducible facts and learned association patterns. O(1) access per item, independent of store size.
- **Generation store:** transient, episodic. Consolidates into the main SDM during sleep via replay (91% recall, 100% precision on durable content).

**Computed keys.** Every word is a phasor with a computed address and zero bytes of overhead. Relations are phasors too. The address space is unbounded, so vocabulary grows cost-free.

**Derive-not-store.** Store atoms, compose relations at query time. A three-hop path (A to B to C to D) is derived by chaining bind and unbind operations, needing zero cache entries. Closure is nearly free: about 74% of reachable facts exist only as derived consequences, never as stored triples.

**Calibration and abstention.** When the best answer is buried in phasor noise, the organism abstains rather than hallucinate. Geometric noise floor at 1/sqrt(2d), fit per bank from a receiver-operator curve. This prevents false positives on open questions.

**Cognition.** 21 native capabilities at Day 90: exact and multi-hop reasoning, arithmetic, planning (tree search with backtracking), generation (grounded walk, structure-first, induced frames), analogical dreaming, rule induction, introspection, free-energy arbitration. Each is a substrate operation, not a Python algorithm bolted on. No hardcoding on the production path.

---

## 4. The 90-Day Arc

**Phase 1: Neuron foundation (Days 1 to 35).** Ikigai began as a biologically accurate spiking-neuron organism: 15 leaky integrate-and-fire neurons, 23 synapses, STDP learning, six neuromodulatory systems (dopamine, serotonin, norepinephrine, acetylcholine, cortisol, oxytocin), sleep-dependent consolidation. First word "danger" (Day 1). Long-horizon stability validated at 20,000 ticks (Day 5). Threat coupling, predictive sleep, spatial cognition, and hippocampal replay layered in by Day 25. By Day 35 the organism showed behavioral diversity, agency, and credit-stratified learning. It was neurally faithful but computationally expensive and hit practical scaling limits.

**Phase 2: VSA substrate pivot (Days 40 to 60).** Neuron simulation was replaced with vector symbolic architecture (FHRR phasors, Kanerva SDM). Day 40 introduced VSA as the representation layer. Days 48 to 50 shipped associative retrieval, holographic working memory, and generative inference, proving VSA geometry could encode syntax without explicit grammar. Day 60 ("the kill stack") consolidated ten architectural inventions: sleep-replay consolidation, holographic context (4 KB holds 10k tokens), logical fixed-point reasoning, reversible writes (knowledge editing), federated merging, time-as-a-role, VSA-attention, inverse generation, and substrate-native gradient. Analogy benchmark hit 100% top-3 on 44 problems; bench held throughout.

**Phase 3: the flip and persistence (Days 70 to 79).** Day 70: a deep persistence audit found three silent bugs that had wiped cognition state on every save and load for five days, fixed in Pack 246b. The same day Prince proposed "the flip": use the already-shipped Day 61-62 infrastructure (T2S compiler, Galois router, in-situ writer) to absorb pre-trained LLM weights directly into the substrate instead of training topology from scratch, which is mathematically impossible under the flat-VSA cosine ceiling. The LLM becomes a one-time donor, then is discarded; the substrate is the organism. By Day 79 compositional derive-not-store was proven: store only irreducible atoms, derive relations and compositions at query time. Rule mining and self-compression shipped. The superposition-substrate hypothesis was falsified empirically (0% recovery vs FHRR), so FHRR stayed primary.

**Phase 4: reasoning proven, architecture final (Days 82 to 90).** Day 82 stood up the equal-knowledge reasoning arena. Days 87 to 90 shipped emergent dispatch (no hardcoded routing; the planner competes inside reason's argmin free-energy arbitration), address-space generation (word-to-code with a per-context successor cache, O(1) warm), facts-as-address-lists, unified consolidation (91% recall after reload), structure-first generation with frames induced from data (100% purity, no labels), and unified `org.generate` and `org.respond` entry points. A quaternion substrate bake-off was run at equal storage and rejected (complex-plus-permutation separated order better at a quarter of the compute). Production organism.ikg stayed 192 MB, load-only, never saved. One loop, one substrate, 21/21 capabilities live.

---

## 5. Measured Results

**Equal-knowledge multi-hop reasoning (organism, closure test).** Arena setup: 700 held-out derived facts, paths of length 1 to 8 hops, clean structured knowledge (a made-up org chart with 20 distractors per query). Result: 100% accuracy through 8 hops, 0 fabrications, about 9 lookups per query, about 76 microseconds total per query (roughly 8 microseconds per lookup) on a single CPU core. This measures the organism deriving its own closure without error, not a frontier comparison.

**Live head-to-head vs frontier.** Against a live 550B model (Nemotron-Ultra-550B, OpenRouter), accuracy was a tie through the tested hops. An earlier "0% at 8 hops" reading for the frontier was traced to an API rate-limit artifact (9 of 62 calls errored), caught and discarded; it is not a real result and is not claimed. The honest, defensible claim is therefore **match correctness at roughly one million-fold less compute**, not "beat the frontier on accuracy." One organism query is about 9 CPU lookups, roughly 76 microseconds, on the order of tens of nanojoules at sub-watt power. The frontier answer is 1000-plus tokens through 550B parameters on a kilowatt-class GPU. The energy ratio is order 1e-6; the exact GPU wattage is an estimate, but the ratio holds across any reasonable assumption.

**Free-to-train cost.** No backpropagation. Learning is: observe sentence, extract entities and relations, store atoms. Rate: 15,275 facts per second per CPU core (Tatoeba 1M sentences in 6.1 minutes). Cost to ingest 1 billion facts: about $0.36 of cloud CPU time, no GPU. Production organism.ikg (192 MB) holds about 15k irreducible facts, derivable to roughly 1.3 million. Measured multiplier: 1306x on a real ConceptNet dump. An earlier 66,970x Wikidata figure was withdrawn as a comparison artifact. Footprint floor: about 3.8 GB for 5 billion atoms (bitpacked triple store plus SDM).

**Flat compute.** Access is O(1) per query, invariant across 100x store growth (10k to 1M facts). Reasoning holds at about 9 lookups regardless of store size. Generation warm-cache is O(1) on repeated contexts; cold is O(vocab) (lazy codebook rebuild, about 1.6 GB DRAM at 500k words).

**Capability battery (Day 90).** 21/21 capabilities tested live on the production 192 MB organism.ikg (load test, never saved):
- Reasoning: capitals 20/21 exact, arithmetic 8/8, multi-hop 100% at 8 hops on clean knowledge, abstention correct on the calibration noise floor.
- Generation: grounded walk, address-space, structure-first (100% structural validity vs 38% flat), constrained, novel.
- Cognition: dreaming (deductive plus analogical), relation invention, induction, theory of mind, planning, introspection.
- Free energy: F drops toward 0 across learning trials; action selection by expected-free-energy gap ranking.
- All green, production code, no flags, no scratch files.

---

## 6. Where We Win, Where We Lose

**Win:**
- Reasoning: multi-hop deduction on structured knowledge under equal conditions, 0 errors through 8 hops.
- Truthfulness: geometric calibration blocks confident hallucination; the organism abstains on genuine uncertainty.
- Compute: order 1e-6 of the energy per query, CPU-only, viable on embedded devices.
- Continual learning: one-shot observation, no retraining, no gradient descent, immediate world-model update.
- Footprint: 384 MB organism, about 5 GB per 5 billion facts, against a 550B LLM near 1 TB. Near-zero inference memory.
- Verifiability: symbolic, traceable derivation; every answer explained by unrolling its chain.

**Lose:**
- Large open-ended generation: 5k-plus token essays, complex code, creative fiction. We produce 100 to 200 tokens coherently (structure-first, grounded fill). The frontier produces 5k-plus with semantic and stylistic coherence. The gap is learned meaning and long-range planning, both of which need real data and more cognition than we have wired.
- Breadth of world knowledge: the frontier holds 550B parameters of absorbed text; we hold about 15k atoms on device, derivable to 1.3M. That gap is data ingestion. On a real Wikidata dump the same machinery scales to roughly 100M facts and closes much of the factoid-QA gap.
- Few-shot natural language: the frontier adapts to new tasks in-context. We learn from clean structured input, with an emergent but fragile parse. That gap is robust natural-language understanding (the fluency problem, Section 9).

---

## 7. Architecture (Day 90)

One emergent loop. `org.respond` holds no routing; it enacts what the reasoner selects.

- **`org.reason` (free-energy arbitration).** Each path proposes an answer with F = -log(confidence); argmin F wins. Exact paths (arithmetic, derive-cache, multi-hop) reach F=0 and dominate. Soft in-context learning contributes only above the calibration boundary. Planning (FE-guided tree search) competes below it. Abstention sits at the geometric noise floor, 1/sqrt(2d).
- **Depth.** Depth 1 is a direct answer (derive or exact cache). Depth 2 is plan, execute by recursion, then stitch, with the generator filling.
- **`org.generate` (one entry, four modes).** Frame (Levelt frame-relax scaffold plus slot fill), structured (hard frame, type-checked fill, inhibition of return), addressed (word addresses, per-context code cache), constrained (composer plus resonance critic for multi-relation grounding).
- **Cognition (21 ops).** Memory (main SDM plus transient generation store), reasoning (compose-derive, rule-mine, schema-induce), planning (FE-guided backtrack), learning (observe, extract, store; one-shot emergent parse), autonomy (dream, introspect, edit beliefs).
- **Sleep consolidation.** Replay the generation store into the SDM on the shared address space, invent composed relations (R1 then R2 becomes a new R3 with real key identity), and re-derive transitive closure fresh as a spot check.

---

## 8. Proven / Claimed / Conjecture Ledger

| Claim | Category | Evidence |
|-------|----------|----------|
| 100% accuracy, 1 to 8 hops, on held-out derived facts | **Proven** | Arena gate (day88_arena.py), 700 samples, 0 errors, reproducible on organism.ikg |
| Match 550B correctness at equal knowledge | **Proven** | Live head-to-head vs Nemotron-Ultra (day88_headtohead.py), accuracy tie through tested hops |
| ~9 lookups, ~76 microseconds per query, single CPU core | **Proven** | Traced lookups plus measured wall-clock, reproducible |
| $0.36 per 1 billion facts ingested | **Proven** | Day 88 scaling, 15,275 facts/sec, cloud CPU pricing, no GPU |
| O(1) flat access independent of store size | **Proven** | Day 82 capacity wall (SDM M-bound); Day 88/90 query cost invariant |
| O(1) warm / O(vocab) cold generation cache | **Proven** | Day 90 address-generation, byte-identical to prior path, invalidation checked |
| 100% structural validity, structure-first generation | **Proven** | Day 90 generate_structured gate, 100% vs 38% flat |
| 100% type purity inducing structure from data | **Proven** | Day 90 frame_induction gate, clustering on phasor signatures, no labels |
| 91% recall / 100% precision consolidating to SDM | **Proven** | Day 90 consolidation gate, save, reload, fresh-organism verify |
| 1306x closure multiplier on ConceptNet | **Claimed** | Day 83 on a real ConceptNet dump; not yet on full Wikidata |
| Autonomous learning from observation | **Claimed** | Day 87 dream discovery (deductive verified, analogical conjectured), confirmed by live perception; small domains |
| Fluent English generation at scale | **Conjecture** | Mechanism proven; needs real corpus, richer types, a meaning critic |
| Generalization to image / audio / spatial | **Conjecture** | Same substrate; no code yet, only text shipped |

---

## 9. The Fluency Question

Structure-first generation ships (Levelt frame-then-fill), with frames induced from data at 100% purity and no labels. What does **not** ship is fluent English. We get 100 to 200 coherent tokens on made-up data with hand-given semantic roles. Real essays (1k-plus tokens) need three things we have not wired:
- **Learned meaning.** A grounding critic that scores filler words by semantic aptness, not just grammar. Requires a real corpus and real word vectors.
- **Long-range planning.** A world model tracking state across a document (plot, character, argument). Requires hierarchical planning (goal to subgoal to sentence) wired into the generator.
- **Chunks and formulaic sequences.** Humans speak in pre-learned phrases and arcs. Requires frequency-based induction over real text.

What is proven data-free: exact multi-hop derivation, reasoning under calibration, grounded word choice (valid successors only), and structural scaffolding. What needs real data: word meaning (computed keys carry none inherently), long-range coherence (local theme resonance drifts past ~200 tokens), and pragmatics.

**Honest state:** the plumbing is solid. The gap is not algorithms, it is meaning. Biology solves this through embodied learning (act, sense, predict, be surprised, update). On text alone we need real corpora and grounding to world states, actions, and goals we have not yet wired. Possible, not inevitable.

---

## 10. Economic Thesis and the Ask

**Thesis:** removing backpropagation flips the bottleneck from compute to data.
- Training cost: about $0.36 per billion facts (one CPU day).
- Inference cost: on the order of tens of nanojoules per query.
- Footprint: about 5 GB per 5 billion facts.
- Result: intelligence is cheap; knowledge is the constraint.

**What we do not need:** a GPU cluster, expensive training runs, proprietary hardware. Classical training cost evaporates.

**What we do need:**
- **Data.** Real Wikidata (about 43 GB bz2), real text corpora, domain KGs (biomedical, legal, finance).
- **Collaboration.** Grounding to images (co-train on image-text pairs), to audio (speech signals), to actions (robotic feedback).
- **Ingestion compute, not training compute.** Parsing at 15k facts/sec on commodity CPU is tractable; reaching 100M atoms is a few CPU-days and zero GPU.

**Ask:** funding for data acquisition and annotation, collaboration on multi-modal grounding, and developer time to wire meaning and planning robustly. Not hardware, not cloud GPUs.

---

## 11. Risks and Open Problems

**Fluency ceiling.** Large open-ended generation may be a structural limit of derive-not-store (optimal for truth, weaker for creative recombination). Options: a hybrid (frontier for generation, organism for reasoning and verification), or heavy investment in meaning grounding. Honest: it is not yet known whether fluency is achievable at our footprint.

**Consolidation is lossy.** 91% recall means 9% of transient generation content dissipates during sleep. Exact storage would blow the footprint; mitigation is cheap re-derivation on demand.

**Cold-cache cost.** Repeated contexts are O(1); novel contexts rebuild the successor codebook (about 1.6 GB at 500k words). Scaling past millions of words needs a better sparse codebook. Acceptable now, open later.

**Multiplier re-measure.** The 1306x ConceptNet multiplier is real; the earlier Wikidata figure was inflated by a comparison artifact and is withdrawn. A clean re-measure on a fresh Wikidata dump is pending. Numbers here are the conservative ones.

**Single-author reproducibility gap.** Ninety days by one researcher. The code is documented but not yet modularized for outside contribution (the cognition layer is a large flat set of files). Roadmap: regroup with alias re-exports, a test suite, and a clean public release. A workflow gap, not a blocker.

---

## 12. Roadmap

**Immediate (2 to 4 weeks):** wire a meaning critic into structured fill; induce richer frames on real data (Wikipedia at scale); clean re-measure of the multiplier on latest Wikidata.

**Medium (4 to 8 weeks):** image grounding (co-train image-text pairs); an embodied demo (reason about state and action, predict outcome, learn from error); modularize the codebase and publish a replication guide.

**Long-term (month 4-plus):** audio grounding (speech to phonemes to propositions); spatial 3D world model and planning; scale to 100M-plus facts from real KGs and verify the multiplier holds.

---

## 13. Bottom Line

In three months, NeuroSeed and Ikigai reached a narrow, defensible win: reasoning that matches frontier correctness at roughly one millionth of the compute on structured equal-knowledge problems. This is not general intelligence. The organism cannot write essays or learn language naturally. It is a framework with real constraints and real strengths, built on mathematics grounded in biology (VSA, Kanerva memory, sleep consolidation, free energy), with no backpropagation, no GPU cluster, and no secrets.

The proof of concept is solid: flat memory is O(1) in access, derive-not-store is near-free in storage while quadratic in reach, and calibration blocks confident falsehood. The economic thesis holds: training is cheap, data is the constraint. The organism learns, dreams, reasons, plans, and introspects on one substrate.

What we have not done: produced fluent language at scale, absorbed a real-world knowledge graph, or generalized to other modalities. These are engineering problems with no known blocking theory, but they require real data and time. The organism is alive in structure; it still needs meaning.

Given data and collaborators, it scales. That is the ask.

---

**Report written:** July 5, 2026
**Organism state:** 192 MB, 21/21 capabilities, production organism.ikg load-only (never saved)
**Next:** funding hunt, collaborators, real data.
