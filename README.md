# Ikigai -- Project NeuroSeed

> **New here?** Start with the **[Quickstart](QUICKSTART.md)** (talk to it in 30
> seconds), then **[How it works](docs/HOW_IT_WORKS.md)** (the ideas in plain
> English), the **[Architecture](docs/ARCHITECTURE.md)** (how the parts interact —
> for collaborators), and the **[API reference](docs/API.md)**.

A digital organism whose knowledge is **composed, not stored**. Facts are held
in a fixed-size hyperdimensional substrate; reasoning is *derived* on demand
rather than looked up. It runs on a CPU -- no GPU, no pretraining, no context
window -- and answers what it can while **saying "I don't know" instead of
confabulating** when it can't.

This repo is the working source of truth: the canonical organism (`ikigai.py`),
the cognition stack (`ikigai/`), the public API (`integrate.py`), a one-command
reproducible benchmark (`benchmark.py`), the wild-deployment server
(`experiments/wild/`), and runnable experiments for each claim.

> Solo research prototype by Prince Siddhpara (17), Mura ALife Labs. No published
> paper yet, but the full write-up (`NeuroSeed_3Month_Report.md`) is in this repo.
> Everything below is something you can run and check yourself -- that is the point.

---

## Try it live

Ikigai is **live on the internet**, running the full organism on a single $5 CPU
box. Talk to it, ask it a fact, or teach it something and watch it remember:

> **https://ikigai.mura-alife.com**

It answers what it can ground, says **"I don't know"** when it can't (no
confabulation), learns continually from whoever talks to it, and knows who it is
("who are you"). Nothing is stubbed -- the same 8-bank Kanerva substrate, derive
engine and calibration you run below is what answers you there. The live server is
`experiments/wild/serve.py`; the from-zero deployment guide is
`experiments/wild/DEPLOY.md` (Docker + VPS + Cloudflare).

---

> **Runs on any PC.** `pip install -r requirements.txt` then `python benchmark.py`.
> CPU only, no GPU, no pretrained body to download (the benchmark trains a fresh
> organism live). Paths are module-relative; there is nothing machine-specific.
> The full trained body (`organism.ikg`) is a **Release asset** (too big for a git
> commit) -- grab it from the Releases tab only if you want to run the pretrained
> organism instead of a fresh one.

---

## The one-command benchmark

```bash
pip install -r requirements.txt
python benchmark.py
```

It boots an **empty** organism, feeds it a small commonsense knowledge graph,
and verifies the headline behaviors end to end -- nothing hardcoded, every line
computed live from the substrate:

```
1) INGEST       31 triples -> 29 kernel atoms in 0.01s
2) MEANING      cat -> isa feline | hasa tail, whiskers | capableof purr, hunt | desires milk
3) MULTI-HOP    cat -> feline -> carnivore -> placental -> mammal -> vertebrate -> chordate -> animal
                (9 hops, DERIVED on demand, never stored)
4) HONEST       knows('flarbnak') = {}        (nonsense -> nothing invented)
5) DISCOVERY    organism learns on its own that IS-A is transitive
6) MULTIPLIER   18 stored IS-A edges -> 83 ancestor facts answerable, none stored
7) FOOTPRINT    29 kernel atoms ~= 0.2 KB marginal store

  10/10 headline checks passed
```

Point it at real data (download separately, see [Real data](#real-data)):

```bash
python benchmark.py --conceptnet path/to/conceptnet-assertions-5.7.0.csv.gz
```

Same code path. On real ConceptNet this ingests ~100k commonsense edges into a
~0.5 MB kernel that answers multi-hop questions and abstains on nonsense.

---

## What makes it different

- **Derive, don't store.** N facts are answered from ~√N stored atoms plus
  learned rules. The transitive closure of a taxonomy (millions of ancestor
  pairs) is *computed*, never materialized. The multiplier **grows** with
  knowledge size -- the opposite of a storage wall.
- **Calibrated honesty.** The organism measures its own substrate noise floor
  and abstains below it, so an unknown query returns nothing instead of a
  confident hallucination. This is the property frontier LLMs lack.
- **Constant-RAM substrate.** Every memory channel lives in superposition on one
  fixed-size body; adding facts doesn't grow it. Zero catastrophic forgetting by
  construction.
- **CPU / on-device.** No GPU, no KV cache, no context window. The compute per
  query is decoupled from the size of what it knows.

**What is *not* claimed:** a broad win on world knowledge, or fluent long-form
prose. The frontier ate the internet; on breadth and generation it wins, and this
repo does not pretend otherwise. What *is* measured: on **equal-knowledge**
multi-hop reasoning (both sides handed the same facts), the organism derives
deeper without error at roughly one millionth the compute per query. See
[Reasoning at near-zero compute](#reasoning-at-near-zero-compute) and
[Honest limitations](#honest-limitations).

---

## Architecture, in one paragraph

~400 biologically-inspired neurons + plastic synapses + 5 neuromodulators +
HPA stress axis + circadian rhythm + homeostatic drives + sleep consolidation +
autobiographical memory, sitting on a **flat ~182 MB VSA-SDM substrate** that
holds every memory channel (co-occurrence, n-grams, IS-A taxonomy, sensory
grounding, properties, verb arithmetic, vision) in superposition without
interference. On top of that substrate sits the **reasoning layer**: a
derive-not-store composition engine (N-hop chaining, inheritance, transitivity),
an autonomous rule miner that runs during sleep, and an empirical calibration
boundary for honest abstention. Constant RAM regardless of data volume. CPU-only.

---

## What's where

| Path | What it is |
|---|---|
| `benchmark.py` | One-command reproducible benchmark (bundled sample, or `--conceptnet`). |
| `ikigai.py` | The canonical biological organism. Single-file neuron/synapse/neuromodulator/HPA/sleep stack. ~1.7 MB by design -- full inspectability. |
| `integrate.py` | Public API. `IkigaiOrganism()` ties every module into one being. `ingest_triples`, `knows`, `say_frame`, ... |
| `ikigai/cognition/compositional.py` | Derive-not-store engine: N-hop chains, inheritance, transitive closure (computed, not stored). |
| `ikigai/cognition/kg_ingest.py` | ConceptNet / N-Triples parsers -- raw KG dump -> triple stream. |
| `ikigai/cognition/calibration.py` | Noise-floor / argmax-safe abstention boundaries (honest unknown). |
| `ikigai/cognition/rule_discovery.py` | Autonomous rule miner (inheritance, synonymy, inverse, transitivity). |
| `ikigai/cognition/frame_relax.py` | Frame-then-fill generator (free-fluency, message-first). |
| `ikigai/cognition/flat_memory.py` | The VSA-SDM substrate. |
| `experiments/` | Runnable demos. Each prints `[PASS]/[FAIL]` per verification + a summary. |
| `experiments/wild/serve.py` | The live-deployment server. Releases the organism into the open world: answers / abstains / learns from every stranger, correct-or-abstain public door, persists its wild life to a separate file (production body stays read-only). |
| `experiments/wild/DEPLOY.md` | From-zero deployment guide: Docker + VPS + Cloudflare. |
| `Dockerfile` | Container for the live server (numpy + psutil only). |
| `NeuroSeed_3Month_Report.md` | Report: goal, measured results, honest wins/losses, roadmap. |

---

## Verified claims (each backed by a runnable experiment)

### Reasoning at near-zero compute

The headline result, and the one honest number on the board: on **equal-knowledge**
multi-hop reasoning (both sides handed the same facts, exactly how ProofWriter /
CLUTRR / bAbI are scored), the organism derives deeper without error, at roughly
one millionth the compute of a frontier forward pass.

- **Equal-knowledge multi-hop arena** -- 100% accuracy through 8 hops on 700
  held-out *derived* facts (never stored), 0 fabrications, ~9 atom lookups and
  ~9 nJ per query on one CPU core. The reasoning is done BY the substrate
  (derive-chaining), not by the harness. (`experiments/nl/day88_arena.py`)
- **Flat access, O(1) in store size** -- recall and chain latency stay flat as the
  store grows 100x (10k -> 1M facts); a query touches its own atom and derivation
  chain, never O(N). (`experiments/nl/day85_flat_access.py`)
- **Faithful, verifiable answers** -- every answer carries a grounded `because`
  chain that must re-derive and verify before it is emitted; unknown queries
  abstain instead of confabulating. (`experiments/nl/day85_explained_answer.py`)
- **Free-to-train scaling** -- ~15k facts/sec/core ingest with zero GPU and zero
  gradient steps (~$0.36 per billion facts projected), flat lookups as the store
  grows 100x. (`experiments/nl/day88_scaling.py`)

**Reasoning / framework**

- **Derive-not-store composition** -- arbitrary N-hop chains and wildcard
  inheritance, derived read-only. (`day80_pack317_derive.py`)
- **Compression multiplier, measured** -- stored kernel x derivation fanout; the
  ratio grows with knowledge size. (`day80_pack321_compression.py`)
- **Rule discovery safe on noisy data** -- the miner promotes true rules, rejects
  spurious ones, and self-compression is lossless (exceptions preserved).
  (`day80_pack323_rule_safety.py`)
- **Self-compression to the kernel** -- ingest a redundant KG, discover rules,
  collapse it to the irreducible kernel, every fact still answerable.
  (`day81_pack325_self_compress.py`)
- **KG ingestion adapter** -- any `(subject, relation, object)` dump with
  arbitrary predicates ingests through one call. (`day81_pack326_ingest.py`)

**Natural-language front end (template-free, no hardcoding)**

A single module (`ikigai/cognition/holo_read.py`) reads plain English and answers
plain English -- no templates, no relation/stopword/wh lists, no stemmer, no
grammar rules. Pure FHRR bind/unbind over a Kanerva SDM, plus distributional
statistics. It is the episodic front door; extracted atoms feed the
derive-not-store engine (the reader never becomes the knowledge store).

- **Template-free holographic reading** -- read any sentence, ask it back with a
  hole, the answer falls out by resonance; honest-unknown when nothing resonates.
  SDM-backed to 100% recall at N=2000. (`experiments/nl/day83_pack333c_module_gate.py`)
- **Plain-English questions, morphology + reordering native** -- "who does X
  report to" with `report`->`reports` for free; same-subject relations
  disambiguated. (`experiments/nl/day83_pack333e_qreader.py`)
- **Emergent atom extraction** -- relations learned by recurrence (no list); text
  -> clean `(subject, relation, object)` -> multi-hop *derived*, not stored.
  (`experiments/nl/day83_pack334_emergent_atoms.py`)
- **Structural relation classifier** -- word-order (edges = arguments, interior =
  relation) lifts mixed-relation prose from 20% to 100%.
  (`experiments/nl/day83_pack337_prose_stress.py`)
- **Reward-driven relation discovery (native dopamine-RL)** -- a rare relation
  below the frequency floor is learned from reward: 0/2 -> 2/2, frequent relations
  intact. (`experiments/nl/day83_pack338_rl_relation.py`)
- **Multi-word entities + emergent question depth.**
  (`experiments/nl/day83_pack336_multitoken.py`, `day83_pack339_emergent_depth.py`)

**Biological / memory substrate**

- **Zero catastrophic forgetting** -- 5 facts retained at 100% through 5,000+
  cross-modal distractors. (`day58_pack129_no_forgetting.py`)
- **Multi-hop structural reasoning via role-binding.** (`day58_pack133_multihop.py`)
- **O(1) per-token generation** -- constant RAM regardless of output length.
  (`day58_pack135_generation_engine.py`)
- **Higher-order n-gram channels at zero substrate cost.** (`day58_pack136_ngrams.py`)
- **Few-shot pattern learning that persists through distractors.** (`day58_pack132_few_shot.py`)
- **Multi-modal on one substrate** -- text, vision (MNIST), arithmetic, taxonomy,
  no interference. (`day58_pack127_vision_channel.py`)
- **Developmental + semantic curriculum** -- learns like a child; meaning sourced
  from WordNet/num2words, not hardcoded. (`day59_pack143_*`, `day59_pack145_*`)
- **Grounded + multi-channel meaning** -- generation consults isa/property
  channels; episodic/affordance/modifier roles native. (`day59_pack146_*`, `day59_pack147_*`)

Run any of them: `python experiments/<name>.py`.

---

## Real data

The benchmark's `--conceptnet` mode reads the ConceptNet 5.7 assertions dump
(commonsense knowledge, CC-BY-SA):

- Download `conceptnet-assertions-5.7.0.csv.gz` from
  <https://github.com/commonsense/conceptnet5/wiki/Downloads>
- `python benchmark.py --conceptnet path/to/conceptnet-assertions-5.7.0.csv.gz`

`ikigai/cognition/kg_ingest.py` also has an N-Triples parser for Wikidata-truthy
/ DBpedia dumps -- same `(subject, relation, object)` contract into the same
`ingest_triples` call.

---

## Honest limitations

This is a proof-of-concept, and it is more useful to you if I'm precise about
the edges:

- **The comparison is equal-knowledge reasoning, not broad knowledge.** In a live
  run against a 550B frontier model, accuracy tied on the hops the rate limit let
  us test, and the organism won on compute by ~1e6x. On a standard *knowledge*
  benchmark (where the frontier can draw on everything it absorbed) it would lose;
  that is a data gap, not a reasoning gap, and we do not claim otherwise.
- **Capacity per role is finite** at a given dimension (~20k facts/role at
  d=400 before recall degrades); scaling means raising the dimension or sharding.
- **Fluent open-ended prose** is mechanism-complete but data-limited -- grammar
  is solved on clean input; shipping fluent generation needs a prose corpus, not
  a new mechanism.
- The shipped trained body (`organism.ikg`, ~190 MB) is **not** a git commit
  (GitHub's 100 MB file cap) -- it ships as a **Release asset** instead. The
  benchmark trains a fresh organism live, so you don't need it to run anything here.

---

## Status

Active solo research prototype by Prince Siddhpara (17), Mura ALife Labs
(formerly Hitoshi AI Labs). No paper yet -- this repo is the source of truth.
Issues and ablations welcome.

## License

MIT -- see [LICENSE](LICENSE).
