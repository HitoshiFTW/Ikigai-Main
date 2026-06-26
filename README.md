# Ikigai -- Project NeuroSeed

A digital organism whose knowledge is **composed, not stored**. Facts are held
in a fixed-size hyperdimensional substrate; reasoning is *derived* on demand
rather than looked up. It runs on a CPU -- no GPU, no pretraining, no context
window -- and answers what it can while **saying "I don't know" instead of
confabulating** when it can't.

This repo is the working source of truth: the canonical organism (`ikigai.py`),
the cognition stack (`ikigai/`), the public API (`integrate.py`), a one-command
reproducible benchmark (`benchmark.py`), and runnable experiments for each claim.

> Solo research prototype by Prince Siddhpara (17), Mura ALife Labs. There is no
> published paper yet. Everything below is something you can run and check
> yourself -- that is the point.

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

**What is *not* claimed:** a head-to-head win against a frontier LLM on a
standard benchmark. That comparison is the next step, not a settled result. See
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

---

## Verified claims (each backed by a runnable experiment)

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

- **Not yet benchmarked head-to-head against a frontier LLM** on a standard
  public eval. The differentiators above (constant-RAM, no-forgetting,
  calibrated abstention, derive-not-store) are demonstrated in isolation; a
  clean comparison run is the next milestone.
- **Capacity per role is finite** at a given dimension (~20k facts/role at
  d=400 before recall degrades); scaling means raising the dimension or sharding.
- **Fluent open-ended prose** is mechanism-complete but data-limited -- grammar
  is solved on clean input; shipping fluent generation needs a prose corpus, not
  a new mechanism.
- The shipped 182 MB trained body (`organism.ikg`) is **not** in this repo
  (GitHub's 100 MB file cap); the benchmark trains a fresh organism live, so you
  don't need it.

---

## Status

Active solo research prototype by Prince Siddhpara (17), Mura ALife Labs
(formerly Hitoshi AI Labs). No paper yet -- this repo is the source of truth.
Issues and ablations welcome.

## License

MIT -- see [LICENSE](LICENSE).
