# Ikigai -- Project NeuroSeed

A biologically grounded digital organism. Cognition, language, and memory emerge from interacting neural systems rather than from explicit symbolic programming or large-scale pretraining.

This repository holds the main code: the canonical organism (`ikigai.py`), the wider cognition stack (`ikigai/`), the public organism API (`integrate.py`), and a set of runnable experiments demonstrating each headline claim.

## Architecture, in one paragraph

~400 biologically-inspired neurons + thousands of plastic synapses + 5 interacting neuromodulators (dopamine, norepinephrine, acetylcholine, oxytocin, cortisol) + a full HPA stress axis + circadian rhythm + homeostatic drives (hunger, sleep, curiosity, safety, social) + sleep consolidation + autobiographical narrative memory, sitting on top of a **flat 192 MB VSA-SDM substrate** that holds every memory channel (co-occurrence, n-grams, IS-A taxonomy, sensory grounding, properties, verb arithmetic, vision, novel patterns) in superposition without interference. Constant RAM regardless of data volume. CPU-only. Zero catastrophic forgetting by construction.

## Quick start

```bash
pip install -r requirements.txt

# 1) boot the organism
python ikigai.py

# 2) run the headline experiments
python experiments/day58_pack129_no_forgetting.py
python experiments/day58_pack133_multihop.py
python experiments/day58_pack135_generation_engine.py
python experiments/day58_pack136_ngrams.py
python experiments/day58_pack127_vision_channel.py
python experiments/day58_pack132_few_shot.py
```

`python ikigai.py` runs the standalone biological organism for `TICKS=1000` ticks (configurable near the bottom of the file). It will birth its neural population, run wake/sleep cycles, form concepts, and write state to `ikigai_state.json`. A saved mature persona is included as `state_demo.json` -- rename to `ikigai_state.json` to boot directly into that organism instead of starting from birth.

## What's where

| Path | What it is |
|---|---|
| `ikigai.py` | The canonical organism. Single-file biology stack: neurons, synapses, neuromodulators, HPA axis, homeostasis, sleep, autobiographical memory. ~1.7 MB by design -- full inspectability over abstraction. |
| `ikigai/core/state.py` | Runtime state container (`IkigaiState`, `IkigaiContext`). |
| `ikigai/cognition/` | Cognition modules: flat memory substrate, multi-role memory, generation engine, reasoning, vision, persona, etc. |
| `integrate.py` | Public organism API. `IkigaiOrganism()` ties every cognition module into one being. |
| `experiments/` | Runnable demos for each headline claim. Each one prints `[PASS] / [FAIL]` per verification and a final summary. |
| `state_demo.json` | A saved mature persona -- optional, lets you skip the boot phase. |

## Headline claims (each backed by a runnable experiment)

- **Zero catastrophic forgetting** -- 5 original facts retained at 100% after 5,000+ distractor writes across text, arithmetic, and vision channels. (`day58_pack129_no_forgetting.py`)
- **Continual learning that beats a CNN by 9.2* on Split-MNIST** -- flatmem 92% vs CNN 10%, no replay buffer, no retraining.
- **Multi-hop structural reasoning** -- 5-hop chains (`cat -> mammal -> forest -> river -> water -> cold`) via role-binding, not prompt iteration. (`day58_pack133_multihop.py`)
- **O(1) per-token generation** -- `org.cogitate(prompt, max_tokens)` generates with constant RAM regardless of output length. Thought-state is a single hypervector that evolves through the substrate. No KV cache, no context window. (`day58_pack135_generation_engine.py`)
- **Online learning mid-generation** -- inject a fact while generating, the next tokens reflect it. No fine-tuning, no RAG.
- **Multi-modal on one 192 MB substrate** -- text, vision (MNIST), arithmetic, taxonomy all share the same memory with no interference. (`day58_pack127_vision_channel.py`)
- **Few-shot pattern learning that persists** -- 5 novel mappings (`flompet -> red`, etc) learned in 2 shots, survive 5K+ distractors across modalities. (`day58_pack132_few_shot.py`)
- **Trigram + 4-gram channels at zero substrate cost** -- higher-order n-grams via permute-then-bind cost no additional RAM. (`day58_pack136_ngrams.py`)
- **Real Wikipedia in 192 MB** -- 200 Simple English Wikipedia articles (8,306 sentences, 79K tokens) absorbed in 68 seconds; substrate stays 192 MB FIXED; learned co-occurrences (north-south +0.55, year-month +0.45, king-queen +0.33) recovered cleanly. Cross-modal channels (vision, math, IS-A) intact after the flood. (`day58_pack137_simple_wiki.py`)
- **RapidTrainer: 7.5x faster training** -- batched cross-sentence flush + stopword filter cuts per-token training cost without changing write semantics. Same substrate, same recall quality, much faster scaling. (`day58_pack138_rapid_trainer.py`)
- **Ablation harness for the bio stack** -- toggle individual mechanisms (cortisol, sleep onset, dopamine suppression, arousal modulation, L23 recovery) and measure behavioral deltas via `exec()` patching of `ikigai.py` -- the canonical organism is never permanently modified. Real signal: killing cortisol drives dopamine up 38%, confirming the cortisol-DA coupling literature in our model. (`day58_pack140_ablation_harness.py`)
- **Goal-state HV for long-gen coherence** -- adds a second, FIXED hypervector (the initial prompt HV, never drifts) alongside the drifting thought. Speak step scores candidates by both: thought explores, goal anchors topic. Improves late-token prompt alignment by 2.77x on toy corpus. (`day59_pack142_goal_anchor.py`)
- **Real Wikipedia at 10K-article scale** -- 10,000 Simple English Wikipedia articles, 323K sentences, **3.35 million tokens** absorbed in one run. Substrate stays 192 MB FIXED. Trigram coverage 137,911 contexts, 4-gram 133,366. Real Wiki cooccur sims hold: north-south +0.57, king-queen +0.40, film-movie +0.39. Checkpoint 100 MB on disk. (`day59_pack141_wiki_10k.py`)
- **FlatTrainer: bounded-RAM trainer for any data size** -- substrate stays 192 MB, side caches (ComputedKey vocab + SDM location cache) stay bounded too via periodic compaction at flush boundaries. Trained 3.35M tokens with peak RSS 3.6 GB on a 16 GB machine. (`ikigai/cognition/flat_trainer.py`)
- **Developmental curriculum (5 stages)** -- organism learns English like a child: alphabet, letter->word anchors ("a for apple"), CVC words, Dolch sight words, simple SVO sentences. Each stage = bounded named effects on named channels. 26/26 letter->word anchors recovered at 97%+ confidence. Substrate stays 192 MB across all stages. (`day59_pack143_developmental_curriculum.py`)
- **Curriculum priming yields 2.32x training speedup** -- A/B test: organism trained on curriculum (stages 1-5) before 500 Wikipedia articles trains the wiki phase in 336s vs 780s for the bare-wiki baseline. Same final corpus, same final vocab (~22K). Cache priming explains it: trigram + word caches are warm by the time wiki starts. Letter->anchor 26/26 also survives the 500-article wiki flood (no-forgetting at scale). (`day59_pack144_curriculum_then_wiki.py`)

## Status

Active solo research prototype by Prince Siddhpara (17), Mura ALife Labs (formerly Hitoshi AI Labs).

There is no published paper yet -- this is the working source of truth. The architecture is under continuous refinement; the experiments folder is the cleanest summary of where things currently are.

If you'd like to give feedback, run an ablation, or just discuss design choices, please open an issue.

## License

MIT -- see [LICENSE](LICENSE).
