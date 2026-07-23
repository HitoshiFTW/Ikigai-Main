# API reference

Two ways to use Ikigai: the **live HTTP API** (talk to the hosted organism, or your
own deployment) and the **Python API** (drive an organism in your own code).

---

## Live HTTP API

Base URL of the hosted organism: **`https://ikigai-api.mura-alife.com`**
(the site `https://ikigai.mura-alife.com` is a browser front end over this.)

If you run your own with `experiments/wild/serve.py`, the same endpoints are served
on your host.

### `POST /` -- ask, teach, or greet

Send one message. The organism decides what to do with it (answer, abstain, learn,
etc.) -- you don't specify an intent.

**Request body** (JSON):

```json
{ "q": "what is the capital of france" }
```

**Response** (JSON):

```json
{
  "answer":     "Paris is the capital of France.",
  "chose":      "answer",
  "learned":    0,
  "valence":    null,
  "latency_ms": 3.14,
  "age":        42
}
```

| Field | Meaning |
|---|---|
| `answer` | The reply text. For an unknown, `"i don't know yet"`. For a teaching, `"Got it -- ..."`. |
| `chose` | Which faculty won: `answer`, `abstain`, `learn`, `solve`, `analogy`, or `identity`. |
| `learned` | How many new facts this message taught the organism (0 unless you taught it something). |
| `valence` | The feeling it attached to this exchange, roughly -1 (bad) to +1 (good), or `null` if it felt nothing. |
| `latency_ms` | How long the organism took, in milliseconds. |
| `age` | How many interactions this organism has had in its life so far. |

**Examples:**

```bash
# answer
curl -s https://ikigai-api.mura-alife.com/ -d '{"q":"what is the capital of france"}'

# abstain (honest unknown -- no fabrication)
curl -s https://ikigai-api.mura-alife.com/ -d '{"q":"what is the capital of zorvexia"}'

# teach, then recall
curl -s https://ikigai-api.mura-alife.com/ -d '{"q":"the capital of qualan is mendaro"}'
curl -s https://ikigai-api.mura-alife.com/ -d '{"q":"what is the capital of qualan"}'

# identity
curl -s https://ikigai-api.mura-alife.com/ -d '{"q":"who are you"}'
```

> **The public door is correct-or-abstain.** Any response the organism can't ground
> is shown as "i don't know yet" -- so the hosted organism never says something it
> can't back up.

### `GET /vitals` -- the organism's live stats

```bash
curl -s https://ikigai-api.mura-alife.com/vitals
```

```json
{
  "age_interactions": 128,
  "alive_seconds": 90514.2,
  "facts_learned_from_strangers": 37,
  "answered": 71,
  "abstained": 40,
  "abstain_rate_pct": 31.3,
  "fabrications": 0,
  "compute": "single CPU core, no GPU, no backprop"
}
```

`fabrications` is `0` by construction -- correct-or-abstain guarantees it.

### `GET /` -- health check

Returns `{"ok": true, "organism": "ikigai", "wild": true}`.

### CORS

The server sends permissive CORS headers, so you can call it from a browser app.

---

## Python API

`from integrate import IkigaiOrganism`

### Create an organism

```python
org = IkigaiOrganism(d=400, flat_only=False)
```

- `d` -- substrate dimension (default 400). Higher = more capacity per memory slot.
- `flat_only` -- skip the full biological body and boot just the memory substrate
  (faster; enough for ingest/answer/teach work).

### `org(x)` -- the one call

```python
result = org('what is the capital of france')
```

You don't tell it what to do; every faculty competes and the organism picks. Alias
for `org.be(x)`. **Returns a dict:**

| Key | Meaning |
|---|---|
| `chose` | Winning faculty: `answer` / `abstain` / `learn` / `solve` / `analogy` / `identity` (or `None` if it couldn't parse the input). |
| `result` | The reply (a string), or `None` on abstain. |
| `learned` | Facts it just absorbed (a triple or list of triples), present when `chose == 'learn'`. |
| `F` | The winning faculty's free-energy score (lower = more confident). |
| `options` | The whole competition: `[(faculty, score), ...]`, sorted best-first. Makes the decision auditable. |

### `org.ingest_triples(triples, discover=False, self_compress=False)`

Teach it structured facts. Each triple is `(subject, relation, object)`.

```python
org.ingest_triples([
    ('paris', 'capital_of', 'france'),
    ('cat', 'isa', 'feline'),
    ('feline', 'isa', 'mammal'),
])
```

- `discover=True` -- after ingest, run the rule miner to learn rules (e.g. that
  `isa` is transitive) from the data itself.
- `self_compress=True` -- collapse redundant facts down to the irreducible kernel
  (everything stays answerable, storage shrinks).

### `org.tell(text)`

Teach it in **plain English** -- a single statement or a whole paragraph. It parses
each fact through the grammar it learned, learns the ones it can verify, and keeps
them.

```python
org.tell('mendaro is the capital of qualan')
org.tell('Qualan is a country. Its currency is the dree. It is large.')
```

Returns `{'text': ..., 'learned': [(subj, rel, obj), ...]}`.

### `org.knows(entity, rels=None)`

Return the "meaning web" the organism holds for an entity -- the **direct**
relations it has stored. Empty dict `{}` for something it doesn't know (honest).

```python
org.knows('cat')
# {'isa': ['feline'], 'hasa': ['tail', 'whiskers'], ...}   <- the direct facts held
org.knows('flarbnak')     # {}  -- invents nothing
```

For the **multi-hop** closure (following an is-a chain across many links), use the
derive engine directly -- see [Multi-hop derivation](#multi-hop-derivation) below.

### Multi-hop derivation

The organism derives transitive chains (a cat is an animal, via feline -> mammal ->
...) on demand rather than storing them. This is the derive engine's job. Once the
organism has discovered a relation is transitive (or you tell it so), reach the
whole chain:

```python
eng = org.general_reasoner.derive_engine

eng.is_transitive('isa')                       # True once discovered from the data
eng.transitive_reach('isa', 'cat')             # ['cat','feline',...,'animal']  (full chain)
eng.transitive_related('isa', 'cat', 'animal') # True   -- ancestor, never stored
eng.transitive_related('isa', 'cat', 'oak')    # False  -- correctly rejected
```

Note the argument order: `(relation, start[, target])`. Transitivity is *discovered*
from the data when there's enough evidence (`ingest_triples(..., discover=True)`), so
a small handful of edges may not trip it yet -- `benchmark.py` step 5 shows the
discovery, step 3 shows the multi-hop reach.

> **Front-door note:** the plain-English front door `org(x)` currently grounds
> **direct** facts; multi-hop is reached through the derive engine as shown above.
> Routing NL questions like "is a cat an animal" through the transitive engine is
> tracked work, not yet wired.

### `org.learn_language(n=500000)`

Induce grammar (function words, question words, relation frames) from a text corpus
-- how the organism learns to *read* before it learns *facts*. The hosted organism
runs this at boot. Needs a corpus file present (`eng_sentences.tsv.bz2` by default).

### `org.load_ikg(path=None)` / `org.save_ikg(path, allow_production=False)`

Load or save a trained body (a `.ikg` file).

```python
org.load_ikg('organism.ikg')      # load the pretrained brain (see Quickstart step 4)
```

- `load_ikg()` with no path looks for `organism.ikg` beside `integrate.py`, or the
  path in the `IKIGAI_IKG` environment variable.
- `save_ikg()` **refuses** to overwrite the shipped production body unless you pass
  `allow_production=True`. This is a guard against clobbering the trained brain by
  accident -- save your own organisms to a different filename.

### `org.recall_affect(word)`

The feeling the organism has attached to a word, roughly -1..+1, or `None` if it
never felt anything about it.

### `org.identity_statement()`

The organism's own description of itself (what "who are you" returns).

---

## Which do I use?

- **Just trying it / building an app against the hosted organism** -> the HTTP API.
- **Embedding an organism in your own Python, feeding it your own data, running the
  benchmark** -> the Python API.

New here? Start with the [Quickstart](../QUICKSTART.md). Want the concepts behind
these calls? [How it works](HOW_IT_WORKS.md).
