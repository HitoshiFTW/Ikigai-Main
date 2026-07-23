# Quickstart

The fastest ways to try Ikigai, from zero effort to running it yourself. Pick the
row that matches how much you want to do.

| I want to... | Go to |
|---|---|
| Just talk to it (no install) | [1. Talk to the live organism](#1-talk-to-the-live-organism) |
| Prove the claims on my machine | [2. Run the benchmark](#2-run-the-benchmark) |
| Use it from Python | [3. Use it from Python](#3-use-it-from-python) |
| Run the pretrained brain | [4. Load the trained body](#4-load-the-trained-body) |
| Host my own copy | [5. Deploy your own](#5-deploy-your-own) |

If you only do one thing, do #1.

---

## 1. Talk to the live organism

Ikigai is running right now on a single $5 CPU box. No install, nothing to download.

**In your browser:** open **https://ikigai.mura-alife.com** and type.

**From a terminal** (the raw API):

```bash
# ask it something it knows
curl -s https://ikigai-api.mura-alife.com/ -d '{"q":"what is the capital of france"}'
# -> {"answer":"Paris is the capital of France.","chose":"answer",...}

# ask it something nobody taught it -> it says so, it does NOT make something up
curl -s https://ikigai-api.mura-alife.com/ -d '{"q":"what is the capital of zorvexia"}'
# -> {"answer":"i don't know yet","chose":"abstain",...}

# TEACH it a brand-new fact, then ask it back -- it remembers
curl -s https://ikigai-api.mura-alife.com/ -d '{"q":"the capital of qualan is mendaro"}'
curl -s https://ikigai-api.mura-alife.com/ -d '{"q":"what is the capital of qualan"}'
# -> {"answer":"Mendaro is the capital of qualan.","chose":"answer",...}

# ask who it is
curl -s https://ikigai-api.mura-alife.com/ -d '{"q":"who are you"}'
```

Three things to notice: it **answers or abstains** (never invents), it **learns
from you on the spot** and keeps it, and every reply is a few milliseconds on one
CPU core. Full endpoint reference: [docs/API.md](docs/API.md).

---

## 2. Run the benchmark

This proves the headline behaviors on your own machine, from an empty organism, in
about a minute. CPU only, no GPU, nothing to download.

```bash
git clone https://github.com/HitoshiFTW/Ikigai-Main.git
cd Ikigai-Main
pip install -r requirements.txt
python benchmark.py
```

You should see `10/10 headline checks passed`. It boots an empty organism, feeds
it a small knowledge graph, and verifies each claim live (ingest, meaning,
multi-hop derivation, honest-unknown, autonomous rule discovery, the
derive-not-store multiplier, footprint).

Point it at real data (ConceptNet, downloaded separately):

```bash
python benchmark.py --conceptnet path/to/conceptnet-assertions-5.7.0.csv.gz
```

---

## 3. Use it from Python

After `pip install -r requirements.txt`:

```python
from integrate import IkigaiOrganism

org = IkigaiOrganism()          # a fresh, empty organism

# teach it facts as triples: (subject, relation, object)
org.ingest_triples([
    ('paris', 'capital_of', 'france'),
    ('france', 'isa', 'country'),
])

# ONE call is the whole organism. You don't tell it what to do -- it decides.
print(org('what is the capital of france'))
# {'chose': 'answer', 'result': 'Paris is the capital of France.', ...}

print(org('what is the capital of narnia'))
# {'chose': 'abstain', 'result': None, ...}   <- honest unknown

# teach it in plain English (it learns and keeps it)
org.tell('mendaro is the capital of qualan')
print(org('what is the capital of qualan'))

# what it holds about an entity (the DIRECT relations)
org.knows('paris')          # {'capital_of': ['france'], ...}
org.knows('flarbnak')       # {}  -- unknown entity, invents nothing
```

> Multi-hop derivation (following an is-a chain across many links) works through
> `org(x)` too -- `org('is a cat an animal')` derives the chain and answers, and
> `benchmark.py` step 3 verifies the same reach. See
> [docs/API.md](docs/API.md#multi-hop-derivation).

`org(x)` always returns a dict: `chose` (which faculty won: `answer` / `abstain` /
`learn` / `solve` / `analogy` / `identity`), `result` (the reply), and `learned`
(any facts it just absorbed). Full method list: [docs/API.md](docs/API.md). What
"faculties compete" actually means: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

---

## 4. Load the trained body

The benchmark trains a fresh organism, so you don't need this to run anything. But
if you want the **pretrained** Ikigai (the same body that's live on the internet),
download it from the Releases tab -- it's too big to be a normal git commit.

```bash
# grab the trained body (~184 MB) from the latest release
curl -L -o organism.ikg \
  https://github.com/HitoshiFTW/Ikigai-Main/releases/download/day-104/organism.ikg
```

```python
from integrate import IkigaiOrganism
org = IkigaiOrganism()
org.load_ikg('organism.ikg')    # load the pretrained brain
print(org('what is the capital of france'))
```

By default `load_ikg()` looks for `organism.ikg` next to `integrate.py`, or wherever
the `IKIGAI_IKG` environment variable points.

> **Note:** the trained body is **load-only**. `save_ikg()` refuses to overwrite it
> unless you explicitly pass `allow_production=True` -- a guard so you never clobber
> the shipped brain by accident.

---

## 5. Deploy your own

Want your own Ikigai live on the internet? `experiments/wild/serve.py` is the
server that runs the public one. The full from-zero guide -- Docker, a $5 VPS, and
Cloudflare -- is in [experiments/wild/DEPLOY.md](experiments/wild/DEPLOY.md).

The short version:

```bash
docker build -t ikigai .
docker run -d --restart=always -p 80:80 ikigai
```

---

## Where next

- **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)** -- the ideas in plain English (no math).
- **[docs/API.md](docs/API.md)** -- every HTTP endpoint and Python method.
- **[README.md](README.md)** -- the full picture, the verified claims, honest limitations.
