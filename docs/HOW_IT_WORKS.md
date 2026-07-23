# How Ikigai works (plain English)

No math here, just the ideas. If you've only ever used LLMs, a few things about
Ikigai will feel backwards. This page explains why, one idea at a time.

Ikigai is **not** a language model. There's no neural network, no training run, no
GPU, no context window. It's a digital *organism*: a body, a memory, and a set of
mental faculties that compete to respond to you. Here's what that means.

---

## 1. It derives facts instead of storing them

An LLM (and a database) works by *storing* answers and looking them up. Ikigai
mostly **works answers out from a few building blocks.**

Say it knows three things:

```
a cat is a feline
a feline is a mammal
a mammal is an animal
```

Give it those three links and it can *derive* that a cat is an animal, following
the chain `cat -> feline -> mammal -> animal` on demand. It never stored "cat is an
animal" -- it works it out from the links it holds. (This is exactly what step 3 of
`benchmark.py` checks: a 9-hop chain, derived, never stored, with no fixed hop limit.)

Why this matters: from a small set of stored facts, it can answer a **much larger**
set of questions. Store 18 "is-a" links and it can correctly answer 83 ancestor
questions, none of which were stored. And that multiplier *grows* as it learns
more -- the opposite of a storage wall. This is the single most important idea:
**derive, don't store.**

---

## 2. It says "I don't know" instead of making things up

This is the property frontier LLMs lack, and it's the whole pitch.

Ask Ikigai something nobody taught it -- "what is the capital of zorvexia" -- and
it answers **"i don't know."** It does not invent a plausible-sounding capital.

How? Its memory has a measurable **noise floor.** When it reaches for a fact, it
checks whether what comes back is a real, confident match or just noise near that
floor. Above the floor: it answers. Below it: it abstains. There's no setting for
"be confident anyway." An answer is only emitted if it can be re-derived and
verified first. **Correct, or silent -- never confidently wrong.**

---

## 3. One memory, fixed size, no forgetting

All of Ikigai's knowledge lives in one fixed-size memory called a **substrate**
(technically a vector-symbolic memory over a Kanerva sparse distributed memory --
8 "banks" for different kinds of knowledge). Two surprising consequences:

- **Adding facts doesn't grow the RAM.** Everything is layered into the same fixed
  body in superposition, like many faint transparencies stacked on one sheet. A
  query pulls its own layer back out without disturbing the others.
- **It doesn't catastrophically forget.** Teach it 5 facts, then bombard it with
  5,000 unrelated ones, and the original 5 are still there at 100%. New learning
  doesn't overwrite old learning the way fine-tuning a neural net does.

The trade-off (we're honest about it): each memory "slot" has a finite capacity at
a given size. Scaling to billions of facts means raising the dimension or sharding
-- a known, mechanical step, not a mystery.

---

## 4. It learns the instant you talk to it -- for life

There's no separate "training phase." Tell Ikigai a fact and it's learned **now**,
in that request, and it keeps it forever. The live organism on the internet is
learning from every stranger who talks to it.

And it doesn't just memorize on the first try. It **reinforces** a fact until it's
actually sure of it -- re-testing itself and strengthening the trace, driven by a
dopamine-style "was I surprised?" signal (biology, not backprop). A brand-new fact
takes a couple of repetitions; a well-worn one is rock solid. It won't claim to
know something it hasn't locked in -- it'll tell you it's still unsure instead.

---

## 5. Its grammar is learned, not programmed

Ikigai reads plain English, but nobody hand-wrote its grammar. It **emerges from
what it reads:**

- **Function words** (the, of, is, a...) fall out of raw word frequency -- the
  handful of words that appear constantly are the connective tissue of the language.
- **Question words** (what, who, is, how...) it picks up by noticing which words
  start sentences that end in a `?`.
- **Relation patterns** ("the X of Y is Z") it works out by self-consistency --
  spotting the shapes that reliably line a subject up with one answer.

None of these are lists we typed in. Feed it a different language's text and it
would induce that language's structure the same way. This is the line that makes it
an organism and not a lookup table: **learned from data, or it doesn't ship.**

---

## 6. "One call, and the organism decides"

When you send Ikigai a message, you don't tell it whether to answer, learn, or
stay quiet. Every **faculty** looks at your input and proposes what it would do,
each attaching a number for how well that fits what the organism actually knows.
The organism picks the best-fitting one and does it:

- Give it a fact it doesn't hold -> **learning** wins.
- Ask something it can ground -> **answering** wins.
- Ask who it is -> **identity** wins.
- Give it nothing it can stand on -> **abstaining** wins.

You can see the whole competition in the response (`chose` is the winner,
`options` lists every faculty and its score), so the decision is auditable, not a
black box. And it's the same regardless of the order faculties were registered in
-- only what the organism *knows* changes the outcome.

---

## The honest summary

Ikigai is **not** better than a frontier LLM at broad world knowledge or writing
long fluent prose -- the big models ate the internet and win there, and we don't
pretend otherwise. What Ikigai does that they can't:

- **Never hallucinate** -- answer or abstain, by construction.
- **Reason at near-zero compute** -- roughly a millionth the cost per query, on a CPU.
- **Learn continually, for life** -- no retraining, no forgetting.
- **Stay tiny** -- constant RAM no matter how much it knows.

Want to see it? [Quickstart](../QUICKSTART.md). Want to see how the pieces fit
together in code? [Architecture](ARCHITECTURE.md). Want the exact API?
[API reference](API.md). Want the measured numbers and limitations?
[README](../README.md).
