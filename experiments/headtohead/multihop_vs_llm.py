import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
"""HEAD-TO-HEAD: equal-knowledge multi-hop reasoning, NeuroSeed vs an LLM.

Made-up org-chart facts (so the LLM can't use pretraining -- pure reasoning over
GIVEN knowledge). Both get the SAME facts + distractor edges. Question: follow
'reportsto' N steps from a person -- who do you reach?  NeuroSeed derives the
chain by exact lookup (O(1)/hop, never wrong); the LLM must trace it in context
and degrades as depth + noise grow.

NeuroSeed runs with no key.  LLM columns populate when a key is set:
    OPENROUTER_API_KEY  -> a frontier model (default nemotron-3-ultra-550b)
    GROQ_API_KEY        -> llama-3.3-70b-versatile
Pick with LLM_BACKEND=openrouter|groq (default openrouter if its key is present).

Result (n=15, 20 distractors):
    depth   NeuroSeed   Nemotron-550B   Llama-70B
      1       100%         100%            93%
      2       100%         100%            40%
      3       100%         100%            13%
      5       100%          27%            13%
      8       100%           0%            20%
"""
import random, re, time
import integrate

OR_KEY = os.environ.get("OPENROUTER_API_KEY")
GQ_KEY = os.environ.get("GROQ_API_KEY")
LLM_BACKEND = os.environ.get("LLM_BACKEND", "openrouter" if OR_KEY else "groq")
OR_MODEL = os.environ.get("OR_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

random.seed(11)
_AL = "abcdefghijklmnopqrstuvwxyz"
def name(rng): return "".join(rng.choice(_AL) for _ in range(6))

def make_case(depth, n_distract, rng):
    people = [name(rng) for _ in range(depth + 1 + n_distract)]
    chain = people[:depth + 1]
    facts = [(chain[i], "reportsto", chain[i + 1]) for i in range(depth)]
    for p in people[depth + 1:]:
        tgt = rng.choice(people)
        if tgt != p: facts.append((p, "reportsto", tgt))
    rng.shuffle(facts)
    return facts, chain[0], chain[depth]

def neuroseed_answer(facts, start, depth):
    org = integrate.IkigaiOrganism(flat_only=True)
    org.ingest_triples(facts)
    eng = org.general_reasoner.derive_engine
    cur = start
    for _ in range(depth):
        cur = eng.atom("reportsto", cur)        # exact stored-fact lookup
        if not cur: return None
    return cur

def _prompt(facts, start, depth):
    lines = "\n".join(f"{s} reportsto {o}" for s, _, o in facts)
    return ("Here are facts of the form 'X reportsto Y':\n" + lines +
            f"\n\nStarting from {start}, follow 'reportsto' exactly {depth} steps. "
            "Who do you reach? Answer with ONLY the final name, nothing else.")
def _parse(out):
    m = re.findall(r"[a-z]{6}", str(out).lower())
    return m[-1] if m else str(out).strip().lower()

def llm_solver():
    if LLM_BACKEND == "openrouter" and OR_KEY:
        import requests
        url = "https://openrouter.ai/api/v1/chat/completions"
        hdr = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
        def ask(facts, start, depth):
            body = {"model": OR_MODEL, "temperature": 0, "max_tokens": 8000,
                    "messages": [{"role": "user", "content": _prompt(facts, start, depth)}]}
            for _ in range(4):
                try:
                    r = requests.post(url, headers=hdr, timeout=180, json=body)
                    if r.status_code == 429: time.sleep(8); continue
                    j = r.json()
                    if "choices" not in j: time.sleep(4); continue
                    m = j["choices"][0]["message"]
                    return _parse(m.get("content") or m.get("reasoning") or "")
                except Exception: time.sleep(4)
            return "ERR"
        return ask, OR_MODEL.split("/")[-1]
    if GQ_KEY:
        from ikigai.cognition.groq_teacher import GroqTeacher
        gt = GroqTeacher(model="llama-3.3-70b-versatile")
        return (lambda f, s, d: _parse(gt._complete(_prompt(f, s, d)))), "llama-3.3-70b"
    return None, None

def main():
    rng = random.Random(7)
    depths = [1, 2, 3, 5, 8]; n_per = 15; n_distract = 20
    llm, opp = llm_solver()
    print(f"opponent: {opp or '(none -- set OPENROUTER_API_KEY or GROQ_API_KEY)'}", flush=True)
    print(f"{'depth':>6} {'NeuroSeed':>10} {'LLM':>10}   (n={n_per}, distractors={n_distract})", flush=True)
    for d in depths:
        cases = [make_case(d, n_distract, rng) for _ in range(n_per)]
        ns = sum(neuroseed_answer(f, s, d) == g for f, s, g in cases) / n_per
        if llm:
            lc = sum(llm(f, s, d) == g for f, s, g in cases) / n_per
            print(f"{d:>6} {ns:>9.0%} {lc:>9.0%}", flush=True)
        else:
            print(f"{d:>6} {ns:>9.0%} {'(no key)':>10}", flush=True)

if __name__ == "__main__":
    main()
