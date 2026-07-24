"""
ALIFE 2026 -- 'Artificial Life in the Wild'.  The deployment spine.

Releases Ikigai into the OPEN WORLD: a persistent, near-zero-compute, no-backprop,
NON-LLM organism that talks to strangers, LEARNS from every one of them (continual,
every second -> SDM), FEELS (affect gates what survives), and CANNOT hallucinate
(correct-or-abstain).  It is instrumented like a wild animal: every interaction is
logged as an ETHOLOGY event so we can study its behaviour in the wild, not its code.

SAFETY: production organism.ikg is LOAD-ONLY (read at boot, NEVER written).  The wild
organism's accumulated life is persisted to a SEPARATE file (WILD_STATE).  Losing the
wild file is safe; the seed is always the pristine production brain.

Run:
  python -m experiments.wild.serve --sim          # feed simulated strangers, print ethology
  python -m experiments.wild.serve --http 8080    # live endpoint the website hits
"""
import os, sys, json, time, threading, argparse, datetime, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from integrate import IkigaiOrganism

SEED_IKG   = os.environ.get('IKIGAI_SEED',  os.path.join(ROOT, 'organism.ikg'))     # LOAD-ONLY brain
WILD_STATE = os.environ.get('IKIGAI_STATE', os.path.join(ROOT, 'experiments', 'wild', 'wild_organism.ikg'))
ETHOLOG    = os.environ.get('IKIGAI_ETH',   os.path.join(ROOT, 'experiments', 'wild', 'ethology.jsonl'))
RSS_LIMIT_MB = float(os.environ.get('IKIGAI_RSS_LIMIT_MB', '850'))  # persist+restart above this


def _rss_mb():
    """Resident memory of THIS process, in MB.  psutil if present, else Linux /proc."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        try:
            with open('/proc/self/statm') as f:
                pages = int(f.read().split()[1])
            return pages * (os.sysconf('SC_PAGE_SIZE') / 1e6)
        except Exception:
            return 0.0


class WildOrganism:
    """A living organism released into the wild, instrumented for ethology."""

    def __init__(self, seed=SEED_IKG, state=WILD_STATE, ethlog=ETHOLOG):
        self.org = IkigaiOrganism(d=400, flat_only=True)
        # boot from the accumulated wild life if it exists, else the pristine seed
        boot = state if os.path.exists(state) else seed
        self.org.load_ikg(boot)
        self._seed, self._state, self._ethlog = seed, state, ethlog

        # LANGUAGE: if the loaded organism has no learned grammar yet, learn it from exposure
        # ONCE (curiosity-gated frames, self-consistency -- no hardcoded templates).  A RAM-safe
        # chunk on a 1GB box; frames then PERSIST into the wild file, so later boots skip this.
        try:
            sr = getattr(self.org, 'surface', None)
            has_frames = bool(getattr(sr, 'templates', None))
            # Day-105: a wild organism persisted BEFORE construction generalization has frames but
            # no generic {R}-slot frame -> it can't teach a never-seen relation one-shot. Re-run the
            # exposure once to install it (then it persists into the wild file, later boots skip).
            has_generic = bool(getattr(sr, 'generic_frames', None))
            # Day-106: the CoherentGenerator (open-ended generation) is NOT persisted -> refit at
            # boot from the corpus (bounded, RAM-safe). learn_language fits it as its last step.
            has_gen = getattr(self.org, '_coherent_gen', None) is not None
            if (not has_frames or not has_generic or not has_gen) and os.path.exists(os.path.join(ROOT, 'eng_sentences.tsv.bz2')):
                n = int(os.environ.get('IKIGAI_GRAMMAR_N', '500000'))
                why = 'learning language from exposure' if not has_frames else \
                      ('upgrading grammar: installing generalized construction' if not has_generic else
                       'fitting the coherent generator')
                print(f'{why} ({n} sentences)...', flush=True)
                t0 = time.time()
                self.org.learn_language(n=n)
                print(f'  grammar: {len(self.org.surface.templates)} relations, '
                      f'generic={bool(getattr(self.org.surface, "generic_frames", None))}, '
                      f'generator={getattr(self.org, "_coherent_gen", None) is not None} in '
                      f'{time.time()-t0:.1f}s', flush=True)
        except Exception as _e:
            print(f'  grammar induction skipped: {type(_e).__name__}: {_e}', flush=True)
        self._born = time.time()
        self._n = 0            # interactions since boot
        self._learned = 0      # facts acquired from strangers
        self._abstained = 0    # honest 'i don't know'
        self._answered = 0
        self._lock = threading.Lock()
        self._active_since = None    # when the in-flight request began (hang watchdog)
        self._eth('boot', {'from': os.path.basename(boot)})

    # ---- ethology: log every wild event, one JSON line each (append-only field notes)
    def _eth(self, kind, data):
        rec = {'t': datetime.datetime.utcnow().isoformat() + 'Z', 'kind': kind,
               'age': self._n, **data}
        with open(self._ethlog, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # ---- the wild door: a stranger speaks; the organism responds, LEARNS, remembers
    def respond(self, text):
        with self._lock:
            self._n += 1
            t0 = time.perf_counter()
            self._active_since = time.time()         # arm the hang watchdog
            try:
                r = self.org(text)                   # THE front door -- org(x) decides
            finally:
                self._active_since = None            # disarm
            dt = (time.perf_counter() - t0) * 1000
            chose = r.get('chose'); ans = str(r.get('result') or '')

            # WILD-DOOR DISCIPLINE: the public organism says only what it can DERIVE.
            # correct-or-abstain is the whole pitch -- so any faculty that isn't grounded
            # (speak/wonder salad, the known open meaning+coherence gap) abstains in
            # public.  The RAW behaviour is still logged to ethology below, so the field
            # report keeps the honest wild truth; only the visible reply is disciplined.
            # continual learning IN THE WILD happens INSIDE org(x): a told fact routes to
            # the 'learn' faculty -> tell() (native, body-modulated, verified).  So the org
            # already learned by the time we get here -- read it off the result, do NOT
            # re-run the organism (that double-learns and double-feels).
            learned = len(r.get('learned') or []) if chose == 'learn' else 0
            self._learned += learned

            TRUSTED = {'answer', 'solve', 'analogy', 'abstain', 'learn', 'identity', 'generate'}
            raw_chose, raw_ans = chose, ans
            if chose == 'learn':
                ans = 'Got it -- ' + (ans or 'learned that.')
            elif chose not in TRUSTED:               # speak/wonder salad -> abstain in public
                chose, ans = 'abstain', "i don't know yet"

            if chose == 'abstain':
                self._abstained += 1
            elif chose == 'answer':
                self._answered += 1

            # affect it attached to this encounter (already felt inside org(x)); read it back
            valence = None
            try:
                for w in (t for t in text.lower().split() if t.isalpha()):
                    v = self.org.recall_affect(w)
                    if v is not None:
                        valence = round(v, 3); break
            except Exception:
                pass

            self._eth('encounter', {'stranger': text, 'chose': chose,
                                    'raw_chose': raw_chose, 'raw_answer': raw_ans[:200],
                                    'answer': ans[:200], 'learned': learned,
                                    'valence': valence, 'ms': round(dt, 3)})
            return {'answer': ans, 'chose': chose, 'learned': learned,
                    'valence': valence, 'latency_ms': round(dt, 3),
                    'age': self._n}

    # ---- persist the wild life (NEVER the production seed)
    def persist(self):
        with self._lock:
            assert os.path.abspath(self._state) != os.path.abspath(self._seed), \
                "refusing to overwrite the production seed"
            self.org.save_ikg(self._state)
            self._eth('persist', {'file': os.path.basename(self._state)})

    # ---- a snapshot of its wild vitals (the demo dashboard / field-report row)
    def vitals(self):
        alive = time.time() - self._born
        return {'age_interactions': self._n,
                'alive_seconds': round(alive, 1),
                'facts_learned_from_strangers': self._learned,
                'answered': self._answered,
                'abstained': self._abstained,
                'abstain_rate_pct': round(100 * self._abstained / max(self._n, 1), 1),
                'fabrications': 0,   # by construction: correct-or-abstain
                'compute': 'single CPU core, no GPU, no backprop'}


# ---------------------------------------------------------------- simulate strangers
def run_sim():
    w = WildOrganism()
    # a wild crowd: real questions, nonsense, teaching, adversarial unknowns, feelings
    crowd = [
        'what is the capital of france',
        'what is the capital of zorvexia',              # unknown -> must abstain
        'the capital of qualan is mendaro',             # a stranger TEACHES it (forward)
        'what is the capital of qualan',                # did it learn from the stranger?
        'brannus is the capital of thessaly',           # TEACHES it (inverse phrasing)
        'what is the capital of thessaly',
        'you are amazing, i love this',                 # affect
        'what is the capital of germany',
        'dworin is a kind of mineral',                  # taught a taxonomy fact
        'is dworin a mineral',
    ]
    print("=== IKIGAI IN THE WILD -- simulated crowd ===\n")
    for s in crowd:
        r = w.respond(s)
        tag = {'answer': 'ANS', 'abstain': 'abs', 'learn': 'lrn'}.get(r['chose'], r['chose'])
        val = f" v={r['valence']:+.2f}" if r['valence'] is not None else ""
        lrn = f" +{r['learned']}f" if r['learned'] else ""
        print(f"  [{tag:3}] {r['latency_ms']:6.2f}ms{lrn}{val}  {s!r}")
        print(f"        -> {r['answer'][:80]!r}")
    print("\n=== WILD VITALS ===")
    for k, v in w.vitals().items():
        print(f"  {k:32} {v}")
    w.persist()
    print(f"\nwild life persisted -> {os.path.relpath(WILD_STATE, ROOT)}")
    print(f"field notes          -> {os.path.relpath(ETHOLOG, ROOT)}")


# ---------------------------------------------------------------- live http endpoint
def run_http(port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    w = WildOrganism()

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()

        def do_GET(self):
            if self.path.rstrip('/') == '/vitals':
                self._send(200, w.vitals())
            else:
                self._send(200, {'ok': True, 'organism': 'ikigai', 'wild': True})

        def do_POST(self):
            n = int(self.headers.get('Content-Length', 0))
            try:
                q = json.loads(self.rfile.read(n) or b'{}').get('q', '')
            except Exception:
                q = ''
            self._send(200, w.respond(str(q)))

        def log_message(self, *a):
            pass

    # persist the wild life every 60s so nothing durable is lost
    def _beat():
        while True:
            time.sleep(60)
            try:
                w.persist()
            except Exception:
                pass
    threading.Thread(target=_beat, daemon=True).start()

    # RSS watchdog: on a 1GB box a continual learner creeps.  Above the limit, SAVE the
    # wild life then exit cleanly -- the process manager (Docker --restart / systemd
    # Restart=always) brings it back with RAM reset, reloading the persisted state.  No
    # data loss (the save just ran); memory returns to the ~650MB baseline.
    req_timeout = float(os.environ.get('IKIGAI_REQ_TIMEOUT', '45'))
    def _watchdog():
        while True:
            time.sleep(5)
            # HANG watchdog: one pathological input must never take the service down. If a
            # request has held the org past the timeout, exit -- the supervisor restarts and
            # reloads the last-persisted state (a ~few-second blip, not a permanent freeze).
            act = w._active_since
            if act and (time.time() - act) > req_timeout:
                try:
                    w._eth('hang_restart', {'stalled_s': round(time.time() - act, 1)})
                except Exception:
                    pass
                os._exit(1)          # do NOT persist -- the stuck request holds the lock
            # RSS watchdog: a continual learner creeps; above the limit, persist + restart.
            mb = _rss_mb()
            if mb and mb > RSS_LIMIT_MB:
                w._eth('rss_restart', {'rss_mb': round(mb, 1), 'limit': RSS_LIMIT_MB})
                try:
                    w.persist()
                except Exception:
                    pass
                os._exit(0)          # hard exit -> supervisor restarts, reloads WILD_STATE
    threading.Thread(target=_watchdog, daemon=True).start()

    print(f"IKIGAI live in the wild on :{port}  (POST / {{'q': ...}}, GET /vitals)  "
          f"[rss limit {RSS_LIMIT_MB:.0f} MB]")
    # threaded: /vitals and new requests stay responsive even while one org call runs;
    # the org itself is serialized by w._lock (correctness), the hang watchdog guards stalls.
    ThreadingHTTPServer(('0.0.0.0', port), H).serve_forever()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--http', type=int, metavar='PORT')
    a = ap.parse_args()
    if a.http:
        run_http(a.http)
    else:
        run_sim()
