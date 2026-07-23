"""ikigai.cognition.predictive_sequence -- Day 92, PATH A: local-learning SEQUENCE model.

The non-backprop bet for open-ended generation, and the direct cure for the Day-91
degenerate walk ("the house in the house ...").  The walk loops because its context is a
DISCRETE last-k token tuple (branch_gen / AddressGenerator): it has no memory of what it
already said, and an order-10 tuple is never seen twice so long-range dependency is
unreachable.  Fix: replace the discrete tuple with a CONTINUOUS recurrent context-state.

Architecture = HDC RESERVOIR COMPUTING (Echo State Network in phase space):
  * RESERVOIR (fixed, NOT trained) -- one running context-state HV that IS the
    sentence-so-far:
            s_t = decay * (s_{t-1} (x) P) + key(tok_t)
    P is a fixed random unit phasor (a positional decorrelator: binding by P once per
    step rotates each past token to a distinct phase, so order matters and a token at age
    a contributes decay^a * (key (x) P^a)).  decay in (0,1) gives graceful recency.  No
    per-component renorm -- that would destroy the decay weighting.
  * READOUT (the ONLY learned part) -- a single complex d x d map W from context-state to
    the next token's key, trained ONLINE by the normalized delta / Widrow-Hoff rule:
            p_t = W s_t ;  e = key(tok_{t+1}) - p_t ;  W += lr * outer(e, conj(s_t)) / ||s_t||^2
    Single layer, local error -- this is PREDICTIVE CODING (surprise minimization), NOT
    backprop (no chain rule through hidden layers).  Reservoir readouts are classically
    trained WITHOUT backprop; this is that, in the organism's own FHRR substrate.

Predict / generate: cleanup W s against the vocab codebook (argmax |VM^H p|) -> next
token; fold the emitted token back into s and continue.  Because s carries history, the
state after emitting a word differs from the state before it, so repetition self-inhibits
-- the structural cure for the loop.

HONEST SCOPE: this is the FIRST BRICK.  Bar = beat a bigram on a long-range dependency
(proof it captures context without backprop) and stop looping on a real corpus.  Winning
open-ended generation outright is far off (needs scale/data).  Reuses phasor_state ops +
the organism's shared ck; adds only the reservoir-readout.  Additive -- touches no
reasoning path.
"""
import numpy as np


class PredictiveSequenceModel:
    """Recurrent HDC context-state + delta-rule readout.  No backprop.

    ck    -- the organism's shared key space (org.unified.ck): same token identities as
             the rest of the substrate.
    decay -- reservoir leak (recency); token at age a keeps weight decay^a.
    lr    -- readout learning rate (normalized LMS step).
    """

    def __init__(self, ck, decay=0.9, lr=0.5, seed=92):
        self.ck = ck
        self.d = int(ck.key('__probe__').shape[0])
        self.decay = float(decay)
        self.lr = float(lr)
        rng = np.random.RandomState(int(seed))
        # fixed positional decorrelator P (a unit phasor; bind-by-P = per-component phase
        # rotation -> distinct phase per position, reversible, cheap)
        self.P = np.exp(1j * rng.uniform(-np.pi, np.pi, self.d)).astype(np.complex64)
        self.W = np.zeros((self.d, self.d), dtype=np.complex64)   # readout (the only learned part)
        self._code = {}          # token -> index
        self._word = []          # index -> token
        self._keys = []          # index -> key (HV)
        self._VM = None          # cached codebook matrix (V x d)

    # ---- codebook (surface <-> key) --------------------------------------------
    def _key(self, tok):
        w = str(tok)
        c = self._code.get(w)
        if c is None:
            c = len(self._word)
            self._code[w] = c
            self._word.append(w)
            self._keys.append(self.ck.key(w).astype(np.complex64))
            self._VM = None
        return self._keys[c]

    def _codebook(self):
        if self._VM is None:
            self._VM = (np.stack(self._keys) if self._keys
                        else np.zeros((0, self.d), dtype=np.complex64))
        return self._VM

    @property
    def vocab_size(self):
        return len(self._word)

    # ---- reservoir (fixed) ------------------------------------------------------
    def reset(self):
        return np.zeros(self.d, dtype=np.complex64)

    def _advance(self, s, tok):
        """One reservoir step: rotate the past by P, decay it, add the new token."""
        return self.decay * (s * self.P) + self._key(tok)

    def _state_of(self, context):
        s = self.reset()
        for t in context:
            s = self._advance(s, t)
        return s

    # ---- learn (readout, delta rule -- no backprop) -----------------------------
    def learn(self, sequences, epochs=1):
        """Roll the reservoir over each sequence; at every step correct the readout W
        toward the actual next token by the normalized delta rule (predictive coding)."""
        for seq in sequences:                      # register vocab up front
            for t in seq:
                self._key(t)
        for _ in range(int(epochs)):
            for seq in sequences:
                seq = [str(t) for t in seq]
                s = self.reset()
                for t in range(len(seq) - 1):
                    s = self._advance(s, seq[t])
                    ns = float(np.real(np.vdot(s, s)))
                    if ns < 1e-9:
                        continue
                    target = self._key(seq[t + 1])
                    pred = self.W @ s
                    err = target - pred
                    self.W += self.lr * np.outer(err, np.conj(s)) / ns
        return len(self._word)

    # ---- predict / generate -----------------------------------------------------
    def _sims(self, s):
        p = self.W @ s
        VM = self._codebook()
        if VM.shape[0] == 0:
            return None
        return np.abs(VM.conj() @ p)

    def predict(self, context, k=6):
        """Top-k next tokens for a context, ranked by readout resonance."""
        sims = self._sims(self._state_of([str(t) for t in context]))
        if sims is None:
            return []
        order = np.argsort(sims)[::-1][:k]
        return [(self._word[i], float(sims[i])) for i in order]

    def generate(self, seed, steps=20, sample=False, temp=0.7, rng_seed=0):
        """Generate by rolling the reservoir: predict -> emit -> fold back into state.
        Deterministic (argmax) or sampled from the softmaxed resonances."""
        import random as _r
        rng = _r.Random(rng_seed)
        path = [str(x) for x in (seed if isinstance(seed, (list, tuple)) else [seed])]
        s = self._state_of(path)
        for _ in range(int(steps)):
            sims = self._sims(s)
            if sims is None:
                break
            if sample:
                z = sims - sims.max()
                w = np.exp(z / max(1e-6, temp))
                w = w / w.sum()
                i = rng.choices(range(len(w)), weights=w.tolist())[0]
            else:
                i = int(np.argmax(sims))
            tok = self._word[i]
            path.append(tok)
            s = self._advance(s, tok)
        return path

    # ---- evaluation helpers -----------------------------------------------------
    def next_token_acc(self, sequences, k=5):
        """Fraction of positions whose true next token is in the readout's top-k."""
        hit = tot = 0
        for seq in sequences:
            seq = [str(t) for t in seq]
            s = self.reset()
            for t in range(len(seq) - 1):
                s = self._advance(s, seq[t])
                if seq[t + 1] not in self._code:
                    continue
                sims = self._sims(s)
                if sims is None:
                    continue
                top = {self._word[i] for i in np.argsort(sims)[::-1][:k]}
                hit += (seq[t + 1] in top)
                tot += 1
        return hit / max(1, tot)

    def predict_at(self, context, k=1):
        """Argmax (or top-k) prediction for a context -- returns token list."""
        return [w for w, _ in self.predict(context, k=k)]

    def score_next(self, context, candidates):
        """Resonance of each candidate as the NEXT token given the context (for composing the
        reservoir's local sequence coherence into a slot fill).  Returns {word: |resonance| or
        None if the word was never seen}.  |resonance| is unnormalised -- caller softmaxes."""
        s = self._state_of([str(t) for t in context])
        p = self.W @ s
        out = {}
        for c in candidates:
            c = str(c)
            i = self._code.get(c)
            out[c] = float(np.abs(np.vdot(self._keys[i], p))) if i is not None else None
        return out
