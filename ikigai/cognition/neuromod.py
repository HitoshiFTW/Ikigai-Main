"""
ikigai.cognition.neuromod -- Neuromodulator Generative Binding.

Day 61 Pack 170. Wires emotional/chemical state directly into the VSA
retrieval + generation engine. The substrate already has the algebra;
this layer makes the algebra MOOD-AWARE.

Mapping (the asked-for surface):
    dopamine / epinephrine  -> retrieval gating threshold + softmax
                                temperature scale (high DA = creative
                                bleeding; high EPI = focused/strict)
    cortisol                -> negative gamma compression on role weights;
                                forces rapid rigid convergence onto verified
                                logical fixed points under conflict/entropy
    serotonin               -> baseline / decay-rate mood floor (added)
    acetylcholine           -> read-side SNR boost (added)

Token-driven spikes: environmental text containing emotionally loaded
tokens (danger, reward, novel, calm, ...) spike the chemical tensors in
real time. The state alters every downstream substrate call until it
decays back to baseline -- naturally during sleep_consolidate().

Public API:
    neuro = NeuroModulators()
    neuro.expose_tokens("danger danger fire")     -> spikes cortisol/epi
    neuro.spike('dopamine', +0.3, reason='reward')
    neuro.decay(dt=1)                             -> step toward baseline
    weights = neuro.attention_weights(roles)      -> per-role modulation
    temp    = neuro.temperature_scale()           -> softmax temp scale
    snr     = neuro.snr_boost()                   -> read sharpening
    pre     = neuro.gamma_preset()                -> 'categorical' under cortisol
    state   = neuro.state()                       -> full snapshot dict

On IkigaiOrganism (wired in integrate.py):
    org.neuro                  -> lazy-built singleton
    org.neuro_spike(chem, dx)  -> manual spike
    org.neuro_expose(text)     -> auto token spike on a sentence
    org.neuro_state()          -> snapshot
    org.attend_modulated(query, candidates, roles=...) -> mood-aware cleanup
    sleep_consolidate(...)     -> additionally decays neuromod toward baseline
"""

import numpy as np


class NeuroModulators:
    """
    Tensor of chemical levels mapped into substrate read/write parameters.
    Each chemical has: baseline, decay_rate, current level.
    """

    # Baselines match ikigai.py runtime setpoints so the flat-substrate
    # neuromod state is a drop-in mirror of the biological organism in
    # ikigai.py (DopamineSystem.setpoint=0.5, SerotoninSystem=0.6,
    # NorepinephrineSystem=0.3, AcetylcholineSystem=0.4, CortisolSystem=0.1,
    # OxytocinSystem=0.3, AdenosineSystem accumulator starts 0).
    # 'epinephrine' bridges to ikigai's NE (norepinephrine).
    BASELINE = {
        'dopamine':      0.50,
        'epinephrine':   0.30,    # mirrors NorepinephrineSystem setpoint
        'cortisol':      0.10,
        'serotonin':     0.60,
        'acetylcholine': 0.40,
        'oxytocin':      0.30,    # ikigai.py oxt setpoint -- social trust
        'adenosine':     0.00,    # ikigai.py ado accumulator -- sleep pressure
    }

    DECAY_RATE = {
        'dopamine':      0.95,
        'epinephrine':   0.90,
        'cortisol':      0.97,
        'serotonin':     0.99,
        'acetylcholine': 0.92,
        'oxytocin':      0.98,
        'adenosine':     1.00,    # NEVER decays during waking; only sleep drops it
    }

    # Names in ikigai.py runtime namespace mapped to our keys.
    IKIGAI_VAR_MAP = {
        'dopamine':      'da',
        'epinephrine':   'ne',     # NorepinephrineSystem
        'cortisol':      'cort',
        'serotonin':     'ht',
        'acetylcholine': 'ach',
        'oxytocin':      'oxt',
        'adenosine':     'ado',
    }

    SPIKE_LEXICON = {
        # threat axis -> cortisol + epinephrine
        'danger':  {'cortisol': +0.25, 'epinephrine': +0.30},
        'threat':  {'cortisol': +0.25, 'epinephrine': +0.30},
        'attack':  {'cortisol': +0.25, 'epinephrine': +0.30},
        'fight':   {'cortisol': +0.20, 'epinephrine': +0.25},
        'fire':    {'cortisol': +0.20, 'epinephrine': +0.25},
        'fear':    {'cortisol': +0.20, 'epinephrine': +0.15},
        'scary':   {'cortisol': +0.18, 'epinephrine': +0.20},
        'pain':    {'cortisol': +0.30, 'epinephrine': +0.20},
        'urgent':  {'cortisol': +0.15, 'epinephrine': +0.25},
        'panic':   {'cortisol': +0.30, 'epinephrine': +0.35},

        # reward axis -> dopamine + serotonin
        'reward':  {'dopamine':  +0.25, 'serotonin':   +0.10},
        'win':     {'dopamine':  +0.20, 'serotonin':   +0.10},
        'joy':     {'dopamine':  +0.20, 'serotonin':   +0.20},
        'love':    {'serotonin': +0.25, 'dopamine':    +0.10},
        'success': {'dopamine':  +0.20, 'serotonin':   +0.15},

        # novelty axis -> dopamine + acetylcholine
        'novel':   {'dopamine':  +0.20, 'acetylcholine': +0.15},
        'curious': {'dopamine':  +0.15, 'acetylcholine': +0.20},
        'explore': {'dopamine':  +0.15, 'acetylcholine': +0.10},
        'discover':{'dopamine':  +0.20, 'acetylcholine': +0.15},

        # focus axis -> acetylcholine
        'focus':   {'acetylcholine': +0.25},
        'sharp':   {'acetylcholine': +0.15},
        'attend':  {'acetylcholine': +0.20},

        # calming axis -> serotonin, lowers cortisol/epi
        'safe':    {'serotonin': +0.15, 'cortisol': -0.15},
        'calm':    {'serotonin': +0.20, 'cortisol': -0.10, 'epinephrine': -0.10},
        'rest':    {'serotonin': +0.15, 'cortisol': -0.15, 'epinephrine': -0.10},
        'peace':   {'serotonin': +0.20, 'cortisol': -0.10},

        # social trust axis -> oxytocin
        'friend':  {'oxytocin': +0.20, 'serotonin': +0.10},
        'trust':   {'oxytocin': +0.25, 'serotonin': +0.10},
        'family':  {'oxytocin': +0.20, 'serotonin': +0.15},
        'hug':     {'oxytocin': +0.30, 'cortisol':  -0.10},
        'bond':    {'oxytocin': +0.20},
        'betray':  {'oxytocin': -0.30, 'cortisol': +0.20},
        'lonely':  {'oxytocin': -0.20, 'serotonin': -0.10},

        # fatigue axis -> adenosine
        'tired':   {'adenosine': +0.20, 'acetylcholine': -0.10},
        'sleepy':  {'adenosine': +0.25, 'acetylcholine': -0.15},
        'exhausted':{'adenosine': +0.30, 'cortisol': +0.10},
    }

    def __init__(self, lexicon=None, baseline=None, decay_rate=None,
                 d=None, seed=14820):
        self.baseline = dict(baseline) if baseline else dict(self.BASELINE)
        self.decay    = dict(decay_rate) if decay_rate else dict(self.DECAY_RATE)
        self.level    = dict(self.baseline)
        self.lexicon  = dict(self.SPIKE_LEXICON)
        if lexicon:
            self.lexicon.update(lexicon)
        self.history       = []          # [(tick, chem, delta, reason)]
        self.cortisol_load = 0.0         # rolling EMA of cortisol excess (allostatic)
        self.allostatic    = dict(self.baseline)  # adaptive set-point (slow EMA)
        self._tick         = 0

        # Pack 170+ phasic/tonic separation (mirrors ikigai.py DopamineSystem).
        # tonic[chem] = slow EMA; phasic[chem] = level - tonic at last spike.
        self.tonic  = dict(self.baseline)
        self.phasic = {k: 0.0 for k in self.baseline}

        # Mood-bound encoding: a per-chemical phasor HV. When the substrate
        # writes a fact, the "mood HV" can be bound INTO the address so the
        # fact is retrievable best when the same mood returns (mood-congruent
        # recall, classic psych phenomenon). Lazy-built when d known.
        self._d = int(d) if d else None
        self._rng_seed = int(seed)
        self._mood_axes = None     # dict chem -> (cos, sin) phasor axis HVs

    # ── core dynamics ────────────────────────────────────────────────────
    def spike(self, chem, delta, reason='manual'):
        if chem not in self.level:
            return
        prev = self.level[chem]
        self.level[chem] = float(np.clip(prev + float(delta), 0.0, 1.0))
        # phasic = instantaneous delta vs tonic baseline
        self.phasic[chem] = self.level[chem] - self.tonic[chem]
        # tonic = slow EMA (allostatic running average)
        self.tonic[chem]  = 0.98 * self.tonic[chem] + 0.02 * self.level[chem]
        # allostatic load tracks each chemical's drift away from baseline
        self.allostatic[chem] = 0.99 * self.allostatic[chem] + 0.01 * self.level[chem]
        self.history.append((self._tick, chem, float(delta), reason))
        self._tick += 1

    def expose_tokens(self, text):
        """Scan tokens. Each lexicon hit spikes its chemicals."""
        if not text:
            return 0
        n_spikes = 0
        for raw in text.lower().split():
            tok = ''.join(ch for ch in raw if ch.isalpha())
            if not tok:
                continue
            spec = self.lexicon.get(tok)
            if not spec:
                continue
            for chem, delta in spec.items():
                self.spike(chem, delta, reason=f'token:{tok}')
                n_spikes += 1
        return n_spikes

    def decay_step(self, dt=1):
        """Exponential decay toward baseline. Update cortisol load EMA."""
        for chem, lvl in list(self.level.items()):
            base = self.baseline[chem]
            rate = self.decay[chem] ** float(dt)
            self.level[chem] = base + (lvl - base) * rate
            # phasic always decays toward 0; tonic catches up to baseline slowly
            self.phasic[chem] = self.phasic[chem] * (rate ** 2)
            self.tonic[chem]  = base + (self.tonic[chem]  - base) * (rate ** 0.5)
        excess = max(0.0, self.level['cortisol'] - self.baseline['cortisol'])
        self.cortisol_load = 0.9 * self.cortisol_load + 0.1 * excess
        # adenosine ALWAYS accumulates during waking ticks
        self.level['adenosine'] = float(np.clip(
            self.level['adenosine'] + 0.002 * float(dt), 0.0, 1.0))
        self._tick += 1

    def sleep_step(self, dt=10):
        """Inverse of decay_step. Used by sleep_consolidate to flush
        adenosine + reset allostatic drift."""
        for chem in self.level:
            base = self.baseline[chem]
            rate = self.decay[chem] ** float(dt)
            self.level[chem] = base + (self.level[chem] - base) * rate
        # adenosine ONLY clears during sleep
        self.level['adenosine'] = self.level['adenosine'] * (0.5 ** float(dt) / 5.0)
        # allostatic drift reset partially
        for chem in self.allostatic:
            self.allostatic[chem] = (0.7 * self.allostatic[chem]
                                     + 0.3 * self.baseline[chem])
        # cortisol_load: exponential decay over sleep duration (deep sleep crash)
        self.cortisol_load *= float(np.exp(-0.08 * float(dt)))
        for chem in self.phasic:
            self.phasic[chem] *= (0.5 ** float(dt))
        self._tick += dt

    def reset(self):
        """Hard reset to baseline. Used by deep sleep."""
        self.level = dict(self.baseline)
        self.cortisol_load = 0.0

    # ── bridge to/from ikigai.py runtime ─────────────────────────────────
    @classmethod
    def from_ikigai_namespace(cls, ns, lexicon=None):
        """Construct a NeuroModulators seeded from an exec()'d ikigai.py
        namespace (the dict you pass into exec() and get back populated
        with `da`, `ne`, `cort`, `ht`, `ach` system objects).

        Example:
            ns = {}
            exec(compile(open('ikigai.py').read(), 'ikigai.py', 'exec'), ns)
            neuro = NeuroModulators.from_ikigai_namespace(ns)
        """
        inst = cls(lexicon=lexicon)
        inst.sync_from_ikigai(ns)
        return inst

    def sync_from_ikigai(self, ns):
        """Pull current `.level` from each ikigai.py chemical system in `ns`."""
        for chem, var in self.IKIGAI_VAR_MAP.items():
            sys_obj = ns.get(var) if isinstance(ns, dict) else getattr(ns, var, None)
            if sys_obj is not None and hasattr(sys_obj, 'level'):
                self.level[chem] = float(sys_obj.level)

    def push_to_ikigai(self, ns):
        """Push our levels back into the ikigai.py runtime systems.
        Use this to make a substrate-driven mood change steer the
        biological organism in the next tick."""
        for chem, var in self.IKIGAI_VAR_MAP.items():
            sys_obj = ns.get(var) if isinstance(ns, dict) else getattr(ns, var, None)
            if sys_obj is not None and hasattr(sys_obj, 'level'):
                sys_obj.level = float(self.level[chem])

    # ── derived signals consumed by substrate calls ──────────────────────
    def temperature_scale(self):
        """High DA  -> hotter softmax (creative coordinate bleeding).
           High EPI -> cooler softmax (focused / strict reading).
           Returned scalar is multiplied INTO the base temperature."""
        da = self.level['dopamine']    - self.baseline['dopamine']
        ep = self.level['epinephrine'] - self.baseline['epinephrine']
        return float(np.clip(1.0 + 2.0 * da - 1.5 * ep, 0.4, 2.5))

    def threshold_shift(self):
        """Additive shift to a selection-confidence threshold.
           Positive under cortisol = more conservative."""
        c = self.level['cortisol'] - self.baseline['cortisol']
        return float(np.clip(0.5 * c, -0.2, 0.5))

    def snr_boost(self):
        """Multiplicative SNR boost on substrate reads. Acetylcholine
           sharpens the read; low ach lets noise through."""
        ach = self.level['acetylcholine'] - self.baseline['acetylcholine']
        return float(np.clip(1.0 + 1.5 * ach, 0.3, 2.5))

    def gamma_preset(self):
        """Map current state to a ConceptSynthesizer preset.
              cortisol high  -> 'categorical'  (isa-dominant rigid convergence)
              dopamine high  -> 'broad'        (channel bleeding, creativity)
              else           -> None (caller keeps its own default)"""
        c  = self.level['cortisol'] - self.baseline['cortisol']
        da = self.level['dopamine'] - self.baseline['dopamine']
        if c > 0.15:
            return 'categorical'
        if da > 0.20:
            return 'broad'
        return None

    def forced_rest(self, threshold=0.20):
        """True when sustained cortisol load demands a sleep cycle."""
        return self.cortisol_load > float(threshold)

    # ── attention head weight modulation ─────────────────────────────────
    def attention_weights(self, roles):
        """Mood-modulated per-role weights for VSAAttention.
              cortisol      -> boost 'isa' (taxonomic certainty)
              dopamine      -> boost 'cooccur' and 'episode' (associative)
              acetylcholine -> boost 'property' (precise read)
           Returned weights are renormalized to sum to 1."""
        c   = self.level['cortisol']      - self.baseline['cortisol']
        da  = self.level['dopamine']      - self.baseline['dopamine']
        ach = self.level['acetylcholine'] - self.baseline['acetylcholine']
        ep  = self.level['epinephrine']   - self.baseline['epinephrine']
        out = []
        for r in roles:
            w = 1.0
            if r == 'isa':       w *= (1.0 + 2.5 * c)
            if r == 'cooccur':   w *= (1.0 + 1.8 * da - 0.5 * c)
            if r == 'episode':   w *= (1.0 + 1.2 * da)
            if r == 'property':  w *= (1.0 + 1.5 * ach + 0.5 * ep)
            if r == 'affordance':w *= (1.0 + 0.8 * ep)
            if r == 'mod':       w *= (1.0 + 0.6 * ach)
            out.append(max(0.0, w))
        s = sum(out) or 1.0
        return [x / s for x in out]

    # ── concept gamma scaling (cortisol = negative compression) ─────────
    def scale_concept_weights(self, base_weights):
        """Return a modulated copy of ConceptSynthesizer weights.
              cortisol  -> compress: amplify isa, suppress cooccur/episode
                           (rigid convergence onto a logical fixed point)
              dopamine  -> spread: amplify episode + cooccur
              ach       -> amplify property + mod"""
        c   = self.level['cortisol']      - self.baseline['cortisol']
        da  = self.level['dopamine']      - self.baseline['dopamine']
        ach = self.level['acetylcholine'] - self.baseline['acetylcholine']
        out = {}
        for role, w in base_weights.items():
            m = 1.0
            if role == 'isa':       m *= (1.0 + 3.0 * c)
            if role == 'cooccur':   m *= max(0.0, 1.0 - 1.5 * c) * (1.0 + 1.5 * da)
            if role == 'episode':   m *= max(0.0, 1.0 - 1.0 * c) * (1.0 + 1.2 * da)
            if role == 'property':  m *= (1.0 + 1.5 * ach)
            if role == 'mod':       m *= (1.0 + 0.8 * ach)
            out[role] = float(w) * m
        return out

    # ── snapshot ─────────────────────────────────────────────────────────
    def state(self):
        out = {
            **dict(self.level),
            'cortisol_load':     self.cortisol_load,
            'temperature_scale': self.temperature_scale(),
            'threshold_shift':   self.threshold_shift(),
            'snr_boost':         self.snr_boost(),
            'gamma_preset':      self.gamma_preset(),
            'forced_rest':       self.forced_rest(),
            'write_strength_x20':self.write_strength(20),
            'tick':              self._tick,
            'phasic':            dict(self.phasic),
            'tonic':             dict(self.tonic),
        }
        ok, why = self.should_act()
        out['should_act']        = ok
        out['should_act_reason'] = why
        return out

    def __repr__(self):
        st = self.state()
        return (f"NeuroModulators(DA={st['dopamine']:.2f} EPI={st['epinephrine']:.2f} "
                f"CORT={st['cortisol']:.2f} 5HT={st['serotonin']:.2f} "
                f"ACH={st['acetylcholine']:.2f} OXY={st['oxytocin']:.2f} "
                f"ADO={st['adenosine']:.2f} temp_x{st['temperature_scale']:.2f} "
                f"snr_x{st['snr_boost']:.2f} preset={st['gamma_preset']})")

    # ── reinforcement-rate modulation (substrate write strength) ─────────
    def write_strength(self, base_n=20):
        """Per-write reinforcement count from chemical state.
              ACh boost                -> more reinforcements (focus = strong encode)
              DA phasic positive       -> reinforce (reward = remember)
              CORT spike               -> reinforce (threat = remember; trauma-strong)
              high adenosine           -> down-scale (tired = weak encode)
           Returns positive integer >= 1."""
        ach = self.level['acetylcholine'] - self.baseline['acetylcholine']
        da_phasic = self.phasic['dopamine']
        c = self.level['cortisol'] - self.baseline['cortisol']
        ado = self.level['adenosine']
        mult = (1.0 + 1.0 * ach + 2.0 * max(0.0, da_phasic)
                    + 2.5 * max(0.0, c) - 1.5 * ado)
        return max(1, int(round(base_n * float(np.clip(mult, 0.1, 4.0)))))

    # ── mood-bound encoding HV (mood-congruent recall) ───────────────────
    def _ensure_mood_axes(self, d):
        """Lazy-build per-chemical phasor axis HVs. Each chemical gets a
        random unit phasor axis; mood_hv = product(axis_chem ** level_chem)."""
        if self._mood_axes is not None and self._d == d:
            return
        self._d = int(d)
        rng = np.random.default_rng(self._rng_seed)
        self._mood_axes = {}
        for chem in self.baseline:
            ph = rng.uniform(-np.pi, np.pi, self._d).astype(np.float32)
            self._mood_axes[chem] = ph    # store phases; mood_hv exponentiates them
        # axes are PHASES so we can scale by level (phase * level = scaled phasor)

    def mood_hv(self, d=None):
        """Return a phasor HV that encodes the CURRENT chemical state.
        Bind this into substrate writes to make recall mood-congruent."""
        if d is None:
            d = self._d
        if d is None:
            raise ValueError("mood_hv requires d; pass d=... or call _ensure_mood_axes")
        self._ensure_mood_axes(d)
        accum_phase = np.zeros(d, dtype=np.float32)
        for chem, ph in self._mood_axes.items():
            # scale each chemical's axis by its DEVIATION from baseline
            dev = self.level[chem] - self.baseline[chem]
            accum_phase = accum_phase + ph.astype(np.float32) * float(dev)
        return np.exp(1j * accum_phase).astype(np.complex64)

    # ── persistence ──────────────────────────────────────────────────────
    def to_dict(self):
        """JSON-safe snapshot for save/load."""
        return {
            'level':         dict(self.level),
            'tonic':         dict(self.tonic),
            'phasic':        dict(self.phasic),
            'allostatic':    dict(self.allostatic),
            'baseline':      dict(self.baseline),
            'decay':         dict(self.decay),
            'cortisol_load': float(self.cortisol_load),
            'tick':          int(self._tick),
            'lexicon':       dict(self.lexicon),
            'd':             int(self._d) if self._d else None,
            'seed':          int(self._rng_seed),
        }

    def save(self, path):
        import json
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path):
        import json
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        inst = cls(lexicon=data.get('lexicon'),
                   baseline=data.get('baseline'),
                   decay_rate=data.get('decay'),
                   d=data.get('d'),
                   seed=data.get('seed', 14820))
        inst.level         = dict(data['level'])
        inst.tonic         = dict(data.get('tonic', dict(inst.baseline)))
        inst.phasic        = dict(data.get('phasic', {k: 0.0 for k in inst.baseline}))
        inst.allostatic    = dict(data.get('allostatic', dict(inst.baseline)))
        inst.cortisol_load = float(data.get('cortisol_load', 0.0))
        inst._tick         = int(data.get('tick', 0))
        return inst

    # ── action gating (refuse to act under crisis) ───────────────────────
    def should_act(self):
        """Return (ok, reason). False under sustained cortisol or extreme
        adenosine -- the organism should sleep / cool down first."""
        if self.forced_rest():
            return False, f'cortisol_load={self.cortisol_load:.3f} > 0.20 (sustained stress)'
        if self.level['adenosine'] > 0.85:
            return False, f'adenosine={self.level["adenosine"]:.3f} > 0.85 (exhaustion)'
        return True, 'ok'

    # ── allostatic state report ──────────────────────────────────────────
    def allostatic_drift(self):
        """Per-chemical drift between current allostatic (slow EMA) and baseline.
        Positive = chronically elevated; negative = chronically depressed."""
        return {chem: self.allostatic[chem] - self.baseline[chem]
                for chem in self.baseline}
