"""
ikigai.cognition.language_teach -- Pack 300.1 Language Template Discovery.

"Teach the teacher to teach LANGUAGE."  Instead of authoring sentence
frames (Pack 300 v0) or absorbing junk n-grams, the organism LEARNS how
to speak a fact by INDUCTION from teacher demonstrations -- the same move
as 304->305 (authored rules -> discovered rules):

    teacher demos (oracle supplies the sentences):
        capital, france, paris  ->  "the capital of france is paris"
        capital, japan,  tokyo  ->  "the capital of japan is tokyo"
    organism (anti-unification, Plotkin LGG via schema_inducer):
        replace slot VALUES with named placeholders, then anti-unify:
        "the capital of {SUBJ} is {VAL}"   (learned template)
    apply to ANY capital fact -> grammatical sentence, NO authored frame.

Banks (Prince: not everything in self bank): learned templates live in
`organism._lang_templates` (persisted), NOT b_self.  Slot fillers are read
from the fact cache.  b_self is never written.

Honesty: this is LEARNED generation by imitation+generalisation, grammar
emerges from the teacher's structure (clean only as far as the teacher is
consistent -- low-specificity inductions are rejected).  Open-ended free
fluency (novel sentences beyond learned templates) is the later n-gram
arc.
"""
import re

from ikigai.cognition.schema_inducer import (
    SchemaInducer, SLOT, apply_schema, schema_specificity)

_SUBJ = '\x00subj\x00'
_VAL = '\x00val\x00'


def _toks(s):
    return [t for t in re.sub(r"[^a-z0-9'\s]", ' ', str(s).lower()).split() if t]


def _placeholder(sent_toks, value, ph):
    """Replace the contiguous run of `value` tokens in sent_toks with the
    single placeholder token `ph`.  Returns new list or None if not found."""
    vt = _toks(value)
    if not vt:
        return sent_toks
    n = len(vt)
    for i in range(len(sent_toks) - n + 1):
        if sent_toks[i:i + n] == vt:
            return sent_toks[:i] + [ph] + sent_toks[i + n:]
    return None


class LanguageTeacher:
    """Induce sentence templates per query-type from teacher demos."""

    def __init__(self, reasoner, oracle_teacher):
        self.gr = reasoner
        self.teacher = oracle_teacher          # RemoteLLMTeacher (bigger max_tokens)
        self.inducer = SchemaInducer()
        self.templates = {}                     # qtype -> schema (list w/ SLOT)

    # ---- demonstrate one fact -> sentence, placeholder-ise -----------

    def _demo_sentence(self, subj, val):
        """Ask the oracle to state a fact as one short sentence."""
        prompt = (f'State this fact as one short, simple sentence. '
                  f'Use the exact words "{subj}" and "{val}". '
                  f'Fact: the answer is {val} for {subj}. Sentence:')
        try:
            data = self.teacher._batch([prompt])
            txt = data['items'][0]['text'] if data else ''
        except Exception:
            return None
        txt = re.sub(r'<think>.*?</think>', ' ', txt, flags=re.DOTALL | re.I)
        for line in txt.splitlines():
            line = line.strip()
            if subj.lower() in line.lower() and val.lower() in line.lower():
                # cut R1 CoT ramble that trails the answer (same leak the
                # 287.5 parser handles): keep up to the first sentence end
                # or ramble marker, as long as subj+val still present.
                cut = re.split(r'(?:\.\s|\bwait\b|\bbut\b|\bthat\'?s\b|'
                               r'\blet me\b|\bactually\b|\bhmm\b|\bso\b|'
                               r'\bhowever\b|\bnote\b)', line, maxsplit=1,
                               flags=re.IGNORECASE)[0].strip()
                if (subj.lower() in cut.lower()
                        and val.lower() in cut.lower()):
                    return cut
                return line
        return None

    def teach(self, qtype, demos, min_examples=2, min_specificity=0.4,
              verbose=False):
        """demos: list of (subj, val) for one query-type.  Oracle states
        each as a sentence; we placeholder-ise subj/val and observe the
        templated token list.  Induce -> store template if specific
        enough.  Returns the learned template or None."""
        for subj, val in demos:
            if not str(val or '').strip():
                continue
            sent = self._demo_sentence(subj, val)
            if not sent:
                continue
            st = _toks(sent)
            st = _placeholder(st, val, _VAL)
            if st is None:
                continue
            st = _placeholder(st, subj, _SUBJ)
            if st is None:
                continue
            self.inducer.observe(qtype, st)
            if verbose:
                printable = ' '.join(
                    '{SUBJ}' if t == _SUBJ else '{VAL}' if t == _VAL else t
                    for t in st)
                print(f'    demo[{qtype}] {subj}/{val}: {printable}')
        schema = self.inducer.induce(qtype, min_examples=min_examples)
        if schema is None:
            return None
        spec = schema_specificity(schema)
        # require both placeholders survived + decent specificity
        if (_SUBJ not in schema or _VAL not in schema
                or spec < min_specificity):
            if verbose:
                print(f'    [{qtype}] REJECT (spec={spec:.2f}, '
                      f'subj={_SUBJ in schema}, val={_VAL in schema})')
            return None
        self.templates[qtype] = schema
        if verbose:
            print(f'    [{qtype}] LEARNED (spec={spec:.2f}, support='
                  f'{self.inducer.support(qtype)}): {self.render(qtype)!r}')
        return schema

    # ---- apply: generate a sentence from a learned template ----------

    def render(self, qtype, subj='{SUBJ}', val='{VAL}'):
        sch = self.templates.get(qtype)
        if not sch:
            return None
        # structural-only trim: drop trailing SLOTs (unfilled positions --
        # language-agnostic, no English word list).  Everything else is the
        # teacher's OWN learned structure.
        while sch and sch[-1] is SLOT:
            sch = sch[:-1]
        out = []
        for t in sch:
            if t is SLOT:
                continue
            out.append(subj if t == _SUBJ else val if t == _VAL else t)
        s = ' '.join(out).strip()
        return s[:1].upper() + s[1:] + '.' if s else None

    def specificity(self, qtype):
        """Fraction of fixed (non-SLOT) tokens in the learned template --
        the induction-quality / grammaticality proxy (the teacher's own
        structure is the grammar authority; no hardcoded word lists)."""
        sch = self.templates.get(qtype)
        return schema_specificity(sch) if sch else 0.0

    def say(self, qtype, subj, val):
        """Generate a grammatical sentence for a NEW fact via the learned
        template.  None if the type wasn't learned."""
        return self.render(qtype, subj=str(subj), val=str(val))

    # ---- persistence (serialisable; SLOT/placeholders survive) -------

    def to_state(self):
        # SLOT (None) -> '\x00slot\x00' marker for json/pickle safety
        ser = {}
        for q, sch in self.templates.items():
            ser[q] = ['\x00slot\x00' if t is SLOT else t for t in sch]
        return ser

    def load_state(self, ser):
        if not isinstance(ser, dict):
            return
        for q, sch in ser.items():
            self.templates[q] = [SLOT if t == '\x00slot\x00' else t
                                 for t in sch]
