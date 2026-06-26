"""
ikigai.cognition.sentence_frame -- Pack 300 Generation v0 (schema-framed).

The organism KNOWS facts (capital of France = Paris) but answered with one
word.  This frames the known answer as a GRAMMATICAL SENTENCE by
transforming the query structure into a declarative and slotting the
substrate-derived answer -- grammatical BY CONSTRUCTION (subject-verb-
object), not by free n-gram sampling (which is degenerate on the current
language substrate; free-LM fluency is the 300.1 follow-on arc).

Honesty: the transforms here are GRAMMAR (interrogative->declarative
rewrites), the same class as the relation phrasings and 293 templates --
no fact is encoded; subject/object/answer all come from the query + the
substrate's answer.

A well-formed sentence = capitalised start, subject + verb + object,
answer present, terminal period.  `is_grammatical` validates that shape
(the b_gram FSM check is the substrate-side validator, noted for v1).
"""
import re


_NUM_OP = re.compile(
    r'^\s*what\s+is\s+(.+?\s+(?:plus|minus|times|divided\s+by)\s+.+?)\s*\??\s*$',
    re.IGNORECASE)
_WHAT_OF = re.compile(
    r'^\s*what\s+is\s+the\s+(.+?)\s+of\s+(.+?)\s*\??\s*$', re.IGNORECASE)
_WHAT_X_IN = re.compile(
    r'^\s*(?:what|which)\s+(\w+)\s+is\s+(.+?)\s+in\s*\??\s*$', re.IGNORECASE)
_IS_IN = re.compile(
    r'^\s*is\s+(.+?)\s+in\s+(.+?)\s*\??\s*$', re.IGNORECASE)
_SAME = re.compile(
    r'^\s*(?:do|does|are|is)\s+(.+?)\s+and\s+(.+?)\s+'
    r'(in|on|have|share|use|speak)\s+the\s+same\s+(\w+)\s*\??\s*$',
    re.IGNORECASE)
_BIGGER = re.compile(
    r'^\s*is\s+(.+?)\s+(bigger|larger|smaller)\s+than\s+(.+?)\s*\??\s*$',
    re.IGNORECASE)


def _cap(s):
    s = s.strip()
    return s[:1].upper() + s[1:] if s else s


def _clean(s):
    return re.sub(r'\s+', ' ', str(s).strip().rstrip('?.')).strip()


class SentenceFramer:
    """Turn (query, answer) into one grammatical sentence."""

    def frame(self, query, answer, method=None):
        """Return a grammatical sentence stating the answer, or a safe
        fallback ('<Query>? <Answer>.') when no transform matches."""
        q = _clean(query)
        a = _clean(answer)
        if not a:
            return None
        al = a.lower()
        yesno = al in ('yes', 'no')

        # arithmetic: "what is 12 plus 7" + 19 -> "12 plus 7 is 19."
        m = _NUM_OP.match(q)
        if m:
            return _cap(f'{m.group(1)} is {a}.')

        # comparison: "do X and Y use the same currency" + yes/no
        m = _SAME.match(q)
        if m and yesno:
            x, y, verb, attr = (m.group(1), m.group(2),
                                m.group(3).lower(), m.group(4))
            v = {'in': 'are in', 'on': 'are on', 'have': 'have',
                 'share': 'share', 'use': 'use', 'speak': 'speak'}.get(verb, verb)
            if al == 'yes':
                return _cap(f'yes, {x} and {y} {v} the same {attr}.')
            neg = {'are in': 'are not in', 'are on': 'are not on',
                   'have': 'do not have', 'share': 'do not share',
                   'use': 'do not use', 'speak': 'do not speak'}.get(v, 'do not ' + v)
            return _cap(f'no, {x} and {y} {neg} the same {attr}.')

        # numeric: "is X bigger than Y" + yes/no
        m = _BIGGER.match(q)
        if m and yesno:
            x, cmp_, y = m.group(1), m.group(2).lower(), m.group(3)
            if al == 'yes':
                return _cap(f'yes, {x} is {cmp_} than {y}.')
            return _cap(f'no, {x} is not {cmp_} than {y}.')

        # yes/no membership: "is the capital of egypt in africa" + yes/no
        m = _IS_IN.match(q)
        if m and yesno:
            subj, obj = m.group(1), m.group(2)
            if al == 'yes':
                return _cap(f'yes, {subj} is in {obj}.')
            return _cap(f'no, {subj} is not in {obj}.')

        # "what continent is X in" + ans -> "X is in <ans>."
        m = _WHAT_X_IN.match(q)
        if m:
            subj = m.group(2)
            return _cap(f'{subj} is in {a}.')

        # "what is the X of Y" + ans -> "The X of Y is <ans>."
        m = _WHAT_OF.match(q)
        if m:
            rel, subj = m.group(1), m.group(2)
            return _cap(f'the {rel} of {subj} is {a}.')

        # generic yes/no
        if yesno:
            return _cap(f'{a}.')
        # fallback: keep it grammatical
        return _cap(f'the answer is {a}.')

    # ---- well-formedness gate ---------------------------------------

    @staticmethod
    def is_grammatical(sentence, answer=None):
        """Structural well-formedness: capitalised start, >=4 tokens, a
        verb present, terminal period, and (if given) the answer appears.
        The substrate b_gram frame-bigram FSM is the deeper validator
        (v1)."""
        if not sentence:
            return False
        s = sentence.strip()
        # valid start = capital letter OR a non-alpha token (e.g. a number
        # opening an arithmetic statement: "12 plus 7 is 19.")
        if not (s[:1].isupper() or not s[:1].isalpha()):
            return False
        if not s.endswith('.'):
            return False
        toks = s.rstrip('.').split()
        if len(toks) < 4:
            return False
        verbs = {'is', 'are', 'was', 'were', 'has', 'have', 'use', 'speak',
                 'share', 'do', 'does'}
        neg = {'not', "n't"}
        low = [t.lower() for t in toks]
        if not (set(low) & verbs):
            return False
        if answer:
            if answer.lower().rstrip('?.') not in s.lower():
                return False
        return True
