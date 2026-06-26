"""
ikigai.cognition.taxonomic_grounding -- Channel 4: IS-A relations.

Day 56 Pack 100 -- captures DIRECTED taxonomic meaning.

Co-occurrence makes cat ~ dog (siblings).
Taxonomy makes cat -> animal (parent-child).

3 mechanisms combined:
    1. Hearst pattern extraction from text (NLP classic)
        "X is a Y", "X are Ys", "Ys such as X", "X and other Ys"
    2. Directed Hebbian drift: hypo moves big toward hyper, hyper moves small toward hypo
    3. Triple binding in graph for explicit query: hypernym_of('cat') -> 'animal'

Filters (avoid spurious pairs):
    - Hypernym must not end in -ing, -ed (verb forms)
    - Must not be property-seed word (red, sad, heavy, etc.)
    - Must not be stopword
    - Must not be number
    - Must not be a known verb (from Pack 97 OperationalGrounding)
"""

import re
from collections import Counter, defaultdict

import numpy as np


# ── filters ──────────────────────────────────────────────────────────────────

_STOPWORDS_HYPER = {
    'a', 'an', 'the', 'and', 'or', 'but', 'so', 'if', 'then', 'now', 'at',
    'in', 'on', 'of', 'to', 'from', 'for', 'with', 'by', 'as', 'over',
    'under', 'about', 'into', 'onto', 'this', 'that', 'these', 'those',
    # Presentative / story-opener words (fire on "Once there was a X")
    'there', 'here', 'once', 'where', 'when', 'always', 'never', 'often',
    'someone', 'everyone', 'nobody', 'anybody', 'whoever', 'whatever',
    'he', 'she', 'it', 'they', 'we', 'i', 'you', 'them', 'us',
    'her', 'his', 'its', 'their', 'our', 'my', 'your',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
    'can', 'could', 'may', 'might', 'must', 'shall',
    'good', 'bad', 'old', 'new', 'best', 'better', 'worse',
    'one', 'two', 'three', 'four', 'five', 'first', 'second', 'last',
    'all', 'some', 'any', 'each', 'every', 'no', 'not',
    'more', 'less', 'most', 'least', 'very',
    # Meta-words (false positives in "X is a kind of Y")
    'kind', 'type', 'sort', 'way', 'piece', 'bit', 'part', 'lot',
    'same', 'thing', 'one', 'place', 'time', 'side', 'end', 'top',
    'bottom', 'name', 'word', 'idea', 'fact', 'reason', 'rest',
    'half', 'whole', 'few', 'many', 'much', 'such', 'lots',
    'self', 'while', 'matter', 'case', 'point',
    # Standalone adjectives without suffix markers (common false hypernyms)
    'deep', 'high', 'low', 'long', 'short', 'wide', 'strong', 'great',
    'real', 'main', 'certain', 'full', 'open', 'free', 'safe', 'true',
    'right', 'wrong', 'hard', 'soft', 'warm', 'cool', 'dark', 'bright',
    'excellent', 'perfect', 'special', 'common', 'normal', 'simple',
    'different', 'similar', 'important', 'popular', 'difficult', 'easy',
    'sure', 'clear', 'close', 'far', 'near', 'late', 'early', 'next',
    'ready', 'afraid', 'alone', 'alive', 'aware', 'able', 'okay', 'fine',
}


# Unambiguous adjective endings: safe to reject as hypernyms
# Keep conservative -- 'al','ant','ent','er' also appear in nouns (animal, plant, river)
_ADJ_SUFFIXES = (
    'ous', 'ious', 'eous', 'uous',  # dangerous, glorious, gorgeous
    'ful',                           # hopeful, beautiful
    'less',                          # careless, useless
    'est',                           # strongest, oldest (superlatives)
    'ish',                           # reddish, childish
    'able', 'ible',                  # capable, visible
)


def _lemmatize(word):
    """Singularize: cats->cat. Keeps adjective-suffix words intact."""
    if len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
        if word == 'is':
            return word
        # Check ORIGINAL word for adjective suffix (not stripped form)
        # e.g. "dangerous".endswith('ous') -> True -> keep as-is
        if any(word.endswith(suf) for suf in _ADJ_SUFFIXES):
            return word     # adjective filter will reject it downstream
        return word[:-1]
    return word

# Property seeds (from Pack 98 SensoryGrounding) -- NOT hypernyms
_PROPERTY_SEEDS = {
    'red', 'crimson', 'scarlet', 'rose', 'blue', 'azure', 'navy', 'cyan',
    'green', 'emerald', 'lime', 'verdant', 'yellow', 'gold', 'golden', 'amber',
    'black', 'dark', 'ebony', 'obsidian', 'white', 'snow', 'ivory', 'pale',
    'heavy', 'weighty', 'massive', 'dense', 'light', 'feather', 'airy',
    'happy', 'joyful', 'glad', 'cheerful', 'smiling', 'laughing',
    'sad', 'crying', 'tears', 'sorrow', 'mournful', 'weeping',
    'angry', 'furious', 'rage', 'mad', 'irate',
    'afraid', 'scared', 'fear', 'terrified', 'frightened',
    'hot', 'warm', 'burning', 'scorching', 'fire',
    'cold', 'freezing', 'icy', 'frosty', 'chilly',
    'big', 'large', 'huge', 'giant', 'enormous', 'vast',
    'small', 'tiny', 'little', 'mini', 'wee',
    'loud', 'booming', 'thunderous', 'roaring',
    'quiet', 'silent', 'whisper', 'hushed',
    'sweet', 'sugary', 'honey', 'syrupy',
    'bitter', 'sour', 'tart',
    'fast', 'quick', 'rapid', 'speedy', 'swift',
    'slow', 'sluggish', 'crawling',
    'pretty', 'beautiful', 'ugly', 'nice',
    'shiny', 'bright', 'dim',
    'clear', 'clean', 'dirty',
}


def _is_verb_form(word):
    """Reject -ing and -ed forms."""
    if len(word) <= 3:
        return False
    return word.endswith('ing') or word.endswith('ed')


def _is_adjective(word):
    """Reject clear adjective forms — not valid hypernyms."""
    for suf in _ADJ_SUFFIXES:
        if word.endswith(suf):
            # 'ish' only counts for longer words: fish/dish/wish are nouns
            if suf == 'ish' and len(word) < 6:
                continue
            return True
    # Truncated adjective stems: regex patterns capture (\w+) before literal 's',
    # so "dangerous" -> m.group = "dangerou" (ends in 'ou', not 'ous').
    # No valid English noun longer than 4 chars ends in bare 'ou'.
    if len(word) > 4 and word.endswith('ou'):
        return True
    return False


def _looks_like_hypernym(word, extra_excluded=None):
    """Filter for hypernym validity."""
    if not word or len(word) < 3:
        return False
    if word in _STOPWORDS_HYPER:
        return False
    if word in _PROPERTY_SEEDS:
        return False
    if _is_verb_form(word):
        return False
    if _is_adjective(word):
        return False
    if word.isdigit():
        return False
    if extra_excluded and word in extra_excluded:
        return False
    return True


# ── Hearst pattern definitions ────────────────────────────────────────────────

# Each pattern: (regex, (hypo_group, hyper_group), name)
# hypo_group / hyper_group: 1-indexed match group positions
HEARST_PATTERNS = [
    # "X is a/an/the Y"
    (re.compile(r'\b(\w+)\s+(?:is|was)\s+(?:a|an|the)\s+(\w+)\b', re.IGNORECASE),
     (1, 2), 'is-a'),
    # "Xs are Ys" (with plurality marker on hypernym)
    (re.compile(r'\b(\w+)s\s+are\s+(\w+)s\b', re.IGNORECASE),
     (1, 2), 'plural-are'),
    # "X are Ys"
    (re.compile(r'\b(\w+)\s+are\s+(\w+)s\b', re.IGNORECASE),
     (1, 2), 'are-Ys'),
    # "Ys such as X"
    (re.compile(r'\b(\w+)s\s+such\s+as\s+(\w+)\b', re.IGNORECASE),
     (2, 1), 'such-as'),
    # "Ys including X"
    (re.compile(r'\b(\w+)s\s+including\s+(\w+)\b', re.IGNORECASE),
     (2, 1), 'including'),
    # "X and other Ys"
    (re.compile(r'\b(\w+)\s+and\s+other\s+(\w+)s?\b', re.IGNORECASE),
     (1, 2), 'and-other'),
    # "kind of Y" -- "cat is a kind of animal"
    (re.compile(r'\b(\w+)\s+(?:is|was)\s+(?:a|an)\s+kind\s+of\s+(\w+)\b', re.IGNORECASE),
     (1, 2), 'kind-of'),
    # "type of Y"
    (re.compile(r'\b(\w+)\s+(?:is|was)\s+(?:a|an)\s+type\s+of\s+(\w+)\b', re.IGNORECASE),
     (1, 2), 'type-of'),
]


def _renorm(hv):
    mags = np.abs(hv)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (hv / mags).astype(np.complex64)


# ── TaxonomicGrounding class ─────────────────────────────────────────────────

class TaxonomicGrounding:
    """
    Channel 4: IS-A taxonomic grounding.

    extract_pairs(text)
        Apply Hearst patterns. Return list of (hyponym, hypernym, pattern_name).

    expose(text, lexicon, drift_rate, hyper_back_rate)
        Extract pairs + drift lexicon: hypo toward hyper (big), hyper toward hypo (small).

    hypernym_of(word)
        Most-frequently-asserted hypernym for word.

    hyponyms_of(word)
        List of recorded hyponyms of word.

    is_a(hypo, hyper, transitive=True)
        Check if hypo IS-A hyper (direct or transitive via graph).

    chain_to_root(word)
        Returns list [word, parent, grandparent, ...] up the IS-A tree.
    """

    def __init__(self, d=2048, excluded_hypernyms=None):
        self.d = int(d)
        self._extra_excluded = set(excluded_hypernyms or [])
        # Pair graph
        self._pair_counts = Counter()       # (hypo, hyper) -> count
        self._hyponyms    = defaultdict(set)   # hyper -> {hypo, hypo, ...}
        self._hypernyms   = defaultdict(Counter)  # hypo -> {hyper: count}
        self._pattern_hits = Counter()      # pattern_name -> count

    # ── extraction ────────────────────────────────────────────────────────

    def extract_pairs(self, text):
        """
        Apply all Hearst patterns. Return list of (hypo, hyper, pattern).
        Pairs with invalid hypernym are filtered.
        """
        text_lower = text.lower()
        out = []
        # Run "kind-of" / "type-of" patterns FIRST so generic "is-a" doesn't grab "kind"
        ordered_patterns = sorted(HEARST_PATTERNS,
                                  key=lambda p: 0 if p[2] in ('kind-of', 'type-of') else 1)
        for pat, (hpos, ypos), name in ordered_patterns:
            for m in pat.finditer(text_lower):
                hypo  = _lemmatize(m.group(hpos).lower())
                hyper = _lemmatize(m.group(ypos).lower())
                if not _looks_like_hypernym(hyper, self._extra_excluded):
                    continue
                if hypo == hyper:
                    continue
                if hypo in _STOPWORDS_HYPER:
                    continue
                out.append((hypo, hyper, name))
        return out

    # ── exposure ──────────────────────────────────────────────────────────

    def expose(self, text, lexicon, drift_rate=0.3, hyper_back_rate=0.05):
        """
        Extract IS-A pairs from text. Drift lexicon: hypo->hyper big, hyper->hypo small.
        Mints missing words.
        """
        pairs = self.extract_pairs(text)
        for hypo, hyper, name in pairs:
            # Record graph
            self._pair_counts[(hypo, hyper)] += 1
            self._hyponyms[hyper].add(hypo)
            self._hypernyms[hypo][hyper] += 1
            self._pattern_hits[name] += 1

            # Drift HVs (must be in lexicon to drift; mint on demand)
            if hypo not in lexicon or hyper not in lexicon:
                continue
            lexicon[hypo] = _renorm(lexicon[hypo] + drift_rate * lexicon[hyper])
            lexicon[hyper] = _renorm(lexicon[hyper] + hyper_back_rate * lexicon[hypo])

        return pairs

    # ── queries ───────────────────────────────────────────────────────────

    def hypernym_of(self, word):
        """Most-asserted hypernym for word, or None."""
        if word not in self._hypernyms:
            return None
        c = self._hypernyms[word]
        if not c:
            return None
        return c.most_common(1)[0][0]

    def hyponyms_of(self, word):
        return list(self._hyponyms.get(word, set()))

    def is_a(self, hypo, hyper, transitive=True, max_depth=10):
        """Check if hypo IS-A hyper. Walk up chain if transitive."""
        if hypo == hyper:
            return True
        if hypo not in self._hypernyms:
            return False
        # Direct check
        if hyper in self._hypernyms[hypo]:
            return True
        if not transitive:
            return False
        # Walk up
        current = hypo
        for _ in range(max_depth):
            parent = self.hypernym_of(current)
            if parent is None:
                return False
            if parent == hyper:
                return True
            current = parent
        return False

    def chain_to_root(self, word, max_depth=10):
        """Return [word, parent, grandparent, ...] up to depth."""
        chain = [word]
        current = word
        seen = {word}
        for _ in range(max_depth):
            parent = self.hypernym_of(current)
            if parent is None or parent in seen:
                break
            chain.append(parent)
            seen.add(parent)
            current = parent
        return chain

    # ── introspection ─────────────────────────────────────────────────────

    @property
    def n_pairs(self):
        return len(self._pair_counts)

    def all_pairs(self):
        return list(self._pair_counts.items())

    def stats(self):
        return {
            'n_pairs':         self.n_pairs,
            'n_hypernyms':     len(self._hyponyms),
            'n_hyponyms':      len(self._hypernyms),
            'pattern_hits':    dict(self._pattern_hits),
        }
