"""
ikigai.cognition.kg_extract -- Pack 292 knowledge-graph extraction.

The organism builds its OWN knowledge graph from the teacher (R1).  Each
fact is an (entity, attribute, value) triple where the attribute is a
short noun -- which maps exactly onto the organism's existing queryable
form: "what is the <attribute> of <entity>" -> <value>.  So an extracted
triple is absorbed straight into the Pack 273 anchor-action cache and is
immediately answerable via general_reasoner.reason() AND composable via
Pack 293 multi-hop.  No new substrate primitive -- the KG IS the cache,
phrased as questions.

Pipeline:
    1. ask the teacher for triples about an entity (parseable format),
    2. parse "attribute | value" lines into (entity, attr, value),
    3. populate_cache_from_text with the canonical query phrasings.
"""
import re


# attribute must be a short lowercase noun phrase; reject junk / verbs.
_ATTR_RE = re.compile(r"^[a-z][a-z \-]{1,24}$")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")

_EXTRACT_SYS = (
    'List key facts about the entity, one per line, in the EXACT format:\n'
    'attribute | value\n'
    'Rules: attribute is a short lowercase noun (capital, continent, '
    'currency, language, population, region). value is the answer only. '
    'No sentences, no commentary, no numbering.\n')


class KGExtractor:
    """Pack 292 -- teacher-driven knowledge-graph builder.

    Construct with an organism and a RemoteLLMTeacher.  `extract` returns
    triples; `absorb` writes them into the anchor-action cache; `learn`
    does both for a list of entities.
    """

    def __init__(self, org, teacher, max_facts=8):
        self.org = org
        self.cat4 = getattr(org, 'cat4', None)
        self.teacher = teacher
        self.max_facts = int(max_facts)
        self.stats = {'queried': 0, 'triples': 0, 'absorbed': 0,
                       'rejected': 0}

    # ---- extraction --------------------------------------------------

    def _clean(self, s):
        return ' '.join(_WORD_RE.findall(str(s).strip().lower()))

    def _parse(self, entity, text):
        """Parse bullet/`attribute: value` (or `|`) lines into
        (entity, attr, value).  R1 emits '- capital: Paris' style; it
        also dumps essay-length list values ('famous works: ...') which
        we reject -- a KG value must be atomic."""
        ent = self._clean(entity)
        triples = []
        seen = set()
        for raw in str(text).splitlines():
            line = re.sub(r'^[\-\*•\d\.\)\s]+', '', raw.strip())
            if '|' in line:
                left, _, right = line.partition('|')
            elif ':' in line:
                left, _, right = line.partition(':')
            else:
                continue
            attr = self._clean(left)
            # drop leading filler in the value before cleaning
            right_l = re.sub(r'^\s*(approximately|about|around|roughly)\s+',
                              '', right.strip(), flags=re.IGNORECASE)
            val = self._clean(right_l)
            if not attr or not val:
                self.stats['rejected'] += 1
                continue
            if not _ATTR_RE.match(attr) or len(attr.split()) > 3:
                self.stats['rejected'] += 1
                continue
            # atomic value only: no lists / sentences
            if ',' in right or len(val.split()) > 4:
                self.stats['rejected'] += 1
                continue
            if val == ent or attr == ent:        # degenerate echo
                self.stats['rejected'] += 1
                continue
            key = (attr, val)
            if key in seen:
                continue
            seen.add(key)
            triples.append((ent, attr, val))
            if len(triples) >= self.max_facts:
                break
        return triples

    def extract(self, entity):
        """Query the teacher for triples about one entity."""
        prompt = f'{_EXTRACT_SYS}Entity: {entity}\nFacts:\n'
        saved_n = self.teacher.max_new_tokens
        try:
            self.teacher.max_new_tokens = 256
            data = self.teacher._batch([prompt])
        finally:
            self.teacher.max_new_tokens = saved_n
        self.stats['queried'] += 1
        if not data or 'items' not in data or not data['items']:
            return []
        text = self.teacher._post(data['items'][0].get('text', ''))
        triples = self._parse(entity, text)
        self.stats['triples'] += len(triples)
        return triples

    # ---- absorb into the queryable cache -----------------------------

    def absorb(self, triples):
        """Write triples into the anchor-action cache as the canonical
        'what is the <attr> of <entity>' -> <value> query form."""
        if self.cat4 is None:
            return 0
        added = 0
        for subj, attr, val in triples:
            for chain in (
                f'What is the {attr} of {subj}\n\n{val}\n\n',
                f'The {attr} of {subj} is\n\n{val}\n\n',
            ):
                added += self.cat4.populate_cache_from_text(chain)
        self.stats['absorbed'] += added
        return added

    def learn(self, entities):
        """Extract + absorb for a list of entities.  Returns the full
        triple list (also written to the cache)."""
        all_triples = []
        for ent in entities:
            triples = self.extract(ent)
            self.absorb(triples)
            all_triples.extend(triples)
        return all_triples
