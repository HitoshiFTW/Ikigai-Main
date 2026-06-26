"""
ikigai.cognition.kg_ingest -- Pack 327: knowledge-graph dump parsers.

Thin adapters that turn a downloaded KG dump into a stream of
(subject, relation, object) string triples ready for
IkigaiOrganism.ingest_triples (Pack 326).  No curated relation lists -- the
generic template handles any predicate.

Currently: ConceptNet 5 assertions CSV.  Wikidata N-Triples + DBpedia are
stubbed with the same generator contract for when those dumps land.

ConceptNet 5.7 assertions format (tab-separated, 5 columns):
    /a/[/r/Rel/,/c/la/start/,/c/la/end/]  /r/Rel  /c/la/start/pos  /c/la/end  {json}
e.g.
    /a/[/r/IsA/,...]  /r/IsA  /c/en/cat/n  /c/en/mammal  {"weight": 2.0}
We keep English concepts (/c/en/...), strip the part-of-speech / sense
suffixes, turn underscores into spaces, lowercase the relation.  Download:
github.com/commonsense/conceptnet5/wiki/Downloads
"""

import gzip
import io


def _quick_weight(meta):
    """Cheap weight extract from the ConceptNet meta JSON string without a full
    json.loads (the per-line bottleneck). Returns 1.0 if not found."""
    i = meta.find('"weight":')
    if i < 0:
        return 1.0
    j = i + 9
    k = j
    n = len(meta)
    while k < n and meta[k] in ' \t-+0123456789.eE':
        k += 1
    try:
        return float(meta[j:k].strip())
    except ValueError:
        return 1.0


def clean_relation(rel_uri):
    """'/r/IsA' -> 'isa'; '/r/HasProperty' -> 'hasproperty'.  Lowercased,
    prefix stripped.  Returns None if not a /r/ relation URI."""
    if not rel_uri or not rel_uri.startswith('/r/'):
        return None
    return rel_uri.split('/')[2].lower()


def clean_concept(uri, lang='en'):
    """'/c/en/ice_cream/n' -> 'ice cream'; '/c/en/cat' -> 'cat'.  Returns
    None if the concept is not in `lang` (so non-English edges drop)."""
    if not uri or not uri.startswith('/c/'):
        return None
    parts = uri.split('/')          # ['', 'c', 'en', 'ice_cream', 'n', ...]
    if len(parts) < 4 or parts[2] != lang:
        return None
    term = parts[3].replace('_', ' ').strip().lower()
    return term or None


def parse_conceptnet(path, relations=None, min_weight=1.0, lang='en',
                     limit=None, skip_self=True):
    """Yield (subject, relation, object) triples from a ConceptNet assertions
    CSV (plain or .gz).

    relations  -- optional set of cleaned relation names to keep (e.g.
                  {'isa','partof','usedfor'}); None keeps all.
    min_weight -- drop edges below this confidence weight (ConceptNet meta).
    lang       -- concept language to keep (both endpoints must match).
    limit      -- stop after this many yielded triples (None = all).
    skip_self  -- drop edges whose subject == object.
    """
    rel_filter = set(relations) if relations else None
    opener = gzip.open if str(path).endswith('.gz') else open
    n = 0
    with opener(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 4:
                continue
            rel = clean_relation(cols[1])
            if rel is None or (rel_filter is not None and rel not in rel_filter):
                continue
            subj = clean_concept(cols[2], lang)
            obj = clean_concept(cols[3], lang)
            if not subj or not obj:
                continue
            if skip_self and subj == obj:
                continue
            if min_weight and len(cols) >= 5:
                if _quick_weight(cols[4]) < min_weight:
                    continue
            yield (subj, rel, obj)
            n += 1
            if limit and n >= limit:
                return


def parse_ntriples(path, label_map=None, limit=None):
    """Yield (subject, predicate, object) from an N-Triples dump (Wikidata
    truthy / DBpedia).  Each line: <s> <p> <o> .  If `label_map` (Qid->label
    dict) is given, resolve entity URIs to labels.  Minimal -- expand when the
    dump lands.  Stub kept to the same generator contract as parse_conceptnet.
    """
    def _term(tok, last_segment=True):
        tok = tok.strip()
        if tok.startswith('<') and tok.endswith('>'):
            uri = tok[1:-1]
            seg = uri.rstrip('/').split('/')[-1]
            if label_map and seg in label_map:
                return label_map[seg]
            return seg if last_segment else uri
        if tok.startswith('"'):
            end = tok.rfind('"')
            return tok[1:end] if end > 0 else tok.strip('"')
        return tok
    opener = gzip.open if str(path).endswith('.gz') else open
    n = 0
    with opener(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or not line.endswith('.'):
                continue
            parts = line[:-1].strip().split(' ', 2)
            if len(parts) < 3:
                continue
            s = _term(parts[0]); p = _term(parts[1])
            o = _term(parts[2].rstrip(' .'))
            if not (s and p and o):
                continue
            yield (str(s).lower(), str(p).lower(), str(o).lower())
            n += 1
            if limit and n >= limit:
                return
