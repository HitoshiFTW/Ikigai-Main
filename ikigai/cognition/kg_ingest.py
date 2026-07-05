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
    if str(path).endswith('.bz2'):
        import bz2
        opener = bz2.open
    elif str(path).endswith('.gz'):
        opener = gzip.open
    else:
        opener = open
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


def parse_ntriples(path, label_map=None, limit=None, obj_entity_only=False,
                   pred_allow=None, progress_every=0):
    """Yield (subject, predicate, object) from an N-Triples dump (Wikidata
    truthy / DBpedia).  Each line: <s> <p> <o> .  If `label_map` (Qid->label
    dict) is given, resolve entity URIs to labels.

    Raw Wikidata truthy is ~90% noise for the derive kernel: every item carries
    ~900 predicates (labels, descriptions, sitelinks, external IDs in every
    language) whose objects are literals, not entities.  Two filters keep only
    the STRUCTURAL backbone -- the part the derive-not-store kernel multiplies:
      obj_entity_only=True : keep a triple only if its object is a <URI> (an
                             entity), dropping every literal/label/ext-id.  This
                             collapses thousands of junk relations to the tens of
                             real entity->entity properties (p31, p279, ...).
      pred_allow={...}     : additionally restrict the predicate's last URI
                             segment to a set (e.g. {'p279','p31'} for the pure
                             subclass/instance taxonomic backbone).
    `limit` counts EMITTED (post-filter) triples, so a backbone run reads as far
    into the dump as needed to collect `limit` real edges.  progress_every>0
    prints a heartbeat (read/kept/rate) to stderr every N read lines.
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
    if str(path).endswith('.bz2'):
        import bz2
        opener = bz2.open
    elif str(path).endswith('.gz'):
        opener = gzip.open
    else:
        opener = open
    import sys as _sys, time as _time, re as _re
    allow = set(pred_allow) if pred_allow else None
    _qid = _re.compile(r'^q\d+$')          # a Wikidata entity object (Q-id)
    n = 0; read = 0; t0 = _time.time()
    with opener(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            read += 1
            if progress_every and read % progress_every == 0:
                dt = _time.time() - t0
                _sys.stderr.write(
                    f'[parse] read {read:,} kept {n:,} '
                    f'({read/max(dt,1e-9):,.0f} lines/s) {dt:.0f}s\n')
                _sys.stderr.flush()
            line = line.strip()
            if not line or line.startswith('#') or not line.endswith('.'):
                continue
            parts = line[:-1].strip().split(' ', 2)
            if len(parts) < 3:
                continue
            otok = parts[2].rstrip(' .').strip()
            if obj_entity_only and not otok.startswith('<'):
                continue                       # object is a literal -> skip
            p = _term(parts[1]).lower()
            if allow is not None and p not in allow:
                continue
            s = _term(parts[0]); o = _term(otok)
            if not (s and p and o):
                continue
            o = str(o).lower()
            if obj_entity_only and not _qid.match(o):
                continue                       # object URI is not a Q-entity
            yield (str(s).lower(), str(p).lower(), o)
            n += 1
            if limit and n >= limit:
                return
