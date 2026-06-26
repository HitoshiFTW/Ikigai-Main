"""
ikigai.cognition.wordnet_absorb -- Pack 247z dictionary absorb (v2).

Day 72. WordNet -> substrate via mr.role_minter.observe. Lights up the
semantic + affordance + world + episodic + grammar banks that
hardcoded-role absorb left empty (Day 71 audit: 5/8 banks at 0 energy).

v2 changes vs v1:
  - Canonical label map: hyponym -> isa, synonym -> similar. Reuses
    existing DEFAULT_ROLES so we don't double semantic capacity.
  - New relations: examples (b_epi), derivationally_related_forms
    (b_gram), member_meronyms/holonyms, substance_meronyms/holonyms,
    similar_tos, pertainyms, also_sees, verb_groups, attribute,
    lexname -> class role.

Relations emitted (label after canonical map -> default bank):

    similar            b_sem  (was 'synonym'; lemma pairs in synset)
    isa                b_sem  (was 'hyponym'; head IS-A hypernym)
    holonym            b_sem  (head PART-OF holonyms)
    meronym            b_sem  (part -> whole)
    member             b_sem  (head MEMBER-OF member_holonyms;
                                  reversed for member_meronyms)
    substance          b_sem  (substance composition)
    pertainym          b_sem  (adj -> noun root, dental->tooth)
    also_see           b_sem  (related concept)
    antonym            b_sem  (existing role)
    definition         b_sem  (head -> first content word of gloss)
    attribute          b_sem  (noun <-> adj)
    similar_to         b_sem  (adjective network, mostly satellite adj)
    troponym           b_aff  (verb manner-of)
    entailment         b_aff  (verb entailment)
    verb_group         b_aff  (verb sense cluster)
    cause              b_world (existing role)
    class              b_sem  (head -> lexname domain, animal/food/...)
    derivation         b_gram  (run<->runner<->running, cross-POS)
    example            b_epi   (head -> first content word of example sentence)

Single-word lemmas only in v1+v2. Multi-word (Canis_familiaris) skipped.
"""

import re

_GLOSS_SPLIT = re.compile(r'[\s,;:\-\.\(\)]+')
_STOP = frozenset((
    'a', 'an', 'the', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with',
    'as', 'is', 'or', 'and', 'but', 'so', 'this', 'that', 'these',
    'those', 'be', 'are', 'was', 'were', 'been', 'being', 'has', 'have',
    'had', 'one', 'two', 'three', 'any', 'all', 'some', 'no', 'not',
    'such', 'which', 'who', 'whom', 'their', 'its', 'his', 'her', 'our',
))


def _clean_lemma(name):
    """WordNet lemma name -> single-token or None."""
    n = str(name).strip().lower()
    if '_' in n or ' ' in n:
        return None
    if not n.isalpha():
        return None
    if len(n) < 2 or len(n) > 20:
        return None
    return n


def _first_content_word(gloss):
    """Pick first non-stopword content token from a gloss string."""
    if not gloss:
        return None
    for tok in _GLOSS_SPLIT.split(gloss):
        t = tok.lower()
        if not t.isalpha():
            continue
        if t in _STOP:
            continue
        if 2 <= len(t) <= 20:
            return t
    return None


# Canonical label map: route incoming WordNet labels to existing
# DEFAULT_ROLES where semantically equivalent. Prevents minting a
# duplicate role (e.g. 'hyponym' alongside 'isa').
_CANONICAL = {
    'hyponym':  'isa',         # X --hyponym--> Y  means X IS-A Y
    'synonym':  'similar',     # synset lemma pairs
    'class':    'class',       # lexname domain category (already a role)
}


def _emit(rm, subj, label, obj, stats):
    if not subj or not obj or subj == obj:
        return
    label = _CANONICAL.get(label, label)
    mode = rm.observe(subj, label, obj)
    stats['triples'] += 1
    stats['by_label'][label] = stats['by_label'].get(label, 0) + 1
    stats['modes'][mode] += 1


def absorb_wordnet(mr, pos=None, max_synsets=None, vocab_filter=None,
                    verbose_every=2000):
    """
    Iterate WordNet synsets, emit triples through mr.role_minter.

    Args:
        mr             -- MultiRoleMemory (with role_minter wired)
        pos            -- None / 'n' / 'v' / 'a' / 'r' (POS filter)
        max_synsets    -- cap synsets visited (None = all)
        vocab_filter   -- set of allowed head-words (None = no filter)
        verbose_every  -- print progress every N synsets (0 = silent)

    Returns:
        stats dict with synsets / triples / by_label / modes / minted_labels
    """
    from nltk.corpus import wordnet as wn

    rm = mr.role_minter
    stats = {
        'synsets':       0,
        'triples':       0,
        'skipped_synsets': 0,
        'by_label':      {},
        'modes':         {'direct': 0, 'buffered': 0, 'minted': 0},
        'minted_labels': [],
    }

    # Pack 247y: import lazily so legacy callers w/o numpy/multirole present
    # don't break. MultiRoleMemory provides make_sense_hv classmethod.
    from ikigai.cognition.multirole_memory import MultiRoleMemory

    for synset in wn.all_synsets(pos):
        if max_synsets and stats['synsets'] >= max_synsets:
            break
        stats['synsets'] += 1

        lemmas = [_clean_lemma(l.name()) for l in synset.lemmas()]
        lemmas = [l for l in lemmas if l]
        if not lemmas:
            stats['skipped_synsets'] += 1
            continue

        head = lemmas[0]
        if vocab_filter is not None and head not in vocab_filter:
            stats['skipped_synsets'] += 1
            continue

        # Pack 247y SENSE DISAMBIGUATION: bind a per-synset frame HV so
        # writes for this sense land at distinct addresses from other
        # senses of the same lemma. (dog.n.01 hyponyms != dog.n.03 hyponyms.)
        sense_name = synset.name()
        sense_hv = MultiRoleMemory.make_sense_hv(sense_name, mr.d)
        mr.set_frame(sense_hv, frame_tag=sense_name)

        def _related_lemmas(syn):
            return (l for l in (_clean_lemma(x.name()) for x in syn.lemmas()) if l)

        # synonym -> 'similar' (canonical)
        if len(lemmas) > 1:
            for i, a in enumerate(lemmas):
                for b in lemmas[i+1:]:
                    _emit(rm, a, 'synonym', b, stats)

        # definition: head -> first content word of gloss
        defw = _first_content_word(synset.definition())
        if defw and defw != head:
            _emit(rm, head, 'definition', defw, stats)

        # hyponym -> 'isa' (canonical)
        for h in synset.hypernyms():
            for hl in _related_lemmas(h):
                _emit(rm, head, 'hyponym', hl, stats)

        # holonyms: part / member / substance
        for ho in synset.part_holonyms():
            for hl in _related_lemmas(ho):
                _emit(rm, head, 'holonym', hl, stats)
        for mh in synset.member_holonyms():
            for hl in _related_lemmas(mh):
                _emit(rm, head, 'member', hl, stats)
        for sh in synset.substance_holonyms():
            for hl in _related_lemmas(sh):
                _emit(rm, head, 'substance', hl, stats)

        # meronyms: part / member / substance (reversed, part -> whole).
        # Pack 247y: inner lemma IS subject, so frame by INNER synset.
        for pm in synset.part_meronyms():
            inner_hv = MultiRoleMemory.make_sense_hv(pm.name(), mr.d)
            with mr.in_frame(inner_hv, frame_tag=pm.name()):
                for pl in _related_lemmas(pm):
                    _emit(rm, pl, 'meronym', head, stats)
        for mm in synset.member_meronyms():
            inner_hv = MultiRoleMemory.make_sense_hv(mm.name(), mr.d)
            with mr.in_frame(inner_hv, frame_tag=mm.name()):
                for pl in _related_lemmas(mm):
                    _emit(rm, pl, 'member', head, stats)
        for sm in synset.substance_meronyms():
            inner_hv = MultiRoleMemory.make_sense_hv(sm.name(), mr.d)
            with mr.in_frame(inner_hv, frame_tag=sm.name()):
                for pl in _related_lemmas(sm):
                    _emit(rm, pl, 'substance', head, stats)

        # antonyms: lemma-level
        for lemma in synset.lemmas():
            a_subj = _clean_lemma(lemma.name())
            if not a_subj:
                continue
            for ant in lemma.antonyms():
                a_obj = _clean_lemma(ant.name())
                if a_obj:
                    _emit(rm, a_subj, 'antonym', a_obj, stats)
            # derivationally related forms (cross-POS)
            for d in lemma.derivationally_related_forms():
                d_obj = _clean_lemma(d.name())
                if d_obj:
                    _emit(rm, a_subj, 'derivation', d_obj, stats)
            # pertainyms (adj -> noun root)
            for p in lemma.pertainyms():
                p_obj = _clean_lemma(p.name())
                if p_obj:
                    _emit(rm, a_subj, 'pertainym', p_obj, stats)

        # also_sees + similar_tos (broad semantic adjacency)
        for asyn in synset.also_sees():
            for al in _related_lemmas(asyn):
                _emit(rm, head, 'also_see', al, stats)
        for sim in synset.similar_tos():
            for sl in _related_lemmas(sim):
                _emit(rm, head, 'similar_to', sl, stats)

        # verb-only relations
        if synset.pos() == 'v':
            for h in synset.hyponyms():
                # Pack 247y: troponym subject = INNER hyponym lemma, so frame
                # by INNER synset (not outer head's synset).
                inner_hv = MultiRoleMemory.make_sense_hv(h.name(), mr.d)
                with mr.in_frame(inner_hv, frame_tag=h.name()):
                    for tl in _related_lemmas(h):
                        _emit(rm, tl, 'troponym', head, stats)
            for e in synset.entailments():
                for el in _related_lemmas(e):
                    _emit(rm, head, 'entailment', el, stats)
            for c in synset.causes():
                for cl in _related_lemmas(c):
                    _emit(rm, head, 'cause', cl, stats)
            for vg in synset.verb_groups():
                for vl in _related_lemmas(vg):
                    _emit(rm, head, 'verb_group', vl, stats)

        # attribute (noun <-> adj)
        for attr in synset.attributes():
            for al in _related_lemmas(attr):
                _emit(rm, head, 'attribute', al, stats)

        # lexname domain category (animal, food, plant, ...) -> existing
        # 'class' role
        try:
            lex = synset.lexname()
        except Exception:
            lex = None
        if lex:
            # lexname is 'noun.animal' / 'verb.motion' / 'adj.all' / ...
            domain = lex.split('.', 1)[-1].lower()
            if domain and domain != head and domain.isalpha():
                _emit(rm, head, 'class', domain, stats)

        # example sentences -> first content word -> b_epi via 'example' role
        try:
            examples = synset.examples()
        except Exception:
            examples = []
        for ex in examples[:3]:   # cap 3 per synset
            exw = _first_content_word(ex)
            if exw and exw != head:
                _emit(rm, head, 'example', exw, stats)

        # Pack 247y: clear sense frame after each synset's writes complete
        # so cross-synset bleed does not happen (e.g. wn iteration order).
        mr.clear_frame()

        if verbose_every and stats['synsets'] % verbose_every == 0:
            print(f'  [absorb] {stats["synsets"]} synsets / '
                   f'{stats["triples"]} triples / '
                   f'minted={rm.status()["minted_count"]}')

    # Defense-in-depth: ensure frame cleared on exit (including break).
    mr.clear_frame()
    # final flush: mint any sub-threshold buffers
    rm.flush()
    stats['minted_labels'] = rm.status()['minted_labels']
    return stats
