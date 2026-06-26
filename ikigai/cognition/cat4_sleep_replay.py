"""
ikigai.cognition.cat4_sleep_replay -- Pack 290.

REVIVE Pack 271 (Day 76).  The first sleep cycle died because the
replay buffer was mixed-topic: replaying "Greece -> euro" chains
right after "Germany -> berlin" chains polluted the Germany
state-archetype with Greek-currency tokens via OPV crosstalk.

Fix via Faldor + Cully 2024 / Zhang HADES 2024 classifier-free
guidance: partition the replay buffer by TOPIC and replay each
sub-population in isolation.  The cache anchor's first 3 state
tokens are a deterministic topic signature -- 'what is the capital
of <X>' clusters all capital queries together; '<X> + <Y>'
clusters all add queries together; etc.

What sleep actually does (organism-level):
  1. Topic cluster pass over the live cache.
  2. For each cluster, schedule M replay iterations:
       - sample a deterministic subset of cluster anchors
       - re-fire absorb_chain(query, answer) so the substrate
         state HV gets reinforced (Hebbian repetition) without
         introducing new vocabulary.
  3. Substrate cleanup similarity on hot anchors goes up; cold
     anchors fade naturally because their state HV doesn't get
     refreshed (Pack 270 pure-decay path).

Biological grounding: McClelland-McNaughton-O'Reilly 1995
complementary learning systems.  Cortex (substrate) and
hippocampus (cache) interplay during sleep -- hippocampus replays
the day's episodes, cortex consolidates.  Our cache IS the
hippocampus analogue; substrate IS the cortex.

USAGE
-----
    sleep = TopicConditionalSleep(org)
    stats = sleep.run(max_clusters=8, iters_per_cluster=3)
"""
import hashlib
import re


_LEADING_TOKS = 3   # cluster key derived from first N state tokens


class TopicConditionalSleep:
    """Pack 290 sleep replay over the Pack 273 anchor-action cache.

    Topic cluster = hash of the first `_LEADING_TOKS` non-stopword
    tokens of the query that originally seeded the cache entry.
    """

    _STOP = frozenset({'a', 'an', 'the', 'is', 'are', 'was', 'be',
                         'of', 'in', 'on', 'at', 'to', 'for',
                         'what', 'which', 'who', 'how'})

    _WORD_RE = re.compile(r"[a-z]+|-?\d+")

    def __init__(self, organism):
        self.org = organism
        self.stats = {
            'clusters_seen': 0,
            'replays': 0,
            'cache_entries_touched': 0,
        }

    # ---- topic clustering ------------------------------------------

    def _cluster_key(self, query_text):
        """Stable topic key for `query_text`.  Strip stopwords from
        the leading tokens to keep cluster keys focused on content."""
        toks = self._WORD_RE.findall(str(query_text).lower())
        content = [t for t in toks if t not in self._STOP][:_LEADING_TOKS]
        if not content:
            content = toks[:_LEADING_TOKS]
        return '|'.join(content)

    def cluster_cache(self, cache, queries=None):
        """Partition the live cache into topic clusters.

        Since the cache stores deterministic blake2b anchors (not the
        original query text), we cannot recover topics from the keys
        alone.  Caller passes `queries` -- the dict of query_text ->
        answer_text accumulated during teaching, OR we fall back to
        cache iteration with empty topic keys (single bucket).

        Returns dict[topic_key, list[(anchor, action_tuple)]].
        """
        clusters = {}
        if queries:
            for query_text, _ in queries.items():
                key = self._cluster_key(query_text)
                clusters.setdefault(key, []).append(query_text)
        else:
            # No query log -- single bucket (no topic distinction)
            clusters['_all'] = list(cache)
        self.stats['clusters_seen'] = len(clusters)
        return clusters

    # ---- replay ----------------------------------------------------

    def run(self, queries_to_answers, iters_per_cluster=3,
              max_clusters=None, verbose=False):
        """Replay the (query, answer) pairs through Cat4Absorb,
        cluster by cluster.

        `queries_to_answers`: dict[query_text, answer_text] from the
        most recent teaching session (TeacherOracle.cache works).

        Each cluster's pairs are absorbed `iters_per_cluster` times.
        Hot anchors get Hebbian reinforcement; cold anchors are
        untouched (preserve their natural decay).
        """
        cat4 = self.org.cat4
        clusters = self.cluster_cache(cat4.anchor_actions,
                                         queries=queries_to_answers)
        keys = list(clusters.keys())
        if max_clusters is not None:
            keys = keys[:int(max_clusters)]
        if verbose:
            print(f'sleep: {len(keys)} clusters')
        for key in keys:
            queries = clusters[key]
            if verbose:
                print(f'  cluster {key!r:<32s} ({len(queries)} q)')
            for _ in range(int(iters_per_cluster)):
                for q in queries:
                    a = queries_to_answers.get(q)
                    if a is None:
                        continue
                    cat4.absorb_chain(f'{q}\n\n{a}\n\n')
                    self.stats['replays'] += 1
                    self.stats['cache_entries_touched'] += 1
        return self.stats
