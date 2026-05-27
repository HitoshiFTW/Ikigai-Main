"""
ikigai.cognition.flat_trainer -- bounded-RAM trainer for the flat substrate.

The architectural promise of the flat substrate is "fixed memory regardless
of data size".  That holds for the SUBSTRATE (192 MB FIXED).  Training-time
RAM, however, leaks through three side caches:

    ComputedKey._cache       -> {word: complex64[d]} grows with vocab
                                  100K vocab * 4 KB = 400 MB
    VSASDM._loc_cache        -> {word: ndarray[k]}   grows with vocab
                                  100K vocab * 512 B = 50 MB
    HF datasets parquet      -> external library; handled by jsonl phase-0

FlatTrainer wraps RapidTrainer with a periodic-compaction policy.  Caches
are normal fast dicts (RapidTrainer-equivalent throughput); at every flush
boundary, if a side cache exceeds the threshold, it is cleared.  The
substrate is permanent; caches are throwaways, so dropping them is safe
(every key is recomputable from the word string alone).

This gives near-RapidTrainer speed (no per-op LRU overhead) AND bounded
training RAM (no unbounded growth across the run).  Trigram cache stays
permanent because it's small-finite-set bounded by char-trigram space.

Public API mirrors RapidTrainer:
    ft = FlatTrainer(organism, batch_size=256, compact_threshold=15000)
    ft.train(sentence_iter)
    ft.flush()
"""

import gc
import numpy as np

from ikigai.cognition.rapid_trainer import RapidTrainer, DEFAULT_STOPWORDS


class FlatTrainer(RapidTrainer):
    """
    Bounded-RAM trainer.  Substrate stays 192 MB FIXED; side caches stay
    bounded too via periodic compaction.

    Parameters (all RapidTrainer args plus):
        compact_threshold -- when ComputedKey._cache exceeds this many
                              entries, all word + location caches are
                              dropped at the next flush boundary.  Default
                              15_000 (~60 MB).  Substrate writes already
                              committed are unaffected.
    """

    def __init__(self, organism, batch_size=256, drop_stop=True,
                 cooccur_w=None, do_cooccur=True, do_ngrams=True,
                 stopwords=None, compact_threshold=15_000):
        super().__init__(organism, batch_size=batch_size, drop_stop=drop_stop,
                         cooccur_w=cooccur_w, do_cooccur=do_cooccur,
                         do_ngrams=do_ngrams, stopwords=stopwords)
        self.compact_threshold = int(compact_threshold)
        self.n_compactions = 0

    def _compact_caches(self):
        """Drop word + location caches, then gc.collect to reclaim numpy
        heap fragments from per-sentence complex128 cumsum allocations."""
        ck = self.mr.ck
        if hasattr(ck, '_cache'):
            ck._cache.clear()
        for bank in (self.mr.sdm, getattr(self.mr, 'sdm_rel', None)):
            if bank is None: continue
            if hasattr(bank, '_loc_cache'):
                bank._loc_cache.clear()
        gc.collect()
        self.n_compactions += 1

    def flush(self):
        super().flush()
        # Compact when word cache exceeds threshold OR every N flushes
        # regardless (catches numpy heap fragmentation even when vocab
        # is small).
        ck = self.mr.ck
        cache_full = (hasattr(ck, '_cache')
                      and len(ck._cache) > self.compact_threshold)
        # gc every flush via internal counter
        if not hasattr(self, '_flush_counter'):
            self._flush_counter = 0
        self._flush_counter += 1
        periodic = (self._flush_counter % 10 == 0)
        if cache_full or periodic:
            self._compact_caches()

    # restore_caches is a no-op now (we never swapped originals)
    def restore_caches(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.flush()
        return False
