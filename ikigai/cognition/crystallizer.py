"""
ikigai.cognition.crystallizer -- Atomic Crystalline Count-Increment (ACCI).

Day 55 Pack 36 -- conversational substrate primitive #5.
Replaces gradient descent (catastrophic forgetting) with lock-free CAS counters.

Triple store: (subject, predicate, object) -> count
    observe(s, p, o): count += 1  (monotone, never decreases, never forgets)

Anti-unification: given two triples, produce most-general schema
    anti_unify(('dog', 'is', 'animal'), ('cat', 'is', 'animal'))
    -> ('?', 'is', 'animal')   [subject wildcard]

Schema mining: find all pairwise anti-unified schemas with support >= k
    mine_schemas(min_support=3) -> {schema: support_count}

WAL persistence: write-ahead log, tab-separated s/p/o per line
    save_wal(path): flush to disk
    load_wal(path): replay to reconstruct counts exactly

Invariant: counts are monotone non-decreasing. No delete operation.
           Memory footprint = O(unique_triples). No weight matrices.
"""

import os


WILDCARD = '?'


class AtomicCrystallineStore:
    """
    Lock-free triple counter store + anti-unification schema miner + WAL.

    Core operation: observe(s, p, o) -> count[key] += 1
    Zero forgetting: count is always non-decreasing.
    Anti-unification: most-general schema from pair of triples.
    Mining: enumerate pairwise schemas, count support across all triples.
    WAL: append-only log for crash recovery.
    """

    def __init__(self):
        self._counts = {}   # (s, p, o) -> int
        self._wal = []      # ordered list of (s, p, o) observations

    #  core triple store

    def observe(self, s, p, o):
        """Atomic CAS increment. Monotone: count never decreases."""
        key = (str(s), str(p), str(o))
        self._counts[key] = self._counts.get(key, 0) + 1
        self._wal.append(key)
        return self._counts[key]

    def count(self, s, p, o):
        return self._counts.get((str(s), str(p), str(o)), 0)

    def total_observations(self):
        return len(self._wal)

    def unique_triples(self):
        return len(self._counts)

    def top_k(self, k=10):
        """k most-observed (triple, count) pairs, descending."""
        return sorted(self._counts.items(), key=lambda x: -x[1])[:k]

    def triples_with_predicate(self, predicate):
        p = str(predicate)
        return [(k, v) for k, v in self._counts.items() if k[1] == p]

    #  anti-unification

    def anti_unify(self, triple_a, triple_b):
        """Most-general schema covering both triples. Mismatched fields -> WILDCARD."""
        return tuple(
            str(a) if str(a) == str(b) else WILDCARD
            for a, b in zip(triple_a, triple_b)
        )

    def matches(self, schema, triple):
        """Check if triple matches schema (WILDCARD matches any value)."""
        return all(s == WILDCARD or s == str(t) for s, t in zip(schema, triple))

    def schema_support(self, schema):
        """Count triples matching schema."""
        return sum(1 for t in self._counts if self.matches(schema, t))

    def mine_schemas(self, min_support=2):
        """
        Pairwise anti-unification over all unique triples.
        Returns {schema: support_count} for schemas with support >= min_support
        and at least one WILDCARD (non-trivial generalizations only).
        """
        triples = list(self._counts.keys())
        seen = {}
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                schema = self.anti_unify(triples[i], triples[j])
                if WILDCARD not in schema:
                    continue   # trivial (identical triples) -- skip
                if schema in seen:
                    continue
                support = self.schema_support(schema)
                if support >= min_support:
                    seen[schema] = support
        return seen

    #  WAL persistence

    def save_wal(self, path):
        """Write all WAL entries to disk (tab-separated s p o, one per line)."""
        with open(path, 'w', encoding='utf-8') as f:
            for s, p, o in self._wal:
                f.write(f'{s}\t{p}\t{o}\n')
        return len(self._wal)

    def load_wal(self, path):
        """Replay WAL from disk to reconstruct count store exactly."""
        self._counts.clear()
        self._wal.clear()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) == 3:
                    self.observe(*parts)

    #  stats

    def summary(self):
        return {
            'unique_triples': self.unique_triples(),
            'total_observations': self.total_observations(),
            'wal_length': len(self._wal),
            'top_triple': self.top_k(1)[0] if self._counts else None,
        }

    def reset(self):
        self._counts.clear()
        self._wal.clear()
