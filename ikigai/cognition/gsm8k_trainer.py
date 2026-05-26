"""
ikigai.cognition.gsm8k_trainer -- Train Ikigai on GSM8K train split.

Day 55 Pack 82 -- training is memory-population + pattern-clustering, NOT gradient descent.

Strategy:
    1. Load N train problems with (question, gold_chain, gold_answer)
    2. Run current V4 solver on each train problem
    3. Cluster ALL train questions via ConceptAtomizer -> M concept atoms
    4. For each atom, log:
        - which V4 method succeeds most often (the cluster's canonical handler)
        - example train problem + its gold chain
        - failure rate of V4 on this cluster
    5. Store gold (question -> answer) pairs in HolographicMemory
    6. At test: question -> retrieve nearest atom -> apply atom's preferred handler
                                                -> fallback to memory lookup
                                                -> fallback to V4 default
"""

import re
import json
from collections import Counter

import numpy as np

from ikigai.cognition.concept_atomizer import ConceptAtomizer
from ikigai.cognition.holographic_memory import HolographicMemory
from ikigai.cognition.gsm8k_solver_v4 import solve_v4


_ANS_PATTERN = re.compile(r'####\s*(-?\d+(?:\.\d+)?)')


def parse_gsm8k_answer(answer_field):
    """Extract numeric ground-truth from GSM8K 'answer' field (which contains chain + '#### N')."""
    m = _ANS_PATTERN.search(answer_field)
    if not m:
        return None
    val = float(m.group(1))
    if val == int(val):
        return int(val)
    return val


def tokenize_question(question):
    """Simple word-level tokenization for question encoding."""
    # Strip punctuation, lowercase
    cleaned = re.sub(r'[^\w\s]', ' ', question.lower())
    return cleaned.split()


def numeric_compare(a, b, tol=1e-3):
    """Compare two numeric answers (handles int/float/str)."""
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < tol
    except (ValueError, TypeError):
        return False


class GSM8KTrainer:
    """
    Trains Ikigai on GSM8K-style data.

    train(records, n_atoms=100, max_train=None)
        records: list of {'question', 'answer'} (raw GSM8K format)
        Returns training report.

    cluster_lookup(question)
        Returns (cluster_id, preferred_method) for a new question.

    memory_lookup(question)
        Returns (gold_answer, similarity) of nearest train problem.

    hybrid_solve(question)
        V4 -> if confidence low, cluster method -> if still low, memory lookup -> fallback.
    """

    def __init__(self, d=400):
        self.d         = d
        self.atomizer  = ConceptAtomizer(d=d)
        self.memory    = HolographicMemory(d=d)
        # Per-atom preferred method (most-correct V4 method for this cluster)
        self._atom_method = {}     # atom_name -> method_name
        # Per-atom success rate
        self._atom_stats  = {}     # atom_name -> {n_total, n_correct}
        # Original train problems
        self._train_qs   = {}      # ep_name -> question
        self._train_ans  = {}      # ep_name -> gold_answer (int/float)
        self.n_train     = 0

    #  training

    def train(self, records, n_atoms=100, max_train=None, verbose=True):
        """
        Train on GSM8K-format records.
        Returns dict of training stats.
        """
        if max_train is not None:
            records = records[:max_train]

        # Phase 1: load problems, run V4, store (q, gold_answer, v4_answer, v4_method)
        v4_correct = 0
        v4_method_counts = Counter()
        v4_method_correct = Counter()
        records_analyzed = []

        if verbose:
            print(f'\n  Phase 1: loading + V4 baseline on {len(records)} train problems...')

        for i, rec in enumerate(records):
            q     = rec['question']
            gold  = parse_gsm8k_answer(rec['answer'])
            if gold is None:
                continue
            v4_ans, v4_method = solve_v4(q)
            correct = numeric_compare(v4_ans, gold)
            if correct:
                v4_correct += 1
                v4_method_correct[v4_method] += 1
            v4_method_counts[v4_method] += 1

            ep_name = f'tr_{i}'
            tokens = tokenize_question(q)
            self.atomizer.record(ep_name, tokens, tick=i)
            self.memory.store(ep_name, tokens, [str(gold)])
            self._train_qs[ep_name]  = q
            self._train_ans[ep_name] = gold
            records_analyzed.append({
                'ep': ep_name, 'q': q, 'gold': gold,
                'v4_ans': v4_ans, 'v4_method': v4_method,
                'correct': correct,
            })

            if verbose and (i + 1) % 500 == 0:
                print(f'    [{i+1}/{len(records)}] V4 acc so far: {v4_correct/(i+1):.3f}')

        self.n_train = len(records_analyzed)
        v4_acc = v4_correct / max(1, self.n_train)

        if verbose:
            print(f'  V4 baseline on train: {v4_correct}/{self.n_train} = {v4_acc:.3f}')

        # Phase 2: cluster the train problems
        if verbose:
            print(f'  Phase 2: clustering into {n_atoms} concept atoms...')
        atoms = self.atomizer.sleep(n_atoms=n_atoms, max_iter=15)
        if verbose:
            print(f'    {len(atoms)} atoms created')

        # Phase 3: per-atom canonical method
        atom_examples = {}    # atom -> list of (ep, gold, v4_ans, v4_method, correct)
        for rec in records_analyzed:
            atom = self.atomizer.cluster_of(rec['ep'])
            if atom is None:
                continue
            atom_examples.setdefault(atom, []).append(rec)

        for atom, examples in atom_examples.items():
            # Most-correct method = method that succeeded most often in this cluster
            method_correct = Counter()
            method_total   = Counter()
            for ex in examples:
                method_total[ex['v4_method']] += 1
                if ex['correct']:
                    method_correct[ex['v4_method']] += 1
            # Pick method with highest correct-rate AND coverage
            if not method_correct:
                preferred = None
            else:
                preferred = max(method_correct, key=lambda m: method_correct[m])
            self._atom_method[atom] = preferred
            self._atom_stats[atom]  = {
                'n_total':   len(examples),
                'n_correct': sum(1 for ex in examples if ex['correct']),
            }

        return {
            'n_train':              self.n_train,
            'v4_correct':           v4_correct,
            'v4_accuracy':          round(v4_acc, 4),
            'n_atoms':              len(atoms),
            'v4_method_counts':     dict(v4_method_counts),
            'v4_method_correct':    dict(v4_method_correct),
            'atom_coverage':        {a: s for a, s in self._atom_stats.items()},
        }

    #  lookup

    def cluster_lookup(self, question):
        """Return (nearest_atom, preferred_method, atom_quality)."""
        tokens = tokenize_question(question)
        results = self.atomizer.recall(tokens, top_k=1)
        if not results:
            return None, None, 0.0
        atom_name, sim = results[0][0], results[0][1]
        method = self._atom_method.get(atom_name)
        stats = self._atom_stats.get(atom_name, {})
        quality = stats.get('n_correct', 0) / max(1, stats.get('n_total', 1))
        return atom_name, method, float(quality)

    def memory_lookup(self, question, top_k=3):
        """Return list of (ep_name, similarity) for nearest train problems."""
        tokens = tokenize_question(question)
        recalled = self.memory.recall(tokens, top_k=top_k)
        return recalled

    #  hybrid solver

    def hybrid_solve(self, question, mem_confidence_min=0.3):
        """
        V4 -> cluster method check -> memory lookup -> default.
        Returns (answer, method_used).
        """
        # 1. Run V4
        v4_ans, v4_method = solve_v4(question)

        # 2. Cluster lookup
        atom, preferred, quality = self.cluster_lookup(question)

        # 3. Trust V4 if its method matches cluster's preferred method (consensus)
        if preferred is not None and v4_method == preferred and v4_ans is not None:
            return v4_ans, f'v4+atom:{atom}'

        # 4. If cluster has high quality but V4 disagrees, try cluster's preferred handler
        if quality > 0.7 and preferred is not None and preferred != v4_method:
            # Re-route through the cluster's preferred handler if it's a special-case
            # (Can't easily re-call individual handlers here; defer to V4 fallback chain.)
            # For now: still trust V4 (cluster is just a tie-breaker hint).
            pass

        # 5. If V4 returned valid answer, use it
        if v4_ans is not None:
            return v4_ans, f'v4:{v4_method}'

        # 6. Last resort: memory lookup (return gold from nearest train problem)
        recalled = self.memory_lookup(question, top_k=1)
        if recalled:
            ep_name, sim = recalled[0]
            if sim >= mem_confidence_min:
                # Need to map ep_name back to its stored gold (memory.recall returns value HV scores).
                # Use _train_ans for direct lookup.
                if ep_name in self._train_ans:
                    return self._train_ans[ep_name], f'memory:{ep_name}'

        return None, 'none'

    #  introspection

    def report_summary(self):
        return {
            'n_train':       self.n_train,
            'n_atoms':       self.atomizer.n_atoms,
            'n_memory_keys': len(self._train_ans),
        }
