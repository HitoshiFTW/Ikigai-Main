"""
ikigai.cognition.remote_llm_teacher -- Day 73. HTTP client to the 3090 VM
vLLM serve (DeepSeek-R1-Distill-Llama-8B). Drop-in replacement for the
local LLMTeacher; same stream_triples signature.

Uses /batch endpoint so vLLM's continuous batching gives concurrent
decode throughput. One HTTP round-trip yields BATCH_SIZE chains.

URL handed over via env var NEUROSEED_3090_URL, e.g.:
    export NEUROSEED_3090_URL=https://xyz-abc-def.trycloudflare.com
"""

import os
import re
import time
import numpy as np

try:
    import requests
except ImportError:
    requests = None

TOKEN_RE = re.compile(r'[a-z]+')


def _clean_tokens(text, min_len=2, max_len=20):
    return [w for w in TOKEN_RE.findall(text.lower())
             if min_len <= len(w) <= max_len]


class RemoteLLMTeacher:
    """HTTP client to the VM-side vLLM serve. Yields substrate-ready
    (prev, cur, actual) triples.

    Usage:
        t = RemoteLLMTeacher(base_url='https://xxx.trycloudflare.com',
                              batch_size=16, max_new_tokens=512)
        t.healthcheck()
        for (prev, cur, actual) in t.stream_triples(
                seed_prompts=SEED_PROMPTS,
                max_tokens=10_000_000,
                vocab_filter=substrate_vocab):
            ...
    """

    def __init__(self,
                  base_url=None,
                  batch_size=16,
                  temperature=0.85,
                  top_p=0.9,
                  repetition_penalty=1.1,
                  max_new_tokens=512,
                  request_timeout=300,
                  seed=248,
                  strip_think=True):
        if requests is None:
            raise ImportError('install requests: pip install requests')
        self.base_url = (base_url or os.environ.get('NEUROSEED_3090_URL') or '')
        self.base_url = self.base_url.rstrip('/')
        if not self.base_url:
            raise ValueError(
                'no base_url provided and NEUROSEED_3090_URL not set')
        self.batch_size = int(batch_size)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.max_new_tokens = int(max_new_tokens)
        self.request_timeout = int(request_timeout)
        self.seed = int(seed)
        self.strip_think = bool(strip_think)
        self._sess = requests.Session()
        self._rng = np.random.default_rng(self.seed)
        self.stats = {'requests': 0, 'errors': 0, 'gen_tokens': 0,
                       'wall_s': 0.0}

    # ---- HTTP --------------------------------------------------------

    def healthcheck(self):
        r = self._sess.get(f'{self.base_url}/healthz',
                            timeout=self.request_timeout)
        r.raise_for_status()
        info = r.json()
        # Also pull /info
        i = self._sess.get(f'{self.base_url}/info',
                            timeout=self.request_timeout)
        if i.ok:
            info.update(i.json())
        return info

    def _batch(self, prompts, logprobs=None):
        body = {
            'prompts': prompts,
            'max_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'repetition_penalty': self.repetition_penalty,
        }
        # Pack 287.5 -- forward top-N logprobs when caller requests.
        # Older serve.py builds ignore the param silently; we
        # tolerate either response shape on read.
        if logprobs:
            body['logprobs'] = int(logprobs)
        t0 = time.perf_counter()
        try:
            r = self._sess.post(f'{self.base_url}/batch', json=body,
                                  timeout=self.request_timeout)
            r.raise_for_status()
            data = r.json()
            dt = time.perf_counter() - t0
            self.stats['requests'] += 1
            self.stats['gen_tokens'] += data.get('total_completion_tokens', 0)
            self.stats['wall_s'] += dt
            return data
        except Exception as e:
            self.stats['errors'] += 1
            print(f'    HTTP batch err: {e}')
            return None

    # ---- text post-processing ----------------------------------------

    _THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)

    def _post(self, text):
        """DeepSeek R1-Distill emits <think>...</think> chains. Default
        keeps both -- the reasoning chain has useful structure for cat-3
        absorb. For cat-1 pure next-token, strip to final answer."""
        if self.strip_think:
            text = self._THINK_RE.sub(' ', text)
        return text

    # ---- streaming ---------------------------------------------------

    def stream_triples(self,
                        seed_prompts,
                        max_tokens=10_000,
                        vocab_filter=None,
                        log_every=2000):
        """Generate text in batches until max_tokens substrate-ready
        triples yielded. Yields (prev, cur, actual)."""
        if not seed_prompts:
            seed_prompts = ['']
        idx = 0
        yielded = 0
        while yielded < max_tokens:
            # Build a batch of prompts
            batch = []
            for _ in range(self.batch_size):
                batch.append(seed_prompts[idx % len(seed_prompts)])
                idx += 1
            data = self._batch(batch)
            if not data:
                continue
            for item in data['items']:
                text = self._post(item['text'])
                toks = _clean_tokens(text)
                if vocab_filter is not None:
                    toks = [t for t in toks if t in vocab_filter]
                if len(toks) < 2:
                    continue
                for i in range(len(toks) - 1):
                    prev = toks[i-1] if i > 0 else None
                    cur = toks[i]; actual = toks[i+1]
                    yielded += 1
                    yield (prev, cur, actual)
                    if yielded >= max_tokens:
                        return
                    if yielded % log_every == 0:
                        gen_rate = (self.stats['gen_tokens'] /
                                     max(self.stats['wall_s'], 1e-6))
                        print(f'    teacher {yielded}/{max_tokens}  '
                               f'gen_rate {gen_rate:.0f} tok/s  '
                               f'errs {self.stats["errors"]}')

    # ---- raw-text mode (gen-to-disk path) ----------------------------

    def stream_raw_chunks(self, seed_prompts, n_chunks):
        """Yield raw decoded text chunks (one per generated sample). Used
        by gen-to-JSONL drivers that decouple gen from absorb."""
        if not seed_prompts:
            seed_prompts = ['']
        idx = 0
        emitted = 0
        while emitted < n_chunks:
            batch = []
            for _ in range(self.batch_size):
                batch.append(seed_prompts[idx % len(seed_prompts)])
                idx += 1
            data = self._batch(batch)
            if not data:
                continue
            for item in data['items']:
                text = self._post(item['text'])
                emitted += 1
                yield text
                if emitted >= n_chunks:
                    return
