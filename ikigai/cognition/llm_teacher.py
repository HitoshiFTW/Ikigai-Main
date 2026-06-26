"""
ikigai.cognition.llm_teacher -- Pack 248 redesigned. LLM-as-data-oracle.

Day 73. Wraps a HuggingFace causal-LM in a generator that yields
substrate-ready (prev, cur, actual) next-token triples. Pairs with
OnPolicyEvaluator (Pack 251) for gated absorb.

Design choices:
    - Decode -> re-tokenize. Qwen BBPE has 151K vocab; substrate uses
      lowercase word-level lemmas (matches wiki + WordNet absorb). We
      decode to text, then split via TOKEN_RE = [a-z]+. Lossy but
      maintains substrate-side vocab continuity.
    - Sampling: nucleus p=0.9 T=0.85 (per research return). Repetition
      penalty 1.1. Anti-repetition windowed dedup also applied at the
      yielded-triple level.
    - Vocab filter: caller passes substrate vocab set; triples with both
      sides in set are yielded. Tokens outside scope dropped silently.
    - CPU default. GPU caused two BSODs on the RTX 3050 during Day 72
      stage-1 SVD; LLM teacher inference is bandwidth-bound and runs
      acceptably on 8-core CPU.

Backpressure: yielding works one triple at a time. The driver decides
how many to absorb before sleep consolidation.
"""

import re
import numpy as np

TOKEN_RE = re.compile(r'[a-z]+')


def _clean_tokens(text, min_len=2, max_len=20):
    """Lowercase word-level tokenizer matching wiki/WordNet absorb conventions."""
    return [w for w in TOKEN_RE.findall(text.lower())
             if min_len <= len(w) <= max_len]


class LLMTeacher:
    """HuggingFace causal-LM wrapper for cat-1 token generation.

    Usage:
        teacher = LLMTeacher(model_id='Qwen/Qwen2.5-1.5B-Instruct',
                              device='cpu', dtype='float16')
        teacher.load()
        for (prev, cur, actual) in teacher.stream_triples(
                seed_prompts=['The history of ',  'In science, '],
                max_tokens=10_000,
                vocab_filter=substrate_vocab):
            ...
    """

    def __init__(self, model_id='Qwen/Qwen2.5-1.5B-Instruct',
                  device='cpu', dtype='float16',
                  temperature=0.85, top_p=0.9, repetition_penalty=1.1,
                  max_new_tokens=128, seed=248):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.max_new_tokens = int(max_new_tokens)
        self.seed = int(seed)
        self.tokenizer = None
        self.model = None

    def load(self):
        """Lazy load Qwen tokenizer + model. Returns self."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch_dtype = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }.get(self.dtype, torch.float16)
        # CPU + float16 is unstable in some ops -- force float32 on CPU.
        if self.device == 'cpu' and torch_dtype == torch.float16:
            torch_dtype = torch.float32
            self.dtype = 'float32'
        print(f'  loading {self.model_id} on {self.device} ({self.dtype}) ...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        torch.manual_seed(self.seed)
        return self

    def _generate_once(self, prompt):
        """One generation call. Returns decoded text (no prompt prefix)."""
        import torch
        ids = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = out[0, ids['input_ids'].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def stream_triples(self, seed_prompts, max_tokens=10_000,
                         vocab_filter=None, log_every=2000):
        """Generate text until max_tokens substrate-ready triples yielded.

        Yields (prev, cur, actual) triples where cur, actual are
        single-token lemmas in vocab_filter (if provided).

        prev is the token preceding cur (may be None at sentence start).
        """
        if not seed_prompts:
            seed_prompts = ['']
        prompt_idx = 0
        yielded = 0
        rng = np.random.default_rng(self.seed)
        while yielded < max_tokens:
            prompt = seed_prompts[prompt_idx % len(seed_prompts)]
            prompt_idx += 1
            try:
                text = self._generate_once(prompt)
            except Exception as e:
                print(f'    gen err: {e}')
                continue
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
                    print(f'    teacher emitted {yielded}/{max_tokens} triples')
