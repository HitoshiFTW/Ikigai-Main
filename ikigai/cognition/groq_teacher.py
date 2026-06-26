"""
ikigai.cognition.groq_teacher -- Pack 308 (Day 80 #2).  Drop-in teacher
backend backed by the Groq cloud API instead of the 3090 vLLM
(DeepSeek-R1-Distill-Llama-8B).

WHY
---
The 3090 R1-distill is the 303 accuracy bottleneck (Day 79): an 8B
reasoning-distill that (a) is weak on obscure facts and (b) wraps every
answer in a <think>...</think> ramble that the parser has to fight (the
whole Pack 287.5 ramble-disease class).  Groq serves INSTRUCT models
(llama-3.1-8b-instant, llama-3.3-70b-versatile) at low latency that
return CLEAN answers -- "rajendra prasad", "christopher nolan" -- no
think block, multi-word, exactly what Pack 307 multiword storage wants.

This is still the LLM-as-data-oracle doctrine: the teacher is a transient
data source.  Capability lives in the absorbed substrate, not the
teacher's weights -- swap the oracle, the organism's body is unchanged.

INTERFACE
---------
Mirrors the subset of RemoteLLMTeacher that TeacherOracle uses:
    .temperature, .max_new_tokens   (read + temporarily overridden)
    ._batch(prompts, logprobs=None) -> {'items': [{'text': str}, ...]}
    ._post(text)                    -> str

No logprobs (Groq chat completions don't expose the vLLM-style top-N
shape Pack 287.5 needs); _batch ignores the arg, so the logprob path
degrades to None exactly like an older serve.py -- the active-learning
path (ask / ask_batch) never needs it.
"""

import os
from concurrent.futures import ThreadPoolExecutor

try:
    from groq import Groq
except ImportError:
    Groq = None


class GroqTeacher:
    """Groq-API teacher backend, interface-compatible with the subset of
    RemoteLLMTeacher that TeacherOracle drives."""

    def __init__(self,
                 model='llama-3.3-70b-versatile',
                 api_key=None,
                 temperature=0.0,
                 top_p=1.0,
                 max_new_tokens=32,
                 request_timeout=60,
                 max_workers=8):
        if Groq is None:
            raise ImportError('install groq: pip install groq')
        api_key = api_key or os.environ.get('GROQ_API_KEY')
        if not api_key:
            raise ValueError(
                'no api_key provided and GROQ_API_KEY not set')
        self.model = str(model)
        self.client = Groq(api_key=api_key, timeout=float(request_timeout))
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.max_workers = int(max_workers)
        self.stats = {'requests': 0, 'errors': 0}

    # ---- single completion ------------------------------------------

    def _complete(self, prompt):
        """One chat completion -> the assistant's text (or '' on error)."""
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=self.max_new_tokens,
                stream=False)
            self.stats['requests'] += 1
            return r.choices[0].message.content or ''
        except Exception as e:
            self.stats['errors'] += 1
            print(f'    Groq err: {str(e)[:120]}')
            return ''

    # ---- batch (parallel single completions) ------------------------

    def _batch(self, prompts, logprobs=None):
        """Groq has no batch endpoint; fan out across a thread pool so a
        teach pass over N prompts isn't N serial round-trips.  Returns the
        same {'items': [{'text': ...}]} shape RemoteLLMTeacher does."""
        if not prompts:
            return {'items': []}
        n = min(self.max_workers, len(prompts))
        with ThreadPoolExecutor(max_workers=n) as ex:
            texts = list(ex.map(self._complete, prompts))
        return {'items': [{'text': t} for t in texts]}

    # ---- post-processing --------------------------------------------

    def _post(self, text):
        """Instruct models return clean answers -- no <think> block to
        strip.  Kept for interface parity with RemoteLLMTeacher."""
        return text
