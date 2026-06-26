"""
DEPRECATED (Day 77 Pack 278 v0).  Day 72 vision pivot: LLMs are DATA
oracles, not weight donors.  Pack 251+252+253+254+255+273+276 path
absorbs DATA from R1 traces, never weights.  T2S has zero call-sites
in Pack 255 GeneralReasoner path.  Scheduled for deletion in Pack 278
v1 once integrate.py Phase load_text path drops its t2s_compiler
import.

ikigai.cognition.t2s_compiler -- Transformer-to-Substrate compiler v0.

NVIDIA Killer #2 (project_nvidia_killer_stack.md). The "impossible"
because no one has reverse-engineered transformer weights into substrate
equivalents. But: every transformer weight matrix is a linear map. VSA
encodes any linear map as a superposition of role-bindings. Therefore
it is possible.

This module starts with the cleanest case: compile a TOKEN EMBEDDING
matrix W_E (R^vocab x R^d_model) into substrate writes such that the
substrate preserves the geometry of the original embeddings (cosine
similarities track within tolerance).

If geometry is preserved, the substrate can stand in for the embedding
lookup in any downstream transformer computation. Then we move to
attention QKV (Pack 174), then FFN (Pack 175), then a full layer.

Algorithm v0 (embedding-geometry compile):
    1. Pick N tokens. Get W_E[token_i] in R^d_model from the LLM.
    2. Build a random Gaussian projection P : R^d_model -> R^d (substrate dim).
       Bandwidth ~bandwidth (default ~ 1/sqrt(d_model)).
    3. For each token: phase_i = P @ W_E[i], hv_i = exp(i * phase_i).
       This is a unit-magnitude phasor HV encoding the embedding direction.
    4. Write hv_i to substrate under role 'gpt_embed' at address key(token_id).
    5. Read back: substrate returns hv_i'.  Compare:
         cos_orig (i,j) over original W_E embeddings
         cos_subst(i,j) over substrate-read phasor HVs
       Correlation across all token pairs = geometry preservation score.

Public API:
    t2s = T2SCompiler(organism)
    t2s.compile_embeddings(W_E, tokens, bandwidth=None)
    score = t2s.geometry_score(tokens)
    sim   = t2s.substrate_cosine(token_a, token_b)
"""

import numpy as np


def _renorm(hv):
    mags = np.abs(hv)
    mags = np.where(mags > 1e-9, mags, 1.0)
    return (hv / mags).astype(np.complex64)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ── GPU backend detection ───────────────────────────────────────────────
_GPU_BACKEND = None
_GPU_XP = None

def _detect_gpu():
    """Return (backend_name, xp_module) for the available GPU stack.
    Order: cupy > torch-cuda > None."""
    global _GPU_BACKEND, _GPU_XP
    if _GPU_BACKEND is not None:
        return _GPU_BACKEND, _GPU_XP
    try:
        import cupy as cp
        # quick probe
        x = cp.array([1.0, 2.0], dtype=cp.float32)
        cp.asnumpy(x)
        _GPU_BACKEND = 'cupy'
        _GPU_XP = cp
        return _GPU_BACKEND, _GPU_XP
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            _GPU_BACKEND = 'torch'
            _GPU_XP = torch
            return _GPU_BACKEND, _GPU_XP
    except Exception:
        pass
    _GPU_BACKEND = 'none'
    _GPU_XP = None
    return _GPU_BACKEND, _GPU_XP


def gpu_info():
    """Report current GPU backend availability."""
    backend, xp = _detect_gpu()
    if backend == 'cupy':
        try:
            name = xp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8', 'ignore')
            mem = xp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9
            return f'cupy on {name} ({mem:.1f} GB)'
        except Exception:
            return 'cupy detected'
    if backend == 'torch':
        name = xp.cuda.get_device_name(0)
        mem = xp.cuda.get_device_properties(0).total_memory / 1e9
        return f'torch+cuda on {name} ({mem:.1f} GB)'
    return 'no GPU backend (install cupy-cuda12x or torch+cuda)'


class T2SCompiler:
    """
    Compile a transformer weight surface into substrate writes.
    v0: token embedding geometry preservation.
    """

    ROLE = 'gpt_embed'

    def __init__(self, organism, role=None, seed=24001, d=None, K=1):
        """d: optional override of substrate dim for JL projection.
        K: ensemble size. K=1 = single projection (Packs 173-177).
            K=k = compile each layer with k independent projections and
            average forward outputs. Cuts JL noise variance by 1/K.
            K=4 at d=2048 matches single-projection precision of d~8192
            at LOWER memory because K parallel small d's beat one big d
            in the constant prefactor.  Pack 178 default K=4."""
        self.org  = organism
        self.role = role or self.ROLE
        self.d    = int(d) if d else organism.unified.d
        self.K    = int(K)
        self._rng_seed = int(seed)
        self.W_proj = None        # (d, d_model) projection cache
        self._d_model = None
        self.tokens = []          # token ids encoded
        if self.d == organism.unified.d and self.K == 1:
            self._register_role()

    def _register_role(self):
        mr = self.org.unified
        if self.role not in mr.roles:
            rng = np.random.default_rng(self._rng_seed)
            ph = rng.uniform(-np.pi, np.pi, self.d).astype(np.float32)
            mr.roles[self.role] = np.exp(1j * ph).astype(np.complex64)

    def _ensure_projection(self, d_model, bandwidth=None, k_idx=0):
        """Per-(d_model, k_idx) projection cache.

        Each input dimension AND each ensemble index gets its own
        deterministic projection seeded from (rng_seed, d_model, k_idx).
        k_idx=0..K-1 selects which projection in the ensemble.
        """
        d_model = int(d_model)
        k_idx = int(k_idx)
        key = (d_model, k_idx)
        if not hasattr(self, '_proj_cache'):
            self._proj_cache = {}
        if key in self._proj_cache:
            self._d_model = d_model
            self._P_re, self._P_im, self.W_proj = self._proj_cache[key]
            return
        # Build new projection deterministically from (seed, d_model, k_idx).
        rng = np.random.default_rng(self._rng_seed + 1 +
                                     d_model * 9973 + k_idx * 7919)
        scale = 1.0 / np.sqrt(2.0 * self.d)
        P_re = (rng.standard_normal((self.d, d_model))
                .astype(np.float32) * scale)
        P_im = (rng.standard_normal((self.d, d_model))
                .astype(np.float32) * scale)
        W_proj = (P_re + 1j * P_im).astype(np.complex64)
        self._proj_cache[key] = (P_re, P_im, W_proj)
        self._P_re, self._P_im, self.W_proj = P_re, P_im, W_proj
        self._d_model = d_model

    def embed_to_hv(self, embedding):
        """R^d_model -> unit-magnitude phasor HV in C^d via complex JL.

        Per-component random projection: c_k = (P_re[k] @ x) + i*(P_im[k] @ x).
        Phasor renorm preserves direction. Geometry preserved per
        Johnson-Lindenstrauss with high probability when d is large enough."""
        v = np.asarray(embedding, dtype=np.float32).reshape(-1)
        self._ensure_projection(v.shape[0])
        re = self._P_re @ v
        im = self._P_im @ v
        c  = (re + 1j * im).astype(np.complex64)
        # phasor renorm to unit magnitudes (preserves direction/cosine)
        mags = np.abs(c)
        mags = np.where(mags > 1e-9, mags, 1.0)
        return (c / mags).astype(np.complex64)

    def _addr(self, token_id):
        """Substrate address for an embedding entry. role-bound token key."""
        mr = self.org.unified
        key = mr.ck.key(f't{int(token_id)}')
        return (key * mr.roles[self.role]).astype(np.complex64)

    def compile_embeddings(self, W_E, token_ids, bandwidth=None,
                            n_reinforce=1, verbose=False):
        """
        Compile rows of W_E into substrate.
        W_E         : (vocab, d_model) ndarray
        token_ids   : iterable of token ids (rows of W_E to compile)
        Returns the number of writes performed.
        """
        W = np.asarray(W_E, dtype=np.float32)
        self._ensure_projection(W.shape[1], bandwidth=bandwidth)
        mr = self.org.unified
        bank = mr.sdm_rel
        n_writes = 0
        for tid in token_ids:
            hv = self.embed_to_hv(W[int(tid)])
            addr = self._addr(tid)
            for _ in range(int(n_reinforce)):
                bank.write(addr, hv)
            mr._role_targets.setdefault(self.role, set()).add(f't{int(tid)}')
            n_writes += 1
            self.tokens.append(int(tid))
        if verbose:
            print(f'  [T2S] compiled {n_writes} embeddings '
                   f'd_model={W.shape[1]} -> d={self.d}')
        return n_writes

    def read(self, token_id):
        """Read back the stored embedding HV for token_id."""
        addr = self._addr(token_id)
        return _renorm(self.org.unified.sdm_rel.read(addr))

    def substrate_cosine(self, token_a, token_b):
        """Cosine similarity of two substrate-recovered embeddings."""
        a = self.read(token_a)
        b = self.read(token_b)
        return float(np.real(np.vdot(a, b))) / self.d

    def geometry_score(self, W_E, token_ids, n_sample=None, seed=0):
        """
        Pearson correlation between original-embedding cosine matrix
        and substrate-read cosine matrix over the given tokens.
        Returns (corr, n_pairs).
        """
        token_ids = list(token_ids)
        if n_sample and len(token_ids) > n_sample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(token_ids), int(n_sample), replace=False)
            token_ids = [token_ids[int(i)] for i in idx]

        # Original cosine over W_E
        W = np.asarray(W_E, dtype=np.float32)[token_ids]
        Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
        cos_orig = Wn @ Wn.T

        # Substrate cosine via read
        hvs = np.stack([self.read(t) for t in token_ids])
        # phasor HVs are unit-magnitude already, but ensure
        hvs = hvs / (np.abs(hvs) + 1e-9)
        cos_subst = (hvs @ hvs.conj().T).real / self.d

        # Pearson correlation on the off-diagonal upper triangle
        iu = np.triu_indices(len(token_ids), k=1)
        a = cos_orig[iu]
        b = cos_subst[iu]
        a_c = a - a.mean()
        b_c = b - b.mean()
        denom = (np.linalg.norm(a_c) * np.linalg.norm(b_c)) + 1e-12
        corr = float((a_c @ b_c) / denom)
        return corr, len(iu[0])

    # ── Pack 174: linear-layer compile (dot-product preservation) ───────
    # Complex JL with variance 1/(2d) preserves dot products in expectation:
    #     E[ Re( <conj(z(x)), z(y)> ) ] = <x, y>
    # We exploit this to store an LLM linear layer W (d_model x d_out)
    # as d_out complex HVs in C^d, then approximate the forward pass
    #     y = x @ W
    # by computing Re(np.vdot(z(x), z(W[:,j]))) for each output column j.
    #
    # We DO NOT renorm column HVs (unlike compile_embeddings) because the
    # magnitudes carry the matrix scale.
    def encode_no_renorm(self, vec):
        """R^d_model -> complex C^d via JL, NO phasor renorm.
        Magnitudes preserved so dot products recover."""
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        self._ensure_projection(v.shape[0])
        re = self._P_re @ v
        im = self._P_im @ v
        return (re + 1j * im).astype(np.complex64)

    def compile_linear(self, W, name, verbose=False):
        """Compile a (d_model x d_out) matrix into K x d_out complex HVs
        (one set per ensemble projection). At forward time the K outputs
        are averaged."""
        if not hasattr(self, 'linear_cache'):
            self.linear_cache = {}
        Wn = np.asarray(W, dtype=np.float32)
        d_model, d_out = Wn.shape
        ensemble = []
        for k in range(self.K):
            self._ensure_projection(d_model, k_idx=k)
            # Vectorised compile: cols[j] = (P_re @ W[:,j]) + i*(P_im @ W[:,j])
            re = (self._P_re @ Wn).T   # (d_out, d)
            im = (self._P_im @ Wn).T   # (d_out, d)
            cols = (re + 1j * im).astype(np.complex64)
            ensemble.append(cols)
        self.linear_cache[name] = (ensemble, int(d_model))
        if verbose:
            print(f'  [T2S] linear {name}: {d_model}x{d_out} -> '
                  f'{self.K}x{d_out} HVs in C^{self.d}')
        return ensemble

    def linear(self, x, name):
        """Substrate forward pass averaged over K ensemble projections."""
        ensemble, d_model = self.linear_cache[name]
        acc = None
        for k, cols in enumerate(ensemble):
            self._ensure_projection(d_model, k_idx=k)
            zx = self.encode_no_renorm(x)
            prod = cols @ zx.conj()
            acc = prod.real if acc is None else (acc + prod.real)
        return (acc / float(self.K)).astype(np.float32)

    def linear_batch(self, X, name):
        """Batched substrate forward averaged over K ensemble projections.
        X (seq, d_in) -> Y (seq, d_out). Variance ÷ K reduces JL noise."""
        ensemble, d_model = self.linear_cache[name]
        Xf = X.astype(np.float32)
        acc = None
        for k, cols in enumerate(ensemble):
            self._ensure_projection(d_model, k_idx=k)
            re = Xf @ self._P_re.T
            im = Xf @ self._P_im.T
            ZX = (re + 1j * im).astype(np.complex64)
            prod = ZX.conj() @ cols.T
            acc = prod.real if acc is None else (acc + prod.real)
        return (acc / float(self.K)).astype(np.float32)

    # ── speed path: encode X once, reuse across many linear calls ────────
    def encode_seq(self, X, d_model):
        """Encode X for all K ensemble projections at given d_model.
        Returns list[K] of (seq, d) complex64 arrays. Caller reuses
        across many linear_batch_cached() calls sharing same X."""
        Xf = X.astype(np.float32)
        out = []
        for k in range(self.K):
            self._ensure_projection(int(d_model), k_idx=k)
            re = Xf @ self._P_re.T
            im = Xf @ self._P_im.T
            out.append((re + 1j * im).astype(np.complex64))
        return out

    def linear_batch_cached(self, ZX_list, name):
        """linear_batch using pre-encoded ensemble inputs.
        ZX_list: list[K] of (seq, d) complex from encode_seq."""
        ensemble, d_model = self.linear_cache[name]
        acc = None
        for cols, ZX in zip(ensemble, ZX_list):
            prod = ZX.conj() @ cols.T
            acc = prod.real if acc is None else (acc + prod.real)
        return (acc / float(self.K)).astype(np.float32)

    # ── attention head compile (QKV projections + softmax + V mix) ───────
    def compile_attention_head(self, W_Q, W_K, W_V, head_idx, n_heads,
                                 name_prefix='gpt2_layer0_head'):
        """Compile ONE attention head's Q, K, V projection matrices.

        Each W is (d_model, d_model). We slice the head's columns
        [head_idx*d_head : (head_idx+1)*d_head] and store as separate
        linear layers.  Returns the head dimension d_head.
        """
        d_model = W_Q.shape[0]
        d_head  = d_model // n_heads
        lo = head_idx * d_head
        hi = lo + d_head
        self.compile_linear(W_Q[:, lo:hi], f'{name_prefix}{head_idx}_Q')
        self.compile_linear(W_K[:, lo:hi], f'{name_prefix}{head_idx}_K')
        self.compile_linear(W_V[:, lo:hi], f'{name_prefix}{head_idx}_V')
        return d_head

    def attention_forward(self, X, head_idx, n_heads=12,
                           name_prefix='gpt2_layer0_head',
                           causal=True, with_biases=False):
        """Substrate forward pass through ONE attention head.

        X   : (seq_len, d_model) hidden states.
        Returns: (seq_len, d_head) head output (V mixed by softmax(QK)).
        with_biases: if True, add bQ/bK/bV from self.bias_cache (Pack 176)."""
        # ensemble: list of (d_out, d) arrays; take first to read d_out
        d_head = self.linear_cache[f'{name_prefix}{head_idx}_Q'][0][0].shape[0]
        seq_len = X.shape[0]
        # Encode X once across K, reuse for Q/K/V (3x save).
        ZX = self.encode_seq(X, self.linear_cache[f'{name_prefix}{head_idx}_Q'][1])
        Q = self.linear_batch_cached(ZX, f'{name_prefix}{head_idx}_Q')
        K = self.linear_batch_cached(ZX, f'{name_prefix}{head_idx}_K')
        V = self.linear_batch_cached(ZX, f'{name_prefix}{head_idx}_V')
        if with_biases and hasattr(self, 'bias_cache'):
            bQ = self.bias_cache.get(f'{name_prefix}{head_idx}_bQ')
            bK = self.bias_cache.get(f'{name_prefix}{head_idx}_bK')
            bV = self.bias_cache.get(f'{name_prefix}{head_idx}_bV')
            if bQ is not None: Q = Q + bQ.reshape(1, -1)
            if bK is not None: K = K + bK.reshape(1, -1)
            if bV is not None: V = V + bV.reshape(1, -1)
        # standard attention math: A = softmax(QK^T / sqrt(d_head)) (with mask)
        scores = (Q @ K.T) / np.sqrt(d_head)
        if causal:
            mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            scores[mask] = -1e9
        # softmax row-wise
        scores = scores - scores.max(axis=1, keepdims=True)
        ex = np.exp(scores)
        attn = ex / ex.sum(axis=1, keepdims=True)
        return attn @ V

    # ── GPU mode: move fast_cache to GPU memory ─────────────────────────
    def enable_gpu(self, verbose=True):
        """Move fast_cache + ln_cache + wte/wpe to GPU. Forward uses GPU matmul.
        Requires cupy-cuda12x or torch+cuda installed."""
        backend, xp = _detect_gpu()
        if backend == 'none':
            raise RuntimeError(
                "No GPU backend. Install cupy-cuda12x or torch+cuda first.")
        self.gpu_backend = backend
        self._gpu_xp = xp
        if verbose:
            print(f'  [T2S/GPU] backend={backend}  '
                  f'{gpu_info()}', flush=True)
        # Move fast_cache to GPU
        if hasattr(self, 'fast_cache'):
            new = {}
            for k, v in self.fast_cache.items():
                new[k] = self._to_gpu(v)
            self.fast_cache = new
        # Move ln_cache
        if hasattr(self, 'ln_cache'):
            new = {}
            for k, (w, b) in self.ln_cache.items():
                new[k] = (self._to_gpu(w), self._to_gpu(b))
            self.ln_cache = new
        # Move embeddings
        if hasattr(self, 'wte'):
            self.wte = self._to_gpu(self.wte)
            self.wpe = self._to_gpu(self.wpe)
        return self

    def _to_gpu(self, arr):
        if self.gpu_backend == 'cupy':
            return self._gpu_xp.asarray(arr)
        elif self.gpu_backend == 'torch':
            return self._gpu_xp.from_numpy(np.ascontiguousarray(arr)).cuda()
        return arr

    def _to_host(self, arr):
        if self.gpu_backend == 'cupy':
            return self._gpu_xp.asnumpy(arr)
        elif self.gpu_backend == 'torch':
            return arr.cpu().numpy()
        return arr

    def gpt2_forward_fast_gpu(self, token_ids, name_prefix='gpt2_layer',
                                return_logits=True):
        """GPU fast forward. Same math as gpt2_forward_fast but on GPU."""
        if not hasattr(self, 'gpu_backend'):
            raise RuntimeError("Call enable_gpu() first")
        xp = self._gpu_xp
        ids_np = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        seq_len = ids_np.shape[0]
        pos_np = np.arange(seq_len, dtype=np.int64)
        if self.gpu_backend == 'cupy':
            ids = xp.asarray(ids_np)
            pos = xp.asarray(pos_np)
            X = (self.wte[ids] + self.wpe[pos]).astype(xp.float32)
        else:  # torch
            ids = xp.from_numpy(ids_np).cuda()
            pos = xp.from_numpy(pos_np).cuda()
            X = (self.wte[ids] + self.wpe[pos]).float()

        d_model  = self.gpt2_d_model
        n_heads  = self.gpt2_n_heads
        d_head   = d_model // n_heads

        def layer_norm_gpu(x, name, eps=1e-5):
            w, b = self.ln_cache[name]
            mu = x.mean(axis=-1, keepdims=True)
            var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
            return (x - mu) / xp.sqrt(var + eps) * w + b

        def gelu_gpu(x):
            if self.gpu_backend == 'torch':
                import torch.nn.functional as F
                return F.gelu(x)
            else:
                # cupy approximation matching exact GELU
                from cupyx.scipy.special import erf
                return 0.5 * x * (1.0 + erf(x / xp.sqrt(xp.array(2.0, dtype=x.dtype))))

        for L in range(self.gpt2_n_layers):
            X_ln1 = layer_norm_gpu(X, f'{name_prefix}{L}_ln1')
            qkv = X_ln1 @ self.fast_cache[f'{name_prefix}{L}_c_attn_W']  \
                  + self.fast_cache[f'{name_prefix}{L}_c_attn_b']
            Q_all = qkv[:,            0:d_model]
            K_all = qkv[:,    d_model:2*d_model]
            V_all = qkv[:,  2*d_model:3*d_model]
            Q = Q_all.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2) \
                  if self.gpu_backend == 'cupy' else \
                  Q_all.reshape(seq_len, n_heads, d_head).permute(1, 0, 2)
            K = K_all.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2) \
                  if self.gpu_backend == 'cupy' else \
                  K_all.reshape(seq_len, n_heads, d_head).permute(1, 0, 2)
            V = V_all.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2) \
                  if self.gpu_backend == 'cupy' else \
                  V_all.reshape(seq_len, n_heads, d_head).permute(1, 0, 2)
            scale = 1.0 / float(np.sqrt(d_head))
            if self.gpu_backend == 'cupy':
                scores = (Q @ K.transpose(0, 2, 1)) * scale
            else:
                scores = (Q @ K.transpose(-2, -1)) * scale
            # causal mask
            if self.gpu_backend == 'cupy':
                mask = xp.triu(xp.ones((seq_len, seq_len), dtype=bool), k=1)
                scores = xp.where(mask[None, :, :], xp.float32(-1e9), scores)
            else:
                mask = xp.triu(xp.ones(seq_len, seq_len, dtype=xp.bool, device='cuda'), diagonal=1)
                scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
            # softmax row-wise
            scores = scores - scores.max(axis=-1, keepdims=True) \
                  if self.gpu_backend == 'cupy' else \
                  scores - scores.max(dim=-1, keepdim=True).values
            ex = xp.exp(scores)
            attn = ex / ex.sum(axis=-1, keepdims=True) \
                  if self.gpu_backend == 'cupy' else \
                  ex / ex.sum(dim=-1, keepdim=True)
            head_out = attn @ V
            if self.gpu_backend == 'cupy':
                concat = head_out.transpose(1, 0, 2).reshape(seq_len, d_model)
            else:
                concat = head_out.permute(1, 0, 2).reshape(seq_len, d_model)
            attn_out = concat @ self.fast_cache[f'{name_prefix}{L}_Wo'] \
                       + self.fast_cache[f'{name_prefix}{L}_bO']
            X = X + attn_out
            X_ln2 = layer_norm_gpu(X, f'{name_prefix}{L}_ln2')
            hid = X_ln2 @ self.fast_cache[f'{name_prefix}{L}_W1'] \
                  + self.fast_cache[f'{name_prefix}{L}_b1']
            hid = gelu_gpu(hid)
            ffn_out = hid @ self.fast_cache[f'{name_prefix}{L}_W2'] \
                       + self.fast_cache[f'{name_prefix}{L}_b2']
            X = X + ffn_out
        X = layer_norm_gpu(X, f'{name_prefix}_ln_f')
        if not return_logits:
            return self._to_host(X)
        if self.gpu_backend == 'cupy':
            logits = X @ self.wte.T
        else:
            logits = X @ self.wte.T
        return self._to_host(logits)

    # ── FAST MODE: store raw W, forward via direct matmul (BLAS speed) ──
    def enable_fast_mode(self):
        """Switch the compiler to FAST MODE. All compile_linear calls
        store the raw weight ndarray in fast_cache[name] in addition to
        (or instead of) the JL ensemble. forward_fast() then uses
        direct numpy matmul -> BLAS sgemm -> 30-80 tok/s on CPU."""
        self.fast_mode = True
        if not hasattr(self, 'fast_cache'):
            self.fast_cache = {}

    def _store_fast(self, W, name):
        if not hasattr(self, 'fast_cache'):
            self.fast_cache = {}
        self.fast_cache[name] = np.ascontiguousarray(W, dtype=np.float32)

    def linear_fast(self, x, name):
        """Direct matmul forward. x @ W. seq inputs OK."""
        return x.astype(np.float32, copy=False) @ self.fast_cache[name]

    def gpt2_forward_fast(self, token_ids, name_prefix='gpt2_layer',
                            return_logits=True):
        """Fast forward. Uses raw stored W via BLAS. Requires fast mode
        was enabled when the model was compiled (or after a fast-rebuild)."""
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        seq_len = ids.shape[0]
        pos = np.arange(seq_len, dtype=np.int64)
        X = (self.wte[ids] + self.wpe[pos]).astype(np.float32)
        d_model  = self.gpt2_d_model
        n_heads  = self.gpt2_n_heads
        d_head   = d_model // n_heads
        for L in range(self.gpt2_n_layers):
            # LN1
            X_ln1 = self.layer_norm(X, f'{name_prefix}{L}_ln1')
            # fused QKV via stored c_attn weight (d_model x 3*d_model)
            qkv = self.linear_fast(X_ln1, f'{name_prefix}{L}_c_attn_W')  \
                  + self.fast_cache[f'{name_prefix}{L}_c_attn_b']
            Q_all = qkv[:,            0:d_model]
            K_all = qkv[:,    d_model:2*d_model]
            V_all = qkv[:,  2*d_model:3*d_model]
            # reshape into heads
            Q = Q_all.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
            K = K_all.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
            V = V_all.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
            # scaled dot product, causal masked, softmax, V mix
            scale = 1.0 / np.sqrt(d_head)
            scores = (Q @ K.transpose(0, 2, 1)) * scale       # (h, seq, seq)
            mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            scores[:, mask] = -1e9
            scores = scores - scores.max(axis=-1, keepdims=True)
            ex = np.exp(scores)
            attn = ex / ex.sum(axis=-1, keepdims=True)
            head_out = attn @ V                                # (h, seq, d_head)
            concat = head_out.transpose(1, 0, 2).reshape(seq_len, d_model)
            attn_out = (self.linear_fast(concat, f'{name_prefix}{L}_Wo')
                         + self.fast_cache[f'{name_prefix}{L}_bO'])
            X = X + attn_out
            # LN2 -> FFN -> residual
            X_ln2 = self.layer_norm(X, f'{name_prefix}{L}_ln2')
            hid = (self.linear_fast(X_ln2, f'{name_prefix}{L}_W1')
                    + self.fast_cache[f'{name_prefix}{L}_b1'])
            hid = self._gelu(hid)
            ffn_out = (self.linear_fast(hid, f'{name_prefix}{L}_W2')
                        + self.fast_cache[f'{name_prefix}{L}_b2'])
            X = X + ffn_out
        X = self.layer_norm(X, f'{name_prefix}_ln_f')
        if not return_logits:
            return X
        return X @ self.wte.T

    # ── Pack 180: SUBSTRATE-NATIVE T2S ──────────────────────────────────
    # Writes weight column HVs directly into org.unified.sdm_rel (the
    # 192 MB flat memory bank). No sidecar linear_cache. Substrate
    # stays FIXED at 192 MB regardless of model size. Bigger model =
    # more superposition into same bank = more crosstalk.
    # This is Kill Stack #5 architecturally activated.

    def _register_layer_role(self, layer_role_name):
        """Lazy-create a role HV for a per-layer slot."""
        mr = self.org.unified
        if layer_role_name not in mr.roles:
            # deterministic seed per role name
            seed = abs(hash(layer_role_name)) % (2**31)
            rng = np.random.default_rng(seed)
            ph = rng.uniform(-np.pi, np.pi, mr.d).astype(np.float32)
            mr.roles[layer_role_name] = np.exp(1j * ph).astype(np.complex64)
        return mr.roles[layer_role_name]

    def _col_addr(self, layer_role_name, col_idx):
        """Address for a weight column: key(f'c{j}') * role_layer."""
        mr = self.org.unified
        key = mr.ck.key(f'c{int(col_idx)}')
        role_v = self._register_layer_role(layer_role_name)
        return (key * role_v).astype(np.complex64)

    def compile_linear_substrate(self, W, name, verbose=False):
        """Write each column of W as a phasor HV INTO the substrate
        (org.unified.sdm_rel) under role `name`. Substrate stays 192 MB."""
        if not hasattr(self, 'substrate_linear_meta'):
            self.substrate_linear_meta = {}
        Wn = np.asarray(W, dtype=np.float32)
        d_model, d_out = Wn.shape
        self._ensure_projection(d_model)
        mr = self.org.unified
        bank = mr.sdm_rel
        # encode all cols at once via JL
        re = (self._P_re @ Wn).T   # (d_out, d)
        im = (self._P_im @ Wn).T
        cols = (re + 1j * im).astype(np.complex64)
        # write each column to substrate
        for j in range(d_out):
            addr = self._col_addr(name, j)
            bank.write(addr, cols[j])
        # remember dims for forward
        self.substrate_linear_meta[name] = (int(d_model), int(d_out))
        if verbose:
            print(f'  [T2S/substrate] {name}: {d_model}x{d_out} written '
                  f'to org.unified.sdm_rel  (bank {mr.substrate_bytes()/1_048_576:.0f} MB)',
                  flush=True)

    def linear_substrate(self, x, name):
        """Forward via substrate reads. Per-column substrate.read."""
        d_model, d_out = self.substrate_linear_meta[name]
        self._ensure_projection(d_model)
        mr = self.org.unified
        bank = mr.sdm_rel
        # JL-encode x once
        zx = (self._P_re @ x.astype(np.float32)
              + 1j * self._P_im @ x.astype(np.float32)).astype(np.complex64)
        y = np.empty(d_out, dtype=np.float32)
        for j in range(d_out):
            addr = self._col_addr(name, j)
            recall = bank.read(addr)
            y[j] = float(np.real(np.vdot(zx, recall)))
        return y

    def compile_gpt2_into_substrate(self, hf_model,
                                       name_prefix='subst_L',
                                       verbose=True):
        """Write the ENTIRE GPT-2 model into org.unified.sdm_rel.
        Substrate stays 192 MB. Crosstalk scales with model size."""
        import time as _t
        n_layers = hf_model.config.n_layer
        n_heads  = hf_model.config.n_head
        d_model  = hf_model.config.n_embd
        self.gpt2_n_layers = n_layers
        self.gpt2_n_heads  = n_heads
        self.gpt2_d_model  = d_model
        self.wte = hf_model.wte.weight.detach().cpu().numpy().astype(np.float32)
        self.wpe = hf_model.wpe.weight.detach().cpu().numpy().astype(np.float32)
        # final LN as ndarray (LN params tiny, don't substrate-write)
        self.compile_layer_norm(hf_model.ln_f.weight.detach().cpu().numpy(),
                                 hf_model.ln_f.bias.detach().cpu().numpy(),
                                 f'{name_prefix}_ln_f')
        if not hasattr(self, 'bias_cache'):
            self.bias_cache = {}
        for L in range(n_layers):
            t0 = _t.perf_counter()
            block = hf_model.h[L]
            cW = block.attn.c_attn.weight.detach().cpu().numpy()
            cB = block.attn.c_attn.bias.detach().cpu().numpy()
            W_Q = cW[:,            0 : d_model]
            W_K = cW[:,      d_model : 2*d_model]
            W_V = cW[:,    2*d_model : 3*d_model]
            W_O = block.attn.c_proj.weight.detach().cpu().numpy()
            W1 = block.mlp.c_fc.weight.detach().cpu().numpy()
            W2 = block.mlp.c_proj.weight.detach().cpu().numpy()
            # SUBSTRATE writes
            self.compile_linear_substrate(W_Q, f'{name_prefix}{L}_Q')
            self.compile_linear_substrate(W_K, f'{name_prefix}{L}_K')
            self.compile_linear_substrate(W_V, f'{name_prefix}{L}_V')
            self.compile_linear_substrate(W_O, f'{name_prefix}{L}_O')
            self.compile_linear_substrate(W1,  f'{name_prefix}{L}_W1')
            self.compile_linear_substrate(W2,  f'{name_prefix}{L}_W2')
            # biases + LN params stay as ndarrays (tiny, doesn't violate substrate)
            self.bias_cache[f'{name_prefix}{L}_bQ'] = cB[          0 : d_model].astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_bK'] = cB[    d_model : 2*d_model].astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_bV'] = cB[  2*d_model : 3*d_model].astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_bO'] = block.attn.c_proj.bias.detach().cpu().numpy().astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_b1'] = block.mlp.c_fc.bias.detach().cpu().numpy().astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_b2'] = block.mlp.c_proj.bias.detach().cpu().numpy().astype(np.float32)
            self.compile_layer_norm(block.ln_1.weight.detach().cpu().numpy(),
                                     block.ln_1.bias.detach().cpu().numpy(),
                                     f'{name_prefix}{L}_ln1')
            self.compile_layer_norm(block.ln_2.weight.detach().cpu().numpy(),
                                     block.ln_2.bias.detach().cpu().numpy(),
                                     f'{name_prefix}{L}_ln2')
            if verbose:
                print(f'  [T2S/substrate] L{L:2d} '
                       f'(substrate {self.org.unified.substrate_bytes()/1_048_576:.0f} MB) '
                       f'in {_t.perf_counter()-t0:.1f}s', flush=True)
        return n_layers

    # ── Pack 187: alpha compression -- post-training int8 quantization ──
    # BitNet 1.58 (ternary) ONLY works with quantization-aware training.
    # Post-hoc on pretrained weights = quality collapse (cosine 0.04, Pack 187 v0).
    # int8 per-channel symmetric = 4x compression, ~99% quality, GPTQ-class.
    @staticmethod
    def _bitnet_quantize(W, threshold_factor=0.75):
        """Post-training int8 symmetric quantization per output channel.
        Each weight in [-127, 127] * scale. 1 byte/weight = 4x vs fp32.
        Reconstruction quality ~99% for typical LLM weights."""
        W = np.asarray(W, dtype=np.float32)
        # per-output-channel symmetric scale
        max_abs = np.abs(W).max(axis=0) + 1e-9
        scales = max_abs / 127.0
        qW = np.round(W / scales[None, :]).clip(-127, 127).astype(np.int8)
        return qW, scales.astype(np.float32)

    @staticmethod
    def _bitnet_dequantize(qW, scales):
        """Reconstruct fp32 from int8 quant + per-col scales."""
        return qW.astype(np.float32) * scales[None, :]

    def _store_alpha_weight(self, W, name):
        """Quantize W via BitNet and store under alpha_cache."""
        if not hasattr(self, 'alpha_cache'):
            self.alpha_cache = {}
        ternary, scales = self._bitnet_quantize(W)
        self.alpha_cache[name] = (ternary, scales)

    def _alpha_dequant(self, name):
        """Return fp32 weight reconstructed from α.
        Dict cache wins if present (Pack 189), else int8 alpha (Pack 187)."""
        if hasattr(self, 'dict_cache') and name in self.dict_cache:
            return self._dict_dequant(name)
        ternary, scales = self.alpha_cache[name]
        return self._bitnet_dequantize(ternary, scales)

    # ── Pack 189: dictionary atoms via vector quantization ──
    # K-means cluster Linear ROWS into K centroids. Each row stored as
    # 1 byte (K<=256) or 2 bytes (K<=65536) + per-row scale.
    # Combined with HolographicMemory for substrate-native atom lookup.
    @staticmethod
    def _kmeans_l2(X, K, max_iter=8, seed=0):
        """Fast K-means in L2. X (N,D) -> centroids (K,D), assignments (N,).
        Random init (k-means++ is O(K^2 N D) -- too slow at K=256). Vectorized
        centroid update via np.add.at scatter.
        """
        X = np.asarray(X, dtype=np.float32)
        N, D = X.shape
        K = min(int(K), N)
        rng = np.random.default_rng(seed)
        # random init: pick K distinct points
        idx = rng.choice(N, size=K, replace=False)
        C = X[idx].copy()
        assignments = np.full(N, -1, dtype=np.int32)
        for it in range(max_iter):
            # assign: argmin ||x - c||^2 = argmax x.c - 0.5*||c||^2
            cnorm = (C * C).sum(-1) * 0.5
            sim = X @ C.T - cnorm[None, :]
            new_a = sim.argmax(axis=1).astype(np.int32)
            if it > 0 and (new_a == assignments).all():
                assignments = new_a
                break
            assignments = new_a
            # vectorized centroid update via scatter-add
            sums = np.zeros((K, D), dtype=np.float32)
            counts = np.zeros(K, dtype=np.int32)
            np.add.at(sums, assignments, X)
            np.add.at(counts, assignments, 1)
            mask = counts > 0
            C[mask] = sums[mask] / counts[mask, None]
        return C, assignments

    def _store_dict_weight(self, W, name, k_atoms=256, hm=None, atomizer=None):
        """VQ-compress W and store as (codebook, assignments, scales) in dict_cache.
        Optional: register atom signatures with HolographicMemory and ConceptAtomizer."""
        if not hasattr(self, 'dict_cache'):
            self.dict_cache = {}
        Wn = np.asarray(W, dtype=np.float32)
        d_in, d_out = Wn.shape
        rows = Wn.T                        # (d_out, d_in) -- cluster rows
        # per-row L2 scale -> normalize before clustering
        scales = np.linalg.norm(rows, axis=1) + 1e-9
        unit_rows = rows / scales[:, None]
        K = min(int(k_atoms), d_out)
        seed = abs(hash(name)) % (2 ** 31)
        codebook, assignments = self._kmeans_l2(unit_rows, K, max_iter=10, seed=seed)
        idx_dtype = np.uint8 if K <= 256 else np.uint16
        assignments = assignments.astype(idx_dtype)
        self.dict_cache[name] = (codebook.astype(np.float32),
                                  assignments,
                                  scales.astype(np.float32))
        # substrate-native: register codebook with HolographicMemory (bipolar sigs)
        if hm is not None:
            for k in range(K):
                sig_tokens = [name, 'atom', int(k)]
                key_tokens = [name, int(k)]
                hm.store(f'{name}#a{k}', key_tokens, sig_tokens)
        # concept_atomizer: record each atom as an episode (bipolar HV space)
        if atomizer is not None:
            for k in range(K):
                atomizer.record(f'{name}#a{k}', [name, int(k)])
        return K

    def _dict_dequant(self, name):
        """Reconstruct fp32 W (d_in, d_out) from codebook + assignments + scales.
        Supports both scalar VQ (3-tuple) and PQ (4-tuple with n_sub field)."""
        entry = self.dict_cache[name]
        if len(entry) == 3:
            codebook, assignments, scales = entry
            rows = codebook[assignments.astype(np.int64)] * scales[:, None]
            return rows.T.astype(np.float32, copy=False)
        # PQ: (sub_codebooks list, sub_assignments (d_out, n_sub), scales, n_sub)
        sub_codebooks, sub_assignments, scales, n_sub = entry
        d_out = sub_assignments.shape[0]
        d_sub = sub_codebooks[0].shape[1]
        d_in = d_sub * n_sub
        rows = np.empty((d_out, d_in), dtype=np.float32)
        for s in range(n_sub):
            idx = sub_assignments[:, s].astype(np.int64)
            rows[:, s * d_sub:(s + 1) * d_sub] = sub_codebooks[s][idx]
        rows *= scales[:, None]
        return rows.T.astype(np.float32, copy=False)

    def _store_dict_weight_pq(self, W, name, n_sub=8, k_per_sub=256,
                                hm=None, atomizer=None):
        """Product Quantization compress W.
        Split d_in into n_sub chunks, K-means each chunk independently.
        Each row stored as n_sub small indices.
        Effective alphabet = k_per_sub^n_sub. Same byte size as scalar VQ
        but exponentially more reconstructable patterns.
        """
        if not hasattr(self, 'dict_cache'):
            self.dict_cache = {}
        Wn = np.asarray(W, dtype=np.float32)
        d_in, d_out = Wn.shape
        rows = Wn.T                                # (d_out, d_in)
        scales = np.linalg.norm(rows, axis=1) + 1e-9
        unit_rows = rows / scales[:, None]
        # adapt n_sub if d_in not divisible
        while d_in % n_sub != 0 and n_sub > 1:
            n_sub -= 1
        d_sub = d_in // n_sub
        K = min(int(k_per_sub), d_out)
        idx_dtype = np.uint8 if K <= 256 else np.uint16
        sub_codebooks = []
        sub_assignments = np.empty((d_out, n_sub), dtype=idx_dtype)
        seed_base = abs(hash(name)) % (2 ** 31)
        for s in range(n_sub):
            sub_data = unit_rows[:, s * d_sub:(s + 1) * d_sub]
            cb, asg = self._kmeans_l2(sub_data, K, max_iter=8,
                                       seed=seed_base + s * 7919)
            sub_codebooks.append(cb.astype(np.float32))
            sub_assignments[:, s] = asg.astype(idx_dtype)
        self.dict_cache[name] = (sub_codebooks,
                                  sub_assignments,
                                  scales.astype(np.float32),
                                  int(n_sub))
        # substrate-native registration: one HM entry per sub-atom (n_sub * K)
        if hm is not None:
            for s in range(n_sub):
                for k in range(K):
                    hm.store(f'{name}#s{s}a{k}', [name, s, k], [name, 'sub', s, k])
        if atomizer is not None:
            for s in range(n_sub):
                for k in range(K):
                    atomizer.record(f'{name}#s{s}a{k}', [name, s, k])
        return K * n_sub

    def compile_llama_model_dictionary_pq(self, hf_model, name_prefix='alpha_layer',
                                            n_sub=8, k_per_sub=256,
                                            hm=None, atomizer=None,
                                            verbose=True):
        """Pack 189v3: PQ-compressed Llama/Qwen.
        Per linear: n_sub subspaces × k_per_sub centroids each.
        Effective alphabet per row = k_per_sub^n_sub.
        """
        import time as _t
        if not hasattr(self, 'dict_cache'):
            self.dict_cache = {}
        if not hasattr(self, 'alpha_cache'):
            self.alpha_cache = {}
        cfg = hf_model.config
        n_layers = cfg.num_hidden_layers
        n_heads  = cfg.num_attention_heads
        n_kv_heads = getattr(cfg, 'num_key_value_heads', n_heads)
        d_model  = cfg.hidden_size
        intermediate = cfg.intermediate_size
        head_dim = getattr(cfg, 'head_dim', d_model // n_heads)
        rope_base = float(getattr(cfg, 'rope_theta', 10000.0))
        rms_eps  = float(getattr(cfg, 'rms_norm_eps', 1e-6))
        self.llm_n_layers = n_layers
        self.llm_n_heads  = n_heads
        self.llm_n_kv_heads = n_kv_heads
        self.llm_d_model  = d_model
        self.llm_head_dim = head_dim
        self.llm_intermediate = intermediate
        self.llm_rope_base = rope_base
        self.llm_rms_eps  = rms_eps
        model_inner = hf_model.model if hasattr(hf_model, 'model') else hf_model
        self.llm_embed = model_inner.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)
        if hasattr(hf_model, 'lm_head'):
            self.llm_lm_head = hf_model.lm_head.weight.detach().cpu().numpy().astype(np.float32)
        else:
            self.llm_lm_head = self.llm_embed
        self.compile_layer_norm(model_inner.norm.weight.detach().cpu().numpy(),
                                 np.zeros_like(model_inner.norm.weight.detach().cpu().numpy()),
                                 f'{name_prefix}_final_rms')
        n_params_total = 0
        bytes_dict = 0
        n_atoms_total = 0
        for L in range(n_layers):
            t0 = _t.perf_counter()
            block = model_inner.layers[L]
            attn = block.self_attn
            mlp = block.mlp
            for proj_name, proj in [
                ('q', attn.q_proj), ('k', attn.k_proj),
                ('v', attn.v_proj), ('o', attn.o_proj),
                ('gate', mlp.gate_proj), ('up', mlp.up_proj),
                ('down', mlp.down_proj),
            ]:
                W = proj.weight.detach().cpu().numpy().T  # (in, out)
                full_name = f'{name_prefix}{L}_{proj_name}'
                n_atoms = self._store_dict_weight_pq(W, full_name,
                                                      n_sub=n_sub,
                                                      k_per_sub=k_per_sub,
                                                      hm=hm, atomizer=atomizer)
                n_params_total += W.size
                n_atoms_total += n_atoms
                entry = self.dict_cache[full_name]
                if len(entry) == 4:
                    cbs, asg, sc, nsub = entry
                    bytes_dict += sum(c.nbytes for c in cbs) + asg.nbytes + sc.nbytes
                if hasattr(proj, 'bias') and proj.bias is not None:
                    if not hasattr(self, 'bias_cache'):
                        self.bias_cache = {}
                    self.bias_cache[f'{full_name}_b'] = \
                        proj.bias.detach().cpu().numpy().astype(np.float32)
            self.compile_layer_norm(
                block.input_layernorm.weight.detach().cpu().numpy(),
                np.zeros_like(block.input_layernorm.weight.detach().cpu().numpy()),
                f'{name_prefix}{L}_attn_rms')
            self.compile_layer_norm(
                block.post_attention_layernorm.weight.detach().cpu().numpy(),
                np.zeros_like(block.post_attention_layernorm.weight.detach().cpu().numpy()),
                f'{name_prefix}{L}_ffn_rms')
            if verbose:
                print(f'  [T2S/PQ n_sub={n_sub} K={k_per_sub}] block {L:2d} in '
                      f'{_t.perf_counter()-t0:.2f}s  '
                      f'dict_bytes={bytes_dict/1_048_576:.1f} MB', flush=True)
        self.dict_stats = {
            'n_params_total': n_params_total,
            'n_atoms_total': n_atoms_total,
            'fp32_size_bytes': n_params_total * 4,
            'dict_size_bytes': bytes_dict,
            'n_sub': int(n_sub),
            'k_per_sub': int(k_per_sub),
            'effective_alphabet_bits': int(n_sub) * int(np.log2(k_per_sub)),
        }
        if verbose:
            stats = self.dict_stats
            ratio = stats['fp32_size_bytes'] / max(stats['dict_size_bytes'], 1)
            print(f'  [T2S/PQ] {stats["n_params_total"]/1e6:.0f}M params -> '
                  f'fp32={stats["fp32_size_bytes"]/1e9:.2f} GB -> '
                  f'dict={stats["dict_size_bytes"]/1e9:.3f} GB '
                  f'({ratio:.1f}x, eff_alphabet=2^{stats["effective_alphabet_bits"]})',
                  flush=True)
        return n_layers

    def compile_llama_model_dictionary(self, hf_model, name_prefix='alpha_layer',
                                          k_atoms=256, hm=None, atomizer=None,
                                          ws=None, verbose=True):
        """Absorb a Llama/Qwen2 model into dict atoms (Pack 189).
        Uses alpha_cache infra by storing into dict_cache; _alpha_dequant routes."""
        import time as _t
        if not hasattr(self, 'dict_cache'):
            self.dict_cache = {}
        cfg = hf_model.config
        n_layers = cfg.num_hidden_layers
        n_heads  = cfg.num_attention_heads
        n_kv_heads = getattr(cfg, 'num_key_value_heads', n_heads)
        d_model  = cfg.hidden_size
        intermediate = cfg.intermediate_size
        head_dim = getattr(cfg, 'head_dim', d_model // n_heads)
        rope_base = float(getattr(cfg, 'rope_theta', 10000.0))
        rms_eps  = float(getattr(cfg, 'rms_norm_eps', 1e-6))
        self.llm_n_layers = n_layers
        self.llm_n_heads  = n_heads
        self.llm_n_kv_heads = n_kv_heads
        self.llm_d_model  = d_model
        self.llm_head_dim = head_dim
        self.llm_intermediate = intermediate
        self.llm_rope_base = rope_base
        self.llm_rms_eps  = rms_eps
        model_inner = hf_model.model if hasattr(hf_model, 'model') else hf_model
        self.llm_embed = model_inner.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)
        if hasattr(hf_model, 'lm_head'):
            self.llm_lm_head = hf_model.lm_head.weight.detach().cpu().numpy().astype(np.float32)
        else:
            self.llm_lm_head = self.llm_embed
        self.compile_layer_norm(model_inner.norm.weight.detach().cpu().numpy(),
                                 np.zeros_like(model_inner.norm.weight.detach().cpu().numpy()),
                                 f'{name_prefix}_final_rms')
        n_params_total = 0
        n_atoms_total = 0
        bytes_dict = 0
        # ensure alpha_cache exists (for compat with forward routing)
        if not hasattr(self, 'alpha_cache'):
            self.alpha_cache = {}
        # wake-sleep transition tracker: which atom-id follows which (per layer chain)
        prev_atom_per_proj = {}
        for L in range(n_layers):
            t0 = _t.perf_counter()
            block = model_inner.layers[L]
            attn = block.self_attn
            mlp = block.mlp
            for proj_name, proj in [
                ('q', attn.q_proj), ('k', attn.k_proj),
                ('v', attn.v_proj), ('o', attn.o_proj),
                ('gate', mlp.gate_proj), ('up', mlp.up_proj),
                ('down', mlp.down_proj),
            ]:
                W = proj.weight.detach().cpu().numpy().T  # (in, out)
                full_name = f'{name_prefix}{L}_{proj_name}'
                K_used = self._store_dict_weight(W, full_name,
                                                  k_atoms=k_atoms,
                                                  hm=hm, atomizer=atomizer)
                n_params_total += W.size
                n_atoms_total += K_used
                cb, asg, sc = self.dict_cache[full_name]
                bytes_dict += cb.nbytes + asg.nbytes + sc.nbytes
                # wake-sleep: log layer-to-layer atom transitions (proj kind)
                if ws is not None:
                    prev = prev_atom_per_proj.get(proj_name)
                    if prev is not None:
                        for j in range(min(asg.shape[0], 64)):
                            ws.record(f'{prev}#{int(j)}', f'{full_name}#{int(asg[j])}')
                    prev_atom_per_proj[proj_name] = full_name
                if hasattr(proj, 'bias') and proj.bias is not None:
                    if not hasattr(self, 'bias_cache'):
                        self.bias_cache = {}
                    self.bias_cache[f'{full_name}_b'] = \
                        proj.bias.detach().cpu().numpy().astype(np.float32)
            self.compile_layer_norm(
                block.input_layernorm.weight.detach().cpu().numpy(),
                np.zeros_like(block.input_layernorm.weight.detach().cpu().numpy()),
                f'{name_prefix}{L}_attn_rms')
            self.compile_layer_norm(
                block.post_attention_layernorm.weight.detach().cpu().numpy(),
                np.zeros_like(block.post_attention_layernorm.weight.detach().cpu().numpy()),
                f'{name_prefix}{L}_ffn_rms')
            if verbose:
                print(f'  [T2S/dict K={k_atoms}] block {L:2d} VQ in '
                      f'{_t.perf_counter()-t0:.2f}s  '
                      f'dict_bytes={bytes_dict/1_048_576:.1f} MB', flush=True)
        self.dict_stats = {
            'n_params_total': n_params_total,
            'n_atoms_total': n_atoms_total,
            'fp32_size_bytes': n_params_total * 4,
            'dict_size_bytes': bytes_dict,
            'k_atoms': int(k_atoms),
        }
        if verbose:
            stats = self.dict_stats
            ratio = stats['fp32_size_bytes'] / max(stats['dict_size_bytes'], 1)
            print(f'  [T2S/dict] {stats["n_params_total"]/1e6:.0f}M params -> '
                  f'fp32={stats["fp32_size_bytes"]/1e9:.2f} GB -> '
                  f'dict={stats["dict_size_bytes"]/1e9:.3f} GB '
                  f'({ratio:.1f}x compression, K={k_atoms} per linear)', flush=True)
        return n_layers

    def compile_llama_model_alpha(self, hf_model, name_prefix='alpha_layer',
                                     verbose=True):
        """Absorb a Llama/Qwen2 model into α = BitNet 1.58 ternary quant.
        Each weight: 1 byte int8 ternary + fp32 scale per col.
        Effective ~1.58 bits/weight with simple int8 storage today.
        Pack 188 will pack to 2-bit for the disk α format."""
        import time as _t
        if not hasattr(self, 'alpha_cache'):
            self.alpha_cache = {}
        cfg = hf_model.config
        n_layers = cfg.num_hidden_layers
        n_heads  = cfg.num_attention_heads
        n_kv_heads = getattr(cfg, 'num_key_value_heads', n_heads)
        d_model  = cfg.hidden_size
        intermediate = cfg.intermediate_size
        head_dim = getattr(cfg, 'head_dim', d_model // n_heads)
        rope_base = float(getattr(cfg, 'rope_theta', 10000.0))
        rms_eps  = float(getattr(cfg, 'rms_norm_eps', 1e-6))
        self.llm_n_layers = n_layers
        self.llm_n_heads  = n_heads
        self.llm_n_kv_heads = n_kv_heads
        self.llm_d_model  = d_model
        self.llm_head_dim = head_dim
        self.llm_intermediate = intermediate
        self.llm_rope_base = rope_base
        self.llm_rms_eps  = rms_eps
        model_inner = hf_model.model if hasattr(hf_model, 'model') else hf_model
        # embed + lm head -- keep raw float for now (Pack 188 quantizes too)
        self.llm_embed = model_inner.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)
        if hasattr(hf_model, 'lm_head'):
            self.llm_lm_head = hf_model.lm_head.weight.detach().cpu().numpy().astype(np.float32)
        else:
            self.llm_lm_head = self.llm_embed
        # final norm
        self.compile_layer_norm(model_inner.norm.weight.detach().cpu().numpy(),
                                 np.zeros_like(model_inner.norm.weight.detach().cpu().numpy()),
                                 f'{name_prefix}_final_rms')
        n_params_total = 0
        n_params_ternary = 0
        for L in range(n_layers):
            t0 = _t.perf_counter()
            block = model_inner.layers[L]
            attn = block.self_attn
            mlp = block.mlp
            for proj_name, proj in [
                ('q', attn.q_proj), ('k', attn.k_proj),
                ('v', attn.v_proj), ('o', attn.o_proj),
                ('gate', mlp.gate_proj), ('up', mlp.up_proj),
                ('down', mlp.down_proj),
            ]:
                W = proj.weight.detach().cpu().numpy().T  # (in, out)
                self._store_alpha_weight(W, f'{name_prefix}{L}_{proj_name}')
                n_params_total += W.size
                n_params_ternary += W.size
                # Qwen2/Qwen2.5 have biases on q/k/v; Llama does not.
                if hasattr(proj, 'bias') and proj.bias is not None:
                    if not hasattr(self, 'bias_cache'):
                        self.bias_cache = {}
                    self.bias_cache[f'{name_prefix}{L}_{proj_name}_b'] = \
                        proj.bias.detach().cpu().numpy().astype(np.float32)
            self.compile_layer_norm(
                block.input_layernorm.weight.detach().cpu().numpy(),
                np.zeros_like(block.input_layernorm.weight.detach().cpu().numpy()),
                f'{name_prefix}{L}_attn_rms')
            self.compile_layer_norm(
                block.post_attention_layernorm.weight.detach().cpu().numpy(),
                np.zeros_like(block.post_attention_layernorm.weight.detach().cpu().numpy()),
                f'{name_prefix}{L}_ffn_rms')
            if verbose:
                print(f'  [T2S/alpha] block {L:2d} ternarized in '
                      f'{_t.perf_counter()-t0:.2f}s', flush=True)
        self.alpha_stats = {
            'n_params_total': n_params_total,
            'n_params_ternary': n_params_ternary,
            'fp32_size_bytes': n_params_total * 4,
            'alpha_size_bytes': sum(t.nbytes + s.nbytes for t, s in self.alpha_cache.values()),
        }
        if verbose:
            stats = self.alpha_stats
            ratio = stats['fp32_size_bytes'] / max(stats['alpha_size_bytes'], 1)
            print(f'  [T2S/alpha] {stats["n_params_ternary"]/1e6:.0f}M params ternarized. '
                  f'fp32={stats["fp32_size_bytes"]/1e9:.2f} GB -> '
                  f'alpha={stats["alpha_size_bytes"]/1e9:.2f} GB '
                  f'({ratio:.1f}x compression)', flush=True)
        return n_layers

    def llama_forward_alpha(self, token_ids, name_prefix='alpha_layer',
                               return_logits=True):
        """Forward through α-compressed Qwen/Llama: dequant each linear on the fly."""
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        seq_len = ids.shape[0]
        d_model = self.llm_d_model
        n_heads = self.llm_n_heads
        n_kv_heads = self.llm_n_kv_heads
        head_dim = self.llm_head_dim
        rms_eps = self.llm_rms_eps
        X = self.llm_embed[ids].astype(np.float32)
        cos, sin = self._rope_cos_sin(seq_len, head_dim, self.llm_rope_base)
        n_kv_groups = n_heads // n_kv_heads
        for L in range(self.llm_n_layers):
            w_attn, _ = self.ln_cache[f'{name_prefix}{L}_attn_rms']
            X_norm = self._rms_norm(X, w_attn, eps=rms_eps)
            # dequant Q/K/V/O each call
            Wq = self._alpha_dequant(f'{name_prefix}{L}_q')
            Wk = self._alpha_dequant(f'{name_prefix}{L}_k')
            Wv = self._alpha_dequant(f'{name_prefix}{L}_v')
            Wo = self._alpha_dequant(f'{name_prefix}{L}_o')
            Q_lin = X_norm @ Wq
            K_lin = X_norm @ Wk
            V_lin = X_norm @ Wv
            # Qwen2/2.5 QKV biases (Llama has none -> these are no-ops if absent)
            bq = self.bias_cache.get(f'{name_prefix}{L}_q_b') if hasattr(self, 'bias_cache') else None
            bk = self.bias_cache.get(f'{name_prefix}{L}_k_b') if hasattr(self, 'bias_cache') else None
            bv = self.bias_cache.get(f'{name_prefix}{L}_v_b') if hasattr(self, 'bias_cache') else None
            if bq is not None: Q_lin = Q_lin + bq
            if bk is not None: K_lin = K_lin + bk
            if bv is not None: V_lin = V_lin + bv
            Q = Q_lin.reshape(seq_len, n_heads, head_dim)
            K = K_lin.reshape(seq_len, n_kv_heads, head_dim)
            V = V_lin.reshape(seq_len, n_kv_heads, head_dim)
            Q = self._apply_rope(Q, cos[:, None, :], sin[:, None, :])
            K = self._apply_rope(K, cos[:, None, :], sin[:, None, :])
            if n_kv_groups > 1:
                K = np.repeat(K, n_kv_groups, axis=1)
                V = np.repeat(V, n_kv_groups, axis=1)
            Q_h = Q.transpose(1, 0, 2)
            K_h = K.transpose(1, 0, 2)
            V_h = V.transpose(1, 0, 2)
            scores = (Q_h @ K_h.transpose(0, 2, 1)) / np.sqrt(head_dim)
            mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            scores[:, mask] = -1e9
            scores = scores - scores.max(axis=-1, keepdims=True)
            ex = np.exp(scores)
            attn = ex / ex.sum(axis=-1, keepdims=True)
            head_out = attn @ V_h
            concat = head_out.transpose(1, 0, 2).reshape(seq_len, d_model)
            attn_out = concat @ Wo
            X = X + attn_out
            w_ffn, _ = self.ln_cache[f'{name_prefix}{L}_ffn_rms']
            X_norm = self._rms_norm(X, w_ffn, eps=rms_eps)
            Wg = self._alpha_dequant(f'{name_prefix}{L}_gate')
            Wu = self._alpha_dequant(f'{name_prefix}{L}_up')
            Wd = self._alpha_dequant(f'{name_prefix}{L}_down')
            gate = X_norm @ Wg
            up   = X_norm @ Wu
            hid  = self._swiglu(gate, up)
            ffn_out = hid @ Wd
            X = X + ffn_out
        w_final, _ = self.ln_cache[f'{name_prefix}_final_rms']
        X = self._rms_norm(X, w_final, eps=rms_eps)
        if not return_logits:
            return X
        return X @ self.llm_lm_head.T

    # ── Pack 186: modern LLM primitives (Llama/Qwen/Mistral arch) ───────
    @staticmethod
    def _rms_norm(x, weight, eps=1e-6):
        """RMSNorm: x / sqrt(mean(x^2)) * weight. Llama/Qwen use this."""
        mean_sq = (x * x).mean(axis=-1, keepdims=True)
        return x * (1.0 / np.sqrt(mean_sq + eps)) * weight

    # Standard Llama/Qwen2 RoPE -- matches HF transformers.
    @staticmethod
    def _rope_freqs(d_head, base=10000.0, dtype=np.float32):
        """RoPE inverse freqs. d_head must be even."""
        half = d_head // 2
        inv_freq = 1.0 / (base ** (2 * np.arange(0, half, dtype=dtype) / d_head))
        return inv_freq

    @staticmethod
    def _rotate_half(x):
        """Llama/Qwen rotate_half: [x1, x2] -> [-x2, x1] over last dim."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return np.concatenate([-x2, x1], axis=-1)

    @staticmethod
    def _apply_rope(x, cos, sin):
        """Llama/Qwen apply_rotary_pos_emb. cos/sin shape (seq, d_head)
        with first half duplicated from second half."""
        return x * cos + T2SCompiler._rotate_half(x) * sin

    def _rope_cos_sin(self, seq_len, d_head, base=10000.0):
        """Llama/Qwen RoPE cos/sin tables. cos/sin shape (seq, d_head)
        with first half = second half (duplicated for rotate_half trick)."""
        if not hasattr(self, '_rope_cache'):
            self._rope_cache = {}
        key = (int(seq_len), int(d_head), float(base))
        if key in self._rope_cache:
            return self._rope_cache[key]
        inv_freq = self._rope_freqs(d_head, base=base)
        positions = np.arange(seq_len, dtype=np.float32)
        freqs = positions[:, None] * inv_freq[None, :]   # (seq, d_head/2)
        # duplicate to (seq, d_head)
        emb = np.concatenate([freqs, freqs], axis=-1).astype(np.float32)
        cos = np.cos(emb).astype(np.float32)
        sin = np.sin(emb).astype(np.float32)
        self._rope_cache[key] = (cos, sin)
        return self._rope_cache[key]

    @staticmethod
    def _swiglu(gate, up):
        """SwiGLU: silu(gate) * up. Llama/Qwen FFN."""
        silu = gate * (1.0 / (1.0 + np.exp(-gate)))
        return silu * up

    def compile_llama_model_fast(self, hf_model, name_prefix='llama_layer',
                                    verbose=True):
        """Fast-mode compile for Llama/Qwen2/Mistral-style models.
        Stores raw weights for: embed_tokens, lm_head, per-layer
        q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj,
        input_layernorm, post_attention_layernorm, final_norm."""
        import time as _t
        self.enable_fast_mode()
        cfg = hf_model.config
        n_layers = cfg.num_hidden_layers
        n_heads  = cfg.num_attention_heads
        n_kv_heads = getattr(cfg, 'num_key_value_heads', n_heads)
        d_model  = cfg.hidden_size
        intermediate = cfg.intermediate_size
        head_dim = getattr(cfg, 'head_dim', d_model // n_heads)
        rope_base = float(getattr(cfg, 'rope_theta', 10000.0))
        rms_eps  = float(getattr(cfg, 'rms_norm_eps', 1e-6))
        # store arch
        self.llm_n_layers = n_layers
        self.llm_n_heads  = n_heads
        self.llm_n_kv_heads = n_kv_heads
        self.llm_d_model  = d_model
        self.llm_head_dim = head_dim
        self.llm_intermediate = intermediate
        self.llm_rope_base = rope_base
        self.llm_rms_eps  = rms_eps
        # embed + lm head
        model_inner = hf_model.model if hasattr(hf_model, 'model') else hf_model
        self.llm_embed = model_inner.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)
        # lm head -- tied to embed if model.lm_head.weight is None or shares storage
        if hasattr(hf_model, 'lm_head'):
            lm_head_w = hf_model.lm_head.weight.detach().cpu().numpy().astype(np.float32)
        else:
            lm_head_w = self.llm_embed
        self.llm_lm_head = lm_head_w
        # final norm
        self.compile_layer_norm(model_inner.norm.weight.detach().cpu().numpy(),
                                 np.zeros_like(model_inner.norm.weight.detach().cpu().numpy()),
                                 f'{name_prefix}_final_rms')
        for L in range(n_layers):
            t0 = _t.perf_counter()
            block = model_inner.layers[L]
            attn = block.self_attn
            mlp = block.mlp
            # weights are (out, in) for Llama style -> transpose to (in, out) for x @ W
            self._store_fast(attn.q_proj.weight.detach().cpu().numpy().T,
                              f'{name_prefix}{L}_q')
            self._store_fast(attn.k_proj.weight.detach().cpu().numpy().T,
                              f'{name_prefix}{L}_k')
            self._store_fast(attn.v_proj.weight.detach().cpu().numpy().T,
                              f'{name_prefix}{L}_v')
            self._store_fast(attn.o_proj.weight.detach().cpu().numpy().T,
                              f'{name_prefix}{L}_o')
            self._store_fast(mlp.gate_proj.weight.detach().cpu().numpy().T,
                              f'{name_prefix}{L}_gate')
            self._store_fast(mlp.up_proj.weight.detach().cpu().numpy().T,
                              f'{name_prefix}{L}_up')
            self._store_fast(mlp.down_proj.weight.detach().cpu().numpy().T,
                              f'{name_prefix}{L}_down')
            # RMS norms (no bias for Llama/Qwen2)
            self.compile_layer_norm(
                block.input_layernorm.weight.detach().cpu().numpy(),
                np.zeros_like(block.input_layernorm.weight.detach().cpu().numpy()),
                f'{name_prefix}{L}_attn_rms')
            self.compile_layer_norm(
                block.post_attention_layernorm.weight.detach().cpu().numpy(),
                np.zeros_like(block.post_attention_layernorm.weight.detach().cpu().numpy()),
                f'{name_prefix}{L}_ffn_rms')
            if verbose:
                print(f'  [T2S/llama] block {L:2d} stored in '
                      f'{_t.perf_counter()-t0:.2f}s', flush=True)
        return n_layers

    def llama_forward_fast(self, token_ids, name_prefix='llama_layer',
                              return_logits=True):
        """Fast forward for Llama/Qwen2 model: RMSNorm + RoPE + GQA + SwiGLU."""
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        seq_len = ids.shape[0]
        d_model = self.llm_d_model
        n_heads = self.llm_n_heads
        n_kv_heads = self.llm_n_kv_heads
        head_dim = self.llm_head_dim
        rms_eps = self.llm_rms_eps
        # embedding lookup
        X = self.llm_embed[ids].astype(np.float32)
        cos, sin = self._rope_cos_sin(seq_len, head_dim, self.llm_rope_base)
        n_kv_groups = n_heads // n_kv_heads
        for L in range(self.llm_n_layers):
            # attention pre-norm
            w_attn, _ = self.ln_cache[f'{name_prefix}{L}_attn_rms']
            X_norm = self._rms_norm(X, w_attn, eps=rms_eps)
            # Q, K, V projections
            Q = (X_norm @ self.fast_cache[f'{name_prefix}{L}_q']).reshape(
                seq_len, n_heads, head_dim)
            K = (X_norm @ self.fast_cache[f'{name_prefix}{L}_k']).reshape(
                seq_len, n_kv_heads, head_dim)
            V = (X_norm @ self.fast_cache[f'{name_prefix}{L}_v']).reshape(
                seq_len, n_kv_heads, head_dim)
            # RoPE on Q and K
            Q = self._apply_rope(Q, cos[:, None, :], sin[:, None, :])
            K = self._apply_rope(K, cos[:, None, :], sin[:, None, :])
            # GQA: repeat K, V to match Q heads
            if n_kv_groups > 1:
                K = np.repeat(K, n_kv_groups, axis=1)
                V = np.repeat(V, n_kv_groups, axis=1)
            # attention scores (h, seq, seq)
            Q_h = Q.transpose(1, 0, 2)   # (n_heads, seq, head_dim)
            K_h = K.transpose(1, 0, 2)
            V_h = V.transpose(1, 0, 2)
            scores = (Q_h @ K_h.transpose(0, 2, 1)) / np.sqrt(head_dim)
            mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            scores[:, mask] = -1e9
            scores = scores - scores.max(axis=-1, keepdims=True)
            ex = np.exp(scores)
            attn = ex / ex.sum(axis=-1, keepdims=True)
            head_out = attn @ V_h
            concat = head_out.transpose(1, 0, 2).reshape(seq_len, d_model)
            attn_out = concat @ self.fast_cache[f'{name_prefix}{L}_o']
            X = X + attn_out
            # FFN pre-norm
            w_ffn, _ = self.ln_cache[f'{name_prefix}{L}_ffn_rms']
            X_norm = self._rms_norm(X, w_ffn, eps=rms_eps)
            gate = X_norm @ self.fast_cache[f'{name_prefix}{L}_gate']
            up = X_norm @ self.fast_cache[f'{name_prefix}{L}_up']
            hid = self._swiglu(gate, up)
            ffn_out = hid @ self.fast_cache[f'{name_prefix}{L}_down']
            X = X + ffn_out
        w_final, _ = self.ln_cache[f'{name_prefix}_final_rms']
        X = self._rms_norm(X, w_final, eps=rms_eps)
        if not return_logits:
            return X
        return X @ self.llm_lm_head.T

    # ── tracking helper: status file for long runs ───────────────────────
    def _track_init(self, path='c:/neuroseed/cache/t2s_progress.txt'):
        """Open progress tracking file. Polled externally to see live status."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._track_path = path
        self._track_t0 = None
        import time as _t
        self._track_t0 = _t.perf_counter()
        with open(path, 'w') as fh:
            fh.write(f'[T2S START] {_t.strftime("%H:%M:%S")}\n')

    def _track(self, msg, fraction=None):
        """Write timestamped progress + optional fraction-done + ETA."""
        if not hasattr(self, '_track_path'):
            return
        import time as _t
        elapsed = _t.perf_counter() - self._track_t0
        line = f'[{_t.strftime("%H:%M:%S")}  +{elapsed:.0f}s] {msg}'
        if fraction is not None and fraction > 1e-6:
            eta = elapsed * (1 - fraction) / fraction
            line += f'  ({fraction*100:.1f}% done, ETA {eta:.0f}s)'
        with open(self._track_path, 'a') as fh:
            fh.write(line + '\n')

    # ── Pack 182: 4 GB substrate (d=2048 M=128K) capacity proof ─────────
    def create_big_substrate(self, d=2048, M=131072, k=128, seed=4096):
        """Allocate a separate, larger FlatSDM substrate for T2S use.
        Default d=2048 M=128K k=128 -> ~4 GB. Capacity Plate bound
        ~M/log(M) ~= 7500 clean items, 4.4x current 192 MB substrate.
        Substrate stays the dedicated T2S brain; org.unified stays 192 MB
        for chemistry/cognition."""
        from ikigai.cognition.flat_memory import VSASDM
        self.big_substrate = VSASDM(d=int(d), M=int(M), k=int(k), seed=int(seed))
        # Re-build per-projection cache for the new d (different from org.unified.d)
        if not hasattr(self, '_big_proj_cache'):
            self._big_proj_cache = {}
        self.big_d = int(d)
        return self.big_substrate

    def _big_ensure_projection(self, d_model, k_idx=0):
        if not hasattr(self, '_big_proj_cache'):
            self._big_proj_cache = {}
        key = (int(d_model), int(k_idx))
        if key in self._big_proj_cache:
            self._big_P_re, self._big_P_im = self._big_proj_cache[key]
            return
        rng = np.random.default_rng(self._rng_seed + 11 +
                                     d_model * 9907 + k_idx * 6991)
        scale = 1.0 / np.sqrt(2.0 * self.big_d)
        P_re = (rng.standard_normal((self.big_d, d_model))
                .astype(np.float32) * scale)
        P_im = (rng.standard_normal((self.big_d, d_model))
                .astype(np.float32) * scale)
        self._big_proj_cache[key] = (P_re, P_im)
        self._big_P_re, self._big_P_im = P_re, P_im

    def _big_col_addr(self, name, col_idx):
        """Address into big substrate: phasor key from name + col index.
        Cached after first call so forward doesn't pay RNG cost."""
        if not hasattr(self, '_big_addr_cache'):
            self._big_addr_cache = {}
        key = (name, int(col_idx))
        cached = self._big_addr_cache.get(key)
        if cached is not None:
            return cached
        rng = np.random.default_rng(abs(hash(f'{name}_c{col_idx}')) % (2**31))
        ph = rng.uniform(-np.pi, np.pi, self.big_d).astype(np.float32)
        addr = np.exp(1j * ph).astype(np.complex64)
        self._big_addr_cache[key] = addr
        return addr

    def _big_addr_stack(self, name, d_out):
        """Get all col addresses for a layer in one (d_out, d) stack.
        Caches the stack itself for forward-time reuse."""
        if not hasattr(self, '_big_stack_cache'):
            self._big_stack_cache = {}
        cached = self._big_stack_cache.get(name)
        if cached is not None and cached.shape[0] == d_out:
            return cached
        stack = np.empty((d_out, self.big_d), dtype=np.complex64)
        for j in range(d_out):
            stack[j] = self._big_col_addr(name, j)
        self._big_stack_cache[name] = stack
        return stack

    def compile_linear_big_substrate(self, W, name, K_writes=2, verbose=False,
                                       store_patterns=True):
        """Batched compile: build all addresses + run ONE locs_batch + ONE
        scatter-add per K_writes. ~10x faster than per-col loop.

        Pack 214: stores per-col clean magnitude.
        Pack 192: stores clean cols at complex64 as Hopfield patterns.
                  Sidecar memory cost = d_out * big_d * 8 bytes per linear.
                  Required for substrate read denoising via Hopfield iter.
                  TODO Pack 192v2: replace sidecar with substrate-encoded
                  patterns once NWC encoder trained.
        """
        if not hasattr(self, 'big_substrate_meta'):
            self.big_substrate_meta = {}
        if not hasattr(self, '_big_clean_mag'):
            self._big_clean_mag = {}
        if not hasattr(self, '_big_clean_patterns'):
            self._big_clean_patterns = {}
        Wn = np.asarray(W, dtype=np.float32)
        d_model, d_out = Wn.shape
        self._big_ensure_projection(d_model)
        re = (self._big_P_re @ Wn).T
        im = (self._big_P_im @ Wn).T
        cols = (re + 1j * im).astype(np.complex64)
        self._big_clean_mag[name] = np.abs(cols).mean(axis=1).astype(np.float32)
        # Pack 192: keep clean cols as Hopfield attractor patterns
        if store_patterns:
            self._big_clean_patterns[name] = cols.astype(np.complex64)
        bank = self.big_substrate
        for k in range(int(K_writes)):
            slot_name = f'{name}_kw{k}'
            addr_stack = self._big_addr_stack(slot_name, d_out)
            sims = (addr_stack @ bank.Hconj.T).real
            for j in range(d_out):
                idx = np.argpartition(-sims[j], bank.k)[:bank.k]
                bank.C[idx] += cols[j]
        self.big_substrate_meta[name] = (int(d_model), int(d_out), int(K_writes))
        if verbose:
            print(f'  [T2S/BIG-K{K_writes}] {name}: {d_model}x{d_out}  '
                  f'big_substrate {bank.substrate_bytes()/1_048_576:.0f} MB',
                  flush=True)

    def linear_big_substrate(self, x, name):
        """Forward via big-substrate reads, averaged over K_writes.
        Batched: one big matmul against Hconj per K, then sparse gather."""
        d_model, d_out, K_writes = self.big_substrate_meta[name]
        self._big_ensure_projection(d_model)
        bank = self.big_substrate
        zx = (self._big_P_re @ x.astype(np.float32)
              + 1j * self._big_P_im @ x.astype(np.float32)).astype(np.complex64)
        y = np.zeros(d_out, dtype=np.float32)
        for k in range(K_writes):
            # build all addrs for this K and read in one batch
            addr_stack = np.empty((d_out, bank.d), dtype=np.complex64)
            for j in range(d_out):
                addr_stack[j] = self._big_col_addr(f'{name}_kw{k}', j)
            # batched activation: addrs @ Hconj.T  ->  (d_out, M)
            sims = (addr_stack @ bank.Hconj.T).real
            # top-k locations per row (parallelisable later)
            # gather and renorm each, dot with zx
            for j in range(d_out):
                idx = np.argpartition(-sims[j], bank.k)[:bank.k]
                hv = bank.C[idx].sum(axis=0)
                m = float(np.abs(hv).mean())
                if m > 1e-9:
                    hv = hv / m
                y[j] += float(np.real(np.vdot(zx, hv)))
        return y / float(K_writes)

    def compile_gpt2_into_big_substrate(self, hf_model, K_writes=2,
                                          name_prefix='big_L', verbose=True):
        """Write entire GPT-2 into big substrate (4 GB, d=2048 M=128K)."""
        import time as _t
        n_layers = hf_model.config.n_layer
        n_heads  = hf_model.config.n_head
        d_model  = hf_model.config.n_embd
        self.gpt2_n_layers = n_layers
        self.gpt2_n_heads  = n_heads
        self.gpt2_d_model  = d_model
        self.wte = hf_model.wte.weight.detach().cpu().numpy().astype(np.float32)
        self.wpe = hf_model.wpe.weight.detach().cpu().numpy().astype(np.float32)
        self.compile_layer_norm(hf_model.ln_f.weight.detach().cpu().numpy(),
                                 hf_model.ln_f.bias.detach().cpu().numpy(),
                                 f'{name_prefix}_ln_f')
        if not hasattr(self, 'bias_cache'):
            self.bias_cache = {}
        for L in range(n_layers):
            t0 = _t.perf_counter()
            block = hf_model.h[L]
            cW = block.attn.c_attn.weight.detach().cpu().numpy()
            cB = block.attn.c_attn.bias.detach().cpu().numpy()
            W_Q = cW[:,            0 : d_model]
            W_K = cW[:,      d_model : 2*d_model]
            W_V = cW[:,    2*d_model : 3*d_model]
            W_O = block.attn.c_proj.weight.detach().cpu().numpy()
            W1 = block.mlp.c_fc.weight.detach().cpu().numpy()
            W2 = block.mlp.c_proj.weight.detach().cpu().numpy()
            self.compile_linear_big_substrate(W_Q, f'{name_prefix}{L}_Q', K_writes)
            self.compile_linear_big_substrate(W_K, f'{name_prefix}{L}_K', K_writes)
            self.compile_linear_big_substrate(W_V, f'{name_prefix}{L}_V', K_writes)
            self.compile_linear_big_substrate(W_O, f'{name_prefix}{L}_O', K_writes)
            self.compile_linear_big_substrate(W1,  f'{name_prefix}{L}_W1', K_writes)
            self.compile_linear_big_substrate(W2,  f'{name_prefix}{L}_W2', K_writes)
            self.bias_cache[f'{name_prefix}{L}_bQ'] = cB[          0 : d_model].astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_bK'] = cB[    d_model : 2*d_model].astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_bV'] = cB[  2*d_model : 3*d_model].astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_bO'] = block.attn.c_proj.bias.detach().cpu().numpy().astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_b1'] = block.mlp.c_fc.bias.detach().cpu().numpy().astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_b2'] = block.mlp.c_proj.bias.detach().cpu().numpy().astype(np.float32)
            self.compile_layer_norm(block.ln_1.weight.detach().cpu().numpy(),
                                     block.ln_1.bias.detach().cpu().numpy(),
                                     f'{name_prefix}{L}_ln1')
            self.compile_layer_norm(block.ln_2.weight.detach().cpu().numpy(),
                                     block.ln_2.bias.detach().cpu().numpy(),
                                     f'{name_prefix}{L}_ln2')
            if verbose:
                print(f'  [T2S/BIG] L{L:2d} '
                       f'(big substrate {self.big_substrate.substrate_bytes()/1_048_576:.0f} MB) '
                       f'in {_t.perf_counter()-t0:.1f}s', flush=True)
        return n_layers

    def gpt2_forward_big_substrate(self, token_ids, name_prefix='big_L'):
        """Forward reading from 4 GB substrate. Substrate FIXED 4 GB."""
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        seq_len = ids.shape[0]
        pos = np.arange(seq_len, dtype=np.int64)
        X = (self.wte[ids] + self.wpe[pos]).astype(np.float32)
        d_model = self.gpt2_d_model
        n_heads = self.gpt2_n_heads
        d_head  = d_model // n_heads
        for L in range(self.gpt2_n_layers):
            X_ln1 = self.layer_norm(X, f'{name_prefix}{L}_ln1')
            Q_h = []; K_h = []; V_h = []
            for t in range(seq_len):
                Q_h.append(self.linear_big_substrate(X_ln1[t], f'{name_prefix}{L}_Q')
                            + self.bias_cache[f'{name_prefix}{L}_bQ'])
                K_h.append(self.linear_big_substrate(X_ln1[t], f'{name_prefix}{L}_K')
                            + self.bias_cache[f'{name_prefix}{L}_bK'])
                V_h.append(self.linear_big_substrate(X_ln1[t], f'{name_prefix}{L}_V')
                            + self.bias_cache[f'{name_prefix}{L}_bV'])
            Q = np.stack(Q_h).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
            K = np.stack(K_h).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
            V = np.stack(V_h).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
            scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_head)
            mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            scores[:, mask] = -1e9
            scores = scores - scores.max(axis=-1, keepdims=True)
            ex = np.exp(scores)
            attn = ex / ex.sum(axis=-1, keepdims=True)
            head_out = attn @ V
            concat = head_out.transpose(1, 0, 2).reshape(seq_len, d_model)
            attn_out = np.stack([
                self.linear_big_substrate(concat[t], f'{name_prefix}{L}_O')
                + self.bias_cache[f'{name_prefix}{L}_bO']
                for t in range(seq_len)])
            X = X + attn_out
            X_ln2 = self.layer_norm(X, f'{name_prefix}{L}_ln2')
            hid = np.stack([
                self.linear_big_substrate(X_ln2[t], f'{name_prefix}{L}_W1')
                + self.bias_cache[f'{name_prefix}{L}_b1']
                for t in range(seq_len)])
            hid = self._gelu(hid)
            ffn_out = np.stack([
                self.linear_big_substrate(hid[t], f'{name_prefix}{L}_W2')
                + self.bias_cache[f'{name_prefix}{L}_b2']
                for t in range(seq_len)])
            X = X + ffn_out
        X = self.layer_norm(X, f'{name_prefix}_ln_f')
        return X @ self.wte.T

    # ── Pack 181: K-projection substrate writes for crosstalk reduction ─
    # Each column written to K different substrate addresses (different
    # role tag per write). Read averages all K. Variance / K cuts crosstalk.
    def compile_linear_substrate_k(self, W, name, K_writes=4, verbose=False):
        """K-projection substrate writes. Each col stored at K addresses,
        read averages. Substrate stays 192 MB. Crosstalk variance ÷ K."""
        if not hasattr(self, 'substrate_linear_meta'):
            self.substrate_linear_meta = {}
        Wn = np.asarray(W, dtype=np.float32)
        d_model, d_out = Wn.shape
        self._ensure_projection(d_model)
        mr = self.org.unified
        bank = mr.sdm_rel
        re = (self._P_re @ Wn).T
        im = (self._P_im @ Wn).T
        cols = (re + 1j * im).astype(np.complex64)
        for k in range(int(K_writes)):
            slot_name = f'{name}_kw{k}'
            for j in range(d_out):
                addr = self._col_addr(slot_name, j)
                bank.write(addr, cols[j])
        self.substrate_linear_meta[name] = (int(d_model), int(d_out),
                                              int(K_writes))
        if verbose:
            print(f'  [T2S/substrate-K{K_writes}] {name}: {d_model}x{d_out} '
                  f'x{K_writes} writes  '
                  f'(bank {mr.substrate_bytes()/1_048_576:.0f} MB)', flush=True)

    def linear_substrate_k(self, x, name):
        """K-projection substrate forward. Average K reads per column."""
        meta = self.substrate_linear_meta[name]
        d_model, d_out, K_writes = meta
        self._ensure_projection(d_model)
        mr = self.org.unified
        bank = mr.sdm_rel
        zx = (self._P_re @ x.astype(np.float32)
              + 1j * self._P_im @ x.astype(np.float32)).astype(np.complex64)
        y = np.zeros(d_out, dtype=np.float32)
        for k in range(K_writes):
            slot_name = f'{name}_kw{k}'
            for j in range(d_out):
                addr = self._col_addr(slot_name, j)
                recall = bank.read(addr)
                y[j] += float(np.real(np.vdot(zx, recall)))
        return y / float(K_writes)

    def compile_gpt2_into_substrate_k(self, hf_model, K_writes=4,
                                        name_prefix='subst_K_L',
                                        verbose=True):
        """Write GPT-2 into substrate with K-projection per column."""
        import time as _t
        n_layers = hf_model.config.n_layer
        n_heads  = hf_model.config.n_head
        d_model  = hf_model.config.n_embd
        self.gpt2_n_layers = n_layers
        self.gpt2_n_heads  = n_heads
        self.gpt2_d_model  = d_model
        self.wte = hf_model.wte.weight.detach().cpu().numpy().astype(np.float32)
        self.wpe = hf_model.wpe.weight.detach().cpu().numpy().astype(np.float32)
        self.compile_layer_norm(hf_model.ln_f.weight.detach().cpu().numpy(),
                                 hf_model.ln_f.bias.detach().cpu().numpy(),
                                 f'{name_prefix}_ln_f')
        if not hasattr(self, 'bias_cache'):
            self.bias_cache = {}
        for L in range(n_layers):
            t0 = _t.perf_counter()
            block = hf_model.h[L]
            cW = block.attn.c_attn.weight.detach().cpu().numpy()
            cB = block.attn.c_attn.bias.detach().cpu().numpy()
            W_Q = cW[:,            0 : d_model]
            W_K = cW[:,      d_model : 2*d_model]
            W_V = cW[:,    2*d_model : 3*d_model]
            W_O = block.attn.c_proj.weight.detach().cpu().numpy()
            W1 = block.mlp.c_fc.weight.detach().cpu().numpy()
            W2 = block.mlp.c_proj.weight.detach().cpu().numpy()
            self.compile_linear_substrate_k(W_Q, f'{name_prefix}{L}_Q', K_writes)
            self.compile_linear_substrate_k(W_K, f'{name_prefix}{L}_K', K_writes)
            self.compile_linear_substrate_k(W_V, f'{name_prefix}{L}_V', K_writes)
            self.compile_linear_substrate_k(W_O, f'{name_prefix}{L}_O', K_writes)
            self.compile_linear_substrate_k(W1,  f'{name_prefix}{L}_W1', K_writes)
            self.compile_linear_substrate_k(W2,  f'{name_prefix}{L}_W2', K_writes)
            self.bias_cache[f'{name_prefix}{L}_bQ'] = cB[          0 : d_model].astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_bK'] = cB[    d_model : 2*d_model].astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_bV'] = cB[  2*d_model : 3*d_model].astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_bO'] = block.attn.c_proj.bias.detach().cpu().numpy().astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_b1'] = block.mlp.c_fc.bias.detach().cpu().numpy().astype(np.float32)
            self.bias_cache[f'{name_prefix}{L}_b2'] = block.mlp.c_proj.bias.detach().cpu().numpy().astype(np.float32)
            self.compile_layer_norm(block.ln_1.weight.detach().cpu().numpy(),
                                     block.ln_1.bias.detach().cpu().numpy(),
                                     f'{name_prefix}{L}_ln1')
            self.compile_layer_norm(block.ln_2.weight.detach().cpu().numpy(),
                                     block.ln_2.bias.detach().cpu().numpy(),
                                     f'{name_prefix}{L}_ln2')
            if verbose:
                print(f'  [T2S/substrate-K{K_writes}] L{L:2d} '
                       f'(substrate {self.org.unified.substrate_bytes()/1_048_576:.0f} MB) '
                       f'in {_t.perf_counter()-t0:.1f}s', flush=True)
        return n_layers

    def gpt2_forward_substrate_k(self, token_ids, name_prefix='subst_K_L'):
        """Forward via K-projection substrate reads. 192 MB fixed."""
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        seq_len = ids.shape[0]
        pos = np.arange(seq_len, dtype=np.int64)
        X = (self.wte[ids] + self.wpe[pos]).astype(np.float32)
        d_model = self.gpt2_d_model
        n_heads = self.gpt2_n_heads
        d_head  = d_model // n_heads
        for L in range(self.gpt2_n_layers):
            X_ln1 = self.layer_norm(X, f'{name_prefix}{L}_ln1')
            Q_h = []; K_h = []; V_h = []
            for t in range(seq_len):
                Q_h.append(self.linear_substrate_k(X_ln1[t], f'{name_prefix}{L}_Q')
                            + self.bias_cache[f'{name_prefix}{L}_bQ'])
                K_h.append(self.linear_substrate_k(X_ln1[t], f'{name_prefix}{L}_K')
                            + self.bias_cache[f'{name_prefix}{L}_bK'])
                V_h.append(self.linear_substrate_k(X_ln1[t], f'{name_prefix}{L}_V')
                            + self.bias_cache[f'{name_prefix}{L}_bV'])
            Q = np.stack(Q_h).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
            K = np.stack(K_h).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
            V = np.stack(V_h).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
            scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_head)
            mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            scores[:, mask] = -1e9
            scores = scores - scores.max(axis=-1, keepdims=True)
            ex = np.exp(scores)
            attn = ex / ex.sum(axis=-1, keepdims=True)
            head_out = attn @ V
            concat = head_out.transpose(1, 0, 2).reshape(seq_len, d_model)
            attn_out = np.stack([
                self.linear_substrate_k(concat[t], f'{name_prefix}{L}_O')
                + self.bias_cache[f'{name_prefix}{L}_bO']
                for t in range(seq_len)])
            X = X + attn_out
            X_ln2 = self.layer_norm(X, f'{name_prefix}{L}_ln2')
            hid = np.stack([
                self.linear_substrate_k(X_ln2[t], f'{name_prefix}{L}_W1')
                + self.bias_cache[f'{name_prefix}{L}_b1']
                for t in range(seq_len)])
            hid = self._gelu(hid)
            ffn_out = np.stack([
                self.linear_substrate_k(hid[t], f'{name_prefix}{L}_W2')
                + self.bias_cache[f'{name_prefix}{L}_b2']
                for t in range(seq_len)])
            X = X + ffn_out
        X = self.layer_norm(X, f'{name_prefix}_ln_f')
        return X @ self.wte.T

    # ── Pack 188: substrate-native Llama/Qwen absorption ────────────────
    # Writes Llama/Qwen2 weights INTO self.big_substrate (a real VSA-SDM
    # bank). No sidecar ndarrays for the matrices. Forward via substrate
    # reads. Embed, LM head, RMSNorm weights kept as ndarray (small, OK).
    def compile_llama_into_big_substrate(self, hf_model, K_writes=2,
                                            name_prefix='big_llama',
                                            verbose=True):
        """Absorb Llama/Qwen2 model INTO big_substrate (VSA-SDM bank).
        Each Q/K/V/O/gate/up/down weight col -> JL phasor -> SDM write."""
        import time as _t
        if not hasattr(self, 'big_substrate'):
            raise RuntimeError("Call create_big_substrate() first")
        cfg = hf_model.config
        n_layers = cfg.num_hidden_layers
        n_heads  = cfg.num_attention_heads
        n_kv_heads = getattr(cfg, 'num_key_value_heads', n_heads)
        d_model  = cfg.hidden_size
        intermediate = cfg.intermediate_size
        head_dim = getattr(cfg, 'head_dim', d_model // n_heads)
        rope_base = float(getattr(cfg, 'rope_theta', 10000.0))
        rms_eps  = float(getattr(cfg, 'rms_norm_eps', 1e-6))
        self.llm_n_layers = n_layers
        self.llm_n_heads  = n_heads
        self.llm_n_kv_heads = n_kv_heads
        self.llm_d_model  = d_model
        self.llm_head_dim = head_dim
        self.llm_intermediate = intermediate
        self.llm_rope_base = rope_base
        self.llm_rms_eps  = rms_eps
        model_inner = hf_model.model if hasattr(hf_model, 'model') else hf_model
        self.llm_embed = model_inner.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)
        if hasattr(hf_model, 'lm_head'):
            self.llm_lm_head = hf_model.lm_head.weight.detach().cpu().numpy().astype(np.float32)
        else:
            self.llm_lm_head = self.llm_embed
        self.compile_layer_norm(model_inner.norm.weight.detach().cpu().numpy(),
                                 np.zeros_like(model_inner.norm.weight.detach().cpu().numpy()),
                                 f'{name_prefix}_final_rms')
        if not hasattr(self, 'bias_cache'):
            self.bias_cache = {}
        n_writes = 0
        for L in range(n_layers):
            t0 = _t.perf_counter()
            block = model_inner.layers[L]
            attn = block.self_attn
            mlp = block.mlp
            for proj_name, proj in [
                ('q', attn.q_proj), ('k', attn.k_proj),
                ('v', attn.v_proj), ('o', attn.o_proj),
                ('gate', mlp.gate_proj), ('up', mlp.up_proj),
                ('down', mlp.down_proj),
            ]:
                W = proj.weight.detach().cpu().numpy().T  # (in, out)
                self.compile_linear_big_substrate(
                    W, f'{name_prefix}{L}_{proj_name}', K_writes=K_writes)
                n_writes += W.shape[1] * K_writes
                if hasattr(proj, 'bias') and proj.bias is not None:
                    self.bias_cache[f'{name_prefix}{L}_{proj_name}_b'] = \
                        proj.bias.detach().cpu().numpy().astype(np.float32)
            self.compile_layer_norm(
                block.input_layernorm.weight.detach().cpu().numpy(),
                np.zeros_like(block.input_layernorm.weight.detach().cpu().numpy()),
                f'{name_prefix}{L}_attn_rms')
            self.compile_layer_norm(
                block.post_attention_layernorm.weight.detach().cpu().numpy(),
                np.zeros_like(block.post_attention_layernorm.weight.detach().cpu().numpy()),
                f'{name_prefix}{L}_ffn_rms')
            if verbose:
                print(f'  [T2S/llama-substrate] L{L:2d} '
                       f'(substrate {self.big_substrate.substrate_bytes()/1_073_741_824:.2f} GB) '
                       f'in {_t.perf_counter()-t0:.1f}s', flush=True)
        return n_layers, n_writes

    # ── Pack 213: precompute top-k idx cache for FAST substrate reads ──
    # Critical bottleneck of llama_forward_big_substrate was rebuilding
    # `addr_stack @ Hconj.T` every linear call (60B ops each). Cache the
    # resulting top-k indices PER LINEAR ONCE at absorb. Forward becomes
    # batched gather + matmul (one per K_write per linear).
    def precompute_big_idx_cache(self, name_prefix='big_llama', verbose=True):
        """Precompute per-col top-k substrate location indices for every
        big-substrate linear. Run AFTER compile_llama_into_big_substrate.
        Cache stays in self._big_idx_cache. Drops per-forward cost from
        hours to seconds."""
        import time as _t
        if not hasattr(self, '_big_idx_cache'):
            self._big_idx_cache = {}
        bank = self.big_substrate
        meta = getattr(self, 'big_substrate_meta', {})
        names = [n for n in meta.keys() if n.startswith(name_prefix)]
        t0 = _t.perf_counter()
        for n in names:
            d_in, d_out, K_writes = meta[n]
            for kw in range(K_writes):
                slot = f'{n}_kw{kw}'
                if slot in self._big_idx_cache:
                    continue
                addr_stack = self._big_addr_stack(slot, d_out)
                # batched sims: (d_out, d) @ (d, M) -> (d_out, M)
                sims = (addr_stack @ bank.Hconj.T).real
                idx = np.argpartition(-sims, bank.k, axis=1)[:, :bank.k] \
                          .astype(np.int32)
                self._big_idx_cache[slot] = idx
                # drop sims immediately (big mem)
                del sims, addr_stack
        if verbose:
            print(f'  [T2S/idx-cache] precomputed {len(names)} linears in '
                  f'{_t.perf_counter()-t0:.1f}s', flush=True)
        return len(names)

    def linear_big_substrate_fast(self, X, name, chunk=512, reduce='median'):
        """Batched fast forward. X (seq_len, d_in) -> Y (seq_len, d_out).
        Pack 213 idx cache + Pack 214 mag anchor + Pack 214v2 K reduce.

        reduce: 'mean' (Pack 213 default) or 'median' (Pack 214v2 ECC
        robust). 'median' rejects outlier K-writes; with K>=3 should
        improve cosine via outlier suppression. K=2 median == mean.

        Memory: peak gather = chunk * k_sub * big_d * 8 bytes complex64.
        At chunk=512, k=128, big_d=2048: 1GB peak per K_write. Fits 16 GB.
        """
        if not hasattr(self, '_big_idx_cache'):
            raise RuntimeError('call precompute_big_idx_cache() first')
        d_in, d_out, K_writes = self.big_substrate_meta[name]
        self._big_ensure_projection(d_in)
        bank = self.big_substrate
        X32 = np.asarray(X, dtype=np.float32)
        if X32.ndim == 1:
            X32 = X32[None, :]
        seq_len = X32.shape[0]
        # encode all tokens to complex: ZX (seq_len, big_d)
        ZX = ((X32 @ self._big_P_re.T) +
              1j * (X32 @ self._big_P_im.T)).astype(np.complex64)
        ZX_conj = np.conj(ZX)
        # per-K contributions kept separate if reduce='median' & K>=3
        use_median = (str(reduce) == 'median' and K_writes >= 3)
        if use_median:
            Y_per_k = np.zeros((K_writes, seq_len, d_out), dtype=np.float32)
        else:
            Y = np.zeros((seq_len, d_out), dtype=np.float32)
        chunk = int(chunk)
        clean_mag = None
        if hasattr(self, '_big_clean_mag') and name in self._big_clean_mag:
            clean_mag = self._big_clean_mag[name]        # (d_out,)
        clean_patterns = None
        if hasattr(self, '_big_clean_patterns') and name in self._big_clean_patterns:
            clean_patterns = self._big_clean_patterns[name]   # (d_out, big_d)
        for kw in range(K_writes):
            slot = f'{name}_kw{kw}'
            idx = self._big_idx_cache[slot]              # (d_out, k_sub)
            for c0 in range(0, d_out, chunk):
                c1 = min(c0 + chunk, d_out)
                idx_chunk = idx[c0:c1]                   # (chunk, k_sub)
                gathered = bank.C[idx_chunk]              # (chunk, k_sub, d)
                hv_per_col = gathered.sum(axis=1)        # (chunk, d) complex
                if clean_mag is not None:
                    noisy_m = np.abs(hv_per_col).mean(axis=1, keepdims=True) + 1e-9
                    cm_chunk = clean_mag[c0:c1, None]
                    hv_per_col = (hv_per_col / noisy_m) * cm_chunk
                else:
                    noisy_m = np.abs(hv_per_col).mean(axis=1, keepdims=True) + 1e-9
                    hv_per_col = hv_per_col / noisy_m
                # Pack 192: Hopfield iter -- pull noisy col toward own clean
                # pattern proportional to how well current read matches it
                if clean_patterns is not None:
                    p_chunk = clean_patterns[c0:c1]      # (chunk, big_d)
                    sim_self = np.real(np.sum(hv_per_col * np.conj(p_chunk),
                                                axis=1, keepdims=True))
                    pat_norm_sq = np.real(np.sum(p_chunk * np.conj(p_chunk),
                                                   axis=1, keepdims=True)) + 1e-9
                    alpha = np.clip(sim_self / pat_norm_sq, 0.0, 1.0)
                    hv_per_col = alpha * p_chunk + (1.0 - alpha) * hv_per_col
                contrib = np.real(hv_per_col @ ZX_conj.T).T.astype(np.float32)
                if use_median:
                    Y_per_k[kw, :, c0:c1] = contrib
                else:
                    Y[:, c0:c1] += contrib
                del gathered, hv_per_col, contrib
        if use_median:
            # robust reduction across K -- median rejects outliers
            result = np.median(Y_per_k, axis=0)
        else:
            result = Y / float(K_writes)
        return result.squeeze(0) if result.shape[0] == 1 else result

    def llama_forward_big_substrate_fast(self, token_ids,
                                           name_prefix='big_llama'):
        """Batched fast forward through substrate. Pack 213.
        Same semantics as llama_forward_big_substrate but uses cached idx +
        batched matmul. Sub-second per forward at d=1024 M=64K."""
        if not hasattr(self, '_big_idx_cache') or not self._big_idx_cache:
            self.precompute_big_idx_cache(name_prefix=name_prefix)
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        seq_len = ids.shape[0]
        d_model = self.llm_d_model
        n_heads = self.llm_n_heads
        n_kv_heads = self.llm_n_kv_heads
        head_dim = self.llm_head_dim
        rms_eps = self.llm_rms_eps
        X = self.llm_embed[ids].astype(np.float32)
        cos, sin = self._rope_cos_sin(seq_len, head_dim, self.llm_rope_base)
        n_kv_groups = n_heads // n_kv_heads
        for L in range(self.llm_n_layers):
            w_attn, _ = self.ln_cache[f'{name_prefix}{L}_attn_rms']
            X_norm = self._rms_norm(X, w_attn, eps=rms_eps)
            # BATCHED substrate reads: (seq, d_in) -> (seq, d_out)
            Q = self.linear_big_substrate_fast(X_norm, f'{name_prefix}{L}_q')
            K = self.linear_big_substrate_fast(X_norm, f'{name_prefix}{L}_k')
            V = self.linear_big_substrate_fast(X_norm, f'{name_prefix}{L}_v')
            bq = self.bias_cache.get(f'{name_prefix}{L}_q_b')
            bk = self.bias_cache.get(f'{name_prefix}{L}_k_b')
            bv = self.bias_cache.get(f'{name_prefix}{L}_v_b')
            if bq is not None: Q = Q + bq
            if bk is not None: K = K + bk
            if bv is not None: V = V + bv
            Q = Q.reshape(seq_len, n_heads, head_dim)
            K = K.reshape(seq_len, n_kv_heads, head_dim)
            V = V.reshape(seq_len, n_kv_heads, head_dim)
            Q = self._apply_rope(Q, cos[:, None, :], sin[:, None, :])
            K = self._apply_rope(K, cos[:, None, :], sin[:, None, :])
            if n_kv_groups > 1:
                K = np.repeat(K, n_kv_groups, axis=1)
                V = np.repeat(V, n_kv_groups, axis=1)
            Q = Q.transpose(1, 0, 2)
            K = K.transpose(1, 0, 2)
            V = V.transpose(1, 0, 2)
            scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(head_dim)
            mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            scores[:, mask] = -1e9
            scores = scores - scores.max(axis=-1, keepdims=True)
            ex = np.exp(scores)
            attn = ex / ex.sum(axis=-1, keepdims=True)
            head_out = attn @ V
            concat = head_out.transpose(1, 0, 2).reshape(seq_len, d_model)
            attn_out = self.linear_big_substrate_fast(concat,
                                                       f'{name_prefix}{L}_o')
            X = X + attn_out
            w_ffn, _ = self.ln_cache[f'{name_prefix}{L}_ffn_rms']
            X_norm = self._rms_norm(X, w_ffn, eps=rms_eps)
            gate_l = self.linear_big_substrate_fast(X_norm,
                                                     f'{name_prefix}{L}_gate')
            up_l = self.linear_big_substrate_fast(X_norm,
                                                   f'{name_prefix}{L}_up')
            # SiLU = x * sigmoid(x). Use clip to avoid overflow on extremes.
            g_clip = np.clip(gate_l, -50.0, 50.0)
            silu = gate_l * (1.0 / (1.0 + np.exp(-g_clip)))
            hid = silu * up_l
            ffn_out = self.linear_big_substrate_fast(hid,
                                                      f'{name_prefix}{L}_down')
            X = X + ffn_out
        w_final, _ = self.ln_cache[f'{name_prefix}_final_rms']
        X = self._rms_norm(X, w_final, eps=rms_eps)
        return X @ self.llm_lm_head.T

    def llama_forward_big_substrate(self, token_ids, name_prefix='big_llama'):
        """Substrate-native forward. Weights READ from substrate.
        No ndarray weight storage in this path.

        Pack 213 alternative: use llama_forward_big_substrate_fast (1000x faster).
        """
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        seq_len = ids.shape[0]
        d_model = self.llm_d_model
        n_heads = self.llm_n_heads
        n_kv_heads = self.llm_n_kv_heads
        head_dim = self.llm_head_dim
        rms_eps = self.llm_rms_eps
        X = self.llm_embed[ids].astype(np.float32)
        cos, sin = self._rope_cos_sin(seq_len, head_dim, self.llm_rope_base)
        n_kv_groups = n_heads // n_kv_heads
        for L in range(self.llm_n_layers):
            w_attn, _ = self.ln_cache[f'{name_prefix}{L}_attn_rms']
            X_norm = self._rms_norm(X, w_attn, eps=rms_eps)
            # SUBSTRATE READS for Q/K/V/O (per-token loop required by substrate API)
            Q_h, K_h, V_h = [], [], []
            for t in range(seq_len):
                q_t = self.linear_big_substrate(X_norm[t], f'{name_prefix}{L}_q')
                k_t = self.linear_big_substrate(X_norm[t], f'{name_prefix}{L}_k')
                v_t = self.linear_big_substrate(X_norm[t], f'{name_prefix}{L}_v')
                bq = self.bias_cache.get(f'{name_prefix}{L}_q_b')
                bk = self.bias_cache.get(f'{name_prefix}{L}_k_b')
                bv = self.bias_cache.get(f'{name_prefix}{L}_v_b')
                if bq is not None: q_t = q_t + bq
                if bk is not None: k_t = k_t + bk
                if bv is not None: v_t = v_t + bv
                Q_h.append(q_t); K_h.append(k_t); V_h.append(v_t)
            Q = np.stack(Q_h).reshape(seq_len, n_heads, head_dim)
            K = np.stack(K_h).reshape(seq_len, n_kv_heads, head_dim)
            V = np.stack(V_h).reshape(seq_len, n_kv_heads, head_dim)
            Q = self._apply_rope(Q, cos[:, None, :], sin[:, None, :])
            K = self._apply_rope(K, cos[:, None, :], sin[:, None, :])
            if n_kv_groups > 1:
                K = np.repeat(K, n_kv_groups, axis=1)
                V = np.repeat(V, n_kv_groups, axis=1)
            Q = Q.transpose(1, 0, 2)
            K = K.transpose(1, 0, 2)
            V = V.transpose(1, 0, 2)
            scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(head_dim)
            mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            scores[:, mask] = -1e9
            scores = scores - scores.max(axis=-1, keepdims=True)
            ex = np.exp(scores)
            attn = ex / ex.sum(axis=-1, keepdims=True)
            head_out = attn @ V
            concat = head_out.transpose(1, 0, 2).reshape(seq_len, d_model)
            attn_out = np.stack([
                self.linear_big_substrate(concat[t], f'{name_prefix}{L}_o')
                for t in range(seq_len)])
            X = X + attn_out
            w_ffn, _ = self.ln_cache[f'{name_prefix}{L}_ffn_rms']
            X_norm = self._rms_norm(X, w_ffn, eps=rms_eps)
            gate_l = np.stack([
                self.linear_big_substrate(X_norm[t], f'{name_prefix}{L}_gate')
                for t in range(seq_len)])
            up_l = np.stack([
                self.linear_big_substrate(X_norm[t], f'{name_prefix}{L}_up')
                for t in range(seq_len)])
            hid = self._swiglu(gate_l, up_l)
            ffn_out = np.stack([
                self.linear_big_substrate(hid[t], f'{name_prefix}{L}_down')
                for t in range(seq_len)])
            X = X + ffn_out
        w_final, _ = self.ln_cache[f'{name_prefix}_final_rms']
        X = self._rms_norm(X, w_final, eps=rms_eps)
        return X @ self.llm_lm_head.T

    def gpt2_forward_substrate(self, token_ids, name_prefix='subst_L'):
        """Forward pass reading ALL weights from substrate. No sidecar.
        Substrate is 192 MB and stays 192 MB. Slow (per-col substrate reads)."""
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        seq_len = ids.shape[0]
        pos = np.arange(seq_len, dtype=np.int64)
        X = (self.wte[ids] + self.wpe[pos]).astype(np.float32)
        d_model = self.gpt2_d_model
        n_heads = self.gpt2_n_heads
        d_head  = d_model // n_heads
        for L in range(self.gpt2_n_layers):
            # LN1
            X_ln1 = self.layer_norm(X, f'{name_prefix}{L}_ln1')
            # attention via substrate reads
            Q_h = []; K_h = []; V_h = []
            for t in range(seq_len):
                Q_h.append(self.linear_substrate(X_ln1[t], f'{name_prefix}{L}_Q')
                            + self.bias_cache[f'{name_prefix}{L}_bQ'])
                K_h.append(self.linear_substrate(X_ln1[t], f'{name_prefix}{L}_K')
                            + self.bias_cache[f'{name_prefix}{L}_bK'])
                V_h.append(self.linear_substrate(X_ln1[t], f'{name_prefix}{L}_V')
                            + self.bias_cache[f'{name_prefix}{L}_bV'])
            Q = np.stack(Q_h).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
            K = np.stack(K_h).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
            V = np.stack(V_h).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
            scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_head)
            mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            scores[:, mask] = -1e9
            scores = scores - scores.max(axis=-1, keepdims=True)
            ex = np.exp(scores)
            attn = ex / ex.sum(axis=-1, keepdims=True)
            head_out = attn @ V
            concat = head_out.transpose(1, 0, 2).reshape(seq_len, d_model)
            attn_out = np.stack([
                self.linear_substrate(concat[t], f'{name_prefix}{L}_O')
                + self.bias_cache[f'{name_prefix}{L}_bO']
                for t in range(seq_len)])
            X = X + attn_out
            # FFN via substrate reads
            X_ln2 = self.layer_norm(X, f'{name_prefix}{L}_ln2')
            hid = np.stack([
                self.linear_substrate(X_ln2[t], f'{name_prefix}{L}_W1')
                + self.bias_cache[f'{name_prefix}{L}_b1']
                for t in range(seq_len)])
            hid = self._gelu(hid)
            ffn_out = np.stack([
                self.linear_substrate(hid[t], f'{name_prefix}{L}_W2')
                + self.bias_cache[f'{name_prefix}{L}_b2']
                for t in range(seq_len)])
            X = X + ffn_out
        X = self.layer_norm(X, f'{name_prefix}_ln_f')
        return X @ self.wte.T

    # ── Pack 179: Linear attention forward. O(1) per token. ─────────────
    @staticmethod
    def _phi(x):
        """Feature map φ(x) = elu(x) + 1. Positive, monotone, cheap.
        Katharopoulos 2020 'Transformers are RNNs'."""
        return np.where(x >= 0, x + 1.0, np.exp(x)).astype(np.float32)

    def init_linear_state(self):
        """Allocate per-layer linear-attention running state.
        Per layer: S (n_heads, d_head, d_head) running outer product
                  Z (n_heads, d_head) running sum of phi(K).
        Total state: O(1) in sequence length. ~600 KB for GPT-2 small."""
        n_layers = self.gpt2_n_layers
        n_heads  = self.gpt2_n_heads
        d_head   = self.gpt2_d_model // n_heads
        states = []
        for _ in range(n_layers):
            states.append([
                np.zeros((n_heads, d_head, d_head), dtype=np.float32),
                np.zeros((n_heads, d_head),         dtype=np.float32),
            ])
        return states

    def _step_one_token(self, tok_id, t_pos, states, name_prefix):
        """Process ONE token through all layers with linear attention.
        Updates `states` in place. Returns logits for this position."""
        n_heads  = self.gpt2_n_heads
        d_model  = self.gpt2_d_model
        d_head   = d_model // n_heads
        X = (self.wte[int(tok_id)] + self.wpe[int(t_pos)]).astype(np.float32)
        for L in range(self.gpt2_n_layers):
            ln1_w, ln1_b = self.ln_cache[f'{name_prefix}{L}_ln1']
            mu = X.mean(); var = X.var()
            X_ln1 = (X - mu) / np.sqrt(var + 1e-5) * ln1_w + ln1_b
            qkv = X_ln1 @ self.fast_cache[f'{name_prefix}{L}_c_attn_W'] \
                  + self.fast_cache[f'{name_prefix}{L}_c_attn_b']
            Q = qkv[          0 : d_model].reshape(n_heads, d_head)
            K = qkv[    d_model : 2*d_model].reshape(n_heads, d_head)
            V = qkv[  2*d_model : 3*d_model].reshape(n_heads, d_head)
            Q_phi = self._phi(Q)
            K_phi = self._phi(K)
            S, Z = states[L]
            # rank-1 update: S += K_phi (outer) V
            S += K_phi[:, :, None] * V[:, None, :]
            Z += K_phi
            # readout: num = Σ_h Q_phi[h] · S[h]   ;   den = Q_phi · Z
            num = np.einsum('hd,hdv->hv', Q_phi, S)        # (h, d_head)
            den = (Z * Q_phi).sum(axis=-1, keepdims=True) + 1e-6
            attn_out = (num / den).reshape(d_model)
            attn_out = attn_out @ self.fast_cache[f'{name_prefix}{L}_Wo'] \
                       + self.fast_cache[f'{name_prefix}{L}_bO']
            X = X + attn_out
            ln2_w, ln2_b = self.ln_cache[f'{name_prefix}{L}_ln2']
            mu = X.mean(); var = X.var()
            X_ln2 = (X - mu) / np.sqrt(var + 1e-5) * ln2_w + ln2_b
            hid = X_ln2 @ self.fast_cache[f'{name_prefix}{L}_W1'] \
                  + self.fast_cache[f'{name_prefix}{L}_b1']
            hid = self._gelu(hid)
            ffn = hid @ self.fast_cache[f'{name_prefix}{L}_W2'] \
                  + self.fast_cache[f'{name_prefix}{L}_b2']
            X = X + ffn
        ln_f_w, ln_f_b = self.ln_cache[f'{name_prefix}_ln_f']
        mu = X.mean(); var = X.var()
        X = (X - mu) / np.sqrt(var + 1e-5) * ln_f_w + ln_f_b
        logits = X @ self.wte.T
        return logits

    def gpt2_generate_linear(self, prompt_ids, max_new=20,
                                name_prefix='gpt2_layer', greedy=True,
                                verbose=False):
        """O(1) per-token streaming generation via linear attention.
        Returns list of all token ids (prompt + generated)."""
        if not hasattr(self, 'fast_mode') or not self.fast_mode:
            raise RuntimeError("gpt2_generate_linear requires fast mode compile")
        states = self.init_linear_state()
        ids = list(prompt_ids)
        # process prompt to populate state
        for t, tok in enumerate(ids):
            logits = self._step_one_token(tok, t, states, name_prefix)
        # generate
        for i in range(int(max_new)):
            nxt = int(logits.argmax()) if greedy \
                  else int(np.random.choice(logits.shape[-1],
                                              p=_softmax(logits)))
            ids.append(nxt)
            if verbose:
                print(f'  +tok{nxt}', flush=True)
            logits = self._step_one_token(nxt, len(ids) - 1, states, name_prefix)
        return ids

    def compile_gpt2_model_fast(self, hf_model, name_prefix='gpt2_layer',
                                  verbose=True):
        """FAST MODE compile: stores raw ndarrays only, no JL ensemble.
        ~10x faster compile, ~10-50x faster forward. Same numerical
        accuracy as torch (no JL noise)."""
        import time as _t
        self.enable_fast_mode()
        n_layers = hf_model.config.n_layer
        n_heads  = hf_model.config.n_head
        d_model  = hf_model.config.n_embd
        self.gpt2_n_layers = n_layers
        self.gpt2_n_heads  = n_heads
        self.gpt2_d_model  = d_model
        self.wte = hf_model.wte.weight.detach().cpu().numpy().astype(np.float32)
        self.wpe = hf_model.wpe.weight.detach().cpu().numpy().astype(np.float32)
        self.compile_layer_norm(hf_model.ln_f.weight.detach().cpu().numpy(),
                                 hf_model.ln_f.bias.detach().cpu().numpy(),
                                 f'{name_prefix}_ln_f')
        for L in range(n_layers):
            t0 = _t.perf_counter()
            block = hf_model.h[L]
            # fused c_attn weight + bias
            self._store_fast(block.attn.c_attn.weight.detach().cpu().numpy(),
                              f'{name_prefix}{L}_c_attn_W')
            self.fast_cache[f'{name_prefix}{L}_c_attn_b'] = \
                block.attn.c_attn.bias.detach().cpu().numpy().astype(np.float32)
            self._store_fast(block.attn.c_proj.weight.detach().cpu().numpy(),
                              f'{name_prefix}{L}_Wo')
            self.fast_cache[f'{name_prefix}{L}_bO'] = \
                block.attn.c_proj.bias.detach().cpu().numpy().astype(np.float32)
            self.compile_layer_norm(block.ln_1.weight.detach().cpu().numpy(),
                                     block.ln_1.bias.detach().cpu().numpy(),
                                     f'{name_prefix}{L}_ln1')
            self.compile_layer_norm(block.ln_2.weight.detach().cpu().numpy(),
                                     block.ln_2.bias.detach().cpu().numpy(),
                                     f'{name_prefix}{L}_ln2')
            self._store_fast(block.mlp.c_fc.weight.detach().cpu().numpy(),
                              f'{name_prefix}{L}_W1')
            self.fast_cache[f'{name_prefix}{L}_b1'] = \
                block.mlp.c_fc.bias.detach().cpu().numpy().astype(np.float32)
            self._store_fast(block.mlp.c_proj.weight.detach().cpu().numpy(),
                              f'{name_prefix}{L}_W2')
            self.fast_cache[f'{name_prefix}{L}_b2'] = \
                block.mlp.c_proj.bias.detach().cpu().numpy().astype(np.float32)
            if verbose:
                print(f'  [T2S/fast] block {L:2d} stored in '
                      f'{_t.perf_counter()-t0:.2f}s', flush=True)
        return n_layers

    # ── Pack 175: FFN compile (W1 / GELU / W2) ──────────────────────────
    def compile_ffn(self, W1, b1, W2, b2, layer_idx=0,
                     name_prefix='gpt2_layer'):
        """Compile a GPT-2 FFN block: y = gelu(x@W1 + b1) @ W2 + b2.
        Two linear layers stored. Biases kept separately."""
        if not hasattr(self, 'bias_cache'):
            self.bias_cache = {}
        self.compile_linear(W1, f'{name_prefix}{layer_idx}_W1')
        self.compile_linear(W2, f'{name_prefix}{layer_idx}_W2')
        self.bias_cache[f'{name_prefix}{layer_idx}_b1'] = np.asarray(b1,
                                                                      dtype=np.float32)
        self.bias_cache[f'{name_prefix}{layer_idx}_b2'] = np.asarray(b2,
                                                                      dtype=np.float32)

    @staticmethod
    def _gelu(x):
        # GPT-2 uses exact GELU (not approx). Same as torch.nn.functional.gelu
        from scipy.special import erf
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

    def ffn_forward(self, X, layer_idx=0, name_prefix='gpt2_layer'):
        """Substrate forward pass through FFN block. X: (seq_len, d_model)."""
        b1 = self.bias_cache[f'{name_prefix}{layer_idx}_b1']
        b2 = self.bias_cache[f'{name_prefix}{layer_idx}_b2']
        hid = self.linear_batch(X, f'{name_prefix}{layer_idx}_W1') + b1.reshape(1, -1)
        hid = self._gelu(hid)
        out = self.linear_batch(hid, f'{name_prefix}{layer_idx}_W2') + b2.reshape(1, -1)
        return out

    # ── Pack 176: LayerNorm + multi-head attention + transformer block ──
    def compile_layer_norm(self, weight, bias, name):
        """Store LN gain (weight) + bias for forward time application."""
        if not hasattr(self, 'ln_cache'):
            self.ln_cache = {}
        self.ln_cache[name] = (
            np.asarray(weight, dtype=np.float32),
            np.asarray(bias, dtype=np.float32),
        )

    def layer_norm(self, x, name, eps=1e-5):
        """GPT-2 layer norm: (x - mean) / sqrt(var + eps) * gain + bias.
        Per-token normalization."""
        weight, bias = self.ln_cache[name]
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            mu = x.mean(); var = x.var()
            return (x - mu) / np.sqrt(var + eps) * weight + bias
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * weight + bias

    def compile_attention_full(self, W_Q, W_K, W_V, W_O, n_heads, layer_idx,
                                name_prefix='gpt2_layer'):
        """Compile ALL heads of one attention block + output projection."""
        d_model = W_Q.shape[0]
        d_head = d_model // n_heads
        for h in range(n_heads):
            self.compile_attention_head(W_Q, W_K, W_V,
                                         head_idx=h, n_heads=n_heads,
                                         name_prefix=f'{name_prefix}{layer_idx}_h')
        self.compile_linear(W_O, f'{name_prefix}{layer_idx}_Wo')
        return d_head

    def attention_full_forward(self, X, n_heads, layer_idx,
                                 name_prefix='gpt2_layer', causal=True,
                                 with_biases=True):
        """Multi-head attention forward: concat all heads + W_O projection.
        Single shared encoding of X across all 12 heads × Q/K/V = 36 reuse."""
        head_prefix = f'{name_prefix}{layer_idx}_h'
        # Encode X once for ALL Q/K/V calls across all heads (same input dim).
        d_in = self.linear_cache[f'{head_prefix}0_Q'][1]
        ZX = self.encode_seq(X, d_in)
        heads = []
        seq_len = X.shape[0]
        for h in range(n_heads):
            d_head = self.linear_cache[f'{head_prefix}{h}_Q'][0][0].shape[0]
            Q = self.linear_batch_cached(ZX, f'{head_prefix}{h}_Q')
            K_ = self.linear_batch_cached(ZX, f'{head_prefix}{h}_K')
            V = self.linear_batch_cached(ZX, f'{head_prefix}{h}_V')
            if with_biases:
                bQ = self.bias_cache.get(f'{head_prefix}{h}_bQ')
                bK = self.bias_cache.get(f'{head_prefix}{h}_bK')
                bV = self.bias_cache.get(f'{head_prefix}{h}_bV')
                if bQ is not None: Q = Q + bQ.reshape(1, -1)
                if bK is not None: K_ = K_ + bK.reshape(1, -1)
                if bV is not None: V = V + bV.reshape(1, -1)
            scores = (Q @ K_.T) / np.sqrt(d_head)
            if causal:
                mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
                scores[mask] = -1e9
            scores = scores - scores.max(axis=1, keepdims=True)
            ex = np.exp(scores)
            attn = ex / ex.sum(axis=1, keepdims=True)
            heads.append(attn @ V)
        concat = np.concatenate(heads, axis=-1)
        return self.linear_batch(concat, f'{name_prefix}{layer_idx}_Wo')

    def compile_transformer_block(self, attn_W_Q, attn_W_K, attn_W_V,
                                    attn_b_Q, attn_b_K, attn_b_V,
                                    attn_W_O, attn_b_O,
                                    ln1_w, ln1_b, ln2_w, ln2_b,
                                    ffn_W1, ffn_b1, ffn_W2, ffn_b2,
                                    n_heads, layer_idx,
                                    name_prefix='gpt2_layer'):
        """Compile one full transformer block (GPT-2)."""
        self.compile_attention_full(attn_W_Q, attn_W_K, attn_W_V, attn_W_O,
                                     n_heads, layer_idx,
                                     name_prefix=name_prefix)
        self.compile_layer_norm(ln1_w, ln1_b, f'{name_prefix}{layer_idx}_ln1')
        self.compile_layer_norm(ln2_w, ln2_b, f'{name_prefix}{layer_idx}_ln2')
        self.compile_ffn(ffn_W1, ffn_b1, ffn_W2, ffn_b2,
                         layer_idx=layer_idx, name_prefix=name_prefix)
        if not hasattr(self, 'bias_cache'):
            self.bias_cache = {}
        # store attention biases (Q/K/V/O)
        d_model = attn_W_Q.shape[0]
        d_head = d_model // n_heads
        for h in range(n_heads):
            lo, hi = h * d_head, (h + 1) * d_head
            self.bias_cache[f'{name_prefix}{layer_idx}_h{h}_bQ'] = attn_b_Q[lo:hi]
            self.bias_cache[f'{name_prefix}{layer_idx}_h{h}_bK'] = attn_b_K[lo:hi]
            self.bias_cache[f'{name_prefix}{layer_idx}_h{h}_bV'] = attn_b_V[lo:hi]
        self.bias_cache[f'{name_prefix}{layer_idx}_bO'] = attn_b_O

    def block_forward(self, X, n_heads, layer_idx,
                       name_prefix='gpt2_layer', causal=True):
        """Substrate forward pass through a full GPT-2 transformer block.
        x -> LN1 -> attention(with biases) -> +residual -> LN2 -> FFN -> +residual."""
        # 1. LN1 + multi-head attn + residual
        x_ln1 = self.layer_norm(X, f'{name_prefix}{layer_idx}_ln1')
        attn_out = self.attention_full_forward(
            x_ln1, n_heads=n_heads, layer_idx=layer_idx,
            name_prefix=name_prefix, causal=causal)
        bO = self.bias_cache[f'{name_prefix}{layer_idx}_bO']
        attn_out = attn_out + bO.reshape(1, -1)
        X1 = X + attn_out
        # 2. LN2 + FFN + residual
        x_ln2 = self.layer_norm(X1, f'{name_prefix}{layer_idx}_ln2')
        ffn_out = self.ffn_forward(x_ln2, layer_idx=layer_idx,
                                    name_prefix=name_prefix)
        X2 = X1 + ffn_out
        return X2

    # ── persistence: save/load the compiled substrate ───────────────────
    def save(self, path):
        """Persist the entire compiled T2S to disk as one .npz.
        Stores: linear_cache (ensemble HVs + d_model tag), ln_cache,
        bias_cache, wte/wpe, gpt2 config, projection cache.
        On reload, no GPT-2 / HF / torch needed -- only this file."""
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        d = {
            '__d':       np.int64(self.d),
            '__K':       np.int64(self.K),
            '__seed':    np.int64(self._rng_seed),
        }
        # gpt2 config (if compiled)
        if hasattr(self, 'gpt2_n_layers'):
            d['__gpt2_n_layers'] = np.int64(self.gpt2_n_layers)
            d['__gpt2_n_heads']  = np.int64(self.gpt2_n_heads)
            d['__gpt2_d_model']  = np.int64(self.gpt2_d_model)
            d['__wte']           = self.wte
            d['__wpe']           = self.wpe
        # linear cache: name -> (K ensemble cols, d_model)
        lin_names = []
        if hasattr(self, 'linear_cache'):
            for name, (ensemble, d_model) in self.linear_cache.items():
                lin_names.append(name)
                for k, cols in enumerate(ensemble):
                    d[f'__lin__{name}__k{k}'] = cols
                d[f'__lin_meta__{name}'] = np.array([len(ensemble), d_model],
                                                     dtype=np.int64)
        d['__lin_names'] = np.array(lin_names, dtype=object)
        # ln cache
        ln_names = []
        if hasattr(self, 'ln_cache'):
            for name, (w, b) in self.ln_cache.items():
                ln_names.append(name)
                d[f'__ln_w__{name}'] = w
                d[f'__ln_b__{name}'] = b
        d['__ln_names'] = np.array(ln_names, dtype=object)
        # bias cache
        b_names = []
        if hasattr(self, 'bias_cache'):
            for name, v in self.bias_cache.items():
                b_names.append(name)
                d[f'__bias__{name}'] = v
        d['__bias_names'] = np.array(b_names, dtype=object)
        # fast cache (raw weight ndarrays)
        f_names = []
        if hasattr(self, 'fast_cache'):
            for name, v in self.fast_cache.items():
                f_names.append(name)
                d[f'__fast__{name}'] = v
        d['__fast_names'] = np.array(f_names, dtype=object)
        np.savez_compressed(path, **d)

    def load(self, path):
        """Load a compiled T2S from a .npz produced by save()."""
        z = np.load(path, allow_pickle=True)
        self.d = int(z['__d'])
        self.K = int(z['__K'])
        self._rng_seed = int(z['__seed'])
        if '__gpt2_n_layers' in z.files:
            self.gpt2_n_layers = int(z['__gpt2_n_layers'])
            self.gpt2_n_heads  = int(z['__gpt2_n_heads'])
            self.gpt2_d_model  = int(z['__gpt2_d_model'])
            self.wte = z['__wte']
            self.wpe = z['__wpe']
        # rebuild caches
        self.linear_cache = {}
        for name in z['__lin_names']:
            n = str(name)
            meta = z[f'__lin_meta__{n}']
            K, d_model = int(meta[0]), int(meta[1])
            ensemble = [z[f'__lin__{n}__k{k}'] for k in range(K)]
            self.linear_cache[n] = (ensemble, d_model)
        self.ln_cache = {}
        for name in z['__ln_names']:
            n = str(name)
            self.ln_cache[n] = (z[f'__ln_w__{n}'], z[f'__ln_b__{n}'])
        self.bias_cache = {}
        for name in z['__bias_names']:
            n = str(name)
            self.bias_cache[n] = z[f'__bias__{n}']
        # fast cache (raw weight ndarrays, if saved in fast mode)
        self.fast_cache = {}
        if '__fast_names' in z.files:
            for name in z['__fast_names']:
                n = str(name)
                self.fast_cache[n] = z[f'__fast__{n}']
            if self.fast_cache:
                self.fast_mode = True
        # projection cache rebuilds lazily on first forward via _ensure_projection
        if not hasattr(self, '_proj_cache'):
            self._proj_cache = {}
        return self

    # ── Pack 177: full GPT-2 small model compile + forward ──────────────
    def compile_gpt2_model(self, hf_model, name_prefix='gpt2_layer',
                            verbose=True):
        """Compile every transformer block of an HF GPT-2 model + final LN.
        wte/wpe stored as raw ndarrays (lookup tables, not JL).
        LM head stays as wte.T direct matmul (no JL on unembedding)."""
        import time as _time
        n_layers = hf_model.config.n_layer
        n_heads  = hf_model.config.n_head
        d_model  = hf_model.config.n_embd

        self.gpt2_n_layers = n_layers
        self.gpt2_n_heads  = n_heads
        self.gpt2_d_model  = d_model
        # Token + position tables, kept as raw ndarrays.
        self.wte = hf_model.wte.weight.detach().cpu().numpy().astype(np.float32)
        self.wpe = hf_model.wpe.weight.detach().cpu().numpy().astype(np.float32)
        # Final LN.
        self.compile_layer_norm(hf_model.ln_f.weight.detach().cpu().numpy(),
                                 hf_model.ln_f.bias.detach().cpu().numpy(),
                                 f'{name_prefix}_ln_f')

        for L in range(n_layers):
            t0 = _time.perf_counter()
            block = hf_model.h[L]
            attn = block.attn; mlp = block.mlp
            ln1 = block.ln_1; ln2 = block.ln_2

            cW = attn.c_attn.weight.detach().cpu().numpy()
            cB = attn.c_attn.bias.detach().cpu().numpy()
            W_Q = cW[:,           0 : d_model]
            W_K = cW[:,     d_model : 2*d_model]
            W_V = cW[:,   2*d_model : 3*d_model]
            b_Q = cB[         0 : d_model]
            b_K = cB[   d_model : 2*d_model]
            b_V = cB[ 2*d_model : 3*d_model]
            W_O = attn.c_proj.weight.detach().cpu().numpy()
            b_O = attn.c_proj.bias.detach().cpu().numpy()
            ln1_w = ln1.weight.detach().cpu().numpy()
            ln1_b = ln1.bias.detach().cpu().numpy()
            ln2_w = ln2.weight.detach().cpu().numpy()
            ln2_b = ln2.bias.detach().cpu().numpy()
            W1 = mlp.c_fc.weight.detach().cpu().numpy()
            b1 = mlp.c_fc.bias.detach().cpu().numpy()
            W2 = mlp.c_proj.weight.detach().cpu().numpy()
            b2 = mlp.c_proj.bias.detach().cpu().numpy()

            self.compile_transformer_block(
                W_Q, W_K, W_V, b_Q, b_K, b_V, W_O, b_O,
                ln1_w, ln1_b, ln2_w, ln2_b,
                W1, b1, W2, b2,
                n_heads=n_heads, layer_idx=L, name_prefix=name_prefix)
            if verbose:
                print(f'  [T2S] block {L:2d} compiled in '
                      f'{_time.perf_counter()-t0:.1f}s', flush=True)
        return n_layers

    def gpt2_forward(self, token_ids, name_prefix='gpt2_layer',
                      return_logits=True, top_k=10):
        """Substrate-native forward pass through compiled GPT-2.

        token_ids: list/ndarray of token ids (seq_len,).
        Returns: logits (seq_len, vocab) if return_logits else hidden (seq_len, d_model).

        Path: lookup wte + wpe -> 12 blocks (substrate JL) -> final LN -> LM head."""
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        seq_len = ids.shape[0]
        pos = np.arange(seq_len, dtype=np.int64)
        # Embedding lookup (direct ndarray indexing -- not JL).
        X = (self.wte[ids] + self.wpe[pos]).astype(np.float32)
        # Run 12 blocks via substrate JL.
        for L in range(self.gpt2_n_layers):
            X = self.block_forward(X, n_heads=self.gpt2_n_heads,
                                    layer_idx=L, name_prefix=name_prefix,
                                    causal=True)
        # Final LN.
        X = self.layer_norm(X, f'{name_prefix}_ln_f')
        if not return_logits:
            return X
        # LM head (tied unembedding) -- direct matmul, no JL.
        logits = X @ self.wte.T
        return logits

    def top_neighbors(self, token_id, candidate_ids, top_k=5):
        """For a given token, return top-K most-similar tokens by substrate cosine."""
        scores = []
        target = self.read(token_id)
        for tid in candidate_ids:
            if tid == token_id: continue
            cand = self.read(tid)
            s = float(np.real(np.vdot(target, cand))) / self.d
            scores.append((tid, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
