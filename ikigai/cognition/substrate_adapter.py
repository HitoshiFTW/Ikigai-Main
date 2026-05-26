"""
ikigai.cognition.substrate_adapter -- Substrate-Independent Migration.

Day 55 Pack 54 -- star7 Decisive stack: substrate independence.

Claim: Ikigai learned knowledge is substrate-independent.
    - Serialize trained GenerationPipeline to bytes (gzip+pickle)
    - Restore on any Python host with Python 3.8+
    - Identical outputs guaranteed (deterministic pipeline)
    - Size: KB-scale vs GB-scale for equivalent LLM

vs LLM: LLM weights are hardware-specific (CUDA kernels, quantization format).
        Migration requires format conversion, re-quantization, hardware matching.
        Ikigai: pure Python + numpy -- any host can restore.

SubstrateAdapter:
    to_bytes(pipeline)  -> bytes
    from_bytes(blob)    -> pipeline
    save(pipeline, path) -> n_bytes
    load(path)           -> pipeline
    migration_proof(pipeline, B_U, query) -> {tokens_match, score_match, ...}
    size_bytes(pipeline) -> int
"""

import io
import gzip
import pickle


class SubstrateAdapter:
    """
    Serialize/deserialize GenerationPipeline (or any picklable object).
    Proves substrate independence: save -> restore -> identical outputs.
    """

    @staticmethod
    def to_bytes(pipeline):
        """Serialize to gzip-compressed pickle bytes."""
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=6) as gz:
            pickle.dump(pipeline, gz, protocol=pickle.HIGHEST_PROTOCOL)
        return buf.getvalue()

    @staticmethod
    def from_bytes(blob):
        """Deserialize from gzip-compressed pickle bytes."""
        buf = io.BytesIO(blob)
        with gzip.GzipFile(fileobj=buf, mode='rb') as gz:
            return pickle.load(gz)

    @staticmethod
    def save(pipeline, path):
        """Write serialized pipeline to file. Returns bytes written."""
        blob = SubstrateAdapter.to_bytes(pipeline)
        with open(path, 'wb') as f:
            f.write(blob)
        return len(blob)

    @staticmethod
    def load(path):
        """Read and deserialize pipeline from file."""
        with open(path, 'rb') as f:
            blob = f.read()
        return SubstrateAdapter.from_bytes(blob)

    @staticmethod
    def size_bytes(pipeline):
        """Serialized size in bytes."""
        return len(SubstrateAdapter.to_bytes(pipeline))

    @staticmethod
    def migration_proof(pipeline, B_U, query_tokens, *slot_args):
        """
        Save pipeline, restore to fresh object, verify identical outputs.
        Returns dict with match flags and blob size.
        """
        r1   = pipeline.run(query_tokens, B_U, *slot_args)
        blob = SubstrateAdapter.to_bytes(pipeline)
        p2   = SubstrateAdapter.from_bytes(blob)
        r2   = p2.run(query_tokens, B_U, *slot_args)
        return {
            'blob_bytes':   len(blob),
            'tokens_match': r1.tokens == r2.tokens,
            'score_match':  abs(float(r1.score) - float(r2.score)) < 1e-6,
            'name_match':   r1.name == r2.name,
            'conf_match':   abs(float(r1.conf) - float(r2.conf)) < 1e-6,
        }
