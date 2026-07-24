# Ikigai -- the wild organism, in a box.
# A container = a frozen mini-computer with Python + the organism + its brain inside,
# so it runs identically on your laptop and on any $5 server.  Build once, run anywhere.

FROM python:3.11-slim
WORKDIR /app

# 1) dependencies first (Docker caches this layer -> rebuilds are fast when only code changes)
COPY experiments/wild/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) the organism: reasoning code + cognition package + the wild server
COPY integrate.py .
COPY ikigai/ ./ikigai/
COPY experiments/wild/ ./experiments/wild/

# 3) the brain (193 MB).  READ-ONLY at runtime -- the pristine seed, never written.
COPY organism.ikg .

# 3b) the language corpus (24 MB) -- the organism learns its grammar from this by exposure
#     (curiosity-gated frames) on first boot; the learned frames then persist to /data.
COPY eng_sentences.tsv.bz2 .

# 3c) broad knowledge (ConceptNet, filtered ~99k facts) -- re-ingested at boot (write_substrate=
#     False facts don't survive the persist round-trip), so the organism knows common things.
COPY data_conceptnet_clean.tsv .

# 4) the wild LIFE (everything it learns from strangers) lives on a mounted volume,
#    so it survives container restarts.  These env vars point the server at /data.
ENV IKIGAI_STATE=/data/wild_organism.ikg \
    IKIGAI_ETH=/data/ethology.jsonl \
    IKIGAI_RSS_LIMIT_MB=850 \
    PYTHONUNBUFFERED=1
RUN mkdir -p /data
VOLUME /data

EXPOSE 8080
# the server: answers, abstains, and LEARNS from every visitor, at near-zero compute
CMD ["python", "-m", "experiments.wild.serve", "--http", "8080"]
