"""Generate the two paper figures from the head-to-head results."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Figure 1: head-to-head accuracy vs reasoning depth ------------------
depth      = [1, 2, 3, 5, 8]
neuroseed  = [100, 100, 100, 100, 100]
llama70b   = [93, 40, 13, 13, 20]      # Groq llama-3.3-70b-versatile
nemotron   = [100, 100, 100, 27, 0]    # OpenRouter nemotron-3-ultra-550b (frontier)

plt.figure(figsize=(7.2, 4.6))
plt.plot(depth, neuroseed, "o-", color="#1a9850", lw=3, ms=9,
         label="NeuroSeed organism (<1 MB, CPU/phone)")
plt.plot(depth, nemotron, "s--", color="#d73027", lw=2.5, ms=8,
         label="Nemotron-Ultra-550B (frontier, GPQA 86.7%)")
plt.plot(depth, llama70b, "^:", color="#fc8d59", lw=2, ms=7,
         label="Llama-3.3-70B")
plt.axhline(50, color="gray", ls=":", lw=0.8)
plt.xlabel("Reasoning depth (number of hops)", fontsize=11)
plt.ylabel("Accuracy (%)", fontsize=11)
plt.title("Equal-knowledge multi-hop reasoning:\norganism stays exact; LLMs collapse as chains deepen",
          fontsize=11)
plt.ylim(-4, 104); plt.xticks(depth)
plt.legend(fontsize=9, loc="center left")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "fig1_headtohead.png"), dpi=150)
plt.close()

# ---- Figure 2: WHY -- error compounds for autoregressive chains ----------
n = np.arange(0, 21)
plt.figure(figsize=(7.2, 4.6))
plt.plot(n, np.ones_like(n) * 100, color="#1a9850", lw=3,
         label="Exact derivation: per-hop error = 0  → accuracy flat")
for p, c in [(0.05, "#fdae61"), (0.15, "#fc8d59"), (0.30, "#d73027")]:
    plt.plot(n, 100 * (1 - p) ** n, "--", color=c, lw=2,
             label=f"Autoregressive: per-hop error {int(p*100)}%  → (1-{p})^n")
plt.xlabel("Reasoning depth (number of hops)", fontsize=11)
plt.ylabel("Chain accuracy (%)", fontsize=11)
plt.title("Why the organism wins deep chains:\nany per-hop error compounds; exact derivation has none",
          fontsize=11)
plt.ylim(0, 104)
plt.legend(fontsize=9)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "fig2_why.png"), dpi=150)
plt.close()
print("wrote fig1_headtohead.png, fig2_why.png")
