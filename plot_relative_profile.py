#!/usr/bin/env python3
"""Figure 4 - Relative metric profile across models.

Reads results_clean.csv (10 configurations x 10 seeds), aggregates per
configuration (mean over seeds), and renders a colour-coded matrix over
EM, F1, ROUGE-L, training time, and F1 per training minute.

Colour = per-column relative favourability (green = better, red = worse);
Train time is inverted (lower = better). Values are printed in bold black
for legibility. Outputs Figure_4.pdf and Figure_4.png.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

RESULTS_FILE = Path(__file__).with_name("results_clean.csv")

LABELS = {
    "neobert": "NeoBERT", "modernbert": "ModernBERT", "spanbert": "SpanBERT",
    "bert": "BERT", "distilbert": "DistilBERT", "partial_ft": "Partial FT",
    "lstm_l3": "BERT+LSTM 3L", "lstm_l1": "BERT+LSTM 1L",
    "lstm_l2": "BERT+LSTM 2L", "linear": "BERT+FF",
}

# ---- aggregate per configuration ----
acc = defaultdict(lambda: defaultdict(list))
with RESULTS_FILE.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        c = row["config"]
        acc[c]["EM"].append(float(row["EM"]))
        acc[c]["F1"].append(float(row["F1"]))
        acc[c]["RL"].append(float(row["ROUGE_L"]))
        acc[c]["TT"].append(float(row["train_minutes"]))

rows = []
for c, d in acc.items():
    em, f1 = np.mean(d["EM"]), np.mean(d["F1"])
    rl, tt = np.mean(d["RL"]), np.mean(d["TT"])
    rows.append((LABELS.get(c, c), em, f1, rl, tt, f1 / tt))
rows.sort(key=lambda r: -r[2])  # by F1 desc

models = [r[0] for r in rows]
data = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows])
cols = ["EM", "F1", "ROUGE-L", "Train time\n(min, lower=better)", "F1/min"]
higher_better = [True, True, True, False, True]
fmt = ["{:.3f}", "{:.3f}", "{:.3f}", "{:.1f}", "{:.3f}"]

# ---- per-column favourability in [0,1] ----
fav = np.zeros_like(data)
for j in range(data.shape[1]):
    col = data[:, j]; lo, hi = col.min(), col.max()
    norm = (col - lo) / (hi - lo) if hi > lo else np.zeros_like(col)
    fav[:, j] = norm if higher_better[j] else 1 - norm

cmap = plt.get_cmap("RdYlGn")
nrow, ncol = data.shape
fig, ax = plt.subplots(figsize=(9.2, 6.4))
ax.set_xlim(-0.2, ncol); ax.set_ylim(-0.9, nrow + 0.6)
ax.invert_yaxis(); ax.axis("off")

for j, c in enumerate(cols):
    ax.text(j + 0.5, -0.75, c, ha="center", va="center", fontsize=10.5, fontweight="bold")
for i, m in enumerate(models):
    ax.text(-0.25, i + 0.5, m, ha="right", va="center", fontsize=10.5, fontweight="bold")

for i in range(nrow):
    for j in range(ncol):
        ax.add_patch(plt.Rectangle((j + 0.06, i + 0.10), 0.88, 0.80,
                     facecolor=cmap(fav[i, j]), edgecolor="white", linewidth=1.4,
                     zorder=1, alpha=0.92))
        ax.text(j + 0.5, i + 0.5, fmt[j].format(data[i, j]),
                ha="center", va="center", fontsize=11.5, fontweight="bold",
                color="black", zorder=3)

ax.set_title("Relative metric profile across models", fontsize=13, fontweight="bold", pad=14)

cax = fig.add_axes([0.32, 0.115, 0.36, 0.020])
sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_ticks([0, 1]); cb.set_ticklabels(["worse", "better"])
cb.ax.tick_params(labelsize=9, length=0)
cb.set_label("Relative favourability (per column)", fontsize=9, labelpad=4)
fig.text(0.5, 0.045,
         "Higher is better: EM, F1, ROUGE-L, F1/min.   Lower is better: Train time.",
         ha="center", fontsize=8.5, style="italic")

plt.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.20)
fig.savefig("Figure_4.pdf", bbox_inches="tight")
fig.savefig("Figure_4.png", dpi=300, bbox_inches="tight")
print("saved Figure_4.pdf / Figure_4.png")
