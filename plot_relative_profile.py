"""Figure 4 - relative metric profile across the ten configurations.

Reads results_clean.csv (10 configurations x 10 seeds) and draws a heatmap of
five indicators: EM, F1, ROUGE-L, wall-clock training time, and F1 per training
minute. Cells are min-max normalised per column for display only; training time
is reverse-scaled so that darker (better) shading marks faster models. The
annotated numbers are the raw seed-averaged values.

Rows are ordered by mean F1, matching Table 4 of the paper.
"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

SEEDS = [13, 42, 73, 101, 123, 202, 333, 777, 1337, 2024]
LABEL = {'neobert': 'NeoBERT', 'modernbert': 'ModernBERT', 'spanbert': 'SpanBERT',
         'bert': 'BERT', 'distilbert': 'DistilBERT', 'partial_ft': 'Partial FT',
         'lstm_l3': 'BERT+LSTM 3L', 'lstm_l1': 'BERT+LSTM 1L',
         'lstm_l2': 'BERT+LSTM 2L', 'linear': 'BERT+FF'}

# ---- load per-seed results -------------------------------------------------
raw = {}
with open('results_clean.csv') as fh:
    for r in csv.reader(fh):
        if r[0] == 'config':
            continue
        raw.setdefault(r[0], []).append(
            {'time': float(r[3]), 'EM': float(r[4]), 'F1': float(r[5]), 'ROUGE-L': float(r[6])})

for cfg, rows in raw.items():
    if len(rows) != len(SEEDS):
        raise SystemExit('%s: expected %d seeds, found %d' % (cfg, len(SEEDS), len(rows)))

mean = lambda cfg, k: sum(d[k] for d in raw[cfg]) / len(raw[cfg])

configs = sorted(raw, key=lambda c: -mean(c, 'F1'))
COLUMNS = ['EM', 'F1', 'ROUGE-L', 'Train time\n(min, lower=better)', 'F1/min']

values = np.array([[mean(c, 'EM'), mean(c, 'F1'), mean(c, 'ROUGE-L'),
                    mean(c, 'time'), mean(c, 'F1') / mean(c, 'time')] for c in configs])

# ---- min-max normalise per column (display only) ---------------------------
norm = np.zeros_like(values)
for j in range(values.shape[1]):
    col = values[:, j]
    lo, hi = col.min(), col.max()
    scaled = (col - lo) / (hi - lo) if hi > lo else np.full_like(col, 0.5)
    norm[:, j] = 1.0 - scaled if COLUMNS[j].startswith('Train time') else scaled

# ---- draw ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.4))
im = ax.imshow(norm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

ax.set_xticks(range(len(COLUMNS)))
ax.set_xticklabels(COLUMNS, fontsize=8)
ax.xaxis.set_ticks_position('top')
ax.set_yticks(range(len(configs)))
ax.set_yticklabels([LABEL[c] for c in configs], fontsize=8, fontweight='bold')
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)

for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        v = values[i, j]
        txt = '%.1f' % v if COLUMNS[j].startswith('Train time') else '%.3f' % v
        ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                color='white' if norm[i, j] < 0.35 else 'black')

cax = fig.add_axes([0.34, 0.02, 0.34, 0.022])
fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[])
cax.text(-0.02, 0.5, 'worse', ha='right', va='center', fontsize=7.5, transform=cax.transAxes)
cax.text(1.02, 0.5, 'better', ha='left', va='center', fontsize=7.5, transform=cax.transAxes)

fig.subplots_adjust(left=0.19, right=0.98, top=0.88, bottom=0.10)
fig.savefig('Figure_4.pdf')
fig.savefig('Figure_4.png', dpi=400)
plt.close(fig)
print('wrote Figure_4 (order: %s)' % ', '.join(LABEL[c] for c in configs))
