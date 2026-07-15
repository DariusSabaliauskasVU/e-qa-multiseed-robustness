#!/usr/bin/env python3
"""Targeted-pairs Wilcoxon/Holm + TOST equivalence (H4/H5) on the F1 column.

Paired tests are valid: same 10 seeds and the same fixed grouped split across all models.
Reports:
  - Friedman omnibus
  - Pre-specified hypothesis pairs (H1, H3) with Wilcoxon + Holm over that small family
  - Paired Cohen's d and Cliff's delta for each pair
  - TOST equivalence at +/- 0.05 F1 for H4 (LSTM depth) and H5 (linear vs LSTM)
"""
import csv
from collections import defaultdict
import math
from scipy import stats

CSV = "results_clean.csv"
METRIC_IDX = 5  # F1 column (0-based): config,seed,epochs,minutes,EM,F1,ROUGE_L
SEEDS = [13, 42, 73, 101, 123, 202, 333, 777, 1337, 2024]

# ---- load ----
data = defaultdict(dict)  # config -> seed -> F1
with open(CSV) as fh:
    r = csv.reader(fh)
    next(r)
    for row in r:
        if not row or len(row) <= METRIC_IDX:
            continue
        cfg, seed = row[0], int(row[1])
        data[cfg][seed] = float(row[METRIC_IDX])

def vec(cfg):
    return [data[cfg][s] for s in SEEDS]

configs = list(data.keys())

def mean_sd(v):
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return m, sd

# ---- leaderboard ----
print("=" * 68)
print("LEADERBOARD (F1, mean +/- sd, n=10 seeds, 30k train)")
print("=" * 68)
board = sorted(configs, key=lambda c: mean_sd(vec(c))[0], reverse=True)
for c in board:
    m, sd = mean_sd(vec(c))
    print(f"  {c:12s} {m:.4f} +/- {sd:.4f}")

# ---- Friedman ----
mat = [vec(c) for c in configs]
chi2, p = stats.friedmanchisquare(*mat)
print("\nFriedman omnibus: chi2 = %.2f, p = %.3e" % (chi2, p))

# ---- effect sizes ----
def paired_cohens_d(a, b):
    diff = [x - y for x, y in zip(a, b)]
    m, sd = mean_sd(diff)
    return m / sd if sd > 0 else float("inf")

def cliffs_delta(a, b):
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))

def holm(pairs):
    # pairs: list of (label, pvalue). returns list of (label, p, p_holm, sig)
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    m = len(pairs)
    out = [None] * len(pairs)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pairs[i][1])
        running = max(running, adj)
        out[i] = running
    return out

# ---- pre-specified hypothesis pairs ----
# H1 modern/adapted encoders > BERT baseline; H3 partial FT > frozen heads
# ModernBERT is treated as a modern encoder (with NeoBERT) in H1.
_ALL_HYP = [
    ("H1", "neobert",    "bert"),
    ("H1", "modernbert", "bert"),
    ("H1", "spanbert",   "bert"),
    ("H1", "neobert",    "spanbert"),
    ("H1", "neobert",    "modernbert"),
    ("H3", "partial_ft", "lstm_l3"),
    ("H3", "partial_ft", "linear"),
]
# only keep pairs whose both configs are present in the CSV
HYP = [(t, a, b) for (t, a, b) in _ALL_HYP if a in data and b in data]
print("\n" + "=" * 68)
print("PRE-SPECIFIED PAIRS (Wilcoxon signed-rank, Holm over this family of 5)")
print("=" * 68)
raw = []
for tag, a, b in HYP:
    va, vb = vec(a), vec(b)
    try:
        w, pw = stats.wilcoxon(va, vb)
    except ValueError:
        w, pw = float("nan"), 1.0
    raw.append((f"{tag}: {a} vs {b}", pw))
holm_p = holm(raw)
for (label, pw), ph, (tag, a, b) in zip(raw, holm_p, HYP):
    d = paired_cohens_d(vec(a), vec(b))
    cd = cliffs_delta(vec(a), vec(b))
    star = "  SIGNIFICANT" if ph < 0.05 else ""
    print(f"  {label:26s} p={pw:.4f}  p_holm={ph:.4f}  d={d:+.2f}  cliff={cd:+.2f}{star}")

# ---- TOST equivalence ----
def tost_paired(a, b, margin):
    diff = [x - y for x, y in zip(a, b)]
    m, sd = mean_sd(diff)
    n = len(diff)
    se = sd / math.sqrt(n)
    if se == 0:
        return (0.0, abs(m) < margin)
    t_lower = (m - (-margin)) / se   # H0: diff <= -margin
    t_upper = (m - margin) / se      # H0: diff >= +margin
    df = n - 1
    p_lower = stats.t.sf(t_lower, df)      # upper tail
    p_upper = stats.t.cdf(t_upper, df)     # lower tail
    p_tost = max(p_lower, p_upper)
    return (p_tost, p_tost < 0.05)

MARGIN = 0.05
print("\n" + "=" * 68)
print(f"TOST EQUIVALENCE (margin +/- {MARGIN} F1)")
print("=" * 68)
print("H4 (LSTM depth negligible):")
for a, b in [("lstm_l1", "lstm_l2"), ("lstm_l1", "lstm_l3"), ("lstm_l2", "lstm_l3")]:
    p_t, equiv = tost_paired(vec(a), vec(b), MARGIN)
    md = mean_sd([x - y for x, y in zip(vec(a), vec(b))])[0]
    verdict = "EQUIVALENT" if equiv else "not equivalent"
    print(f"  {a} vs {b:8s}  mean_diff={md:+.4f}  p_TOST={p_t:.4f}  -> {verdict}")
print("H5 (linear vs LSTM heads):")
for b in ["lstm_l1", "lstm_l2", "lstm_l3"]:
    p_t, equiv = tost_paired(vec("linear"), vec(b), MARGIN)
    md = mean_sd([x - y for x, y in zip(vec("linear"), vec(b))])[0]
    verdict = "EQUIVALENT" if equiv else "not equivalent"
    print(f"  linear vs {b:8s}  mean_diff={md:+.4f}  p_TOST={p_t:.4f}  -> {verdict}")
