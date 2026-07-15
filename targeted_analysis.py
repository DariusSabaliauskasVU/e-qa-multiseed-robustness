#!/usr/bin/env python3
"""Confirmatory statistics for the RE-QA multi-seed study (F1 column).

Paired tests are valid: the same ten seeds and the same fixed grouped train/val split
are reused across all models (a common-random-numbers design), so per-seed scores are
matched across configurations.

Reports, matching the manuscript exactly:
  * Friedman omnibus.
  * Confirmatory pairwise Wilcoxon signed-rank tests with Holm correction applied
    WITHIN each pre-specified hypothesis family (H1, H3) to preserve power.
  * Mean difference with 95% CI, paired Cohen's d, and matched-pairs rank-biserial
    correlation (the effect size appropriate for the Wilcoxon signed-rank test).
  * TOST equivalence for H4 (LSTM depth) and H5 (linear vs LSTM heads) via the 90% CI
    of the mean difference, with a sensitivity analysis at margins of +/-0.02, +/-0.03, +/-0.05.
"""
import csv, math
from collections import defaultdict
from scipy import stats

CSV = "results_clean.csv"
METRIC_IDX = 5  # F1
SEEDS = [13,42,73,101,123,202,333,777,1337,2024]

data = defaultdict(dict)
for r in csv.reader(open(CSV)):
    if not r or r[0]=="config" or len(r)<=METRIC_IDX: continue
    data[r[0]][int(r[1])] = float(r[METRIC_IDX])
def vec(c): return [data[c][s] for s in SEEDS]

def ci(a, b):
    df=[x-y for x,y in zip(vec(a),vec(b))]; n=len(df); m=sum(df)/n
    sd=math.sqrt(sum((x-m)**2 for x in df)/(n-1)); se=sd/math.sqrt(n)
    return m, (m-stats.t.ppf(.975,n-1)*se, m+stats.t.ppf(.975,n-1)*se), \
              (m-stats.t.ppf(.95,n-1)*se,  m+stats.t.ppf(.95,n-1)*se)   # mean, 95% CI, 90% CI
def cohend(a, b):
    df=[x-y for x,y in zip(vec(a),vec(b))]; m=sum(df)/len(df)
    return m/math.sqrt(sum((x-m)**2 for x in df)/(len(df)-1))
def rank_biserial(a, b):
    df=[x-y for x,y in zip(vec(a),vec(b)) if x-y!=0]; n=len(df)
    rk=stats.rankdata([abs(x) for x in df])
    Tp=sum(r for r,x in zip(rk,df) if x>0); Tn=sum(r for r,x in zip(rk,df) if x<0)
    return (Tp-Tn)/(n*(n+1)/2)
def wilcox(a, b):
    try: return stats.wilcoxon(vec(a),vec(b))[1]
    except ValueError: return 1.0
def holm(ps):
    order=sorted(range(len(ps)),key=lambda i:ps[i]); m=len(ps); out=[0]*len(ps); run=0
    for rank,i in enumerate(order): run=max(run,min(1.0,(m-rank)*ps[i])); out[i]=run
    return out

configs=list(data.keys())
chi2,p=stats.friedmanchisquare(*[vec(c) for c in configs])
print("Friedman omnibus (F1): chi2 = %.2f, df = %d, p = %.3e\n" % (chi2, len(configs)-1, p))

FAMILIES = {
 "H1 (modern encoders > BERT)": [("neobert","bert"),("modernbert","bert"),("spanbert","bert")],
 "H3 (encoder adaptation > frozen heads)": [("partial_ft","lstm_l3"),("partial_ft","linear")],
}
print("=== Confirmatory pairwise tests (Holm within each hypothesis family) ===")
print("%-24s %8s %8s %10s %-18s %7s %7s" % ("comparison","raw p","Holm p","mean diff","95% CI","d","r_rb"))
for fam,pairs in FAMILIES.items():
    print("--", fam)
    hp=holm([wilcox(a,b) for a,b in pairs])
    for (a,b),h in zip(pairs,hp):
        m,c95,_=ci(a,b)
        print("%-24s %8.4f %8.4f %+10.4f [%+.3f, %+.3f] %+7.2f %+7.2f" %
              (a+" vs "+b, wilcox(a,b), h, m, c95[0], c95[1], cohend(a,b), rank_biserial(a,b)))

print("\n=== Equivalence (TOST via 90% CI of mean diff), margin sensitivity ===")
for label,pairs in [("H4 LSTM depth",[("lstm_l1","lstm_l2"),("lstm_l1","lstm_l3"),("lstm_l2","lstm_l3")]),
                    ("H5 linear vs LSTM",[("linear","lstm_l1"),("linear","lstm_l2"),("linear","lstm_l3")])]:
    print("--", label)
    for a,b in pairs:
        m,_,c90=ci(a,b)
        eq=lambda mg: "equiv" if (c90[0]>-mg and c90[1]<mg) else "  no "
        print("%-20s mean %+.4f 90%% CI [%+.3f, %+.3f]  ±0.02:%s ±0.03:%s ±0.05:%s" %
              (a+" vs "+b, m, c90[0], c90[1], eq(.02), eq(.03), eq(.05)))

# Sensitivity check that does not rely on seed pairing: unpaired Mann-Whitney U
# on the per-seed F1 distributions, for the confirmatory (H1, H3) contrasts.
print("\n=== Unpaired sensitivity (Mann-Whitney U), does NOT assume seed pairing ===")
print("%-24s %10s %12s" % ("comparison", "paired p", "unpaired p"))
for a, b in FAMILIES["H1 (modern encoders > BERT)"] + FAMILIES["H3 (encoder adaptation > frozen heads)"]:
    _, up = stats.mannwhitneyu(vec(a), vec(b), alternative="two-sided")
    print("%-24s %10.4f %12.4f" % (a + " vs " + b, wilcox(a, b), up))
