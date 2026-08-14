#!/usr/bin/env python3
"""Paired statistical re-analysis of results_per_seed.csv.
Applicable when every model was run on the same fixed train/val split (only the
model-init seed varied) -> results are matched by seed label, so paired tests are used.

Outputs: Friedman omnibus, pairwise Wilcoxon signed-rank with Holm correction,
Cohen's d (paired), and a summary table. An unpaired Kruskal-Wallis test is also
reported as a sensitivity check.
"""
import argparse, csv, collections, itertools, math
from scipy import stats

def load(path):
    by=collections.defaultdict(dict)   # config -> seed -> {metric:val}
    with open(path) as f:
        for r in csv.DictReader(f):
            by[r["config"]][int(r["seed"])] = {m: float(r[m]) for m in ("EM","F1","ROUGE_L")}
    return by

def cohend_paired(a, b):
    d=[x-y for x,y in zip(a,b)]; n=len(d)
    m=sum(d)/n; sd=(sum((x-m)**2 for x in d)/(n-1))**0.5
    return m/sd if sd else float("nan")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--in",dest="inp",default="results_clean.csv")
    ap.add_argument("--metric",default="F1",choices=["EM","F1","ROUGE_L"]); a=ap.parse_args()
    by=load(a.inp); configs=sorted(by)
    seeds=sorted(set.intersection(*[set(by[c]) for c in configs]))
    if len(seeds)<3:
        print("Need >=3 shared seeds for paired tests."); return
    vec={c:[by[c][s][a.metric] for s in seeds] for c in configs}
    print(f"Metric={a.metric}  seeds={len(seeds)}  configs={len(configs)}\n")
    # Friedman (paired omnibus)
    fr=stats.friedmanchisquare(*[vec[c] for c in configs])
    kw=stats.kruskal(*[vec[c] for c in configs])
    print(f"Friedman (paired):  chi2={fr.statistic:.2f}  p={fr.pvalue:.2e}")
    print(f"Kruskal-Wallis (unpaired, for reference): H={kw.statistic:.2f}  p={kw.pvalue:.2e}\n")
    # pairwise Wilcoxon + Holm
    pairs=list(itertools.combinations(configs,2)); raw=[]
    for x,y in pairs:
        try: w=stats.wilcoxon(vec[x],vec[y]).pvalue
        except ValueError: w=1.0
        raw.append(((x,y),w,cohend_paired(vec[x],vec[y])))
    order=sorted(range(len(raw)),key=lambda i:raw[i][1]); m=len(raw); holm=[None]*len(raw)
    for rank,i in enumerate(order):
        holm[i]=min(1.0, raw[i][1]*(m-rank))
    for rank in range(1,len(order)):
        holm[order[rank]]=max(holm[order[rank]],holm[order[rank-1]])
    print("Pairwise Wilcoxon signed-rank (Holm-corrected), paired Cohen's d:")
    print(f"  {'comparison':32s} {'p_raw':>9} {'p_holm':>9} {'d':>7} sig")
    for i,((x,y),p,d) in enumerate(raw):
        sig='*' if holm[i]<0.05 else ''
        print(f"  {x+' vs '+y:32s} {p:9.4f} {holm[i]:9.4f} {d:+7.2f} {sig}")


if __name__=="__main__":
    main()
