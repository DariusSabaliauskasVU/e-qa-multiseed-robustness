#!/usr/bin/env python3
"""
BERTScore and per-category / per-document breakdown from saved predictions (reviewer R1 #12).

Input : prediction files written by `run_experiments.py --save-preds DIR`
Output: semantic_metrics_per_run.csv   one row per configuration and seed (EM, F1, ROUGE-L, BERTScore P/R/F1)
        semantic_metrics_summary.csv   mean and SD over seeds per configuration
        breakdown_by_category.csv      mean F1 / EM / BERTScore-F1 per configuration and requirement category
        breakdown_by_document.csv      the same per source document

Usage:
    pip install bert-score --no-deps
    python compute_semantic_metrics.py --preds preds
    python compute_semantic_metrics.py --preds preds --no-bertscore     # overlap metrics and breakdowns only
"""
import argparse, csv, glob, json, os, statistics as st, collections
from run_experiments import compute_em, compute_f1, rouge_l


def load_meta(path, n):
    cat, doc = ["?"] * n, ["?"] * n
    if os.path.exists(path):
        for i, r in enumerate(csv.DictReader(open(path, newline="", encoding="utf-8"))):
            if i < n:
                cat[i] = r.get("requirement_category", "?") or "?"
                doc[i] = r.get("source_document", "?") or "?"
    return cat, doc


def mean_sd(v):
    v = list(v)
    if any(x != x for x in v):                       # NaN: BERTScore was not computed
        return float("nan"), float("nan")
    return round(st.mean(v), 4), (round(st.stdev(v), 4) if len(v) > 1 else 0.0)


def write(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds")
    ap.add_argument("--meta", default=os.path.join("data", "RE120_provenance.csv"))
    ap.add_argument("--model-type", default="roberta-large", help="BERTScore backbone")
    ap.add_argument("--no-bertscore", action="store_true")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.preds, "*.json")))
    if not files:
        ap.error(f"no prediction files in {args.preds}")
    runs = [json.load(open(f, encoding="utf-8")) for f in files]
    n = len(runs[0]["items"])
    cat, doc = load_meta(args.meta, n)

    scorer = None
    if not args.no_bertscore:
        from bert_score import BERTScorer
        scorer = BERTScorer(model_type=args.model_type, lang="en", rescale_with_baseline=True)
        print("BERTScore hash:", scorer.hash)

    per_run, per_item = [], []          # per_item: (config, seed, q, f1, em, bs_f1)
    for run in runs:
        preds = [it["prediction"] for it in run["items"]]
        golds = [it["gold"] for it in run["items"]]
        em = [compute_em(p, g) for p, g in zip(preds, golds)]
        f1 = [compute_f1(p, g) for p, g in zip(preds, golds)]
        rg = [rouge_l(p, g) for p, g in zip(preds, golds)]
        if scorer:
            safe = [p if p.strip() else "." for p in preds]          # an empty prediction cannot be embedded
            P, Rr, F = scorer.score(safe, golds)
            P, Rr, F = P.tolist(), Rr.tolist(), F.tolist()
            for i, p in enumerate(preds):
                if not p.strip():
                    P[i] = Rr[i] = F[i] = 0.0
        else:
            P = Rr = F = [float("nan")] * n
        per_run.append(dict(config=run["config"], seed=run["seed"], n_questions=n,
                            EM=round(st.mean(em), 4), F1=round(st.mean(f1), 4), ROUGE_L=round(st.mean(rg), 4),
                            BERTScore_P=round(st.mean(P), 4), BERTScore_R=round(st.mean(Rr), 4),
                            BERTScore_F1=round(st.mean(F), 4),
                            empty_predictions=sum(1 for p in preds if not p.strip())))
        for q in range(n):
            per_item.append((run["config"], run["seed"], q, f1[q], em[q], F[q]))
        print(per_run[-1], flush=True)
    write("semantic_metrics_per_run.csv", per_run)

    summ = []
    by = collections.defaultdict(list)
    for r in per_run:
        by[r["config"]].append(r)
    for c, rs in by.items():
        row = dict(config=c, seeds=len(rs))
        for k in ("EM", "F1", "ROUGE_L", "BERTScore_P", "BERTScore_R", "BERTScore_F1"):
            row[k + "_mean"], row[k + "_sd"] = mean_sd([x[k] for x in rs])
        summ.append(row)
    write("semantic_metrics_summary.csv", summ)

    for label, key, path in (("category", cat, "breakdown_by_category.csv"), ("document", doc, "breakdown_by_document.csv")):
        acc = collections.defaultdict(lambda: collections.defaultdict(list))     # (config, group) -> seed -> [(f1, em, bs)]
        for c, s, q, f, e, b in per_item:
            acc[(c, key[q])][s].append((f, e, b))
        rows = []
        for (c, g), seeds in sorted(acc.items()):
            f1s = [st.mean(x[0] for x in v) for v in seeds.values()]
            ems = [st.mean(x[1] for x in v) for v in seeds.values()]
            bss = [st.mean(x[2] for x in v) for v in seeds.values()]
            r = dict(config=c, **{label: g}, n_questions=len(next(iter(seeds.values()))), seeds=len(seeds))
            r["F1_mean"], r["F1_sd"] = mean_sd(f1s)
            r["EM_mean"], r["EM_sd"] = mean_sd(ems)
            r["BERTScore_F1_mean"], r["BERTScore_F1_sd"] = mean_sd(bss)
            rows.append(r)
        write(path, rows)
    print("\nconfig        F1       EM       ROUGE-L  BERTScore-F1")
    for r in summ:
        print(f"{r['config']:<12}  {r['F1_mean']:.4f}   {r['EM_mean']:.4f}   {r['ROUGE_L_mean']:.4f}   {r['BERTScore_F1_mean']:.4f}")


if __name__ == "__main__":
    main()
