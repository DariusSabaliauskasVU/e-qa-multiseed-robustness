#!/usr/bin/env python3
"""
Learning-rate sensitivity of the frozen-encoder prediction heads (reviewer R1 #11).

The paper trains the frozen BERT heads with lr = 2e-5. This script retrains the
three-layer LSTM head and the feed-forward head with several learning rates on a
few seeds, with everything else unchanged (same cached BERT representations, same
data split, batch size, early stopping and evaluation on RE-120).

Usage:
    python lr_sensitivity.py                         # lstm_l3 and linear, lr 2e-5 / 1e-4 / 1e-3, seeds 13 42 73
    python lr_sensitivity.py --configs lstm_l3 --lrs 2e-5 1e-3 --seeds 13 42

Output: lr_sensitivity_per_seed.csv (appended after every seed) and
        lr_sensitivity_summary.csv (mean and SD per configuration and learning rate).
"""
import argparse, csv, os, statistics as st, collections
import run_experiments as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["lstm_l3", "linear"])
    ap.add_argument("--lrs", nargs="+", type=float, default=[2e-5, 1e-4, 1e-3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 73])
    ap.add_argument("--squad-limit", type=int, default=30000)
    ap.add_argument("--data", default="requirements-test-120-fixed.json")
    ap.add_argument("--out", default="lr_sensitivity_per_seed.csv")
    ap.add_argument("--save-preds", default=None, metavar="DIR")
    args = ap.parse_args()
    R.DETERMINISTIC = False                    # same seeding as the runs reported in the paper
    R.PRED_DIR = args.save_preds
    for c in args.configs:
        if R.CONFIGS[c]["kind"] != "frozen":
            ap.error(f"{c} is not a frozen-encoder configuration")
    for c in args.configs:
        for lr in args.lrs:
            name = f"{c}@lr={lr:g}"
            R.CONFIGS[name] = dict(R.CONFIGS[c], lr=lr)
            print(f"\n===== {name} =====", flush=True)
            R.run_config(name, args.seeds, args.squad_limit, args.data, args.out)
    rows = list(csv.DictReader(open(args.out, newline="")))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["config"]].append(r)
    summ = []
    for name, rs in by.items():
        base, lr = name.split("@lr=")
        row = dict(config=base, learning_rate=lr, seeds=len(rs))
        for k in ("EM", "F1", "ROUGE_L", "epochs_at_convergence", "train_minutes"):
            v = [float(x[k]) for x in rs]
            row[k + "_mean"] = round(st.mean(v), 4)
            row[k + "_sd"] = round(st.stdev(v), 4) if len(v) > 1 else 0.0
        summ.append(row)
    with open("lr_sensitivity_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    print("\nconfig      lr        F1 (mean ± SD)     EM       epochs")
    for r in summ:
        print(f"{r['config']:<10} {r['learning_rate']:<8} {r['F1_mean']:.4f} ± {r['F1_sd']:.4f}   {r['EM_mean']:.4f}   {r['epochs_at_convergence_mean']:.1f}")


if __name__ == "__main__":
    main()
