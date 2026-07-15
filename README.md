# Reproducible pipeline — RE-QA multi-seed robustness study

Code and data accompanying the paper **"Assessing Answer Span Detection of BERT-Based
Architectures for Requirements Engineering QA Tasks: A Multi-Seed Robustness Study"**
(Computer Standards & Interfaces).

One script runs **all ten configurations under identical conditions**, varying only the
model/adaptation and the random seed. This replaces the earlier per-model scripts and
fixes the methodology issues raised in review.

## What is fixed / controlled
1. **RE-120** test set loaded from `requirements-test-120-fixed.json` with exact `answer_start` spans.
2. **Sliding-window** tokenisation (`doc_stride` + overflow) — answers past the window are no
   longer mislabelled as position 0.
3. **Unified `MAX_LEN = 384`** for every config (train *and* eval).
4. **One fixed, grouped train/val split** (grouped by SQuAD `title`, no leakage); only the
   model-init seed varies across the 10 runs (`SPLIT_SEED` is constant).
5. **Constrained joint span decoding** (best valid `start <= end`, max answer length).
6. **Official SQuAD normalisation** for EM/F1 (+ ROUGE-L secondary).

## Configurations (10)
`bert, distilbert, spanbert, neobert, modernbert` (full fine-tuning) ·
`lstm_l1, lstm_l2, lstm_l3` (frozen BERT + LSTM decoder, depth 1/2/3) ·
`linear` (frozen BERT + linear decoder) · `partial_ft` (last two BERT layers unfrozen + LSTM-3).

## Install
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Validate before spending GPU hours
```
python run_experiments.py --selftest              # pure-function unit tests (no GPU/model)
python run_experiments.py --config bert --smoke   # tiny CPU end-to-end
```

## Run (all 10 models, 10 seeds each)
```
python run_experiments.py --config all --squad-limit 30000
# or one model, all 10 seeds:
python run_experiments.py --config modernbert --squad-limit 30000
```
Results are appended per seed to `results_per_seed.csv` (columns:
`config, seed, epochs_at_convergence, train_minutes, EM, F1, ROUGE_L`).
Seeds: 13, 42, 73, 101, 123, 202, 333, 777, 1337, 2024.

Frozen models cache the BERT embeddings once and train only the head per seed
(~20 GB RAM at `--squad-limit 30000`). Keep `--squad-limit` identical for every model.

## Analyse (paired tests — valid because the split is fixed)
```
python analyze_results.py --metric F1     # Friedman + Holm-Wilcoxon + Cohen's d
python targeted_analysis.py               # per-hypothesis Holm families + rank-biserial + TOST + unpaired sensitivity
```

## Reproduce the figures
```
python mkfigs2.py            # Figure 2 (F1) and Figure 3 (ROUGE-L) box plots
python graphical_abstract.py # graphical abstract
```

## Files
| File | Description |
|---|---|
| `run_experiments.py` | Unified training/evaluation pipeline (all 10 configs). |
| `analyze_results.py` | Friedman + pairwise Wilcoxon/Holm + Cohen's d. |
| `targeted_analysis.py` | Per-hypothesis-family Holm–Wilcoxon + matched-pairs rank-biserial + TOST equivalence, with an unpaired (Mann–Whitney) sensitivity check. |
| `mkfigs2.py`, `graphical_abstract.py` | Figure / graphical-abstract generation. |
| `requirements-test-120-fixed.json` | RE-120 extractive-QA test set (120 items, verified spans). |
| `results_clean.csv` | Final per-seed results used in the paper (10 models × 10 seeds). |
| `requirements.txt` | Pinned dependency versions. |
| `test_frozen.py`, `test_windowing.py` | Unit tests for the frozen-cache and windowing logic. |

## Environment
Trained on a single NVIDIA GeForce RTX 5090 Laptop GPU (24 GB), 64 GB RAM. Pin the model
revisions for exact reproducibility; the Hugging Face model IDs are `bert-base-cased`,
`distilbert-base-cased`, `SpanBERT/spanbert-base-cased`, `chandar-lab/NeoBERT`,
`answerdotai/ModernBERT-base`.

## Citation
If you use this code or the RE-120 test set, please cite the paper and this repository.

## License
Code: MIT (see `LICENSE`). Data (`requirements-test-120-fixed.json`): CC BY 4.0.
