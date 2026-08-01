# Reproducible pipeline — RE-QA multi-seed robustness study

Code and data accompanying the paper "Assessing Answer Span Detection of
BERT-Based Architectures for Requirements Engineering QA Tasks: A Multi-Seed
Robustness Study."

A unified pipeline runs all ten configurations using the same fixed grouped
training/validation split, the RE-120 test set, the same evaluation procedure,
and the same predefined seed labels. Configuration-specific training
hyperparameters follow the experimental protocol reported in the manuscript.

## Experimental controls

1. **RE-120** test set loaded from `requirements-test-120-fixed.json` with exact
   `answer_start` spans.
2. **Sliding-window** tokenisation (`doc_stride` + overflow), so answers beyond
   the first window keep their correct span labels.
3. **Unified `MAX_LEN = 384`** for every config, both training and evaluation.
4. **One fixed, grouped train/val split** (grouped by SQuAD `title`, no leakage);
   across the ten seeds only the model-initialisation seed varies
   (`SPLIT_SEED` is constant).
5. **Constrained joint span decoding** (best valid `start <= end`, maximum
   answer length).
6. **Official SQuAD normalisation** for EM/F1, with ROUGE-L as a secondary
   metric.

## Configurations (10)

`bert, distilbert, spanbert, neobert, modernbert` (full fine-tuning) ·
`lstm_l1, lstm_l2, lstm_l3` (frozen BERT + LSTM decoder, depth 1/2/3) ·
`linear` (frozen BERT + feed-forward decoder) ·
`partial_ft` (last two BERT layers unfrozen + LSTM-3).

Each configuration is run across ten seeds — one hundred runs in total.

Seeds: 13, 42, 73, 101, 123, 202, 333, 777, 1337, 2024.

## Install

Requires Python 3.10 or newer. Pinned dependencies: `torch==2.3.1`,
`transformers==4.44.2`, `datasets==2.20.0`, `numpy==1.26.4`, `scipy==1.13.1`.
For GPU, install a CUDA-matched torch build (see https://pytorch.org).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Validate before spending GPU hours

```bash
python run_experiments.py --selftest              # pure-function unit tests (no GPU/model)
python run_experiments.py --config bert --smoke   # tiny CPU end-to-end
```

## Run (all 10 models, 10 seeds each)

```bash
python run_experiments.py --config all --squad-limit 30000

# or one model, all 10 seeds:
python run_experiments.py --config modernbert --squad-limit 30000
```

The full sweep took approximately 33 hours of training time on the hardware
listed under [Environment](#environment) — 100 runs, 19.6 minutes per run on
average. Approximate cost per configuration:

| Configuration | Hours (10 seeds) | | Configuration | Hours (10 seeds) |
|---|---|---|---|---|
| `neobert` | 7.3 | | `distilbert` | 2.6 |
| `modernbert` | 5.1 | | `lstm_l3` | 1.8 |
| `partial_ft` | 4.9 | | `lstm_l2` | 1.4 |
| `spanbert` | 4.0 | | `linear` | 1.1 |
| `bert` | 3.6 | | `lstm_l1` | 1.0 |

Results are appended per seed to `results_per_seed.csv` (columns:
`config, seed, epochs_at_convergence, train_minutes, EM, F1, ROUGE_L`). This
file is generated locally; the version used in the paper is committed as
`results_clean.csv`.

Frozen models cache the BERT embeddings once and train only the head per seed
(~20 GB RAM at `--squad-limit 30000`). Keep `--squad-limit` identical for every
model.

## Analyse (primary seed-blocked tests and unpaired sensitivity analysis)

Seed labels are used as common experimental blocks in the primary analysis.
Because identical numerical seeds do not imply identical stochastic trajectories
across architectures, an unpaired Mann-Whitney sensitivity analysis is also
reported.

The analysis scripts read `results_clean.csv`, which is `results_per_seed.csv`
renamed after a validation check (10 configurations x 10 unique seeds = 100 rows,
no duplicate config+seed, no missing values); no numerical values are changed.
It contains exactly the 100 runs reported in the paper.

```bash
python analyze_results.py --metric F1     # exploratory all-pair Friedman + Holm-Wilcoxon + Cohen's d_z
python targeted_analysis.py               # manuscript confirmatory: per-hypothesis Holm families + paired d_z + matched-pairs r_rb + TOST + Mann-Whitney sensitivity
```

## Reproduce the figures

```bash
python mkfigs2.py                 # Figure 2: F1 distributions
python plot_efficiency_pareto.py  # Figure 3: performance-time trade-off (Pareto)
python plot_relative_profile.py   # Figure 4: relative metric profile
python graphical_abstract.py      # graphical abstract
```

## Files

| Path | Description |
|---|---|
| `run_experiments.py` | Unified training/evaluation pipeline (all 10 configs). |
| `analyze_results.py` | Optional exploratory all-pair Friedman + Wilcoxon/Holm + Cohen's d_z. |
| `targeted_analysis.py` | Per-hypothesis-family Holm–Wilcoxon + matched-pairs rank-biserial + TOST equivalence, with an unpaired (Mann–Whitney) sensitivity check. |
| `mkfigs2.py`, `plot_efficiency_pareto.py`, `plot_relative_profile.py`, `graphical_abstract.py` | Figure and graphical-abstract generation (Figures 2, 3, 4, and the graphical abstract). |
| `test_frozen.py`, `test_windowing.py` | Unit tests for the frozen-cache and windowing logic. |
| `requirements-test-120-fixed.json` | RE-120 extractive-QA test set (120 items, verified spans). |
| `results_clean.csv` | Per-seed results reported in the paper (10 configurations × 10 seeds). |
| `requirements.txt` | Pinned dependency versions. |
| `Figure_1.pdf` | BPMN diagram of the experimental workflow (data preparation, model configuration, multi-seed training, evaluation). |
| `data/` | RE-120 provenance table (per-item source, category, span status) and the per-document summary table. |
| `supplementary/` | Per-seed result tables for all ten configurations, plus full confirmatory statistics and the equivalence-margin sensitivity analysis. |
| `LICENSE` | MIT licence for the code. |

## Environment

Trained on a single NVIDIA GeForce RTX 5090 Laptop GPU (24 GB), 64 GB RAM.

Random seeds are controlled for the Python hash seed, NumPy state, PyTorch and
CUDA seeds, and training-data shuffle order, with deterministic cuDNN settings;
the training/validation partition is held fixed.

Hugging Face model identifiers:

| Configuration | Model ID |
|---|---|
| `bert`, frozen and partial variants | `bert-base-cased` |
| `distilbert` | `distilbert-base-cased` |
| `spanbert` | `SpanBERT/spanbert-base-cased` |
| `neobert` | `chandar-lab/NeoBERT` |
| `modernbert` | `answerdotai/ModernBERT-base` |

## Citation

If you use this code or the RE-120 test set, please cite this repository:

```bibtex
@software{sabaliauskas_reqa_multiseed,
  author = {Sabaliauskas, Darius and Miliauskaitė, Jolanta},
  title  = {Reproducible pipeline: RE-QA multi-seed robustness study},
  year   = {2026},
  url    = {https://github.com/DariusSabaliauskasVU/e-qa-multiseed-robustness}
}
```

## License

Code: MIT (see `LICENSE`).

Data: the RE-120 questions, answer spans and annotations are released under
CC BY 4.0.

The context passages are **not** ours. They are quoted from four requirements
specifications distributed in the PURE dataset of public requirements documents
(https://zenodo.org/records/7118517) and remain under the terms of that record:
CCTNS SRS (law enforcement, 30 items), GAMMA-J Web Store SRS (e-commerce, 29),
IUFA SRS (university IT asset management, 31), and PeaZip SRS (desktop utility,
30). Per-item provenance, including source document and URL, is recorded in
`data/RE120_provenance.csv`.

Item categories: 78 functional, 13 non-functional, 13 behavioural, 16
data-related. Span status: 47 human-confirmed, 41 exact matches, 32 case-fixed.
