# Reproducible pipeline — RE-QA multi-seed robustness study

Code and data accompanying the paper "Assessing Answer Span Detection of
BERT-Based Architectures for Requirements Engineering QA Tasks: A Multi-Seed
Robustness Study."

A single script runs all ten configurations under identical conditions, varying
only the model/adaptation and the random seed. This supersedes the earlier
per-model scripts and standardises the protocol across the full comparison.

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

Requires Python ⟨FILL: 3.x⟩. Pinned dependencies: `torch==2.3.1`,
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

## Analyse (paired tests — valid because the split is fixed)

The analysis scripts read `results_clean.csv`, which is the generated
`results_per_seed.csv` after ⟨FILL: describe the step, or write "no changes
other than the filename" if that is the case⟩. It contains exactly the 100 runs
reported in the paper.

```bash
python analyze_results.py --metric F1     # Friedman + Holm-Wilcoxon + Cohen's d
python targeted_analysis.py               # per-hypothesis Holm families + rank-biserial + TOST + unpaired sensitivity
```

## Reproduce the figures

```bash
python mkfigs2.py                 # Figure 2 (F1) and Figure 3 (ROUGE-L) box plots
python plot_efficiency_pareto.py  # accuracy / training-time trade-off
python graphical_abstract.py      # graphical abstract
```

## Files

| Path | Description |
|---|---|
| `run_experiments.py` | Unified training/evaluation pipeline (all 10 configs). |
| `analyze_results.py` | Friedman + pairwise Wilcoxon/Holm + Cohen's d. |
| `targeted_analysis.py` | Per-hypothesis-family Holm–Wilcoxon + matched-pairs rank-biserial + TOST equivalence, with an unpaired (Mann–Whitney) sensitivity check. |
| `mkfigs2.py`, `plot_efficiency_pareto.py`, `graphical_abstract.py` | Figure and graphical-abstract generation. |
| `test_frozen.py`, `test_windowing.py` | Unit tests for the frozen-cache and windowing logic. |
| `requirements-test-120-fixed.json` | RE-120 extractive-QA test set (120 items, verified spans). |
| `results_clean.csv` | Per-seed results reported in the paper (10 configurations × 10 seeds). |
| `requirements.txt` | Pinned dependency versions. |
| `Figure_1.pdf` | ⟨FILL: one line — what this figure shows⟩ |
| `data/` | ⟨FILL: one line — what this folder contains⟩ |
| `supplementary/` | ⟨FILL: one line — what this folder contains⟩ |
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
CC BY 4.0. The context passages are quoted from four publicly available software
requirements specifications and remain under their original terms: the CCTNS SRS
(law enforcement, 30 items), the GAMMA-J Web Store SRS (e-commerce, 29), the IUFA
SRS (university IT asset management, 31), and the PeaZip SRS (desktop utility, 30).
