# Reproducible pipeline — RE-QA multi-seed robustness study

Code and data accompanying the paper "Assessing Answer Span Detection of
BERT-Based Architectures for Requirements Engineering QA Tasks: A Multi-Seed
Robustness Study."

A single script runs all ten configurations under matched conditions, varying
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

The results in the paper were produced on Windows 11 in an Anaconda base
environment with Python 3.13, PyTorch 2.12.0.dev20260218+cu128 (CUDA 12.8
nightly), Transformers 4.55.0, Datasets 4.0.0, NumPy 2.1.3, SciPy 1.15.3 and
xformers 0.0.34 (see `requirements.txt` for the full pinned list). RTX 50-series
GPUs require a CUDA 12.8 build of PyTorch 2.7 or newer. NeoBERT is loaded from the
pinned Hugging Face snapshot `5424c8efeea6491b151d62dee55a752165407430`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
pip install xformers==0.0.34 --no-deps
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
`results_per_seed.csv` with a single smoke-test row removed; no other changes were
made. It contains exactly the 100 runs reported in the paper.

```bash
python analyze_results.py --metric F1     # Friedman + Holm-Wilcoxon + Cohen's d
python targeted_analysis.py               # per-hypothesis Holm families + rank-biserial + TOST + unpaired sensitivity
```

## Reproduce the figures

```bash
python mkfigs2.py                 # Figure 2 (F1 box plots) + supplementary ROUGE-L box plots
python plot_efficiency_pareto.py  # Figure 3 (accuracy / training-time trade-off)
python plot_relative_profile.py   # Figure 4 (relative metric profile across models)
python graphical_abstract.py      # graphical abstract
```

## Files

| Path | Description |
|---|---|
| `run_experiments.py` | Unified training/evaluation pipeline (all 10 configs). |
| `analyze_results.py` | Friedman + pairwise Wilcoxon/Holm + Cohen's d. |
| `targeted_analysis.py` | Per-hypothesis-family Holm–Wilcoxon + matched-pairs rank-biserial + TOST equivalence, with an unpaired (Mann–Whitney) sensitivity check. |
| `mkfigs2.py`, `plot_efficiency_pareto.py`, `plot_relative_profile.py`, `graphical_abstract.py` | Figure and graphical-abstract generation (Figures 2, 3 and 4). |
| `test_frozen.py`, `test_windowing.py` | Unit tests for the frozen-cache and windowing logic. |
| `requirements-test-120-fixed.json` | RE-120 extractive-QA test set (120 items, verified spans). |
| `results_clean.csv` | Per-seed results reported in the paper (10 configurations × 10 seeds). |
| `requirements.txt` | Pinned dependency versions. |
| `Figure_1.pdf` | Figure 1 of the paper: high-level workflow (data preparation, model configuration, training, multi-seed evaluation, statistical analysis). |
| `data/` | RE-120 provenance table (per-item source, category, span status) and the per-document summary table. |
| `supplementary/` | Per-seed result tables for all ten configurations, plus full confirmatory statistics and the equivalence-margin sensitivity analysis. |
| `LICENSE` | MIT licence for the code. |

## Inference cost (Table 8)

Inference latency and peak GPU memory do not depend on the values of the
trained weights, so they are measured without checkpoints: each configuration
is built exactly as in `run_experiments.py` (same backbone, head, `MAX_LEN`,
sliding window and span decoding) and RE-120 is passed through it one question
at a time in fp32.

```bash
pip install -r requirements-inference.txt
python measure_inference.py --config all --device cuda --repeats 5
```

The script first runs a GPU burn-in until the GPU is in its maximum-performance
state (P0), warms each configuration up for 15 s, times the forward pass and the
span decoding for all 120 questions in five repeats, and records the GPU clock,
temperature, power and P-state after every repeat. On a laptop, plug in the
charger and select the maximum-performance power mode.

Outputs (the files used in the paper are committed):

- `inference_summary.csv` - mean and SD over repeats per configuration
- `inference_per_run.csv` - one row per configuration and repeat
- `inference_env.json` - GPU, driver, CUDA, PyTorch and Transformers versions

Measured on Windows 11, Python 3.12.10, PyTorch 2.8.0+cu128,
Transformers 4.57.1, xformers 0.0.32.post2 (required by NeoBERT),
NVIDIA GeForce RTX 5090 Laptop GPU (driver 610.74).

## Learning-rate sensitivity of the frozen heads (Table 9)

```bash
python lr_sensitivity.py        # lstm_l3 and linear, lr 2e-5 / 1e-4 / 1e-3, seeds 13 42 73
```

Retrains the three-layer LSTM head and the feed-forward head over frozen BERT
with three learning rates; everything else is unchanged. Outputs:
`lr_sensitivity_per_seed.csv` and `lr_sensitivity_summary.csv`.

## BERTScore and breakdown by requirement category (Tables 10 and 11)

```bash
python run_experiments.py --config all --seeds 13 42 73 --squad-limit 30000 \
       --paper-seeding --save-preds preds --out results_preds_run.csv
pip install bert-score==0.3.13 --no-deps
python compute_semantic_metrics.py --preds preds
```

`--save-preds` stores the per-question predictions of every run in
`preds/<config>_seed<seed>.json`; it does not change training or scoring.
`compute_semantic_metrics.py` computes BERTScore (roberta-large, rescaled with
baseline; hash `roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.55.0)-rescaled`)
and the breakdown by requirement category and source document. Outputs:
`semantic_metrics_per_run.csv`, `semantic_metrics_summary.csv`,
`breakdown_by_category.csv`, `breakdown_by_document.csv`.

DistilBERT and the four frozen-encoder configurations reproduce the seed-level
scores in `results_clean.csv` exactly; the other encoder-training configurations
differ slightly between reruns because cuDNN determinism was not enforced.

## Environment

Trained on a single NVIDIA GeForce RTX 5090 Laptop GPU (24 GB), 64 GB RAM.

Python, NumPy, PyTorch and CUDA random seeds and the training-data shuffle order
are fixed for each seed. The runs reported in the paper used cuDNN with its
default settings (benchmark mode off, non-deterministic kernels allowed), so
reruns with the same seed may differ slightly. The cuDNN determinism settings in
`set_all_seeds` were added after those runs; `--paper-seeding` disables them to
reproduce the paper's seeding exactly. The training/validation partition is
held fixed.

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
