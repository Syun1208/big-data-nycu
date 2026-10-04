# Analysis of SVD for Fine-Tuning LLMs

## Overview

PiSSA fine-tuning of `meta-llama/Llama-3.2-1B` on MetaMathQA, evaluated on GSM8K and MATH.
Refactored from [nycubigdata-2026fall/BigData2026-Assignment1](https://github.com/nycubigdata-2026fall/BigData2026-Assignment1).

## Highlights

- Q1, Q2: PiSSA rank sweep (16, 64, 256) with training time
- Q3: rank 16 adapters initialized from different singular value ranges (`p25`, `p50`, `bottom`)
- Q4: SVD spectrum and minimum rank for 50% energy of `q_proj`

## Installation

| Item | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.5.1 + CUDA 12.1 |

```bash
conda create -n bigdata-hw1 python=3.10 -y
conda activate bigdata-hw1
pip install -r requirements.txt
```

| `.env` key (in `../.env`) | Used for |
|---|---|
| `HF_TOKEN` | Llama-3.2-1B download |

## Dataset Preparation

Downloaded automatically from Hugging Face on the first run.

| Split | Source | Rows |
|---|---|---|
| Train | `meta-math/MetaMathQA` | 25,000 |
| Eval | `fxmeng/pissa-dataset` (GSM8K, MATH) | 1,000 each |

## Training

| Task | Command | Output |
|---|---|---|
| Q1, Q2 | `bash scripts/q1_q2_rank_sweep.sh` | `outputs/<date>_q1_q2_rank_sweep/` |
| Q3 | `bash scripts/q3_singular_ranges.sh` | `outputs/<date>_q3_singular_ranges/` |
| Q4 | `bash scripts/q4_spectrum.sh` | `outputs/<date>_q4_spectrum/` |
| All | `bash scripts/run_all.sh` | Q4, then Q1/Q2, then Q3 |

Run Q1/Q2 before Q3.

Quick check:

```bash
bash scripts/finetune.sh --run 16 --train-limit 64 --max-eval-examples 8 --output-dir outputs/quick_check --no-save-full-model
```

`--run RANK:COMPONENT`

| Component | Singular values |
|---|---|
| `default` | top `RANK` |
| `p25` | from `len(S)//4` |
| `p50` | from `len(S)//2` |
| `bottom` | last `RANK` |

## Evaluation / Inference

```bash
python main.py report \
    --results outputs/<date>_q1_q2_rank_sweep/benchmark_results.csv outputs/<date>_q3_singular_ranges/benchmark_results.csv \
    --output-dir outputs/<date>_report
```

```bash
bash scripts/test.sh
```

| File | Content |
|---|---|
| `benchmark_results.csv` | `gsm8k_accuracy`, `math_accuracy`, `train_minutes` per run |
| `q4_minimum_rank.csv` | minimum rank per layer |

## Results

| Model | Rank | Component | GSM8K | MATH | Train (min) |
|---|---|---|---|---|---|
| Llama-3.2-1B (no fine-tune) | | | 0.035 | 0.001 | |
| PiSSA | 16 | `default` | | | |
| PiSSA | 64 | `default` | | | |
| PiSSA | 256 | `default` | | | |
| PiSSA | 16 | `p25` | 0.121 | 0.034 | 24.4 |
| PiSSA | 16 | `p50` | 0.119 | 0.021 | 35.2 |
| PiSSA | 16 | `bottom` | | | |

| Layer | Module | Minimum rank (50% energy) |
|---|---|---|
| 0 | `q_proj` | 62 |
| 7 | `q_proj` | 120 |
| 15 | `q_proj` | 84 |

## Acknowledgements

- [nycubigdata-2026fall/BigData2026-Assignment1](https://github.com/nycubigdata-2026fall/BigData2026-Assignment1)
- PiSSA, Hugging Face `transformers`, `datasets`
