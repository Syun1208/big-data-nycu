# Predicting GitHub User Activity

## Overview

For each test user, predict whether they open a PR, open an issue, comment on an issue, review a PR, comment on a PR diff and push code in the week from 2026-11-11 16:00 UTC.
Data and evaluator come from [nycubigdata-2026fall/BigData2026-Assignment2](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2).
Score is `(macro-F1 + micro-F1) / 2` over the 6 events.

## Highlights

- Weekly training samples that copy the test setup: cutoff every Wednesday 16:00 UTC, features until 4 hours before the cutoff, population picked 14 days earlier (same as `test_input.csv` released on 10/28)
- Leak-free features: every row is used only after its row time, and late-known columns (`merged_at`, `closed_at`) only when they are earlier than the feature time
- Q2 relational features: repository activity of other users, open PRs and issues, reviews and comments received, open PRs the user reviewed, collaborators in the same threads
- Q3 temporal features: counts over 7 to 91 days, exponential decay, days since last event, active weeks, week-over-week trend
- Behavior features over 13 weeks: weekly active rate, streak, chance to stay active or come back next week, burstiness, gap between active days, hour and weekend shares, event mix, repository entropy, own-repo pushes, approve share, self-merge share
- Model 1, `xgboost`: 6 XGBoost classifiers on the engineered features (CUDA, early stopping)
- Model 2, `rdl_gnn` (Relational Deep Learning, Lecture 5): heterogeneous temporal GNN on the relational entity graph at time T. Nodes are users, repositories, PRs and issues with raw point-in-time columns. Fact tables (pushes, comments, reviews) become composite messages between the two entities they join (RelGNN atomic routes) with event type and time encoding (age, decay, weekly phase). No hand-made features
- Model 3, `rdl_gnn_hybrid`: the same GNN with the engineered features added to the user nodes
- All models share the same weekly training table (user, cutoff, 6 labels), threshold tuning for the combined score and holdout report
- Strict time split: train weeks, 1 calibration week (early stopping and thresholds), 1 holdout week (report only). No refit on calibration or holdout weeks
- Q1 schema graph and relational entity graph with Graphviz

## Installation

| Item | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.5.1 + CUDA 12.1 |
| XGBoost | 3.2.0 (CUDA) |
| pandas | 2.2.3 |
| Graphviz | `dot` from conda-forge |

```bash
conda create -n bigdata-hw2 python=3.10 -y
conda activate bigdata-hw2
conda install -c conda-forge graphviz -y
pip install -r requirements.txt
```

| `.env` key (in `../.env`) | Used for |
|---|---|
| `STORAGE_ROOT` | Disk for data, models, outputs, logs |
| `BIG_DATA_NYCU_STORAGE` | `${STORAGE_ROOT}/big-data-nycu`, used by the configs |

## Dataset Preparation

The TA repository is replaced every day at 21:00 Taiwan time. The script deletes the old clone and runs `git clone --depth 1` again. With `--keep-snapshot` the old `data/db` is copied to `${BIG_DATA_NYCU_STORAGE}/data/hw2/snapshots/` first.

```bash
bash scripts/download_data.sh
```

| Path | Content |
|---|---|
| `${BIG_DATA_NYCU_STORAGE}/data/hw2/assignment2/data/db` | 8 Parquet tables |
| `${BIG_DATA_NYCU_STORAGE}/data/hw2/assignment2/evaluation` | `evaluate.py`, examples, `test_input.csv` (from 10/28) |

Label rules (checked against `sample_truth.csv`):

| Event | Table | Who | When |
|---|---|---|---|
| `open_pr` | PullRequests | `author_id` | `created_at` |
| `open_issue` | Issues, `is_pull_request` false | `author_id` | `created_at` |
| `issue_comment` | IssueComments | `author_id` | `created_at` |
| `pr_review` | PRReviews | `reviewer_id` | `submitted_at` |
| `pr_review_comment` | PRReviewComments | `author_id` | `created_at` |
| `push` | Pushes | `user_id` | `created_at` |

## Training

Set `GPU_IDS` in the script (at most 2 GPUs) first.

```bash
bash scripts/train.sh
```

It builds the shared weekly feature cache once, trains every config in `CONFIGS` (one job per GPU), writes each model's `output_314540080.csv` and compares the holdout scores.

| Config | Model |
|---|---|
| `configs/xgboost/exp1.yaml` | `xgboost` |
| `configs/rdl_gnn/exp1.yaml` | `rdl_gnn` |
| `configs/rdl_gnn/exp2.hybrid.yaml` | `rdl_gnn_hybrid` |

| Option | Meaning |
|---|---|
| `--run-name` | Output folder name, default `<date>_hw2.<config path>` |
| `--no-cache` | Rebuild the weekly feature tables instead of reading the cache |
| `--no-predict` | Skip writing `output_314540080.csv` after training |

| Output | Path |
|---|---|
| Submission of each model | `${BIG_DATA_NYCU_STORAGE}/outputs/hw2/<run>/output_314540080.csv` |
| `metrics.json`, `feature_importance.csv`, `holdout_predictions.parquet` | `${BIG_DATA_NYCU_STORAGE}/outputs/hw2/<run>/` |
| Comparison table | `${BIG_DATA_NYCU_STORAGE}/outputs/hw2/<date>_hw2.comparison/comparison.{csv,md}` |
| Best model's submission | `${BIG_DATA_NYCU_STORAGE}/outputs/hw2/submission/output_314540080.csv` |
| Models, thresholds | `${BIG_DATA_NYCU_STORAGE}/models/hw2/<run>/` |
| Feature cache | `${BIG_DATA_NYCU_STORAGE}/cache/hw2/features/` |
| Logs | `${BIG_DATA_NYCU_STORAGE}/logs/hw2/` |

The train weeks move forward with the data: the last complete week becomes the holdout week and the one before it the calibration week.

## Evaluation / Inference

```bash
bash scripts/predict.sh
```

For each config it loads the latest trained run, writes `${BIG_DATA_NYCU_STORAGE}/outputs/hw2/<run>/output_314540080.csv`, checks it with the TA `evaluate.py`, then copies the output of the model with the best holdout score to `outputs/hw2/submission/`. Before 10/28 there is no `test_input.csv`, so the users come from the same population rule.


Score every trained method with the TA `evaluate.py` on the holdout week (its `holdout_output.csv` against `holdout_truth.csv`) and check the format of every `output_314540080.csv`:

```bash
python main.py benchmark
```

The table goes to `${BIG_DATA_NYCU_STORAGE}/outputs/hw2/<date>_hw2.benchmark/benchmark.{csv,md}`. The test week itself has no answers until 11/19.

Final submission on 11/11 after the 21:00 release:

```bash
bash scripts/download_data.sh
bash scripts/train.sh
```

Q1 graphs (users 22996444 and 28983449, 2026-09-15 to 09-17 UTC):

```bash
bash scripts/render_graphs.sh
```

| Output | Content |
|---|---|
| `images/schema_graph.png` | Tables, primary keys, foreign keys, row time |
| `images/entity_graph.png` | Users, repositories, PRs, issues and the activity edges between them |

Tests, then the TA `evaluate.py` on the best submission with the score logged (set `PRED` and `TRUTH` in the script, empty `TRUTH` only checks the format and the test UserIDs):

```bash
bash scripts/test.sh
```

## Results

Data release of 2026-10-21. Calibration week 09-30, holdout week 10-07 (4,931 users). The submission is the model with the best calibration score, so the holdout score is an honest estimate.

| Model | Train weeks | Holdout score | macro-F1 | micro-F1 | Calibration score |
|---|---|---|---|---|---|
| Push only | | 0.3057 | 0.1000 | 0.5114 | |
| Active in last 7 days | | 0.4825 | 0.4168 | 0.5483 | |
| `rdl_gnn` (raw columns only) | 06-03 to 09-23 | 0.5078 | 0.4328 | 0.5828 | 0.5502 |
| `xgboost.exp1` | 06-03 to 09-23 | 0.5200 | 0.4521 | 0.5879 | 0.5739 |
| `xgboost.exp6` (+1 and 3 day windows) | 03-04 to 09-23 | 0.5278 | 0.4573 | 0.5983 | 0.5765 |
| `rdl_gnn` hybrid, mean (2 seeds) | 03-04 to 09-23 | 0.5290, 0.5363 | | | 0.5722, 0.5740 |
| `rdl_gnn` hybrid, GAT (2 seeds) | 03-04 to 09-23 | 0.5289, 0.5323 | | | 0.5751, 0.5732 |
| `rdl_gnn` hybrid, Graph Transformer (2 seeds) | 03-04 to 09-23 | 0.5255, 0.5316 | | | 0.5731, 0.5761 |
| **Ensemble** (0.4 + 0.2 XGBoost, 0.4 GAT) | | **0.5323** | 0.4620 | 0.6027 | **0.5836** |

What helped: behavior features, a longer training window (from 03-04), 1 and 3 day windows, the GNN with engineered user features, and the ensemble. What did not help: training only on recent weeks (0.5061), 2 calibration weeks, stronger XGBoost regularization. Mean, GAT and Graph Transformer aggregation are within the seed noise (about 0.01) of each other.

Ceiling: thresholds tuned on the holdout week itself (diagnostic only) reach 0.557, and predicting next week as this week's true labels gives 0.493. Holdout AUROC is 82 to 92 per event, so ranking is strong. The F1 is limited by the 2 to 5% positive rate of 5 of the 6 events.

## Acknowledgements

```bibtex
@misc{bigdata2026assignment2,
  title  = {BigData2026 Assignment 2: Predicting GitHub User Activity},
  author = {NYCU Big Data 2026 Fall TAs},
  year   = {2026},
  url    = {https://github.com/nycubigdata-2026fall/BigData2026-Assignment2}
}

@inproceedings{robinson2024relbench,
  title     = {RelBench: A Benchmark for Deep Learning on Relational Databases},
  author    = {Robinson, Joshua and Ranjan, Rishabh and Hu, Weihua and Huang, Kexin and Han, Jiaqi and Dobles, Alejandro and Fey, Matthias and Lenssen, Jan Eric and Yuan, Yiwen and Zhang, Zecheng and He, Xinwei and Leskovec, Jure},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  pages     = {21330--21341},
  year      = {2024}
}

@article{chen2025relgnn,
  title   = {RelGNN: Composite Message Passing for Relational Deep Learning},
  author  = {Chen, Tianlang and Kanatsoulis, Charilaos and Leskovec, Jure},
  journal = {arXiv preprint arXiv:2502.06784},
  year    = {2025}
}

@inproceedings{schlichtkrull2018rgcn,
  title     = {Modeling Relational Data with Graph Convolutional Networks},
  author    = {Schlichtkrull, Michael and Kipf, Thomas N. and Bloem, Peter and van den Berg, Rianne and Titov, Ivan and Welling, Max},
  booktitle = {European Semantic Web Conference},
  pages     = {593--607},
  year      = {2018}
}

@inproceedings{chen2016xgboost,
  title     = {XGBoost: A Scalable Tree Boosting System},
  author    = {Chen, Tianqi and Guestrin, Carlos},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {785--794},
  year      = {2016}
}

@misc{gharchive,
  title = {GH Archive},
  url   = {https://www.gharchive.org}
}
```
