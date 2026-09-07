# Reproduce the CTR study

The study uses the approved March 2026 partition of FlyRank's warehouse release v20260703.
Access is gated at https://huggingface.co/datasets/FlyRank/internship-warehouse.
Accept the data-use terms in your own account. Download
`fact_content_daily_performance/month=2026-03/data_0.parquet` and store it as
`data/warehouse/march_2026.parquet`. Alternatively set `FLYRANK_DATA_DIR` to the
folder containing that filename. The file is 124,233,710 bytes; its SHA-256 is
recorded in `work/outputs/capstone_metrics.json`.

The dataset is not committed or hosted with the paper. Do not redistribute it.
Only one partition is read locally; raw daily rows are aggregated in DuckDB before
being returned to pandas. No Hugging Face token is needed for local-file execution.

## Environment

Use Python 3.11 and install:

```sh
python -m pip install -r requirements.txt
python -m pip install nbformat nbclient ipykernel
```

The executed run's exact package versions are recorded in the metrics JSON. The
five study dependencies are duckdb, pandas, numpy, scikit-learn and matplotlib.
For the exact recorded environment, see `work/study-requirements.txt`.

## Run

From the repository root:

```sh
python work/execute_study_notebooks.py
```

This executes ML-04, ML-07 and the capstone top to bottom, in order. ML-03 uses the
bundled starter CSV and is executed separately by the same script. Approximate
resource use: one 124 MB partition, a DuckDB memory limit of 2 GB and four threads.
No notebook makes an external model/API call.

The capstone calls `work/run_study.py`, which fits fixed candidates, selects the
method using validation-client MAE, evaluates the sealed test-client group, and
regenerates charts and aggregate receipts. Repeated execution is a reproducibility
check, not further tuning of the frozen methods.

`work/outputs/baseline_action_score.csv` and `top10_review.csv` are local-only,
gitignored regenerable artifacts. The executed notebooks show ten concise
pseudonymized review recommendations, not a raw export. `docs/` hosts the research
paper; `submission/paper_url.txt` records its deployed address.

## Study boundaries

Feature window: March 1–15. Outcome window: March 16–31. Client split: SHA-256 of
`ctr-v1:` plus client pseudonym, first eight hex digits modulo five; folds 2–4 train,
fold 1 validation and fold 0 final test. A page's client cannot cross partitions.
All five features precede outcomes. Final evaluation requires 500+ later impressions
and complete outcome availability; the operational queue never uses that filter.

This is a one-month, unseen-client study. It does not establish generalization to
other months, causal benefits from editing, or independently judged actionability.
The last warehouse month was not used in development or in the reported results.
