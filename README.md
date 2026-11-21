# Care Demand ML Lab

A small, reproducible Python command-line project that explores whether **synthetic operational service signals** can predict which incoming care-service requests need human review. It demonstrates a complete baseline ML workflow: data checks, feature preparation, stratified holdout evaluation, a majority-class baseline, a text-plus-tabular Logistic Regression model, metrics, and documented limitations.

This is a portfolio exercise, not a clinical system. It does not use patient records, infer diagnoses, or make care decisions. All sample rows are generated and artificial.

## Why this example

The workflow mirrors the role's focus on turning an idea into software, analyzing unstructured information, finding useful features, evaluating models, and documenting results. The target label is an artificial operational review flag; it must not be interpreted as a health risk.

## Requirements and run

Python 3.11+ and the dependency in `pyproject.toml`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
care-demand-lab generate --output examples/requests.csv --rows 360
care-demand-lab evaluate examples/requests.csv --report reports/evaluation.json --predictions reports/predictions.csv
```

Alternatively run the module directly:

```powershell
python -m care_demand_ml_lab.cli evaluate examples/requests.csv
```

## What it does

- Validates required columns, labels, missing text and numeric fields.
- Combines short free-text descriptions (`TfidfVectorizer`) with structured categorical/numeric features (`OneHotEncoder`, `StandardScaler`).
- Compares a stratified majority-class baseline against Logistic Regression on a fixed stratified holdout split.
- Reports accuracy, precision, recall, F1, confusion matrix and class support; writes machine-readable JSON and per-row predictions.
- Uses a seeded synthetic data generator so the example is reproducible.

## Data fields

`request_id`, `description`, `service_area`, `channel`, `days_waiting`, `open_items`, `review_required`. `review_required` is generated from explicit synthetic rules and is only a toy operational label.

## Limits and responsible use

The synthetic records encode simple patterns and do not represent real care operations. A random holdout score is not evidence of real-world quality. Before any real use, a team would need lawful data access, privacy/security review, representative sampling, leakage analysis, subgroup/fairness checks, human oversight, monitoring and validation with domain experts. No individual-level health information should be added to this demo.

## Layout

- `src/care_demand_ml_lab/`: CLI, validation, data generation, model and reporting.
- `examples/requests.csv`: synthetic sample data.
- `reports/`: example output directory (generated on request; not committed).
- `docs/experiment-notes.md`: experiment plan and interpretation checklist.
