# Experiment notes

## Question
Can a simple model combine request-description text and operational fields to reproduce a synthetic, human-review routing label better than a majority-class baseline?

## Method
- Generate artificial records with a fixed seed. The target is a toy process label, not a medical condition.
- Keep identifiers out of the feature set; use TF-IDF for the short description, one-hot categories, and scaled numeric fields.
- Use a stratified, seeded holdout split. Fit all transforms inside a scikit-learn pipeline to prevent train/test leakage from preprocessing.
- Report precision, recall, F1 and confusion matrix for the review class alongside accuracy and baseline metrics.

## Interpretation checklist
1. Compare F1 and recall with the baseline; do not report accuracy alone.
2. Inspect false negatives and false positives in `predictions.csv`.
3. Treat feature behavior as exploratory; synthetic patterns are deliberately simple.
4. For any real dataset, establish a valid label with domain experts and test temporal and subgroup robustness before considering deployment.

## Known limits
Small synthetic data, artificial label construction, no external validation, no calibration analysis, no privacy/security assessment, and no real health or care outcome. Results are only a reproducible coding demonstration.
