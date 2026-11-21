from __future__ import annotations

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TEXT = "description"
CATEGORICAL = ["service_area", "channel"]
NUMERIC = ["days_waiting", "open_items"]
FEATURES = [TEXT, *CATEGORICAL, *NUMERIC]


def evaluate(rows: list[dict[str, str]]) -> tuple[dict, list[dict[str, object]]]:
    y = [int(r["review_required"]) for r in rows]
    train_idx, test_idx = train_test_split(range(len(rows)), test_size=0.25, random_state=41, stratify=y)
    train, test = [rows[i] for i in train_idx], [rows[i] for i in test_idx]
    y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

    def frame(items):
        return pd.DataFrame([{k: (int(v) if k in NUMERIC else v) for k, v in row.items()} for row in items])

    prep = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1), TEXT),
        ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numeric", StandardScaler(), NUMERIC),
    ])
    model = Pipeline([("features", prep), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=41))])
    baseline = DummyClassifier(strategy="most_frequent").fit([[0]] * len(y_train), y_train)
    model.fit(frame(train), y_train)
    predicted = model.predict(frame(test)).tolist()
    base_pred = baseline.predict([[0]] * len(y_test)).tolist()

    def metrics(actual, pred):
        tn, fp, fn, tp = confusion_matrix(actual, pred, labels=[0, 1]).ravel().tolist()
        return {"accuracy": round(accuracy_score(actual, pred), 4),
                "precision_review": round(precision_score(actual, pred, zero_division=0), 4),
                "recall_review": round(recall_score(actual, pred, zero_division=0), 4),
                "f1_review": round(f1_score(actual, pred, zero_division=0), 4),
                "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]]}

    result = {"experiment": "synthetic operational review flag", "rows_total": len(rows),
              "train_rows": len(train), "test_rows": len(test), "positive_rate": round(sum(y) / len(y), 4),
              "split": "stratified 75/25; random_state=41", "features": FEATURES,
              "baseline_most_frequent": metrics(y_test, base_pred),
              "logistic_regression_tfidf_tabular": metrics(y_test, predicted),
              "limitations": ["Synthetic labels and records only.", "Holdout results do not establish real-world performance.", "Not for diagnosis, risk scoring, or care decisions."]}
    probs = model.predict_proba(frame(test))[:, 1].tolist()
    out = [{"request_id": rows[i]["request_id"], "actual_review_required": y[i],
            "predicted_review_required": int(predicted[j]), "estimated_probability": round(float(probs[j]), 4)}
           for j, i in enumerate(test_idx)]
    return result, out


