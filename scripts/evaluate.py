"""Evaluation script for the Fake News Detection model.

Loads a trained model and runs it against the sample dataset, printing
and saving evaluation metrics.

Usage
-----
    python scripts/evaluate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    from src.utils import load_config
    from src.preprocessing import TextProcessor
    from src.features import FeatureExtractor
    from src.models import ClassicalModels

    config = load_config()
    models_dir = _PROJECT_ROOT / "models"

    # ---- Load artefacts ----
    print("Loading feature extractor…")
    extractor = FeatureExtractor(config)
    extractor.load(models_dir / "vectorizers")

    print("Loading classifiers…")
    classical_models = ClassicalModels(config)
    classical_models.load(models_dir / "classifiers")

    # ---- Load sample data ----
    sample_path = _PROJECT_ROOT / "data" / "sample" / "sample_news.csv"
    print(f"Loading sample data from: {sample_path}")
    df = pd.read_csv(sample_path)

    texts = df["text"].fillna("").tolist()
    labels = df["label"].tolist()
    y_true = np.array(labels)

    # ---- Pre-process ----
    print("Pre-processing texts…")
    processor = TextProcessor()
    processed = [processor.process(t) for t in texts]

    # ---- Transform + predict ----
    print("Running predictions…")
    X = extractor.transform(processed)
    y_pred = classical_models.predict(X)

    # ---- Metrics ----
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred, target_names=["REAL", "FAKE"], zero_division=0
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy  : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"Precision : {prec:.4f}  (weighted)")
    print(f"Recall    : {rec:.4f}  (weighted)")
    print(f"F1 Score  : {f1:.4f}  (weighted)")
    print(f"\nConfusion Matrix:\n{np.array(cm)}")
    print(f"\nClassification Report:\n{report}")
    print("=" * 60)

    # ---- Save report ----
    eval_report = {
        "accuracy": round(float(acc), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1_score": round(float(f1), 6),
        "confusion_matrix": cm,
        "n_samples": len(y_true),
    }
    report_path = models_dir / "evaluation_report.json"
    models_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(eval_report, fh, indent=2)
    print(f"Evaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
