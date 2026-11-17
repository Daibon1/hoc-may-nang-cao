"""Audit exported predictions against raw test labels and recompute all table metrics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    directory = root / 'reports/model_comparison'
    report = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    raw = pd.read_csv(root / 'data/raw/KDDTest+.txt', header=None)
    truth = raw[41].str.strip().str.lower().str.removesuffix('.').ne('normal').astype(int).to_numpy()
    names = {'autoencoder': 'Autoencoder', 'random_forest': 'Random Forest',
             'isolation_forest': 'Isolation Forest', 'one_class_svm': 'One-Class SVM'}
    table = pd.read_csv(directory / 'comparison.csv').set_index('model')
    for slug, name in names.items():
        frame = pd.read_csv(directory / f'{slug}_predictions.csv')
        np.testing.assert_array_equal(frame.row_id, np.arange(len(raw)))
        np.testing.assert_array_equal(frame.is_attack, truth)
        assert np.isfinite(frame.score).all()
        np.testing.assert_allclose(frame.threshold, report['model_metadata'][name]['threshold'])
        predictions = (frame.score >= frame.threshold).astype(int)
        np.testing.assert_array_equal(frame.prediction.eq('Attack').astype(int), predictions)
        matrix = confusion_matrix(truth, predictions, labels=[0, 1])
        np.testing.assert_array_equal(matrix, report['test_metrics'][name]['confusion_matrix'])
        checks = {
            'accuracy': accuracy_score(truth, predictions),
            'precision_attack': precision_score(truth, predictions, zero_division=0),
            'recall_attack': recall_score(truth, predictions, zero_division=0),
            'f1_attack': f1_score(truth, predictions, zero_division=0),
            'macro_f1': f1_score(truth, predictions, average='macro', zero_division=0),
            'false_positive_rate': float(predictions[truth == 0].mean()),
            'pr_auc_average_precision': average_precision_score(truth, frame.score),
            'roc_auc': roc_auc_score(truth, frame.score),
        }
        for key, value in checks.items():
            np.testing.assert_allclose(value, table.loc[name, key], rtol=1e-8, atol=1e-10)
        print(f'PASS: {name}, {len(frame)} raw test labels, predictions and metrics')


if __name__ == '__main__':
    verify()
