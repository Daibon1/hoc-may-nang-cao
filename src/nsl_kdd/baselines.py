"""Baseline anomaly scores and validation-only operating threshold selection."""
import numpy as np
from sklearn.metrics import f1_score


QUANTILES = (0.97, 0.98, 0.99, 0.995, 0.999)


def anomaly_scores(model, values, kind):
    """Larger always means more suspicious; SVM scores may be negative."""
    if kind == 'Random Forest':
        positive = np.flatnonzero(model.classes_ == 1)
        if len(positive) != 1:
            raise ValueError('Expected a binary classifier with Attack=1')
        result = model.predict_proba(values)[:, positive[0]]
    elif kind == 'Isolation Forest':
        result = -model.score_samples(values)
    elif kind == 'One-Class SVM':
        result = -model.decision_function(values)
    else:
        raise ValueError(f'Unsupported model: {kind}')
    result = np.asarray(result, dtype=np.float64).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite anomaly scores')
    return result


def choose_threshold(normal_calibration, selection_scores, selection_y, fpr_budget=0.01):
    calibration = np.asarray(normal_calibration, dtype=np.float64)
    scores = np.asarray(selection_scores, dtype=np.float64)
    labels = np.asarray(selection_y)
    if (calibration.ndim != 1 or len(calibration) == 0 or scores.ndim != 1
            or scores.shape != labels.shape or not np.isfinite(calibration).all()
            or not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}):
        raise ValueError('Expected finite calibration scores and binary labeled selection rows')
    if not 0 <= fpr_budget <= 1:
        raise ValueError('FPR budget must be in [0, 1]')
    rows = []
    for quantile in QUANTILES:
        threshold = float(np.quantile(calibration, quantile))
        predictions = (scores >= threshold).astype(np.int8)
        rows.append({'quantile': quantile, 'threshold': threshold,
                     'macro_f1': float(f1_score(labels, predictions, average='macro', zero_division=0)),
                     'false_positive_rate': float(predictions[labels == 0].mean())})
    eligible = [row for row in rows if row['false_positive_rate'] <= fpr_budget]
    selected = (max(eligible, key=lambda row: (row['macro_f1'], row['quantile'])) if eligible
                else min(rows, key=lambda row: (row['false_positive_rate'], -row['macro_f1'])))
    return selected, rows, bool(eligible)
