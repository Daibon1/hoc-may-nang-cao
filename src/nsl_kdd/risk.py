"""Empirical normal-reference percentile; never an attack probability."""
import numpy as np


def validate_errors(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected finite nonnegative 1-D errors')
    return values


def risk_scores(errors, reference):
    errors = validate_errors(errors)
    reference = np.sort(validate_errors(reference))
    if not len(reference):
        raise ValueError('Normal reference must be nonempty')
    return 100.0 * np.searchsorted(reference, errors, side='right') / len(reference)


def alert_levels(errors, medium_threshold, high_threshold):
    errors = validate_errors(errors)
    if not np.isfinite([medium_threshold, high_threshold]).all() or not 0 <= medium_threshold < high_threshold:
        raise ValueError('Expected 0 <= medium < high thresholds')
    return np.where(errors >= high_threshold, 'High',
                    np.where(errors >= medium_threshold, 'Medium', 'Low'))
