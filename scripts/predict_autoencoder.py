"""Predict Normal/Attack for a headerless NSL-KDD file (41 or 43 columns)."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
import torch

from inspect_nsl_kdd import FEATURES
from nsl_kdd.autoencoder import load_detector, predict_from_errors, reconstruction_error
from nsl_kdd.risk import risk_scores, alert_levels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/raw/KDDTest+.txt')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'artifacts/models/autoencoder_optimized')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/risk_optimization/predictions.csv')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError('Output must differ from input')
    torch.set_num_threads(4)
    model, pipeline, threshold = load_detector(args.model_dir)
    raw = pd.read_csv(args.input, header=None)
    if raw.shape[1] not in (41, 43):
        raise ValueError('Expected a headerless NSL-KDD file with 41 or 43 columns')
    features = raw.iloc[:, :41].copy()
    features.columns = list(FEATURES)
    if features.isna().any().any():
        raise ValueError('Missing input features')
    transformed = pipeline.transform(features).astype(np.float32)
    errors = reconstruction_error(model, transformed)
    predicted = predict_from_errors(errors, threshold)
    result = pd.DataFrame({'row_id': np.arange(len(raw)), 'reconstruction_error': errors,
                           'threshold': threshold,
                           'prediction': np.where(predicted, 'Attack', 'Normal')})
    risk_path = args.model_dir / 'risk_config.json'
    if risk_path.exists():
        config = json.loads(risk_path.read_text(encoding='utf-8'))
        if config['high_threshold'] != threshold:
            raise ValueError('Risk and binary thresholds differ')
        reference = np.load(args.model_dir / 'normal_reference.npy', allow_pickle=False)
        result['risk_score'] = risk_scores(errors, reference)
        result['alert_level'] = alert_levels(errors, config['medium_threshold'], config['high_threshold'])
        print(result.alert_level.value_counts().to_string())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.prediction.value_counts().to_string())
    print(f'Saved {len(result)} predictions to {args.output}')


if __name__ == '__main__':
    main()
