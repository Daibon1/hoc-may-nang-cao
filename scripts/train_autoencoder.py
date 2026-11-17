"""Train on normal traffic, calibrate P99 on validation, then evaluate test once."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader

from nsl_kdd.autoencoder import (Autoencoder, load_detector, predict_from_errors,
                                 reconstruction_error, select_threshold)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, errors, threshold, groups):
    predicted = predict_from_errors(errors, threshold)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        'threshold': threshold, 'confusion_matrix_labels': ['Normal', 'Attack'],
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'precision_attack': float(precision_score(y, predicted, zero_division=0)),
        'recall_attack': float(recall_score(y, predicted, zero_division=0)),
        'f1_attack': float(f1_score(y, predicted, zero_division=0)),
        'macro_f1': float(f1_score(y, predicted, average='macro', zero_division=0)),
        'false_positive_rate': float(fp / (tn + fp)),
        'pr_auc_average_precision': float(average_precision_score(y, errors)),
        'roc_auc': float(roc_auc_score(y, errors)),
        'groups': {str(group): {'support': int((groups == group).sum()),
                                'flagged_attack_rate': float(predicted[groups == group].mean())}
                   for group in np.unique(groups)},
    }


def train(args):
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError('epochs, patience, batch size and learning rate must be positive')
    if not 0 < args.quantile < 1:
        raise ValueError('quantile must be in (0, 1)')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    data_path = ROOT / 'data/processed/nsl_kdd.npz'
    if not data_path.exists():
        raise FileNotFoundError('Run scripts/prepare_nsl_kdd.py first')
    with np.load(data_path, allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    for name in ['train', 'validation']:
        if not np.isfinite(data[f'X_{name}']).all():
            raise ValueError('Non-finite training/validation features')
    assert set(data['row_train']).isdisjoint(data['row_validation'])
    normal_train = data['X_train'][data['y_train'] == 0]
    # Split validation normals to separate early stopping from threshold calibration.
    normal_val_ids = np.flatnonzero(data['y_validation'] == 0)
    stop_ids, calibration_ids = train_test_split(normal_val_ids, test_size=0.5,
                                                 random_state=args.seed)
    stopping = data['X_validation'][stop_ids]
    loader = DataLoader(torch.from_numpy(normal_train), batch_size=args.batch_size,
                        shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    model = Autoencoder(normal_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss_function = nn.MSELoss()
    best_loss, best_epoch, stale = float('inf'), 0, 0
    best_state = None
    history = []
    started = time.perf_counter()
    initial_loss = float(reconstruction_error(model, normal_train).mean())
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(batch)
        val_loss = float(reconstruction_error(model, stopping).mean())
        history.append({'epoch': epoch, 'train_mse': total / len(normal_train),
                        'validation_normal_mse': val_loss})
        print(f'Epoch {epoch:03d}: train={total / len(normal_train):.6f} val_normal={val_loss:.6f}', flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    calibration_errors = reconstruction_error(model, data['X_validation'][calibration_ids])
    threshold = select_threshold(calibration_errors, args.quantile)
    elapsed = time.perf_counter() - started
    output = ROOT / 'artifacts/models/autoencoder'
    report_dir = ROOT / 'reports/autoencoder'
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_path = ROOT / 'artifacts/preprocessing/pipeline.joblib'
    preprocessing_metadata = ROOT / 'artifacts/preprocessing/metadata.json'
    meta = json.loads(preprocessing_metadata.read_text(encoding='utf-8'))
    import joblib
    fitted_pipeline = joblib.load(preprocessing_path)
    # Detect a stale/mismatched NPZ before packaging the detector.
    from prepare_nsl_kdd import FEATURES, load_data
    raw_train = load_data(ROOT / 'data/raw/KDDTrain+.txt')
    probe = fitted_pipeline.transform(raw_train.loc[data['row_train'][:32], list(FEATURES)])
    np.testing.assert_allclose(probe, data['X_train'][:32], rtol=1e-6, atol=1e-6)
    assert len(meta['features']) == normal_train.shape[1]
    torch.save({'state_dict': model.state_dict(), 'input_dim': normal_train.shape[1],
                'threshold': threshold, 'feature_names': meta['features']}, output / 'model.pt')
    shutil.copy2(preprocessing_path, output / 'pipeline.joblib')
    shutil.copy2(preprocessing_metadata, output / 'preprocessing_metadata.json')
    restored, _, restored_threshold = load_detector(output)
    np.testing.assert_allclose(reconstruction_error(restored, stopping[:32]),
                               reconstruction_error(model, stopping[:32]), rtol=0, atol=0)
    assert restored_threshold == threshold
    scores = {name: reconstruction_error(model, data[f'X_{name}'])
              for name in ['train', 'validation', 'test']}
    evaluations = {name: metrics(data[f'y_{name}'], scores[name], threshold, data[f'group_{name}'])
                   for name in ['validation', 'test']}
    report = {
        'architecture': [normal_train.shape[1], 64, 32, 16, 32, 64, normal_train.shape[1]],
        'activation': 'ReLU hidden layers; linear bottleneck and output',
        'config': vars(args), 'device': 'cpu', 'best_epoch': best_epoch,
        'epochs_run': len(history), 'training_seconds': elapsed,
        'normal_training_rows': len(normal_train), 'early_stopping_rows': len(stop_ids),
        'normal_calibration_rows': len(calibration_ids),
        'initial_train_mse': initial_loss,
        'best_train_mse': float(scores['train'][data['y_train'] == 0].mean()),
        'best_early_stopping_mse': best_loss,
        'threshold_policy': 'P99 by default, normal calibration subset of validation only; Attack if MSE >= threshold',
        'threshold': threshold,
        'calibration_false_positive_rate': float((calibration_errors >= threshold).mean()),
        'validation_note': 'Validation includes rows used for early stopping and calibration; test is independent.',
        'versions': {'python': sys.version, 'torch': str(torch.__version__),
                     'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'sha256': {'data_npz': sha256(data_path), 'pipeline': sha256(preprocessing_path),
                   'model': sha256(output / 'model.pt')},
        'evaluation': evaluations,
    }
    (output / 'config.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (report_dir / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(history).to_csv(report_dir / 'history.csv', index=False)
    np.savez_compressed(output / 'validation_partition.npz',
                        early_stopping_row=data['row_validation'][stop_ids],
                        calibration_row=data['row_validation'][calibration_ids])
    for name in ['validation', 'test']:
        predictions = predict_from_errors(scores[name], threshold)
        pd.DataFrame({'row_id': data[f'row_{name}'], 'label': data[f'label_{name}'],
                      'group': data[f'group_{name}'], 'is_attack': data[f'y_{name}'],
                      'reconstruction_error': scores[name],
                      'prediction': np.where(predictions, 'Attack', 'Normal')}).to_csv(
                          report_dir / f'{name}_predictions.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    curves = pd.DataFrame(history).set_index('epoch')
    curves.plot(ax=axes[0], logy=True, title='Normal-only training / early stopping')
    axes[0].set_ylabel('MSE (log scale)')
    for label in [0, 1]:
        axes[1].hist(np.log10(scores['test'][data['y_test'] == label] + 1e-12),
                     bins=60, alpha=0.5, label='Attack' if label else 'Normal')
    axes[1].axvline(np.log10(threshold + 1e-12), color='black', linestyle='--', label='Validation threshold')
    axes[1].set_xlabel('log10 reconstruction MSE'); axes[1].set_title('Held-out test')
    axes[1].legend(); fig.tight_layout()
    fig.savefig(report_dir / 'training_and_errors.png', dpi=160); plt.close(fig)
    print(json.dumps({'best_epoch': best_epoch, 'threshold': threshold,
                      'test': evaluations['test']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quantile', type=float, default=0.99)
    train(parser.parse_args())
