"""Select latent dimension by validation AP, threshold by F1 under a 1% FPR budget."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from nsl_kdd.autoencoder import load_detector, reconstruction_error, select_threshold
from nsl_kdd.risk import risk_scores, alert_levels
from nsl_kdd.training import fit_normal
from train_autoencoder import metrics
from prepare_nsl_kdd import FEATURES, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError('epochs/patience must be positive')
    torch.set_num_threads(1)
    source = ROOT / 'data/processed/nsl_kdd.npz'
    with np.load(source, allow_pickle=False) as data:
        # Do not read test arrays until the model and thresholds have been frozen.
        x = data['X_train'][data['y_train'] == 0]
        val, y, groups = data['X_validation'], data['y_validation'], data['group_validation']
        val_rows, train_rows = data['row_validation'], data['row_train']
        train_probe = data['X_train'][:32]
    if set(train_rows) & set(val_rows):
        raise ValueError('Train/validation overlap')
    pipeline_path = ROOT / 'artifacts/preprocessing/pipeline.joblib'
    pipeline = joblib.load(pipeline_path)
    raw_train = load_data(ROOT / 'data/raw/KDDTrain+.txt')
    np.testing.assert_allclose(pipeline.transform(raw_train.loc[train_rows[:32], list(FEATURES)]),
                               train_probe, rtol=1e-6, atol=1e-6)
    # Disjoint early-stopping (40%), calibration (30%), selection (30%) roles.
    stopping_ids, rest = train_test_split(np.arange(len(y)), test_size=0.6,
                                         stratify=y, random_state=args.seed)
    calibration_ids, selection_ids = train_test_split(rest, test_size=0.5,
                                                       stratify=y[rest], random_state=args.seed)
    stopping_ids = stopping_ids[y[stopping_ids] == 0]
    calibration_ids = calibration_ids[y[calibration_ids] == 0]
    stopping, calibration, selection = val[stopping_ids], val[calibration_ids], val[selection_ids]
    report_dir = ROOT / 'reports/risk_optimization'
    model_dir = ROOT / 'artifacts/models/autoencoder_optimized'
    trials_dir = ROOT / 'artifacts/models/latent_trials'
    for path in [report_dir, model_dir, trials_dir]:
        path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_dir / 'validation_partition.npz',
                        early_stopping_row=val_rows[stopping_ids],
                        calibration_row=val_rows[calibration_ids], selection_row=val_rows[selection_ids])
    trials, curves, models = [], {}, {}
    for latent in [8, 16, 32]:
        model, history, diagnostics = fit_normal(x, stopping, latent, args.epochs, args.patience, args.seed)
        curves[latent] = pd.DataFrame(history)
        curves[latent].to_csv(report_dir / f'history_latent_{latent}.csv', index=False)
        selection_errors = reconstruction_error(model, selection)
        calibration_errors = reconstruction_error(model, calibration)
        ap = float(average_precision_score(y[selection_ids], selection_errors))
        diagnostics.update(latent_dim=latent, validation_selection_ap=ap,
                           generalization_gap=diagnostics['early_stopping_mse'] - diagnostics['train_mse'],
                           stop_train_ratio=diagnostics['early_stopping_mse'] / max(diagnostics['train_mse'], 1e-12))
        trials.append(diagnostics)
        models[latent] = (model, selection_errors, calibration_errors)
        torch.save({'input_dim': x.shape[1], 'latent_dim': latent,
                    'state_dict': model.state_dict()}, trials_dir / f'latent_{latent}.pt')
    # AP is the prespecified primary metric. Smaller latent size breaks exact ties.
    chosen = max(trials, key=lambda t: (t['validation_selection_ap'], -t['latent_dim']))
    latent = chosen['latent_dim']
    model, selection_errors, reference = models[latent]
    threshold_rows = []
    for q in [0.95, 0.97, 0.98, 0.99, 0.995, 0.999]:
        threshold = select_threshold(reference, q)
        evaluation = metrics(y[selection_ids], selection_errors, threshold, groups[selection_ids])
        threshold_rows.append({'quantile': q, **{k: evaluation[k] for k in
                               ['threshold', 'precision_attack', 'recall_attack', 'macro_f1', 'false_positive_rate']}})
    table = pd.DataFrame(threshold_rows)
    table.to_csv(report_dir / 'threshold_analysis.csv', index=False)
    # Reserve P95 for Medium; optimize the High boundary on independent selection rows.
    candidates = [r for r in threshold_rows if r['quantile'] > 0.95 and r['false_positive_rate'] <= 0.01]
    budget_met = bool(candidates)
    if candidates:
        selected = max(candidates, key=lambda r: (r['macro_f1'], r['quantile']))
    else:
        selected = min(threshold_rows[1:], key=lambda r: (r['false_positive_rate'], -r['macro_f1']))
    high, medium = selected['threshold'], select_threshold(reference, 0.95)
    if medium >= high:
        raise ValueError('Degenerate calibration errors: cannot define three alert levels')
    torch.save({'input_dim': x.shape[1], 'latent_dim': latent, 'state_dict': model.state_dict(),
                'feature_names': list(pipeline.get_feature_names_out()), 'threshold': high}, model_dir / 'model.pt')
    shutil.copy2(pipeline_path, model_dir / 'pipeline.joblib')
    np.save(model_dir / 'normal_reference.npy', np.sort(reference), allow_pickle=False)
    risk_config = {'score': '100 * empirical CDF of normal calibration errors (right-inclusive)',
                   'medium_threshold': medium, 'high_threshold': high,
                   'medium_quantile': 0.95, 'high_quantile': selected['quantile'],
                   'alert_rule': 'Low: error < medium; Medium: medium <= error < high; High: error >= high',
                   'binary_rule': 'High = Attack; Low/Medium = Normal',
                   'not_probability': True}
    (model_dir / 'risk_config.json').write_text(json.dumps(risk_config, indent=2), encoding='utf-8')
    # Only now access test, after architecture and both thresholds are fixed.
    with np.load(source, allow_pickle=False) as data:
        test, test_y, test_groups, test_rows = (data[k] for k in ['X_test', 'y_test', 'group_test', 'row_test'])
    restored, restored_pipeline, restored_threshold = load_detector(model_dir)
    np.testing.assert_array_equal(reconstruction_error(restored, selection[:32]),
                                   reconstruction_error(model, selection[:32]))
    assert restored_threshold == high
    raw_test = load_data(ROOT / 'data/raw/KDDTest+.txt')
    np.testing.assert_allclose(restored_pipeline.transform(raw_test.loc[:31, list(FEATURES)]),
                               test[:32], rtol=1e-6, atol=1e-6)
    errors = reconstruction_error(restored, test)
    levels = alert_levels(errors, medium, high)
    scores = risk_scores(errors, reference)
    test_metrics = metrics(test_y, errors, high, test_groups)
    result = pd.DataFrame({'row_id': test_rows, 'is_attack': test_y, 'group': test_groups,
                           'reconstruction_error': errors, 'risk_score': scores,
                           'alert_level': levels, 'prediction': np.where(errors >= high, 'Attack', 'Normal')})
    result.to_csv(report_dir / 'test_risk_predictions.csv', index=False)
    summary = result.groupby(['group', 'alert_level']).size().unstack(fill_value=0)
    summary.to_csv(report_dir / 'risk_by_group.csv')
    report = {'protocol': vars(args), 'trials': trials, 'selected_latent_dim': latent,
              'model_selection': 'maximum AP on validation selection; smaller latent breaks ties',
              'threshold_selection': 'maximum macro-F1 on validation selection with FPR <= 1%, quantile > P95; fallback minimum FPR',
              'selection_fpr_budget_met': budget_met, 'selected_threshold': selected,
              'risk': risk_config, 'test': test_metrics,
              'counts': {'normal_train': len(x), 'normal_stopping': len(stopping),
                         'normal_calibration': len(calibration), 'selection': len(selection)},
              'sha256': {'data': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'pipeline': hashlib.sha256(pipeline_path.read_bytes()).hexdigest()},
              'versions': {'torch': str(torch.__version__), 'numpy': np.__version__},
              'limitations': ['Single seed: exploratory comparison, not a multi-seed confidence estimate.',
                             'Validation selection metrics are optimistic after model/threshold selection.',
                             'NSL-KDD test was already inspected in the previous stage; no test metrics used in this search.',
                             'Score is a normal-reference percentile, not attack probability or damage severity.']}
    (report_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    shutil.copy2(report_dir / 'summary.json', model_dir / 'config.json')
    pd.DataFrame(trials).to_csv(report_dir / 'latent_comparison.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for dim, curve in curves.items():
        axes[0].plot(curve.epoch, curve.train_online_mse, linestyle='--', label=f'{dim} train')
        axes[0].plot(curve.epoch, curve.early_stopping_mse, label=f'{dim} validation')
    axes[0].set_yscale('log'); axes[0].set_title('Loss / overfitting'); axes[0].legend(fontsize=7)
    for name in ['recall_attack', 'false_positive_rate', 'macro_f1']:
        axes[1].plot(table['quantile'] * 100, table[name], marker='o', label=name)
    axes[1].set_title('Validation selection threshold tradeoff'); axes[1].legend(fontsize=7)
    for label in [0, 1]:
        axes[2].hist(scores[test_y == label], bins=np.linspace(0, 100, 41), alpha=0.5,
                     label='Attack' if label else 'Normal')
    axes[2].set_title('Test risk score (percentile)'); axes[2].legend()
    fig.tight_layout(); fig.savefig(report_dir / 'optimization.png', dpi=160); plt.close(fig)
    from build_risk_report import build_report
    build_report(ROOT)
    print(json.dumps({'selected_latent': latent, 'threshold': high, 'test': test_metrics}, indent=2))


if __name__ == '__main__':
    main()
