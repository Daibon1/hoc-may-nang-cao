"""Fit three baselines and compare four frozen models on the same KDDTest+."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_curve, roc_curve
import torch

from nsl_kdd.autoencoder import load_detector, reconstruction_error
from nsl_kdd.baselines import anomaly_scores, choose_threshold
from prepare_nsl_kdd import FEATURES, load_data
from train_autoencoder import metrics


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    source = ROOT / 'data/processed/nsl_kdd.npz'
    ae_dir = ROOT / 'artifacts/models/autoencoder_optimized'
    ae_config = json.loads((ae_dir / 'config.json').read_text(encoding='utf-8'))
    pipeline_path = ROOT / 'artifacts/preprocessing/pipeline.joblib'
    if sha256(source) != ae_config['sha256']['data'] or sha256(pipeline_path) != ae_config['sha256']['pipeline']:
        raise ValueError('Data/preprocessing changed since AE optimization; rerun optimization first')
    ae, ae_pipeline, ae_threshold = load_detector(ae_dir)
    if sha256(ae_dir / 'pipeline.joblib') != sha256(pipeline_path):
        raise ValueError('AE preprocessing differs from baseline preprocessing')
    with np.load(source, allow_pickle=False) as data:
        train, train_y, val, val_y = (data[key] for key in ['X_train', 'y_train', 'X_validation', 'y_validation'])
        train_rows, val_rows = data['row_train'], data['row_validation']
    if set(train_rows) & set(val_rows):
        raise ValueError('Train/validation overlap')
    with np.load(ae_dir / 'validation_partition.npz', allow_pickle=False) as split:
        partitions = {key: split[key] for key in split.files}
    # Reuse exact calibration and selection rows from AE optimization.
    val_index = pd.Index(val_rows)
    calibration_ids = val_index.get_indexer(partitions['calibration_row'])
    selection_ids = val_index.get_indexer(partitions['selection_row'])
    if (calibration_ids < 0).any() or (selection_ids < 0).any():
        raise ValueError('Saved validation IDs missing from current split')
    if set(calibration_ids) & set(selection_ids) or not (val_y[calibration_ids] == 0).all():
        raise ValueError('Invalid validation roles')
    pipeline = joblib.load(pipeline_path)
    raw_train = load_data(ROOT / 'data/raw/KDDTrain+.txt')
    np.testing.assert_allclose(pipeline.transform(raw_train.loc[train_rows[:32], list(FEATURES)]),
                               train[:32], rtol=1e-6, atol=1e-6)
    output = ROOT / 'reports/model_comparison'
    model_dir = ROOT / 'artifacts/models/baselines'
    for directory in [output, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pipeline_path, model_dir / 'pipeline.joblib')
    shutil.copy2(ae_dir / 'validation_partition.npz', model_dir / 'validation_partition.npz')
    specifications = [
        ('Random Forest', 'random_forest', RandomForestClassifier(
            n_estimators=300, class_weight='balanced', random_state=42, n_jobs=4), True),
        ('Isolation Forest', 'isolation_forest', IsolationForest(
            n_estimators=300, max_samples=256, contamination='auto', random_state=42, n_jobs=4), False),
        ('One-Class SVM', 'one_class_svm', OneClassSVM(
            kernel='rbf', gamma='scale', nu=0.05, cache_size=1024, max_iter=-1), False),
    ]
    fitted, threshold_tables = [], []
    for name, slug, model, supervised in specifications:
        fit_values = train if supervised else train[train_y == 0]
        print(f'Fitting {name}: {len(fit_values)} rows, {fit_values.shape[1]} features', flush=True)
        started = time.perf_counter()
        if supervised:
            model.fit(fit_values, train_y)
        else:
            model.fit(fit_values)
        train_seconds = time.perf_counter() - started
        if getattr(model, 'fit_status_', 0) != 0:
            raise RuntimeError(f'{name} did not converge')
        calibration_scores = anomaly_scores(model, val[calibration_ids], name)
        selection_scores = anomaly_scores(model, val[selection_ids], name)
        selected, rows, budget_met = choose_threshold(calibration_scores, selection_scores, val_y[selection_ids])
        for row in rows:
            threshold_tables.append({'model': name, **row})
        bundle = {'model': model, 'kind': name, 'threshold': selected['threshold'],
                  'feature_names': list(pipeline.get_feature_names_out()),
                  'training_rows': len(fit_values), 'training_policy': 'supervised' if supervised else 'normal-only',
                  'parameters': model.get_params(), 'selected_validation': selected,
                  'selection_fpr_budget_met': budget_met, 'train_seconds': train_seconds,
                  'fit_status': int(getattr(model, 'fit_status_', 0)),
                  'support_vectors': int(len(model.support_)) if hasattr(model, 'support_') else None}
        path = model_dir / f'{slug}.joblib'
        joblib.dump(bundle, path)
        restored = joblib.load(path)
        np.testing.assert_allclose(anomaly_scores(restored['model'], val[selection_ids[:16]], name),
                                   selection_scores[:16], rtol=0, atol=0)
        fitted.append((name, slug, model, bundle))
        print(f'{name}: trained {train_seconds:.2f}s; threshold={selected["threshold"]:.6f}; '
              f'validation FPR budget met={budget_met}', flush=True)
    pd.DataFrame(threshold_tables).to_csv(output / 'validation_thresholds.csv', index=False)
    # All estimators and thresholds are fixed before reading the held-out test arrays.
    with np.load(source, allow_pickle=False) as data:
        test, test_y, test_groups, test_rows = (data[key] for key in ['X_test', 'y_test', 'group_test', 'row_test'])
    raw_test = load_data(ROOT / 'data/raw/KDDTest+.txt')
    if len(test) != len(raw_test):
        raise ValueError('Test must contain the complete KDDTest+')
    np.testing.assert_allclose(ae_pipeline.transform(raw_test.loc[:31, list(FEATURES)]),
                               test[:32], rtol=1e-6, atol=1e-6)
    scores = {}
    started = time.perf_counter()
    scores['Autoencoder'] = reconstruction_error(ae, test)
    inference_seconds = {'Autoencoder': time.perf_counter() - started}
    prior_predictions = pd.read_csv(ROOT / 'reports/risk_optimization/test_risk_predictions.csv')
    np.testing.assert_array_equal(test_rows, prior_predictions.row_id)
    np.testing.assert_allclose(scores['Autoencoder'], prior_predictions.reconstruction_error, rtol=1e-6, atol=1e-8)
    metadata = {'Autoencoder': {'training_policy': 'normal-only', 'training_rows': int((train_y == 0).sum()),
                               'threshold': ae_threshold,
                               'selection_fpr_budget_met': ae_config['selection_fpr_budget_met'],
                               'parameters': {'latent_dim': ae_config['selected_latent_dim']}}}
    for name, slug, model, bundle in fitted:
        started = time.perf_counter()
        scores[name] = anomaly_scores(model, test, name)
        inference_seconds[name] = time.perf_counter() - started
        metadata[name] = {key: value for key, value in bundle.items() if key != 'model'}
    rows, results, group_rows = [], {}, []
    for name, values in scores.items():
        threshold = metadata[name]['threshold']
        predicted = (values >= threshold).astype(np.int8)
        evaluation = metrics(test_y, values, threshold, test_groups)
        evaluation.update(accuracy=float(accuracy_score(test_y, predicted)),
                          balanced_accuracy=float(balanced_accuracy_score(test_y, predicted)),
                          inference_seconds=inference_seconds[name])
        results[name] = evaluation
        rows.append({'model': name, 'training_policy': metadata[name]['training_policy'],
                     'training_rows': metadata[name]['training_rows'],
                     **{key: evaluation[key] for key in ['accuracy', 'precision_attack', 'recall_attack', 'f1_attack',
                                                        'macro_f1', 'false_positive_rate', 'pr_auc_average_precision', 'roc_auc']}})
        for group, item in evaluation['groups'].items():
            group_rows.append({'model': name, 'group': group, **item})
        slug = name.lower().replace(' ', '_').replace('-', '_')
        pd.DataFrame({'row_id': test_rows, 'is_attack': test_y, 'group': test_groups,
                      'score': values, 'threshold': threshold,
                      'prediction': np.where(predicted, 'Attack', 'Normal')}).to_csv(output / f'{slug}_predictions.csv', index=False)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / 'comparison.csv', index=False)
    pd.DataFrame(group_rows).to_csv(output / 'group_detection.csv', index=False)
    report = {'test_rows': len(test), 'test_attack_rows': int(test_y.sum()),
              'validation_calibration_normal_rows': len(calibration_ids),
              'validation_selection_rows': len(selection_ids),
              'sha256': {'data': sha256(source), 'pipeline': sha256(pipeline_path),
                         'autoencoder_model': sha256(ae_dir / 'model.pt')},
              'protocol': 'Same 77 features and official test; RF labeled train, IF/OCSVM full Normal train; '
                          'AE frozen from optimization; calibration/selection rows shared; '
                          'P97/P98/P99/P99.5/P99.9, maximize selection macro-F1 under FPR<=1%; '
                          'fallback minimum FPR if no feasible candidate; Attack if score >= threshold.',
              'versions': {'sklearn': sklearn.__version__, 'numpy': np.__version__, 'torch': str(torch.__version__)},
              'model_metadata': metadata, 'test_metrics': results}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (model_dir / 'metadata.json').write_text(json.dumps({key: report[key] for key in
                                                       ['protocol', 'versions', 'sha256', 'model_metadata']}, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, values in scores.items():
        precision, recall, _ = precision_recall_curve(test_y, values)
        fpr, tpr, _ = roc_curve(test_y, values)
        axes[0].plot(recall, precision, label=name)
        axes[1].plot(fpr, tpr, label=name)
    axes[0].axhline(test_y.mean(), color='gray', linestyle='--', label='Attack prevalence')
    axes[0].set(xlabel='Recall', ylabel='Precision', title='KDDTest+ precision–recall')
    axes[1].set(xlabel='False positive rate', ylabel='True positive rate', title='KDDTest+ ROC')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / 'comparison_curves.png', dpi=160); plt.close(fig)
    from build_comparison_report import build_report
    build_report(ROOT)
    from verify_model_comparison import verify
    verify(ROOT)
    print(comparison.to_string(index=False))


if __name__ == '__main__':
    main()
