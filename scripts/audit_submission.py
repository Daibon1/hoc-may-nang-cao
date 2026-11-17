"""Replay preprocessing and saved-model inference without modifying training artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import torch

from scripts.inspect_nsl_kdd import DATA_FILES, github_blob_sha
from scripts.prepare_nsl_kdd import FEATURES, load_data
from scripts.verify_model_comparison import verify
from nsl_kdd.baselines import anomaly_scores
from nsl_kdd.dashboard import load_bundle, parse_csv, score_frame


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    checks = []
    for name, expected in DATA_FILES.items():
        assert github_blob_sha((ROOT / 'data/raw' / name).read_bytes()) == expected
    checks.append('Pinned raw file hashes verified')
    train = load_data(ROOT / 'data/raw/KDDTrain+.txt')
    test = load_data(ROOT / 'data/raw/KDDTest+.txt')
    fit_ids, val_ids = train_test_split(train.index.to_numpy(), test_size=0.2,
                                       random_state=42, stratify=train.is_attack)
    pipeline = joblib.load(ROOT / 'artifacts/preprocessing/pipeline.joblib')
    replay = clone(pipeline).fit(train.loc[fit_ids].query('is_attack == 0')[list(FEATURES)])
    with np.load(ROOT / 'data/processed/nsl_kdd.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(fit_ids, data['row_train'])
        np.testing.assert_array_equal(val_ids, data['row_validation'])
        assert set(fit_ids).isdisjoint(val_ids)
        for name, frame in [('train', train.loc[fit_ids]), ('validation', train.loc[val_ids]), ('test', test)]:
            np.testing.assert_allclose(replay.transform(frame[list(FEATURES)]).astype(np.float32),
                                       data['X_' + name], rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(frame.is_attack, data['y_' + name])
        normal_train = data['X_train'][data['y_train'] == 0]
        np.testing.assert_array_equal(normal_train, data['X_train_normal'])
        test_values = data['X_test']
    checks.append('Re-fit preprocessing on Normal train; all train/validation/test matrices and labels match')
    optimized = ROOT / 'artifacts/models/autoencoder_optimized'
    with np.load(optimized / 'validation_partition.npz', allow_pickle=False) as partition:
        roles = [set(partition[key]) for key in partition.files]
        for index, role in enumerate(roles):
            assert role <= set(val_ids)
            assert all(role.isdisjoint(other) for other in roles[index + 1:])
    checks.append('Validation roles disjoint and contained in validation split')
    parsed, _ = parse_csv((ROOT / 'data/raw/KDDTest+.txt').read_bytes())
    result, unknown = score_frame(parsed, load_bundle(optimized))
    expected = pd.read_csv(ROOT / 'reports/risk_optimization/test_risk_predictions.csv')
    np.testing.assert_allclose(result.reconstruction_error, expected.reconstruction_error, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(result.risk_score, expected.risk_score)
    np.testing.assert_array_equal(result.alert_level, expected.alert_level)
    np.testing.assert_array_equal(result.prediction, expected.prediction)
    assert result.risk_score.between(0, 100).all()
    checks.append('Full raw CSV -> schema -> saved AE -> error -> risk -> alert -> exported CSV matches 22544 rows')
    reordered = parsed.loc[:99, list(reversed(FEATURES))].to_csv(index=False).encode('utf-8-sig')
    ordered, _ = parse_csv(reordered)
    sample, _ = score_frame(ordered, load_bundle(optimized))
    np.testing.assert_allclose(sample.risk_score, result.risk_score[:100])
    checks.append('41-column reordered UTF-8 BOM upload matches full raw inference')
    for slug in ['random_forest', 'isolation_forest', 'one_class_svm']:
        bundle = joblib.load(ROOT / 'artifacts/models/baselines' / f'{slug}.joblib')
        assert bundle['fit_status'] == 0
        values = anomaly_scores(bundle['model'], test_values, bundle['kind'])
        previous = pd.read_csv(ROOT / 'reports/model_comparison' / f'{slug}_predictions.csv')
        np.testing.assert_allclose(values, previous.score, rtol=1e-6, atol=1e-8)
        np.testing.assert_array_equal(values >= bundle['threshold'], previous.prediction.eq('Attack'))
        checks.append(f'Saved {slug}: full test inference reproduced')
    verify(ROOT)
    checks.append('All four comparison tables independently recomputed against raw test labels')
    notebook = nbformat.read(ROOT / 'notebooks/01_eda_preprocessing.ipynb', as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is not None
            assert not any(output.output_type == 'error' for output in cell.outputs)
    checks.append('EDA notebook valid, all code cells have saved executions without errors')
    output = ROOT / 'deliverables'
    output.mkdir(exist_ok=True)
    report = {'status': 'PASS', 'checks': checks, 'seconds': time.perf_counter() - started,
              'test_rows': len(test), 'unknown_test_categories': unknown,
              'scope': 'Replay preprocessing from raw and reproduce all saved estimators; no retraining or retuning.',
              'data_sha256': hashlib.sha256((ROOT / 'data/processed/nsl_kdd.npz').read_bytes()).hexdigest()}
    (output / 'pipeline_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
