"""Reproducible, normal-only preprocessing for NSL-KDD Autoencoders."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from inspect_nsl_kdd import FEATURES, classify_label, download_dataset

ROOT = Path(__file__).resolve().parents[1]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [name for name in FEATURES if name not in CATEGORICAL]


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != 43:
        raise ValueError(f"Expected 43 columns: {path}")
    frame.columns = [*FEATURES, "label", "difficulty"]
    if frame.isna().any().any():
        raise ValueError(f"Missing values: {path}")
    frame[NUMERIC] = frame[NUMERIC].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[NUMERIC].to_numpy()).all():
        raise ValueError(f"Non-finite features: {path}")
    frame["label"] = frame.label.str.strip().str.lower().str.removesuffix(".")
    frame["group"] = frame.label.map(classify_label)
    if frame.group.eq("unknown").any():
        raise ValueError("Unmapped attack labels")
    frame["is_attack"] = frame.label.ne("normal").astype("int8")
    return frame


def prepare(root: Path = ROOT, seed: int = 42, validation_size: float = 0.2):
    download_dataset(root / "data/raw")  # verifies pinned hashes; downloads only if absent
    train = load_data(root / "data/raw/KDDTrain+.txt")
    test = load_data(root / "data/raw/KDDTest+.txt")
    fit_ids, val_ids = train_test_split(
        train.index.to_numpy(), test_size=validation_size,
        random_state=seed, stratify=train.is_attack,
    )
    normal_ids = fit_ids[train.loc[fit_ids, "is_attack"].to_numpy() == 0]
    pipeline = Pipeline([("features", ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                       dtype=np.float32), CATEGORICAL),
    ], remainder="drop"))])
    pipeline.fit(train.loc[normal_ids, list(FEATURES)])
    splits = {"train": train.loc[fit_ids], "validation": train.loc[val_ids], "test": test}
    arrays = {}
    for name, frame in splits.items():
        values = pipeline.transform(frame[list(FEATURES)]).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite transformed values in {name}")
        arrays[f"X_{name}"] = values
        arrays[f"y_{name}"] = frame.is_attack.to_numpy()
        arrays[f"group_{name}"] = frame.group.to_numpy(dtype=str)
        arrays[f"label_{name}"] = frame.label.to_numpy(dtype=str)
        arrays[f"row_{name}"] = frame.index.to_numpy()
    arrays["X_train_normal"] = arrays["X_train"][arrays["y_train"] == 0]
    arrays["X_validation_normal"] = arrays["X_validation"][arrays["y_validation"] == 0]
    output = root / "data/processed"
    artifacts = root / "artifacts/preprocessing"
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifacts / "pipeline.joblib")
    restored = joblib.load(artifacts / "pipeline.joblib")
    np.testing.assert_allclose(restored.transform(test.loc[:9, list(FEATURES)]),
                               pipeline.transform(test.loc[:9, list(FEATURES)]))
    np.savez_compressed(output / "nsl_kdd.npz", **arrays)
    metadata = {
        "seed": seed, "validation_size": validation_size,
        "fit_policy": "normal rows of training partition only",
        "scaling": "StandardScaler; unbounded values; use linear AE output",
        "unknown_categories": "ignored (all-zero one-hot block)",
        "features": pipeline.get_feature_names_out().tolist(),
        "shapes": {key: list(value.shape) for key, value in arrays.items()},
        "versions": {name: __import__(name).__version__ for name in ["numpy", "pandas", "sklearn", "joblib"]},
    }
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return train, test, pipeline, arrays, metadata


if __name__ == "__main__":
    print(json.dumps(prepare()[-1]["shapes"], indent=2))
