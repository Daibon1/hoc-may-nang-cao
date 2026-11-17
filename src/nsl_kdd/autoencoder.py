"""Normal-only dense Autoencoder and reconstruction-error inference."""
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        if latent_dim not in (8, 16, 32):
            raise ValueError('latent_dim must be 8, 16 or 32')
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(),
                                     nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def reconstruction_error(model, values, batch_size=1024):
    """Per-row mean squared error, accumulated in float64 for large outliers."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected a finite 2-D feature matrix')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    model.eval()
    errors = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            prediction = model(torch.from_numpy(batch)).numpy()
            errors.append(np.mean((batch.astype(np.float64) - prediction) ** 2, axis=1))
    result = np.concatenate(errors) if errors else np.empty(0, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite reconstruction error')
    return result


def select_threshold(normal_errors, quantile=0.99):
    errors = np.asarray(normal_errors, dtype=np.float64)
    if errors.ndim != 1 or not len(errors) or not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError('Expected nonempty finite nonnegative normal errors')
    if not 0 < quantile < 1:
        raise ValueError('quantile must be in (0, 1)')
    return float(np.quantile(errors, quantile))


def predict_from_errors(errors, threshold):
    """Boundary convention: error >= threshold means Attack (1)."""
    return (np.asarray(errors) >= threshold).astype(np.int8)


def load_detector(directory: Path):
    checkpoint = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
    model = Autoencoder(checkpoint['input_dim'], checkpoint.get('latent_dim', 16))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    preprocessing = joblib.load(directory / 'pipeline.joblib')
    if list(preprocessing.get_feature_names_out()) != checkpoint['feature_names']:
        raise ValueError('Model and preprocessing feature schemas differ')
    return model, preprocessing, checkpoint['threshold']
