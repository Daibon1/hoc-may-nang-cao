"""Normal-only fitting with early stopping, shared across latent-size trials."""
import copy
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .autoencoder import Autoencoder, reconstruction_error


def fit_normal(train, stopping, latent_dim, epochs=100, patience=12, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = Autoencoder(train.shape[1], latent_dim)
    loader = DataLoader(torch.from_numpy(train), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    best_loss, stale = float('inf'), 0
    best_state, best_epoch, history = None, 0, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(batch)
        stop_mse = float(reconstruction_error(model, stopping).mean())
        history.append({'epoch': epoch, 'train_online_mse': total / len(train),
                        'early_stopping_mse': stop_mse})
        if stop_mse < best_loss:
            best_loss, stale, best_epoch = stop_mse, 0, epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(f'latent={latent_dim} epoch={epoch} train={total / len(train):.6f} stop={stop_mse:.6f}', flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, history, {'best_epoch': best_epoch, 'epochs_run': len(history),
                            'train_mse': float(reconstruction_error(model, train).mean()),
                            'early_stopping_mse': best_loss,
                            'training_seconds': time.perf_counter() - started}
