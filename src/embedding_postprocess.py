"""Embedding post-processing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


def fit_pca_whitening(
    matrix: torch.Tensor,
    pca_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit PCA on training embeddings and return transformed embeddings + payload."""
    if not bool(pca_cfg.get("enabled", False)):
        return matrix, {"enabled": False}

    if matrix.ndim != 2:
        raise ValueError(f"Expected matrix shape (N,D), got {tuple(matrix.shape)}")

    n_items, feature_dim = int(matrix.size(0)), int(matrix.size(1))
    requested = int(pca_cfg.get("n_components", 128))
    effective = min(requested, feature_dim, n_items)
    if effective < requested:
        print(
            f"WARNING: PCA n_components={requested} exceeds feature_dim={feature_dim} "
            f"or train_embeddings={n_items}; using n_components={effective}."
        )

    whiten = bool(pca_cfg.get("whiten", True))
    l2_after = bool(pca_cfg.get("l2_after", True))
    pca = PCA(n_components=effective, whiten=whiten, svd_solver="auto", random_state=0)
    x_np = matrix.detach().cpu().numpy().astype(np.float32)
    transformed_np = pca.fit_transform(x_np).astype(np.float32)
    transformed = torch.from_numpy(transformed_np).float()
    if l2_after:
        transformed = F.normalize(transformed, dim=-1)

    payload = {
        "enabled": True,
        "requested_n_components": requested,
        "n_components": effective,
        "whiten": whiten,
        "l2_after": l2_after,
        "input_feature_dim": feature_dim,
        "pca": pca,
        "mean": torch.from_numpy(pca.mean_.astype(np.float32)),
        "components": torch.from_numpy(pca.components_.astype(np.float32)),
        "explained_variance": torch.from_numpy(pca.explained_variance_.astype(np.float32)),
        "explained_variance_ratio": torch.from_numpy(pca.explained_variance_ratio_.astype(np.float32)),
    }
    return transformed, payload


def apply_pca_whitening_tensor(
    features: torch.Tensor,
    pca_payload: dict[str, Any] | None,
) -> torch.Tensor:
    """Apply saved PCA whitening stats to one or more embeddings."""
    if not pca_payload or not bool(pca_payload.get("enabled", False)):
        return features

    squeeze = False
    if features.ndim == 1:
        features = features.unsqueeze(0)
        squeeze = True
    if features.ndim != 2:
        raise ValueError(f"Expected feature shape (D,) or (N,D), got {tuple(features.shape)}")

    device = features.device
    dtype = features.dtype
    mean = pca_payload["mean"].to(device=device, dtype=dtype)
    components = pca_payload["components"].to(device=device, dtype=dtype)
    out = (features - mean) @ components.T

    if bool(pca_payload.get("whiten", True)):
        variance = pca_payload["explained_variance"].to(device=device, dtype=dtype)
        out = out / torch.sqrt(torch.clamp(variance, min=1e-12))
    if bool(pca_payload.get("l2_after", True)):
        out = F.normalize(out, dim=-1)
    return out.squeeze(0) if squeeze else out
