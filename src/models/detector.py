"""
k‑NN anomaly detector operating in embedding space.
Stores all training embeddings and computes the *mean* cosine distance to the
k nearest neighbours of a test embedding.
"""

from typing import List
import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np


class KNNDetector:
    def __init__(self, k: int = 3, normalize: bool = False):
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        self.normalize = bool(normalize)
        self.memory: np.ndarray | None = None
        
    @staticmethod
    def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(norm, eps)

    def fit(self, features: List[torch.Tensor]):
        def pool_to_vec(x: torch.Tensor) -> torch.Tensor:
            # Accept shapes: (D,), (1,D), (T,D), (B,T,D)
            if x.ndim == 1:
                return x
            if x.ndim == 2:
                return x.squeeze(0) if x.size(0) == 1 else x.mean(dim=0)
            if x.ndim == 3:
                x = x.mean(dim=1)  # (B,T,D) -> (B,D)
                return x.squeeze(0) if x.size(0) == 1 else x.mean(dim=0)
            raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")

        pooled = [pool_to_vec(f).cpu().numpy() for f in features]
        if self.normalize:
            pooled = [self._l2_normalize_np(p) for p in pooled]
        self.memory = np.stack(pooled, axis=0)
        self.nn.fit(self.memory)

    def score(self, features: torch.Tensor) -> float:
        """Return positive anomaly score (higher == more abnormal)."""
        if self.memory is None:
            raise RuntimeError("Detector has not been fitted!")

        if features.ndim == 1:
            pooled = features.unsqueeze(0)
        elif features.ndim == 2:
            pooled = features if features.size(0) == 1 else features.mean(dim=0, keepdim=True)
        elif features.ndim == 3:
            pooled = features.mean(dim=1)
            pooled = pooled if pooled.size(0) == 1 else pooled.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported feature shape: {tuple(features.shape)}")

        pooled = pooled.cpu().numpy()
        if self.normalize:
            pooled = self._l2_normalize_np(pooled)
        dists, _ = self.nn.kneighbors(pooled)
        return float(dists.mean())
        
class MahalanobisDetector:
    def fit(self, X):
        self.mu = torch.stack(X).mean(0)
        Σ = torch.cov(torch.stack(X).T)
        self.invΣ = torch.linalg.inv(Σ + 1e-6 * torch.eye(Σ.shape[0]))
    def score(self, q):
        d = (q - self.mu)
        return (d @ self.invΣ @ d.T).sqrt().item()
