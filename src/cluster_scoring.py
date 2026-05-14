"""Clustered kNN scoring helpers with optional cluster-local calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.asarray([0.0], dtype=np.float32)

    q25, q75 = np.percentile(values, [25, 75])
    return {
        "mean_knn_distance": float(np.mean(values)),
        "std_knn_distance": float(np.std(values)),
        "median_knn_distance": float(np.median(values)),
        "iqr_knn_distance": float(q75 - q25),
    }


def _knn_mean_distance(
    query: torch.Tensor,
    bank: torch.Tensor,
    k: int,
    distance: str,
) -> tuple[float, int]:
    if query.ndim == 1:
        query = query.unsqueeze(0)
    if query.ndim != 2 or query.size(0) != 1:
        raise ValueError(f"Expected one query embedding with shape (1,D), got {tuple(query.shape)}")

    if distance == "cosine":
        d = 1.0 - (query @ bank.T)
    else:
        d = torch.cdist(query, bank)
    k_eff = min(int(k), int(d.size(1)))
    topk = torch.topk(d, k=k_eff, dim=1, largest=False).values
    return float(topk.mean().item()), k_eff


def compute_global_leave_one_out_distances(
    mem_bank: torch.Tensor,
    k: int,
    distance: str,
    chunk: int = 1024,
) -> np.ndarray:
    """Compute leave-one-out mean kNN distances for every memory entry."""
    values = []
    n_items = int(mem_bank.size(0))
    for start in range(0, n_items, int(chunk)):
        sub = mem_bank[start : start + int(chunk)]
        if distance == "cosine":
            d = 1.0 - (sub @ mem_bank.T)
        else:
            d = torch.cdist(sub, mem_bank)
        for row in range(d.size(0)):
            global_idx = start + row
            d[row, global_idx] = float("inf")
        k_eff = min(int(k), max(1, n_items - 1))
        topk = torch.topk(d, k=k_eff, dim=1, largest=False).values
        values.append(topk.mean(dim=1).detach().cpu().numpy())
    return np.concatenate(values, axis=0).astype(np.float32)


def compute_cluster_score_stats(
    mem_bank: torch.Tensor,
    cluster_labels: torch.Tensor,
    k: int,
    distance: str,
    min_cluster_size: int,
) -> dict[str, Any]:
    """Build global and cluster-local leave-one-out distance statistics."""
    mem_bank = mem_bank.detach().cpu().float()
    cluster_labels = cluster_labels.detach().cpu().long()
    global_dists = compute_global_leave_one_out_distances(mem_bank, k=k, distance=distance)
    global_summary = _summary(global_dists)
    stats: dict[str, Any] = {
        "global_mean_knn_distance": global_summary["mean_knn_distance"],
        "global_std_knn_distance": global_summary["std_knn_distance"],
        "global_median_knn_distance": global_summary["median_knn_distance"],
        "global_iqr_knn_distance": global_summary["iqr_knn_distance"],
        "clusters": {},
    }

    for cluster_id in sorted(int(x) for x in torch.unique(cluster_labels).tolist()):
        selected = torch.nonzero(cluster_labels == cluster_id, as_tuple=False).flatten()
        cluster_values = []
        for idx in selected.tolist():
            local = selected[selected != int(idx)]
            if int(local.numel()) < max(1, min(int(k), int(min_cluster_size))):
                continue
            raw, _ = _knn_mean_distance(
                mem_bank[int(idx) : int(idx) + 1],
                mem_bank[local],
                k=k,
                distance=distance,
            )
            cluster_values.append(raw)

        if cluster_values:
            summary = _summary(np.asarray(cluster_values, dtype=np.float32))
        else:
            summary = global_summary.copy()

        stats["clusters"][int(cluster_id)] = {
            "cluster_mean_knn_distance": summary["mean_knn_distance"],
            "cluster_std_knn_distance": summary["std_knn_distance"],
            "cluster_median_knn_distance": summary["median_knn_distance"],
            "cluster_iqr_knn_distance": summary["iqr_knn_distance"],
            "cluster_size": int(selected.numel()),
        }

    return stats


def normalize_cluster_score(
    raw_score: float,
    normalization: str,
    stats: dict[str, Any] | None,
    cluster_id: int | None,
    eps: float,
    fallback_global: bool,
) -> tuple[float, str]:
    mode = str(normalization or "none").lower()
    if mode == "none":
        return float(raw_score), "none"
    if not stats:
        return float(raw_score), "missing_stats"

    use_global = bool(fallback_global) or cluster_id is None
    if use_global:
        local = {
            "cluster_mean_knn_distance": stats.get("global_mean_knn_distance", 0.0),
            "cluster_std_knn_distance": stats.get("global_std_knn_distance", 1.0),
            "cluster_median_knn_distance": stats.get("global_median_knn_distance", 0.0),
            "cluster_iqr_knn_distance": stats.get("global_iqr_knn_distance", 1.0),
        }
        source = "global"
    else:
        clusters = stats.get("clusters", {})
        local = clusters.get(int(cluster_id), clusters.get(str(int(cluster_id))))
        source = "cluster"
        if local is None:
            local = {
                "cluster_mean_knn_distance": stats.get("global_mean_knn_distance", 0.0),
                "cluster_std_knn_distance": stats.get("global_std_knn_distance", 1.0),
                "cluster_median_knn_distance": stats.get("global_median_knn_distance", 0.0),
                "cluster_iqr_knn_distance": stats.get("global_iqr_knn_distance", 1.0),
            }
            source = "global_missing_cluster"

    eps = float(eps)
    if mode == "mean_ratio":
        value = float(raw_score) / (float(local["cluster_mean_knn_distance"]) + eps)
    elif mode == "zscore":
        value = (float(raw_score) - float(local["cluster_mean_knn_distance"])) / (
            float(local["cluster_std_knn_distance"]) + eps
        )
    elif mode == "robust_zscore":
        value = (float(raw_score) - float(local["cluster_median_knn_distance"])) / (
            float(local["cluster_iqr_knn_distance"]) + eps
        )
    else:
        raise ValueError(f"Unsupported cluster score normalization: {normalization}")

    return float(value), source


def score_clustered_knn(
    query: torch.Tensor,
    mem_bank: torch.Tensor,
    cluster_labels: torch.Tensor,
    centroids: torch.Tensor,
    k: int,
    distance: str,
    min_cluster_size: int,
    score_normalization: str = "none",
    cluster_stats: dict[str, Any] | None = None,
    eps: float = 1e-6,
    exclude_index: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Score one embedding by nearest centroid, local kNN, and optional calibration."""
    if query.ndim == 1:
        query = query.unsqueeze(0)
    if query.ndim != 2 or query.size(0) != 1:
        raise ValueError(f"Expected one query embedding with shape (1,D), got {tuple(query.shape)}")

    q_norm = F.normalize(query, dim=-1)
    c_norm = F.normalize(centroids, dim=-1)
    centroid_dists = 1.0 - (q_norm @ c_norm.T)
    cluster_id = int(torch.argmin(centroid_dists, dim=1).item())
    centroid_distance = float(centroid_dists[0, cluster_id].item())

    selected = torch.nonzero(cluster_labels == cluster_id, as_tuple=False).flatten()
    cluster_size = int(selected.numel())
    fallback = cluster_size < max(1, int(min_cluster_size)) or cluster_size < int(k)

    if not fallback and exclude_index is not None:
        selected = selected[selected != int(exclude_index)]
        fallback = int(selected.numel()) < int(k)

    if fallback:
        selected = torch.arange(mem_bank.size(0), device=mem_bank.device)
        if exclude_index is not None and selected.numel() > 1:
            selected = selected[selected != int(exclude_index)]

    raw_score, k_eff = _knn_mean_distance(query, mem_bank[selected], k=k, distance=distance)
    score, norm_source = normalize_cluster_score(
        raw_score=raw_score,
        normalization=score_normalization,
        stats=cluster_stats,
        cluster_id=cluster_id,
        eps=eps,
        fallback_global=fallback,
    )
    debug = {
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
        "centroid_distance": centroid_distance,
        "fallback_global": fallback,
        "k_used": k_eff,
        "raw_score": raw_score,
        "score": score,
        "score_normalization": str(score_normalization or "none").lower(),
        "normalization_source": norm_source,
    }
    return score, debug
