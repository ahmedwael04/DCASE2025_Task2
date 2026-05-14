#!/usr/bin/env python
"""
Compute AUC_source, AUC_target, and pAUC@FPR0.1 on dev_data test set for each machine.
Usage:
  python scripts/compute_dev_metrics.py --config configs/default.yaml
"""

import argparse, glob, os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# Allow running as: python scripts/compute_dev_metrics.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.file_utils     import load_config
from src.models.beats_backbone import BEATsBackbone
from src.utils.audio_utils    import load_audio_mono
from src.augmentations import (
    AugmentationConfig,
    DomainGeneralizationAugmentor,
    aggregate_augmented_scores,
    set_augmentation_seed,
)
from src.pipeline import (
    bank_matches_memory_config,
    bank_matches_pipeline,
    describe_pipeline,
    get_bank_path,
    get_embedding_mode,
    get_embedding_postprocess_config,
    get_memory_bank_config,
    get_pipeline_mode,
    get_top_windows,
    get_window_params,
    validate_pipeline_memory_compatibility,
)
from src.cluster_scoring import score_clustered_knn as score_clustered_knn_impl
from src.embedding_postprocess import apply_pca_whitening_tensor


def make_window_embeddings(
    temporal_feats: torch.Tensor,
    win_frames: int,
    hop_frames: int,
) -> torch.Tensor:
    if temporal_feats.ndim == 3:
        if temporal_feats.size(0) != 1:
            raise ValueError(f"Expected B==1 temporal feats, got {tuple(temporal_feats.shape)}")
        x = temporal_feats[0]
    elif temporal_feats.ndim == 2:
        x = temporal_feats
    else:
        raise ValueError(f"Unsupported temporal feats shape: {tuple(temporal_feats.shape)}")

    if x.size(0) < win_frames:
        return x.mean(dim=0, keepdim=True)

    windows = []
    for start in range(0, x.size(0) - win_frames + 1, hop_frames):
        windows.append(x[start : start + win_frames].mean(dim=0))
    return torch.stack(windows, dim=0)


def score_clustered_knn(
    query: torch.Tensor,
    mem_bank: torch.Tensor,
    cluster_labels: torch.Tensor,
    centroids: torch.Tensor,
    k: int,
    distance: str,
    min_cluster_size: int,
    score_normalization: str = "none",
    cluster_stats: dict | None = None,
    eps: float = 1e-6,
    exclude_index: int | None = None,
) -> float:
    score, _ = score_clustered_knn_impl(
        query=query,
        mem_bank=mem_bank,
        cluster_labels=cluster_labels,
        centroids=centroids,
        k=k,
        distance=distance,
        min_cluster_size=min_cluster_size,
        score_normalization=score_normalization,
        cluster_stats=cluster_stats,
        eps=eps,
        exclude_index=exclude_index,
    )
    return float(score)


def _harmonic_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, 1e-12, None)
    return float(len(arr) / np.sum(1.0 / arr))


def load_bank_checkpoint(bank_path: Path) -> dict:
    ckpt = None
    try:
        ckpt = torch.load(bank_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = None
    except Exception:
        ckpt = None

    if not (isinstance(ckpt, dict) and "memory" in ckpt and "paths" in ckpt):
        ckpt = torch.load(bank_path, map_location="cpu")
    return ckpt


def compute_threshold_and_mem_dists(
    mem_bank: torch.Tensor,
    k: int,
    distance: str,
    pct: float,
) -> tuple[float, np.ndarray]:
    mem_dists = []
    for i in range(mem_bank.size(0)):
        query = mem_bank[i : i + 1]
        if distance == "cosine":
            d = 1.0 - (query @ mem_bank.T)
        else:
            d = torch.cdist(query, mem_bank)
        d[0, i] = float("inf")
        topk = torch.topk(d, k=k, dim=1, largest=False).values
        mem_dists.append(float(topk.mean().item()))
    mem_dists_np = np.asarray(mem_dists, dtype=np.float32)
    return float(np.percentile(mem_dists_np, pct)), mem_dists_np


def compute_clustered_threshold_and_mem_dists(
    mem_bank: torch.Tensor,
    cluster_labels: torch.Tensor,
    centroids: torch.Tensor,
    k: int,
    distance: str,
    pct: float,
    min_cluster_size: int,
    score_normalization: str = "none",
    cluster_stats: dict | None = None,
    eps: float = 1e-6,
) -> tuple[float, np.ndarray]:
    mem_dists = [
        score_clustered_knn(
            query=mem_bank[i : i + 1],
            mem_bank=mem_bank,
            cluster_labels=cluster_labels,
            centroids=centroids,
            k=k,
            distance=distance,
            min_cluster_size=min_cluster_size,
            score_normalization=score_normalization,
            cluster_stats=cluster_stats,
            eps=eps,
            exclude_index=i,
        )
        for i in range(mem_bank.size(0))
    ]
    mem_dists_np = np.asarray(mem_dists, dtype=np.float32)
    return float(np.percentile(mem_dists_np, pct)), mem_dists_np


def binary_metrics(y_true: list[int], scores: list[float], threshold: float) -> tuple[float, float, float]:
    preds = (np.asarray(scores, dtype=np.float32) > float(threshold)).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        np.asarray(y_true, dtype=np.int64),
        preds,
        average="binary",
        zero_division=0,
    )
    return float(precision * 100), float(recall * 100), float(f1 * 100)


def compute_metrics(
    cfg_path: str,
    machine_filter: str | None = None,
    return_results: bool = False,
    embedding_mode_override: str | None = None,
):
    # 1) load config, set up device
    cfg    = load_config(cfg_path)
    embedding_mode = get_embedding_mode(cfg, embedding_mode_override)
    pipeline_mode = get_pipeline_mode(cfg)
    win_frames, hop_frames = get_window_params(cfg)
    top_windows = get_top_windows(cfg)
    memory_cfg = get_memory_bank_config(cfg)
    memory_mode = memory_cfg["mode"]
    post_cfg = get_embedding_postprocess_config(cfg)
    pca_cfg = post_cfg["pca_whitening"]
    validate_pipeline_memory_compatibility(pipeline_mode, memory_mode)
    aug_config = AugmentationConfig.from_config(cfg)
    set_augmentation_seed(aug_config.seed)
    aug_rng = torch.Generator()
    if aug_config.seed is not None:
        aug_rng.manual_seed(int(aug_config.seed))
    else:
        aug_rng.seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) load backbone & k-NN bank
    backbone = BEATsBackbone(cfg["model"]["embedding"], embedding_mode=embedding_mode).to(device).eval()
    augmentor = DomainGeneralizationAugmentor(aug_config, generator=aug_rng)
    test_aug_views = aug_config.num_views_test if aug_config.enabled else 0
    spec_augment_active = (
        aug_config.enabled
        and test_aug_views > 0
        and (aug_config.freq_mask_max > 0 or aug_config.time_mask_max > 0)
        and not backbone.expect_waveform
    )
    k = cfg.get("detector", {}).get("k", cfg.get("model", {}).get("k", 3))
    distance = str(cfg.get("detector", {}).get("distance", "cosine")).lower()
    normalize = bool(cfg.get("model", {}).get("normalize", False))
    if distance == "cosine" and not normalize:
        print("[WARN]  distance=cosine but model.normalize=false; forcing L2 normalization.")
        normalize = True
    print(f"Detector: k={k}, distance={distance}, normalize={normalize}")
    print(f"Embedding mode: {embedding_mode}")
    for line in describe_pipeline(pipeline_mode, win_frames, hop_frames, top_windows):
        print(line)
    print(f"Memory bank mode: {memory_mode}")

    # 3) locate dev_data test files
    root = Path(cfg["data"]["root"]) / "dev_data" / "raw"
    machines = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if machine_filter:
        machines = [m for m in machines if m == machine_filter]
        if not machines:
            raise RuntimeError(f"Machine {machine_filter!r} was not found under {root}")
    bank_dir = Path(cfg["logging"]["bank_out"])

    results = {}

    for m in machines:
        bank_path = get_bank_path(
            bank_dir,
            m,
            pipeline_mode,
            memory_mode,
            memory_cfg["num_clusters"],
            memory_cfg["cluster_score_normalization"],
            pca_cfg["n_components"] if pca_cfg["enabled"] else None,
            embedding_mode,
        )
        if not bank_path.exists():
            raise RuntimeError(
                f"Missing {pipeline_mode} bank for {m}: {bank_path}\n"
                f"Run: python scripts/train_knn.py --config {cfg_path} --machine {m} --pipeline-mode {pipeline_mode}"
            )

        bank_ckpt = load_bank_checkpoint(bank_path)
        meta = bank_ckpt.get("meta", {})
        matches, reason = bank_matches_pipeline(meta, pipeline_mode, win_frames, hop_frames)
        if not matches:
            raise RuntimeError(
                f"Bank {bank_path.name} does not match active pipeline: {reason}\n"
                f"Run: python scripts/train_knn.py --config {cfg_path} --machine {m} --pipeline-mode {pipeline_mode}"
            )

        def to_vec(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 1:
                return x
            if x.ndim == 2 and x.size(0) == 1:
                return x.squeeze(0)
            if x.ndim == 2:
                return x.mean(dim=0)
            raise ValueError(f"Unsupported bank entry shape: {tuple(x.shape)}")

        mem_bank = torch.stack([to_vec(x) for x in bank_ckpt["memory"]], dim=0).to(device)
        pca_payload = bank_ckpt.get("embedding_postprocess", {}).get("pca_whitening", {"enabled": False})
        bank_pca_enabled = bool(pca_payload.get("enabled", False))
        if normalize and not bank_pca_enabled:
            mem_bank = F.normalize(mem_bank, dim=-1)
        matches, reason = bank_matches_memory_config(
            meta,
            memory_mode,
            memory_cfg["num_clusters"],
            int(mem_bank.size(1)),
            cluster_score_normalization=memory_cfg["cluster_score_normalization"],
            pca_enabled=pca_cfg["enabled"],
            pca_n_components=pca_cfg["n_components"] if pca_cfg["enabled"] else None,
            embedding_mode=embedding_mode,
        )
        if not matches:
            raise RuntimeError(
                f"Bank {bank_path.name} does not match active memory-bank config: {reason}\n"
                f"Run: python scripts/train_knn.py --config {cfg_path} --machine {m} --pipeline-mode {pipeline_mode}"
            )

        cluster_labels = None
        centroids = None
        if memory_mode == "clustered":
            if "cluster_labels" not in bank_ckpt or "centroids" not in bank_ckpt:
                raise RuntimeError(f"Clustered bank {bank_path.name} is missing cluster_labels/centroids")
            cluster_labels = bank_ckpt["cluster_labels"].to(device=device, dtype=torch.long)
            centroids = bank_ckpt["centroids"].to(device=device, dtype=mem_bank.dtype)
            if normalize and not bank_pca_enabled:
                centroids = F.normalize(centroids, dim=-1)
            cluster_stats = bank_ckpt.get("cluster_score_stats")
            if memory_cfg["cluster_score_normalization"] != "none" and not cluster_stats:
                raise RuntimeError(f"Clustered bank {bank_path.name} is missing cluster_score_stats")
        else:
            cluster_stats = None

        threshold_percentile = float(cfg.get("threshold", {}).get("percentile", 90))
        if memory_mode == "clustered":
            threshold, _ = compute_clustered_threshold_and_mem_dists(
                mem_bank=mem_bank,
                cluster_labels=cluster_labels,
                centroids=centroids,
                k=k,
                distance=distance,
                pct=threshold_percentile,
                min_cluster_size=memory_cfg["min_cluster_size"],
                score_normalization=memory_cfg["cluster_score_normalization"],
                cluster_stats=cluster_stats,
                eps=memory_cfg["cluster_score_eps"],
            )
        else:
            threshold, _ = compute_threshold_and_mem_dists(mem_bank, k, distance, threshold_percentile)

        print(
            f"Preparing dev metrics for {m}: bank={bank_path.name}, entries={mem_bank.size(0)}, "
            f"score_norm={memory_cfg['cluster_score_normalization']}, "
            f"pca={'on' if bank_pca_enabled else 'off'}, "
            f"threshold_p{threshold_percentile:g}={threshold:.6f}"
        )

        # gather all test clips for this machine
        pattern = str(root / m / "test" / "**" / "*.wav")
        files = glob.glob(pattern, recursive=True)

        y_src, s_src = [], []
        y_tgt, s_tgt = [], []
        y_all, s_all = [], []

        for f in files:
            # ground truth: anomaly if filename contains 'anomaly'
            label = 1 if "anomaly" in os.path.basename(f) else 0

            # run inference
            wav, sr = load_audio_mono(f)
            def score_view(wav_i: torch.Tensor, is_augmented: bool) -> float:
                mel_aug = augmentor.augment_spectrogram if (is_augmented and spec_augment_active) else None
                if pipeline_mode == "clip":
                    feat = backbone(wav_i.to(device), sr, spec_augment=mel_aug)
                    if normalize:
                        feat = F.normalize(feat, dim=-1)
                    feat = apply_pca_whitening_tensor(feat, pca_payload)
                    if memory_mode == "clustered":
                        return score_clustered_knn(
                            query=feat,
                            mem_bank=mem_bank,
                            cluster_labels=cluster_labels,
                            centroids=centroids,
                            k=k,
                            distance=distance,
                            min_cluster_size=memory_cfg["min_cluster_size"],
                            score_normalization=memory_cfg["cluster_score_normalization"],
                            cluster_stats=cluster_stats,
                            eps=memory_cfg["cluster_score_eps"],
                        )
                    if distance == "cosine":
                        d = 1.0 - (feat @ mem_bank.T)
                    else:
                        d = torch.cdist(feat, mem_bank)
                    topk = torch.topk(d, k=k, dim=1, largest=False).values
                    return float(topk.mean().item())

                temporal = backbone(
                    wav_i.to(device),
                    sr,
                    return_temporal=True,
                    spec_augment=mel_aug,
                )
                win_emb = make_window_embeddings(temporal, win_frames, hop_frames)
                if normalize:
                    win_emb = F.normalize(win_emb, dim=-1)
                win_emb = apply_pca_whitening_tensor(win_emb, pca_payload)
                if distance == "cosine":
                    d = 1.0 - (win_emb @ mem_bank.T)
                else:
                    d = torch.cdist(win_emb, mem_bank)
                topk = torch.topk(d, k=k, dim=1, largest=False).values
                win_scores = topk.mean(dim=1)
                n_top = min(top_windows, win_scores.numel())
                return float(torch.topk(win_scores, k=n_top, largest=True).values.mean().item())

            view_scores = []
            for wav_i, _, is_augmented in augmentor.waveform_views(wav, int(sr), test_aug_views):
                view_scores.append(score_view(wav_i, is_augmented))
            score = aggregate_augmented_scores(view_scores, aug_config.test_score_agg)

            if "_source_test_" in f:
                y_src.append(label)
                s_src.append(score)
            elif "_target_test_" in f:
                y_tgt.append(label)
                s_tgt.append(score)
            y_all.append(label)
            s_all.append(score)

        # sanity check
        if not y_src or not y_tgt:
            raise RuntimeError(f"No source/target clips found for {m} under {pattern}")

        # compute metrics (as percents)
        auc_all = roc_auc_score(y_all, s_all) * 100
        auc_source = roc_auc_score(y_src, s_src) * 100
        auc_target = roc_auc_score(y_tgt, s_tgt) * 100
        # partial AUC at FPR<=0.1, normalized to [0,1], then *100
        pauc = roc_auc_score(y_tgt, s_tgt, max_fpr=0.1) * 100
        precision_source, recall_source, f1_source = binary_metrics(y_src, s_src, threshold)
        precision_target, recall_target, f1_target = binary_metrics(y_tgt, s_tgt, threshold)
        official_score = _harmonic_mean([auc_source / 100.0, auc_target / 100.0, pauc / 100.0]) * 100

        results[m] = {
            "auc_all": float(auc_all),
            "auc_source": float(auc_source),
            "auc_target": float(auc_target),
            "pauc": float(pauc),
            "precision_source": float(precision_source),
            "precision_target": float(precision_target),
            "recall_source": float(recall_source),
            "recall_target": float(recall_target),
            "f1_source": float(f1_source),
            "f1_target": float(f1_target),
            "official_score": float(official_score),
        }

    # 4) print YAML snippet
    print("results:")
    print("  development_dataset:")
    for m, vals in results.items():
        print(f"    {m}:")
        print(f"      auc_all: {vals['auc_all']:.2f}")
        print(f"      auc_source: {vals['auc_source']:.2f}")
        print(f"      auc_target: {vals['auc_target']:.2f}")
        print(f"      pauc: {vals['pauc']:.2f}")
        print(f"      precision_source: {vals['precision_source']:.2f}")
        print(f"      precision_target: {vals['precision_target']:.2f}")
        print(f"      recall_source: {vals['recall_source']:.2f}")
        print(f"      recall_target: {vals['recall_target']:.2f}")
        print(f"      f1_source: {vals['f1_source']:.2f}")
        print(f"      f1_target: {vals['f1_target']:.2f}")
        print(f"      official_score: {vals['official_score']:.2f}")

    if return_results:
        return results
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to your inference/training config"
    )
    ap.add_argument("--machine", default=None, help="Optional machine filter, e.g. AutoTrash")
    ap.add_argument(
        "--embedding-mode",
        default=None,
        choices=[
            "last_layer_mean",
            "last_layer_cls",
            "last_layer_mean_std",
            "last4_layers_mean",
            "middle_layer_mean",
            "middle_layer_mean_std",
        ],
        help="Override model.embedding_mode from config.",
    )
    args = ap.parse_args()
    compute_metrics(args.config, machine_filter=args.machine, embedding_mode_override=args.embedding_mode)
