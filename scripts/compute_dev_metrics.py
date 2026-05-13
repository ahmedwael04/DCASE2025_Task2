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
from sklearn.metrics import roc_auc_score

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
    get_memory_bank_config,
    get_pipeline_mode,
    get_top_windows,
    get_window_params,
    validate_pipeline_memory_compatibility,
)


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
) -> float:
    if query.ndim == 1:
        query = query.unsqueeze(0)

    q_norm = F.normalize(query, dim=-1)
    c_norm = F.normalize(centroids, dim=-1)
    centroid_dists = 1.0 - (q_norm @ c_norm.T)
    cluster_id = int(torch.argmin(centroid_dists, dim=1).item())

    selected = torch.nonzero(cluster_labels == cluster_id, as_tuple=False).flatten()
    if int(selected.numel()) < max(int(k), int(min_cluster_size)):
        selected = torch.arange(mem_bank.size(0), device=mem_bank.device)

    bank = mem_bank[selected]
    if distance == "cosine":
        d = 1.0 - (query @ bank.T)
    else:
        d = torch.cdist(query, bank)
    topk = torch.topk(d, k=min(int(k), int(d.size(1))), dim=1, largest=False).values
    return float(topk.mean().item())


def compute_metrics(cfg_path: str):
    # 1) load config, set up device
    cfg    = load_config(cfg_path)
    pipeline_mode = get_pipeline_mode(cfg)
    win_frames, hop_frames = get_window_params(cfg)
    top_windows = get_top_windows(cfg)
    memory_cfg = get_memory_bank_config(cfg)
    memory_mode = memory_cfg["mode"]
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
    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()
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
        print("⚠️  distance=cosine but model.normalize=false; forcing L2 normalization.")
        normalize = True
    print(f"Detector: k={k}, distance={distance}, normalize={normalize}")
    for line in describe_pipeline(pipeline_mode, win_frames, hop_frames, top_windows):
        print(line)
    print(f"Memory bank mode: {memory_mode}")

    # 3) locate dev_data test files
    root = Path(cfg["data"]["root"]) / "dev_data" / "raw"
    machines = sorted([d.name for d in root.iterdir() if d.is_dir()])
    bank_dir = Path(cfg["logging"]["bank_out"])

    results = {}

    for m in machines:
        bank_path = get_bank_path(
            bank_dir,
            m,
            pipeline_mode,
            memory_mode,
            memory_cfg["num_clusters"],
        )
        if not bank_path.exists():
            raise RuntimeError(
                f"Missing {pipeline_mode} bank for {m}: {bank_path}\n"
                f"Run: python scripts/train_knn.py --config {cfg_path} --machine {m} --pipeline-mode {pipeline_mode}"
            )

        bank_ckpt = torch.load(bank_path, map_location="cpu")
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
        if normalize:
            mem_bank = F.normalize(mem_bank, dim=-1)
        matches, reason = bank_matches_memory_config(
            meta,
            memory_mode,
            memory_cfg["num_clusters"],
            int(mem_bank.size(1)),
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
            if normalize:
                centroids = F.normalize(centroids, dim=-1)

        print(f"Preparing dev metrics for {m}: bank={bank_path.name}, entries={mem_bank.size(0)}")

        # gather all test clips for this machine
        pattern = str(root / m / "test" / "**" / "*.wav")
        files = glob.glob(pattern, recursive=True)

        y_src, s_src = [], []
        y_tgt, s_tgt = [], []

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
                    if memory_mode == "clustered":
                        return score_clustered_knn(
                            query=feat,
                            mem_bank=mem_bank,
                            cluster_labels=cluster_labels,
                            centroids=centroids,
                            k=k,
                            distance=distance,
                            min_cluster_size=memory_cfg["min_cluster_size"],
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

        # sanity check
        if not y_src or not y_tgt:
            raise RuntimeError(f"No source/target clips found for {m} under {pattern}")

        # compute metrics (as percents)
        auc_source = roc_auc_score(y_src, s_src) * 100
        auc_target = roc_auc_score(y_tgt, s_tgt) * 100
        # partial AUC at FPR<=0.1, normalized to [0,1], then *100
        pauc = roc_auc_score(y_tgt, s_tgt, max_fpr=0.1) * 100

        results[m] = (auc_source, auc_target, pauc)

    # 4) print YAML snippet
    print("results:")
    print("  development_dataset:")
    for m, (asrc, atgt, p) in results.items():
        print(f"    {m}:")
        print(f"      auc_source: {asrc:.2f}")
        print(f"      auc_target: {atgt:.2f}")
        print(f"      pauc: {p:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to your inference/training config"
    )
    args = ap.parse_args()
    compute_metrics(args.config)
