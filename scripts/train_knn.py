#!/usr/bin/env python
"""
Build k-NN memory bank for DCASE-2025 Task-2.
"""

import argparse
import sys
from pathlib import Path
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans

# Allow running as: python scripts/train_knn.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.file_utils import load_config
from src.data.dcase_dataset import DCASETask2Dataset
from src.models.beats_backbone import BEATsBackbone
from src.augmentations import (
    AugmentationConfig,
    DomainGeneralizationAugmentor,
    describe_augmentation,
    set_augmentation_seed,
)
from src.pipeline import (
    describe_pipeline,
    get_bank_path,
    get_embedding_mode,
    get_embedding_postprocess_config,
    get_legacy_bank_paths,
    get_memory_bank_config,
    get_pipeline_mode,
    get_stale_bank_paths,
    get_window_params,
    validate_pipeline_memory_compatibility,
)
from src.cluster_scoring import compute_cluster_score_stats
from src.embedding_postprocess import fit_pca_whitening


def collate(batch):
    return batch      # keep list-of-tuples


def collect_wavs(root: str, split: str, stage: str = "both", machine: str | None = None) -> list[str]:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    if stage not in {"dev_data", "eval_data", "both"}:
        raise ValueError("stage must be 'dev_data', 'eval_data', or 'both'")

    stages = ("dev_data", "eval_data") if stage == "both" else (stage,)

    wavs: list[str] = []
    for st in stages:
        if machine:
            patt = Path(root, st, "raw", machine, split, "**", "*.wav")
        else:
            patt = Path(root, st, "raw", "**", split, "**", "*.wav")
        wavs += glob.glob(str(patt), recursive=True)

    wavs = sorted(set(wavs))
    if not wavs:
        scope = f"stage={stage}, machine={machine or '*'}"
        raise RuntimeError(f"No wavs found under {root} for split='{split}' ({scope}).")
    return wavs


def make_window_embeddings(
    temporal_feats: torch.Tensor,
    win_frames: int,
    hop_frames: int,
) -> torch.Tensor:
    """Convert temporal features to window embeddings.

    Args:
        temporal_feats: (B,T,D) or (T,D). Assumes B==1 if 3D.
        win_frames: window size in frames.
        hop_frames: hop size in frames.

    Returns:
        (W,D) window embeddings (mean over frames per window).
    """
    if temporal_feats.ndim == 3:
        if temporal_feats.size(0) != 1:
            raise ValueError(f"Expected B==1 temporal feats, got {tuple(temporal_feats.shape)}")
        x = temporal_feats[0]
    elif temporal_feats.ndim == 2:
        x = temporal_feats
    else:
        raise ValueError(f"Unsupported temporal feats shape: {tuple(temporal_feats.shape)}")

    T, D = x.shape
    if T == 0:
        raise ValueError("Empty temporal features")

    win_frames = int(win_frames)
    hop_frames = int(hop_frames)
    if win_frames <= 0 or hop_frames <= 0:
        raise ValueError("win_frames and hop_frames must be > 0")

    if T < win_frames:
        return x.mean(dim=0, keepdim=True)

    windows = []
    for start in range(0, T - win_frames + 1, hop_frames):
        seg = x[start : start + win_frames]
        windows.append(seg.mean(dim=0))
    return torch.stack(windows, dim=0)  # (W,D)


def _to_vec(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.size(0) == 1:
        return x.squeeze(0)
    if x.ndim == 2:
        return x.mean(dim=0)
    raise ValueError(f"Unsupported feature shape for bank entry: {tuple(x.shape)}")


def build_kmeans_bank(
    feats: list[torch.Tensor],
    requested_clusters: int,
    min_cluster_size: int,
    seed: int | None,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, list[int]]:
    """Cluster clip-level memory embeddings and return labels/centroids."""
    if not feats:
        raise RuntimeError("Cannot cluster an empty memory bank")

    matrix = torch.stack([_to_vec(f).detach().cpu() for f in feats], dim=0).float()
    if normalize:
        matrix = F.normalize(matrix, dim=-1)

    n_embeddings = int(matrix.size(0))
    n_clusters = int(requested_clusters)
    if n_clusters > n_embeddings:
        print(
            f"WARNING: memory_bank.num_clusters={n_clusters} > embeddings={n_embeddings}; "
            f"using num_clusters={n_embeddings}."
        )
        n_clusters = n_embeddings

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=None if seed is None else int(seed),
        n_init=10,
    )
    labels_np = kmeans.fit_predict(matrix.numpy())
    centroids = torch.from_numpy(kmeans.cluster_centers_).float()
    if normalize:
        centroids = F.normalize(centroids, dim=-1)

    cluster_counts = np.bincount(labels_np, minlength=n_clusters).astype(int).tolist()
    for cluster_id, count in enumerate(cluster_counts):
        if count < min_cluster_size:
            print(
                f"WARNING: cluster {cluster_id} has {count} embedding(s), "
                f"below min_cluster_size={min_cluster_size}; inference will fallback if needed."
            )

    return matrix, torch.from_numpy(labels_np).long(), centroids, n_clusters, cluster_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--machine",
        default=None,
        help="If set, build a memory bank only for this machine (e.g., AutoTrash).",
    )
    ap.add_argument(
        "--bank-level",
        default=None,
        choices=["clip", "window"],
        help="Deprecated alias for --pipeline-mode.",
    )
    ap.add_argument(
        "--pipeline-mode",
        default=None,
        choices=["clip", "window"],
        help="Override pipeline.mode from config.",
    )
    ap.add_argument(
        "--win-frames",
        type=int,
        default=None,
        help="Window size in frames when pipeline mode is window.",
    )
    ap.add_argument(
        "--hop-frames",
        type=int,
        default=None,
        help="Hop size in frames when pipeline mode is window.",
    )
    ap.add_argument(
        "--stage",
        default="both",
        choices=["dev_data", "eval_data", "both"],
        help="Which dataset stage to use for building the bank.",
    )
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

    cfg = load_config(args.config)
    embedding_mode = get_embedding_mode(cfg, args.embedding_mode)
    pipeline_override = args.pipeline_mode or args.bank_level
    pipeline_mode = get_pipeline_mode(cfg, pipeline_override)
    win_frames, hop_frames = get_window_params(cfg, args.win_frames, args.hop_frames)
    memory_cfg = get_memory_bank_config(cfg)
    memory_mode = memory_cfg["mode"]
    post_cfg = get_embedding_postprocess_config(cfg)
    pca_cfg = post_cfg["pca_whitening"]
    validate_pipeline_memory_compatibility(pipeline_mode, memory_mode)
    aug_config = AugmentationConfig.from_config(cfg)
    seed = aug_config.seed
    set_augmentation_seed(seed)
    aug_rng = torch.Generator()
    if seed is not None:
        aug_rng.manual_seed(int(seed))
    else:
        aug_rng.seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = cfg["data"]["root"]
    out_dir = Path(cfg["logging"]["bank_out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = get_bank_path(
        out_dir,
        args.machine,
        pipeline_mode,
        memory_mode,
        memory_cfg["num_clusters"],
        memory_cfg["cluster_score_normalization"],
        pca_cfg["n_components"] if pca_cfg["enabled"] else None,
        embedding_mode,
    )

    if args.machine is None and args.stage == "both":
        dataset = DCASETask2Dataset(root, split="train")
    else:
        files = collect_wavs(root=root, split="train", stage=args.stage, machine=args.machine)
        dataset = DCASETask2Dataset(root, split="train", files=files)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    backbone = BEATsBackbone(
        checkpoint=cfg["model"]["embedding"],
        use_layer_stack=cfg["model"].get("use_layer_stack", False),
        embedding_mode=embedding_mode,
    ).to(device).eval()

    augmentation_enabled = aug_config.enabled
    augmentation_views = aug_config.num_views_train if augmentation_enabled else 0
    augmentor = DomainGeneralizationAugmentor(aug_config, generator=aug_rng)
    spec_augment_active = (
        augmentation_enabled
        and augmentation_views > 0
        and (aug_config.freq_mask_max > 0 or aug_config.time_mask_max > 0)
        and not backbone.expect_waveform
    )
    for line in describe_augmentation(aug_config, phase="train-memory"):
        print(line)
    if augmentation_enabled and augmentation_views > 0 and spec_augment_active:
        print("  log-mel masking: active for augmented spectrogram views")
    elif augmentation_enabled and augmentation_views > 0:
        print("  log-mel masking: inactive for waveform-input backbone")

    k = cfg.get("detector", {}).get("k", cfg.get("model", {}).get("k", 3))
    distance = str(cfg.get("detector", {}).get("distance", "cosine")).lower()
    normalize = bool(cfg.get("model", {}).get("normalize", False))
    if distance == "cosine" and not normalize:
        print("[WARN]  distance=cosine but model.normalize=false; forcing L2 normalization.")
        normalize = True

    threshold_percentile = float(cfg.get("threshold", {}).get("percentile", 90))
    print("FINAL ATTEMPT CONFIG")
    print(f"- pipeline mode: {pipeline_mode}")
    print(f"- memory bank mode: {memory_mode}")
    print(f"- num clusters: {memory_cfg['num_clusters']}")
    print(f"- score normalization mode: {memory_cfg['cluster_score_normalization']}")
    print(f"- embedding mode: {embedding_mode}")
    print(f"- PCA whitening: {'enabled' if pca_cfg['enabled'] else 'disabled'}")
    if pca_cfg["enabled"]:
        print(f"- PCA components: {pca_cfg['n_components']}")
    print(f"- threshold percentile: {threshold_percentile:g}")
    print(f"- distance metric: {distance}")
    print(f"- kNN K: {k}")
    print(f"- bank path: {out_path}")

    print(f"Detector: k={k}, distance={distance}, normalize={normalize}")
    print(f"Augmented memory mode: {aug_config.memory_mode}")
    for line in describe_pipeline(pipeline_mode, win_frames, hop_frames):
        print(line)
    print(f"Memory bank mode: {memory_mode}")
    if memory_mode == "clustered":
        print(f"  num_clusters: {memory_cfg['num_clusters']}")
        print(f"  cluster_method: {memory_cfg['cluster_method']}")
        print(f"  cluster_score_mode: {memory_cfg['cluster_score_mode']}")
        print(f"  cluster_score_normalization: {memory_cfg['cluster_score_normalization']}")
        print(f"  cluster_score_eps: {memory_cfg['cluster_score_eps']}")
        print(f"  min_cluster_size: {memory_cfg['min_cluster_size']}")
    print(f"  cleanup stale banks: {memory_cfg['cleanup_stale_banks']}")
    print(f"Bank output path: {out_path}")

    feats, paths = [], []
    clean_entries = 0
    total_entries = 0

    def _extract_clip_embedding(wav_i: torch.Tensor, sr_i: int, is_augmented: bool) -> torch.Tensor:
        mel_aug = augmentor.augment_spectrogram if (is_augmented and spec_augment_active) else None
        feat_i = backbone(wav_i.to(device), sr_i, spec_augment=mel_aug)
        if normalize:
            feat_i = F.normalize(feat_i, dim=-1)
        return feat_i

    def _extract_window_embeddings(wav_i: torch.Tensor, sr_i: int, is_augmented: bool) -> torch.Tensor:
        mel_aug = augmentor.augment_spectrogram if (is_augmented and spec_augment_active) else None
        temporal = backbone(
            wav_i.to(device),
            sr_i,
            return_temporal=True,
            spec_augment=mel_aug,
        )
        win_emb_i = make_window_embeddings(temporal, win_frames, hop_frames)
        if normalize:
            win_emb_i = F.normalize(win_emb_i, dim=-1)
        return win_emb_i

    for batch in tqdm(loader, desc="Extracting embeddings"):
        wav, sr, path = batch[0]
        variants = augmentor.waveform_views(wav, int(sr), augmentation_views)

        if pipeline_mode == "clip":
            view_feats = [
                _extract_clip_embedding(wav_i, int(sr), is_augmented)
                for wav_i, _, is_augmented in variants
            ]

            if augmentation_enabled and augmentation_views > 0 and aug_config.memory_mode == "average":
                feat = torch.stack(view_feats, dim=0).mean(dim=0)
                if normalize:
                    feat = F.normalize(feat, dim=-1)
                feats.append(feat.detach().cpu())
                paths.append(f"{path}#augavg={len(view_feats)}")
                clean_entries += 1
                total_entries += 1
            else:
                for feat, (_, view_tag, _) in zip(view_feats, variants):
                    feats.append(feat.detach().cpu())
                    paths.append(path if view_tag == "clean" else f"{path}#{view_tag}")
                    total_entries += 1
                    if view_tag == "clean":
                        clean_entries += 1
        else:
            view_windows = [
                _extract_window_embeddings(wav_i, int(sr), is_augmented)
                for wav_i, _, is_augmented in variants
            ]

            if augmentation_enabled and augmentation_views > 0 and aug_config.memory_mode == "average":
                min_windows = min(win.size(0) for win in view_windows)
                win_emb = torch.stack([win[:min_windows] for win in view_windows], dim=0).mean(dim=0)
                if normalize:
                    win_emb = F.normalize(win_emb, dim=-1)
                for w_i in range(win_emb.size(0)):
                    feats.append(win_emb[w_i].detach().cpu())
                    paths.append(f"{path}#augavg={len(view_windows)}#win={w_i}")
                clean_entries += int(view_windows[0].size(0))
                total_entries += int(win_emb.size(0))
            else:
                for win_emb, (_, view_tag, _) in zip(view_windows, variants):
                    path_i = path if view_tag == "clean" else f"{path}#{view_tag}"
                    for w_i in range(win_emb.size(0)):
                        feats.append(win_emb[w_i].detach().cpu())
                        paths.append(f"{path_i}#win={w_i}")
                    total_entries += int(win_emb.size(0))
                    if view_tag == "clean":
                        clean_entries += int(win_emb.size(0))

    removed_stale = []
    cleanup_enabled = bool(cfg.get("pipeline", {}).get("cleanup_stale_banks", True)) or bool(
        memory_cfg["cleanup_stale_banks"]
    )
    if cleanup_enabled:
        stale_paths = get_stale_bank_paths(
            out_dir,
            args.machine,
            pipeline_mode,
            memory_mode,
            memory_cfg["num_clusters"],
            memory_cfg["cluster_score_normalization"],
            pca_cfg["n_components"] if pca_cfg["enabled"] else None,
            embedding_mode,
        )
        stale_paths.extend(get_legacy_bank_paths(out_dir, args.machine, pipeline_mode))
        for stale_path in sorted(set(stale_paths)):
            if stale_path.exists() and stale_path.resolve() != out_path.resolve():
                print(f"Removed stale bank: {stale_path}")
                stale_path.unlink()
                removed_stale.append(str(stale_path))
    if cleanup_enabled and not removed_stale:
        print("No stale banks to clean.")

    cluster_payload = {}
    effective_clusters = None
    cluster_counts = None
    pca_payload = {"enabled": False}
    if pca_cfg["enabled"]:
        matrix = torch.stack([_to_vec(f).detach().cpu() for f in feats], dim=0).float()
        matrix, pca_payload = fit_pca_whitening(matrix, pca_cfg)
        feats = [row.detach().cpu() for row in matrix]
        print(
            "PCA whitening fitted: "
            f"requested={pca_payload['requested_n_components']}, "
            f"effective={pca_payload['n_components']}, "
            f"whiten={pca_payload['whiten']}, l2_after={pca_payload['l2_after']}"
        )

    if memory_mode == "clustered":
        matrix, cluster_labels, centroids, effective_clusters, cluster_counts = build_kmeans_bank(
            feats=feats,
            requested_clusters=int(memory_cfg["num_clusters"]),
            min_cluster_size=int(memory_cfg["min_cluster_size"]),
            seed=seed,
            normalize=normalize and not pca_cfg["enabled"],
        )
        feats = [row.detach().cpu() for row in matrix]
        cluster_score_stats = None
        if memory_cfg["cluster_score_normalization"] != "none":
            print("Computing cluster-local leave-one-out kNN distance statistics.")
            cluster_score_stats = compute_cluster_score_stats(
                mem_bank=matrix,
                cluster_labels=cluster_labels,
                k=int(k),
                distance=distance,
                min_cluster_size=int(memory_cfg["min_cluster_size"]),
            )
        cluster_payload = {
            "cluster_labels": cluster_labels,
            "centroids": centroids,
            "cluster_counts": cluster_counts,
            "cluster_score_stats": cluster_score_stats,
        }

    torch.save(
        {
            "memory": feats,
            "paths": paths,
            **cluster_payload,
            "embedding_postprocess": {
                "pca_whitening": pca_payload,
            },
            "meta": {
                "machine": args.machine,
                "stage": args.stage,
                "split": "train",
                "pipeline_mode": pipeline_mode,
                "memory_bank_mode": memory_mode,
                "bank_level": pipeline_mode,
                "win_frames": win_frames if pipeline_mode == "window" else None,
                "hop_frames": hop_frames if pipeline_mode == "window" else None,
                "num_clusters": effective_clusters,
                "requested_num_clusters": memory_cfg["num_clusters"] if memory_mode == "clustered" else None,
                "cluster_method": memory_cfg["cluster_method"] if memory_mode == "clustered" else None,
                "cluster_score_mode": memory_cfg["cluster_score_mode"] if memory_mode == "clustered" else None,
                "cluster_score_normalization": (
                    memory_cfg["cluster_score_normalization"] if memory_mode == "clustered" else "none"
                ),
                "cluster_score_eps": memory_cfg["cluster_score_eps"] if memory_mode == "clustered" else None,
                "min_cluster_size": memory_cfg["min_cluster_size"] if memory_mode == "clustered" else None,
                "cluster_counts": cluster_counts,
                "distance": distance,
                "k": k,
                "embedding_model": cfg["model"].get("embedding"),
                "embedding_mode": embedding_mode,
                "feature_dim": int(_to_vec(feats[0]).numel()) if feats else None,
                "embedding_postprocess": {
                    "pca_whitening": {
                        "enabled": bool(pca_payload.get("enabled", False)),
                        "requested_n_components": pca_payload.get("requested_n_components"),
                        "n_components": pca_payload.get("n_components"),
                        "whiten": pca_payload.get("whiten"),
                        "l2_after": pca_payload.get("l2_after"),
                        "input_feature_dim": pca_payload.get("input_feature_dim"),
                    }
                },
                "augmentation": {
                    "enabled": augmentation_enabled,
                    "num_views_train": augmentation_views,
                    "memory_mode": aug_config.memory_mode,
                    "seed": seed,
                    "spec_augment_active": spec_augment_active,
                    "clean_entries": clean_entries,
                    "total_entries": total_entries,
                },
            },
        },
        out_path,
    )
    print(f"[OK] memory bank -> {out_path}")
    print(f"Memory bank size before augmentation: {clean_entries}")
    print(f"Memory bank size after augmentation:  {total_entries}")
    if memory_mode == "clustered":
        print(f"Clustered bank: requested_clusters={memory_cfg['num_clusters']}, effective_clusters={effective_clusters}")
        print(f"Cluster sizes: {cluster_counts}")


if __name__ == "__main__":
    main()
