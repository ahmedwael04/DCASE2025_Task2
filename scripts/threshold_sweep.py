#!/usr/bin/env python
"""Threshold sweep for the clustered k=6 clip ASD pipeline.

The script computes anomaly scores once, computes memory-bank percentile
thresholds once, then applies each threshold to the same scores.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm.auto import tqdm

# Allow running as: python scripts/threshold_sweep.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.infer import compute_clustered_threshold_and_mem_dists, score_clustered_knn
from src.augmentations import (
    AugmentationConfig,
    DomainGeneralizationAugmentor,
    aggregate_augmented_scores,
    set_augmentation_seed,
)
from src.models.beats_backbone import BEATsBackbone
from src.pipeline import (
    bank_matches_memory_config,
    bank_matches_pipeline,
    get_bank_path,
    get_memory_bank_config,
    get_pipeline_mode,
    get_stale_bank_paths,
    get_window_params,
)
from src.utils.audio_utils import load_audio_mono
from src.utils.file_utils import load_config


REQUIRED_PIPELINE_MODE = "clip"
REQUIRED_MEMORY_MODE = "clustered"
REQUIRED_NUM_CLUSTERS = 6


def require_best_architecture(cfg: dict) -> tuple[dict, str]:
    pipeline_mode = get_pipeline_mode(cfg)
    memory_cfg = get_memory_bank_config(cfg)
    errors = []
    if pipeline_mode != REQUIRED_PIPELINE_MODE:
        errors.append(f"pipeline.mode must be '{REQUIRED_PIPELINE_MODE}', got '{pipeline_mode}'")
    if memory_cfg["mode"] != REQUIRED_MEMORY_MODE:
        errors.append(f"memory_bank.mode must be '{REQUIRED_MEMORY_MODE}', got '{memory_cfg['mode']}'")
    if int(memory_cfg["num_clusters"]) != REQUIRED_NUM_CLUSTERS:
        errors.append(
            f"memory_bank.num_clusters must be {REQUIRED_NUM_CLUSTERS}, got {memory_cfg['num_clusters']}"
        )
    if errors:
        raise RuntimeError(
            "Threshold sweep is locked to the current best architecture:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )
    return memory_cfg, pipeline_mode


def to_vec(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.size(0) == 1:
        return x.squeeze(0)
    if x.ndim == 2:
        return x.mean(dim=0)
    raise ValueError(f"Unsupported bank entry shape: {tuple(x.shape)}")


def infer_domain(path: str) -> str | None:
    s = str(path).lower()
    if "_source_test_" in s:
        return "source"
    if "_target_test_" in s:
        return "target"
    return None


def infer_dev_label(path: str) -> int:
    return 1 if "anomaly" in os.path.basename(path).lower() else 0


def find_machine_dir(stage_raw: Path, machine: str) -> Path | None:
    exact = stage_raw / machine
    if exact.exists():
        return exact
    if not stage_raw.exists():
        return None
    machine_lower = machine.lower()
    for directory in stage_raw.iterdir():
        if directory.is_dir() and directory.name.lower() == machine_lower:
            return directory
    return None


def find_machine_test_wavs(cfg: dict, machine: str, allow_unlabeled_eval: bool) -> tuple[list[str], str, bool]:
    """Find test wavs, using labeled dev files unless eval fallback is explicit."""
    root = Path(cfg["data"]["root"])

    dev_machine_dir = find_machine_dir(root / "dev_data" / "raw", machine)
    if dev_machine_dir is not None:
        directory = dev_machine_dir / "test"
        wavs = sorted(glob.glob(str(directory / "**" / "*.wav"), recursive=True))
        if wavs:
            return wavs, "dev_data", True

    if not allow_unlabeled_eval:
        searched = root / "dev_data" / "raw" / machine / "test"
        available = []
        dev_raw = root / "dev_data" / "raw"
        if dev_raw.exists():
            available = sorted(d.name for d in dev_raw.iterdir() if d.is_dir())
        raise RuntimeError(
            f"No labeled dev test wavs found for machine '{machine}' under {searched}.\n"
            "Threshold optimization requires labeled dev data. "
            "Use a dev machine name, or pass --allow-unlabeled-eval for score-only eval output.\n"
            f"Available dev machines: {available}"
        )

    eval_machine_dir = find_machine_dir(root / "eval_data" / "raw", machine)
    if eval_machine_dir is not None:
        directory = eval_machine_dir / "test"
        wavs = sorted(glob.glob(str(directory / "**" / "*.wav"), recursive=True))
        if wavs:
            if machine.lower() == "autotrash":
                print("Threshold sweep is disabled for AutoTrash because eval data is unlabeled.")
            else:
                print("Unlabeled eval data cannot be used for threshold optimization.")
            return wavs, "eval_data", False

    searched = [
        root / "dev_data" / "raw" / machine / "test",
        root / "eval_data" / "raw" / machine / "test",
    ]
    raise RuntimeError(f"No test wavs found for {machine}. Searched: {searched}")


def harmonic_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if float(v) > 0.0]
    if len(vals) != len(values):
        return 0.0
    return float(len(vals) / sum(1.0 / v for v in vals))


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


def cleanup_conflicting_banks(bank_dir: Path, machine: str, active_bank: Path, enabled: bool) -> list[str]:
    if not enabled:
        print("Stale bank cleanup disabled.")
        return []

    stale_paths = get_stale_bank_paths(
        bank_dir=bank_dir,
        machine=machine,
        pipeline_mode=REQUIRED_PIPELINE_MODE,
        memory_mode=REQUIRED_MEMORY_MODE,
        num_clusters=REQUIRED_NUM_CLUSTERS,
    )

    # Global legacy bank is ambiguous and can hide old experiments.
    legacy_global = bank_dir / "memory_bank.pt"
    if legacy_global.exists():
        stale_paths.append(legacy_global)

    removed = []
    for path in sorted(set(stale_paths)):
        if path.resolve() == active_bank.resolve():
            continue
        if path.exists():
            path.unlink()
            removed.append(str(path))
            print(f"Removed stale bank: {path}")

    if not removed:
        print("No stale banks to clean.")
    return removed


def load_clustered_bank(
    cfg: dict,
    cfg_path: str,
    machine: str,
    bank_path: Path,
    memory_cfg: dict,
    device: torch.device,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if not bank_path.exists():
        if not bool(cfg.get("pipeline", {}).get("auto_rebuild_bank", True)):
            raise RuntimeError(
                f"Missing clustered k6 bank: {bank_path}\n"
                "Set pipeline.auto_rebuild_bank=true or run train_knn.py first."
            )
        print(f"Missing clustered k6 bank: {bank_path}")
        print("Rebuilding clustered k6 clip bank before threshold sweep.")
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "train_knn.py"),
                "--config",
                cfg_path,
                "--machine",
                machine,
                "--pipeline-mode",
                REQUIRED_PIPELINE_MODE,
            ],
            check=True,
        )

    ckpt = load_bank_checkpoint(bank_path)
    meta = ckpt.get("meta", {})
    win_frames, hop_frames = get_window_params(cfg)
    matches, reason = bank_matches_pipeline(meta, REQUIRED_PIPELINE_MODE, win_frames, hop_frames)
    if not matches:
        raise RuntimeError(f"Bank {bank_path.name} does not match clip pipeline: {reason}")

    if "cluster_labels" not in ckpt or "centroids" not in ckpt:
        raise RuntimeError(f"Clustered bank {bank_path.name} is missing cluster_labels/centroids")

    mem_bank = torch.stack([to_vec(x) for x in ckpt["memory"]], dim=0).to(device).float()
    if normalize:
        mem_bank = F.normalize(mem_bank, dim=-1)

    matches, reason = bank_matches_memory_config(
        meta,
        REQUIRED_MEMORY_MODE,
        REQUIRED_NUM_CLUSTERS,
        int(mem_bank.size(1)),
    )
    if not matches:
        raise RuntimeError(f"Bank {bank_path.name} does not match clustered k6 config: {reason}")

    cluster_labels = ckpt["cluster_labels"].to(device=device, dtype=torch.long)
    centroids = ckpt["centroids"].to(device=device, dtype=mem_bank.dtype)
    if normalize:
        centroids = F.normalize(centroids, dim=-1)
    return mem_bank, cluster_labels, centroids, meta


def score_cache_valid(cache_path: Path, bank_path: Path, data_stage: str) -> bool:
    if not cache_path.exists() or not bank_path.exists():
        return False
    try:
        cache = np.load(cache_path, allow_pickle=False)
        cached_bank = str(cache["bank_path"].item())
        cached_mtime = float(cache["bank_mtime"].item())
        cached_stage = str(cache["data_stage"].item())
        return (
            cached_bank == str(bank_path)
            and cached_stage == str(data_stage)
            and abs(cached_mtime - bank_path.stat().st_mtime) < 1e-6
        )
    except Exception:
        return False


def compute_or_load_scores(
    cfg: dict,
    machine: str,
    bank_path: Path,
    cache_path: Path,
    mem_bank: torch.Tensor,
    cluster_labels: torch.Tensor,
    centroids: torch.Tensor,
    memory_cfg: dict,
    device: torch.device,
    normalize: bool,
    recompute: bool,
    allow_unlabeled_eval: bool,
) -> dict[str, np.ndarray]:
    wavs, stage, labeled_stage = find_machine_test_wavs(cfg, machine, allow_unlabeled_eval)
    print(f"Found {len(wavs)} test clips for {machine} under {stage}.")

    if not recompute and score_cache_valid(cache_path, bank_path, stage):
        print(f"Reusing cached anomaly scores: {cache_path}")
        cache = np.load(cache_path, allow_pickle=False)
        return {
            "names": cache["names"],
            "scores": cache["scores"],
            "labels": cache["labels"],
            "domains": cache["domains"],
            "data_stage": np.asarray(str(stage)),
            "labeled_stage": np.asarray(bool(labeled_stage)),
        }

    print("Computing anomaly scores once for threshold sweep.")
    aug_config = AugmentationConfig.from_config(cfg)
    set_augmentation_seed(aug_config.seed)
    aug_rng = torch.Generator()
    if aug_config.seed is not None:
        aug_rng.manual_seed(int(aug_config.seed))
    else:
        aug_rng.seed()
    augmentor = DomainGeneralizationAugmentor(aug_config, generator=aug_rng)
    test_aug_views = aug_config.num_views_test if aug_config.enabled else 0
    spec_augment_active = False

    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()
    spec_augment_active = (
        aug_config.enabled
        and test_aug_views > 0
        and (aug_config.freq_mask_max > 0 or aug_config.time_mask_max > 0)
        and not backbone.expect_waveform
    )

    names, scores, labels, domains = [], [], [], []
    k = int(cfg.get("detector", {}).get("k", cfg.get("model", {}).get("k", 3)))
    distance = str(cfg.get("detector", {}).get("distance", "cosine")).lower()

    for path in tqdm(wavs, desc="Scoring dev clips", unit="clip"):
        wav, sr = load_audio_mono(path)

        view_scores = []
        for wav_i, _, is_augmented in augmentor.waveform_views(wav, int(sr), test_aug_views):
            mel_aug = augmentor.augment_spectrogram if (is_augmented and spec_augment_active) else None
            feat = backbone(wav_i.to(device), sr, spec_augment=mel_aug)
            if normalize:
                feat = F.normalize(feat, dim=-1)
            score, _ = score_clustered_knn(
                query=feat,
                mem_bank=mem_bank,
                cluster_labels=cluster_labels,
                centroids=centroids,
                k=k,
                distance=distance,
                min_cluster_size=memory_cfg["min_cluster_size"],
            )
            view_scores.append(score)

        p = Path(path)
        names.append(p.name)
        scores.append(aggregate_augmented_scores(view_scores, aug_config.test_score_agg))
        if labeled_stage:
            labels.append(infer_dev_label(path))
            domain = infer_domain(path)
            if domain is None:
                raise RuntimeError(f"Could not infer source/target domain from labeled dev file: {path}")
            domains.append(domain)
        else:
            labels.append(-1)
            domains.append("unknown")

    payload = {
        "names": np.asarray(names),
        "scores": np.asarray(scores, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "domains": np.asarray(domains),
        "data_stage": np.asarray(str(stage)),
        "labeled_stage": np.asarray(bool(labeled_stage)),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        **payload,
        bank_path=np.asarray(str(bank_path)),
        bank_mtime=np.asarray(float(bank_path.stat().st_mtime)),
        data_stage=np.asarray(str(stage)),
        labeled_stage=np.asarray(bool(labeled_stage)),
    )
    print(f"Saved anomaly-score cache: {cache_path}")
    return payload


def evaluate_threshold(scores: np.ndarray, labels: np.ndarray, domains: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores > float(threshold)).astype(np.int64)
    result: dict[str, float] = {}
    f1_values = []

    for domain in ("source", "target"):
        mask = domains == domain
        if not np.any(mask):
            raise RuntimeError(f"No {domain} clips available for threshold sweep")
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels[mask],
            preds[mask],
            average="binary",
            zero_division=0,
        )
        result[f"precision_{domain}"] = float(precision)
        result[f"recall_{domain}"] = float(recall)
        result[f"f1_{domain}"] = float(f1)
        f1_values.append(float(f1))

    result["arithmetic_mean_f1"] = float(np.mean(f1_values))
    result["harmonic_mean_f1"] = harmonic_mean(f1_values)
    return result


def score_only_threshold(scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores > float(threshold)).astype(np.int64)
    return {
        "precision_source": float("nan"),
        "precision_target": float("nan"),
        "recall_source": float("nan"),
        "recall_target": float("nan"),
        "f1_source": float("nan"),
        "f1_target": float("nan"),
        "arithmetic_mean_f1": float("nan"),
        "harmonic_mean_f1": float("nan"),
        "num_predicted_anomaly": int(preds.sum()),
        "predicted_anomaly_rate": float(preds.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--machine", default="AutoTrash")
    ap.add_argument("--recompute-scores", action="store_true")
    ap.add_argument(
        "--allow-unlabeled-eval",
        action="store_true",
        help="Allow score-only sweeps on anonymized eval_data. This cannot optimize thresholds.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    memory_cfg, pipeline_mode = require_best_architecture(cfg)
    threshold_cfg = cfg.get("threshold", {})
    if str(threshold_cfg.get("method", "percentile")).lower() != "percentile":
        raise RuntimeError("Threshold sweep only supports threshold.method=percentile")
    if not bool(threshold_cfg.get("sweep_enabled", False)):
        print("threshold.sweep_enabled=false; running sweep because threshold_sweep.py was invoked explicitly.")
    sweep_values = threshold_cfg.get("sweep_values", [80, 85, 90, 92.5, 95])
    sweep_values = [float(v) for v in sweep_values]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distance = str(cfg.get("detector", {}).get("distance", "cosine")).lower()
    k = int(cfg.get("detector", {}).get("k", cfg.get("model", {}).get("k", 3)))
    normalize = bool(cfg.get("model", {}).get("normalize", False))
    if distance == "cosine" and not normalize:
        print("WARNING: distance=cosine but model.normalize=false; forcing L2 normalization.")
        normalize = True

    bank_dir = Path(cfg["logging"]["bank_out"])
    bank_path = get_bank_path(
        bank_dir,
        args.machine,
        pipeline_mode,
        memory_cfg["mode"],
        memory_cfg["num_clusters"],
    )
    removed = cleanup_conflicting_banks(
        bank_dir=bank_dir,
        machine=args.machine,
        active_bank=bank_path,
        enabled=bool(memory_cfg.get("cleanup_stale_banks", True)),
    )

    mem_bank, cluster_labels, centroids, meta = load_clustered_bank(
        cfg=cfg,
        cfg_path=args.config,
        machine=args.machine,
        bank_path=bank_path,
        memory_cfg=memory_cfg,
        device=device,
        normalize=normalize,
    )
    print(f"Clustered k6 bank preserved/loaded: {bank_path}")
    print(f"Embeddings: {mem_bank.size(0)}, feature_dim: {mem_bank.size(1)}")

    print("Computing memory-bank distance distribution once.")
    _, mem_dists = compute_clustered_threshold_and_mem_dists(
        mem_bank=mem_bank,
        cluster_labels=cluster_labels,
        centroids=centroids,
        k=k,
        distance=distance,
        pct=float(threshold_cfg.get("percentile", 90)),
        min_cluster_size=memory_cfg["min_cluster_size"],
    )

    out_dir = Path("results") / "threshold_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_cache_path = out_dir / f"{args.machine}_clustered_k6_scores.npz"
    sweep_csv = out_dir / f"{args.machine}_clustered_k6_threshold_sweep.csv"

    payload = compute_or_load_scores(
        cfg=cfg,
        machine=args.machine,
        bank_path=bank_path,
        cache_path=score_cache_path,
        mem_bank=mem_bank,
        cluster_labels=cluster_labels,
        centroids=centroids,
        memory_cfg=memory_cfg,
        device=device,
        normalize=normalize,
        recompute=args.recompute_scores,
        allow_unlabeled_eval=args.allow_unlabeled_eval,
    )
    scores = payload["scores"].astype(np.float32)
    labels = payload["labels"].astype(np.int64)
    domains = payload["domains"]
    labeled_stage = bool(payload.get("labeled_stage", np.asarray(False)).item())

    source_mask = domains == "source"
    target_mask = domains == "target"
    labeled = labeled_stage and np.all(labels >= 0) and np.any(source_mask) and np.any(target_mask)
    if labeled:
        auc_source = float(roc_auc_score(labels[source_mask], scores[source_mask]))
        auc_target = float(roc_auc_score(labels[target_mask], scores[target_mask]))
        pauc_target = float(roc_auc_score(labels[target_mask], scores[target_mask], max_fpr=0.1))
    else:
        auc_source = float("nan")
        auc_target = float("nan")
        pauc_target = float("nan")
        print(
            "WARNING: Unlabeled eval data cannot be used for threshold optimization. "
            "The sweep will save threshold values and predicted-anomaly rates, but "
            "precision/recall/F1/AUC/pAUC cannot be computed without labels."
        )

    rows = []
    for pct in sweep_values:
        threshold_value = float(np.percentile(mem_dists, pct))
        metrics = evaluate_threshold(scores, labels, domains, threshold_value) if labeled else score_only_threshold(
            scores,
            threshold_value,
        )
        row = {
            "threshold_percentile": pct,
            "threshold_value": threshold_value,
            **metrics,
            "auc_source": auc_source,
            "auc_target": auc_target,
            "pauc_target": pauc_target,
            "official_score": "",
        }
        rows.append(row)

    fieldnames = [
        "threshold_percentile",
        "threshold_value",
        "precision_source",
        "precision_target",
        "recall_source",
        "recall_target",
        "f1_source",
        "f1_target",
        "arithmetic_mean_f1",
        "harmonic_mean_f1",
        "auc_source",
        "auc_target",
        "pauc_target",
        "num_predicted_anomaly",
        "predicted_anomaly_rate",
        "official_score",
    ]
    with sweep_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nThreshold sweep:")
    print("pct, threshold, f1_source, f1_target, mean_f1, harmonic_f1")
    for row in rows:
        print(
            f"{row['threshold_percentile']:>5g}, {row['threshold_value']:.6f}, "
            f"{row['f1_source']:.4f}, {row['f1_target']:.4f}, "
            f"{row['arithmetic_mean_f1']:.4f}, {row['harmonic_mean_f1']:.4f}"
        )

    if labeled:
        best_hmean = max(rows, key=lambda r: r["harmonic_mean_f1"])
        best_target = max(rows, key=lambda r: r["f1_target"])
        print(f"\nBest threshold by harmonic mean F1: {best_hmean['threshold_percentile']:g}")
        print(f"Best threshold by target F1: {best_target['threshold_percentile']:g}")
        print(
            "AUC/pAUC are fixed across threshold rows: "
            f"source AUC={auc_source:.4f}, target AUC={auc_target:.4f}, pAUC={pauc_target:.4f}"
        )
    else:
        print("\nBest F1 threshold is unavailable because labels are not available for these files.")
    print(f"Saved threshold sweep CSV: {sweep_csv}")
    if removed:
        print("Cleaned stale banks:")
        for path in removed:
            print(f"  {path}")


if __name__ == "__main__":
    main()
