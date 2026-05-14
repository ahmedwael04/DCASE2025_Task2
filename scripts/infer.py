#!/usr/bin/env python
"""
GPU-accelerated inference for DCASE-2025 Task 2 (eval set only).

Produces:
  • anomaly_score_<machine>_section_<XX>_test.csv
  • decision_result_<machine>_section_<XX>_test.csv
under your csv_out_dir (configs/default.yaml).
"""

import argparse, sys, glob, subprocess
from pathlib import Path

# Allow running as: python scripts/infer.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.utils.file_utils      import load_config
from src.models.beats_backbone import BEATsBackbone
from src.utils.audio_utils import load_audio_mono
from src.augmentations import (
    AugmentationConfig,
    DomainGeneralizationAugmentor,
    aggregate_augmented_scores,
    describe_augmentation,
    set_augmentation_seed,
)
from src.pipeline import (
    bank_matches_memory_config,
    bank_matches_pipeline,
    describe_pipeline,
    get_bank_path,
    get_legacy_bank_paths,
    get_memory_bank_config,
    get_pipeline_mode,
    get_stale_bank_paths,
    get_top_windows,
    get_window_params,
    validate_pipeline_memory_compatibility,
)


def infer_domain_from_name(name: str) -> str | None:
    """Infer domain (source/target) from a filename/path string."""
    s = str(name).lower()
    if "_source_" in s:
        return "source"
    if "_target_" in s:
        return "target"
    return None


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


def compute_threshold_and_mem_dists(
    mem_bank: torch.Tensor,
    k: int,
    distance: str,
    pct: float,
    chunk: int = 1024,
) -> tuple[float, np.ndarray]:
    """Compute kNN mean-distance threshold and per-entry within-bank distances.

    Returns:
        threshold: percentile threshold over per-entry kNN mean distances.
        mem_dists: shape (N,) array of per-entry kNN mean distances.
    """
    N = mem_bank.size(0)
    mem_dists = []

    for i in tqdm(
        range(0, N, chunk),
        desc="  threshold chunks",
        unit="chunk",
        file=sys.stdout,
        dynamic_ncols=True,
    ):
        sub = mem_bank[i : i + chunk]
        if distance == "cosine":
            d = 1.0 - (sub @ mem_bank.T)
        else:
            d = torch.cdist(sub, mem_bank)

        for j in range(d.size(0)):
            global_idx = i + j
            if global_idx < N:
                d[j, global_idx] = float("inf")

        topk = torch.topk(d, k=k, dim=1, largest=False).values
        mem_dists.append(topk.mean(dim=1).cpu().numpy())

    mem_dists = np.concatenate(mem_dists, axis=0)
    threshold = float(np.percentile(mem_dists, pct))
    return threshold, mem_dists


def score_clustered_knn(
    query: torch.Tensor,
    mem_bank: torch.Tensor,
    cluster_labels: torch.Tensor,
    centroids: torch.Tensor,
    k: int,
    distance: str,
    min_cluster_size: int,
    exclude_index: int | None = None,
) -> tuple[float, dict]:
    """Score one clip embedding by nearest centroid, then kNN inside that cluster."""
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

    bank = mem_bank[selected]
    if distance == "cosine":
        d = 1.0 - (query @ bank.T)
    else:
        d = torch.cdist(query, bank)

    k_eff = min(int(k), int(d.size(1)))
    topk = torch.topk(d, k=k_eff, dim=1, largest=False).values
    score = float(topk.mean().item())
    debug = {
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
        "centroid_distance": centroid_distance,
        "fallback_global": fallback,
        "k_used": k_eff,
        "score": score,
    }
    return score, debug


def compute_clustered_threshold_and_mem_dists(
    mem_bank: torch.Tensor,
    cluster_labels: torch.Tensor,
    centroids: torch.Tensor,
    k: int,
    distance: str,
    pct: float,
    min_cluster_size: int,
) -> tuple[float, np.ndarray]:
    mem_dists = []
    for i in tqdm(
        range(mem_bank.size(0)),
        desc="  clustered threshold",
        unit="entry",
        file=sys.stdout,
        dynamic_ncols=True,
    ):
        score, _ = score_clustered_knn(
            query=mem_bank[i : i + 1],
            mem_bank=mem_bank,
            cluster_labels=cluster_labels,
            centroids=centroids,
            k=k,
            distance=distance,
            min_cluster_size=min_cluster_size,
            exclude_index=i,
        )
        mem_dists.append(score)

    mem_dists_np = np.asarray(mem_dists, dtype=np.float32)
    threshold = float(np.percentile(mem_dists_np, pct))
    return threshold, mem_dists_np


def _format_number(value) -> str:
    value_f = float(value)
    if value_f.is_integer():
        return str(int(value_f))
    return str(value_f)


def _is_autotrash(machine: str | None) -> bool:
    return machine is not None and str(machine).lower() == "autotrash"


def print_final_active_pipeline(
    cfg: dict,
    pipeline_mode: str,
    memory_cfg: dict,
    k: int,
    distance: str,
    threshold_method: str,
    threshold_percentile: float,
) -> None:
    threshold_cfg = cfg.get("threshold", {})
    augmentation_enabled = bool(cfg.get("augmentation", {}).get("enabled", False))
    temporal_enabled = bool(cfg.get("detector", {}).get("temporal", {}).get("enabled", False))
    sweep_enabled = bool(threshold_cfg.get("sweep_enabled", False))

    print("FINAL ACTIVE PIPELINE")
    print(f"- pipeline mode: {pipeline_mode}")
    print(f"- memory bank mode: {memory_cfg['mode']}")
    print(f"- clusters: {memory_cfg['num_clusters']}")
    print(f"- distance metric: {distance}")
    print(f"- kNN K: {k}")
    print(f"- threshold method: {threshold_method}")
    print(f"- threshold percentile: {_format_number(threshold_percentile)}")
    print("- threshold tuning: disabled (unlabeled eval data)")

    non_final = [
        pipeline_mode != "clip",
        memory_cfg["mode"] != "clustered",
        int(memory_cfg["num_clusters"]) != 6,
        str(distance).lower() != "cosine",
        int(k) != 3,
        str(threshold_method).lower() != "percentile",
        float(threshold_percentile) != 90.0,
        sweep_enabled,
        augmentation_enabled,
        temporal_enabled,
    ]
    if any(non_final):
        print("WARNING: Running non-final experimental configuration.")


def main():
    # 1) Argparse & config
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--stage",
        default="eval_data",
        choices=["dev_data", "eval_data"],
        help="Which dataset stage to score.",
    )
    p.add_argument(
        "--machine",
        default=None,
        help="If set, run inference only for this machine under eval_data/raw/<machine>/test.",
    )
    p.add_argument(
        "--score-level",
        default=None,
        choices=["clip", "window"],
        help="Deprecated alias for --pipeline-mode.",
    )
    p.add_argument(
        "--pipeline-mode",
        default=None,
        choices=["clip", "window"],
        help="Override pipeline.mode from config.",
    )
    p.add_argument(
        "--win-frames",
        type=int,
        default=None,
        help="Window size in frames when pipeline mode is window.",
    )
    p.add_argument(
        "--hop-frames",
        type=int,
        default=None,
        help="Hop size in frames when pipeline mode is window.",
    )
    p.add_argument(
        "--top-windows",
        type=int,
        default=None,
        help="Aggregate by mean of top-N window scores when pipeline mode is window.",
    )
    args   = p.parse_args()
    cfg    = load_config(args.config)
    pipeline_override = args.pipeline_mode or args.score_level
    pipeline_mode = get_pipeline_mode(cfg, pipeline_override)
    win_frames, hop_frames = get_window_params(cfg, args.win_frames, args.hop_frames)
    top_windows = get_top_windows(cfg, args.top_windows)
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

    # 2) Paths
    bank_dir = Path(cfg["logging"]["bank_out"])
    csv_dir  = Path(cfg["logging"]["csv_out_dir"])
    csv_dir.mkdir(parents=True, exist_ok=True)

    # 3) Load backbone
    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()
    augmentor = DomainGeneralizationAugmentor(aug_config, generator=aug_rng)
    test_aug_views = aug_config.num_views_test if aug_config.enabled else 0
    spec_augment_active = (
        aug_config.enabled
        and test_aug_views > 0
        and (aug_config.freq_mask_max > 0 or aug_config.time_mask_max > 0)
        and not backbone.expect_waveform
    )
    for line in describe_augmentation(aug_config, phase="test-scoring"):
        print(line)
    if aug_config.enabled and test_aug_views > 0 and spec_augment_active:
        print("  log-mel masking: active for augmented spectrogram views")
    elif aug_config.enabled and test_aug_views > 0:
        print("  log-mel masking: inactive for waveform-input backbone")

    # 4) Detector settings
    K = cfg.get("detector", {}).get("k", cfg.get("model", {}).get("k", 3))
    distance = str(cfg.get("detector", {}).get("distance", "cosine")).lower()
    normalize = bool(cfg.get("model", {}).get("normalize", False))
    threshold_cfg = cfg.get("threshold", {})
    threshold_method = str(threshold_cfg.get("method", "percentile")).lower()
    pct = float(threshold_cfg.get("percentile", 90))
    if (
        bool(threshold_cfg.get("sweep_enabled", False))
        and _is_autotrash(args.machine)
        and args.stage == "eval_data"
    ):
        print("Threshold sweep is disabled for AutoTrash because eval data is unlabeled.")
    print_final_active_pipeline(
        cfg=cfg,
        pipeline_mode=pipeline_mode,
        memory_cfg=memory_cfg,
        k=K,
        distance=distance,
        threshold_method=threshold_method,
        threshold_percentile=pct,
    )
    if distance == "cosine" and not normalize:
        print("⚠️  distance=cosine but model.normalize=false; forcing L2 normalization.")
        normalize = True
    print(f"Detector: k={K}, distance={distance}, normalize={normalize}")
    print(f"Test-time augmentation views: {test_aug_views}, aggregation={aug_config.test_score_agg}")
    for line in describe_pipeline(pipeline_mode, win_frames, hop_frames, top_windows):
        print(line)
    print(f"Memory bank mode: {memory_mode}")
    if memory_mode == "clustered":
        print(f"  scoring method: {memory_cfg['cluster_score_mode']}")
        print(f"  requested clusters: {memory_cfg['num_clusters']}")
        print(f"  min_cluster_size: {memory_cfg['min_cluster_size']}")
    else:
        print("  scoring method: global kNN")
    chunk = 1024  # adjust to fit your GPU
    verbose_debug = str(cfg.get("logging", {}).get("level", "info")).lower() == "debug"

    # 5) Per-machine cache: {machine: {mem_banks, thresholds, dom_stats}}
    cache = {}
    auto_rebuild_bank = bool(cfg.get("pipeline", {}).get("auto_rebuild_bank", True))
    cleanup_stale_banks = bool(cfg.get("pipeline", {}).get("cleanup_stale_banks", True)) or bool(
        memory_cfg["cleanup_stale_banks"]
    )

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

    def rebuild_bank(machine: str, reason: str) -> None:
        if not auto_rebuild_bank:
            raise RuntimeError(
                f"{reason}\n"
                "Set pipeline.auto_rebuild_bank=true or rebuild explicitly with:\n"
                f"  python scripts/train_knn.py --config {args.config} --machine {machine} --pipeline-mode {pipeline_mode}"
            )

        print(f"WARNING: {reason}")
        print(f"Rebuilding {pipeline_mode} memory bank for {machine} before scoring.")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_knn.py"),
            "--config",
            args.config,
            "--machine",
            machine,
            "--stage",
            "both",
            "--pipeline-mode",
            pipeline_mode,
            "--win-frames",
            str(win_frames),
            "--hop-frames",
            str(hop_frames),
        ]
        subprocess.run(cmd, check=True)

    def ensure_bank(machine: str) -> Path:
        bank_path = get_bank_path(
            bank_dir,
            machine,
            pipeline_mode,
            memory_mode,
            memory_cfg["num_clusters"],
        )
        legacy_paths = [p for p in get_legacy_bank_paths(bank_dir, machine, pipeline_mode) if p != bank_path]

        stale_existing = get_stale_bank_paths(
            bank_dir,
            machine,
            pipeline_mode,
            memory_mode,
            memory_cfg["num_clusters"],
        )
        stale_existing.extend([p for p in legacy_paths if p.exists()])
        stale_existing = sorted(set(stale_existing))
        for stale_path in stale_existing:
            print(
                f"WARNING: Active pipeline is {pipeline_mode}/{memory_mode}, ignoring mismatched/stale bank: "
                f"{stale_path.name}"
            )
            if cleanup_stale_banks:
                print(f"Removing stale/mismatched memory bank: {stale_path}")
                stale_path.unlink()

        if not bank_path.exists():
            hints = []
            for stale_path in stale_existing:
                hints.append(f"found stale bank: {stale_path.name}")
            hint_text = f" ({'; '.join(hints)})" if hints else ""
            rebuild_bank(
                machine,
                f"Required {pipeline_mode} bank is missing: {bank_path.name}{hint_text}",
            )

        ckpt = load_bank_checkpoint(bank_path)
        meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
        matches, reason = bank_matches_pipeline(meta, pipeline_mode, win_frames, hop_frames)
        if not matches:
            rebuild_bank(machine, f"Memory bank {bank_path.name} is stale/mismatched: {reason}")
        matches, reason = bank_matches_memory_config(
            meta,
            memory_mode,
            memory_cfg["num_clusters"],
        )
        if not matches:
            rebuild_bank(machine, f"Memory bank {bank_path.name} is stale/mismatched: {reason}")

        if not bank_path.exists():
            raise RuntimeError(f"Expected rebuilt bank was not created: {bank_path}")
        return bank_path

    def prepare_machine(machine: str):
        bank_path = ensure_bank(machine)
        cache_key = f"{machine}__{pipeline_mode}__{memory_mode}__k{memory_cfg['num_clusters']}"
        if cache_key in cache:
            return cache[cache_key]

        ckpt = load_bank_checkpoint(bank_path)
        raw_mem = ckpt["memory"]
        raw_paths = ckpt.get("paths", [""] * len(raw_mem))
        bank_aug_meta = ckpt.get("meta", {}).get("augmentation", {}) if isinstance(ckpt, dict) else {}
        meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

        # Allow entries as (D,), (1,D), etc.
        def to_vec(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 1:
                return x
            if x.ndim == 2 and x.size(0) == 1:
                return x.squeeze(0)
            if x.ndim == 2:
                return x.mean(dim=0)
            raise ValueError(f"Unsupported bank entry shape: {tuple(x.shape)}")

        # Split the bank by domain (source/target) using saved paths.
        domains = [infer_domain_from_name(p) for p in raw_paths]
        idx_all = list(range(len(raw_mem)))
        idx_source = [i for i, d in enumerate(domains) if d == "source"]
        idx_target = [i for i, d in enumerate(domains) if d == "target"]

        def _build_bank(idxs: list[int]) -> torch.Tensor:
            if not idxs:
                idxs = idx_all
            bank = torch.stack([to_vec(raw_mem[i]) for i in idxs], dim=0).to(device)
            if normalize:
                bank = F.normalize(bank, dim=-1)
            return bank

        mem_banks = {
            "__all__": _build_bank(idx_all),
            "source": _build_bank(idx_source),
            "target": _build_bank(idx_target),
            "counts": {
                "__all__": len(idx_all),
                "source": len(idx_source),
                "target": len(idx_target),
            },
        }

        feature_dim = int(mem_banks["__all__"].size(1))
        matches, reason = bank_matches_memory_config(
            meta,
            memory_mode,
            memory_cfg["num_clusters"],
            feature_dim,
        )
        if not matches:
            rebuild_bank(machine, f"Memory bank {bank_path.name} is stale/mismatched: {reason}")
            return prepare_machine(machine)

        cluster_labels = None
        centroids = None
        if memory_mode == "clustered":
            if "cluster_labels" not in ckpt or "centroids" not in ckpt:
                rebuild_bank(machine, f"Clustered bank {bank_path.name} is missing cluster_labels/centroids")
                return prepare_machine(machine)
            cluster_labels = ckpt["cluster_labels"].to(device=device, dtype=torch.long)
            centroids = ckpt["centroids"].to(device=device, dtype=mem_banks["__all__"].dtype)
            if normalize:
                centroids = F.normalize(centroids, dim=-1)

        print(
            f"▶ Preparing {machine} bank ({bank_path.name}): "
            f"source={mem_banks['counts']['source']}, target={mem_banks['counts']['target']}, all={mem_banks['counts']['__all__']}"
        )

        if bank_aug_meta:
            print(
                "  bank augmentation: "
                f"enabled={bank_aug_meta.get('enabled')}, "
                f"views={bank_aug_meta.get('num_views_train', bank_aug_meta.get('copies_per_sample', 0))}, "
                f"mode={bank_aug_meta.get('memory_mode', 'expand')}, "
                f"before={bank_aug_meta.get('clean_entries', 'unknown')}, "
                f"after={bank_aug_meta.get('total_entries', len(raw_mem))}"
            )
        print(f"  loaded bank path: {bank_path}")
        if memory_mode == "clustered":
            print(
                f"  clustered bank: clusters={centroids.size(0)}, "
                f"score_mode={memory_cfg['cluster_score_mode']}, min_cluster_size={memory_cfg['min_cluster_size']}"
            )

        # Unified threshold/stats: score uses both domains jointly.
        thresholds: dict[str, float] = {}
        dom_stats: dict[str, tuple[float, float]] = {}

        print(f"▶ Computing unified threshold for {machine} (__all__) …")
        if memory_mode == "clustered":
            thr, mem_dists = compute_clustered_threshold_and_mem_dists(
                mem_bank=mem_banks["__all__"],
                cluster_labels=cluster_labels,
                centroids=centroids,
                k=K,
                distance=distance,
                pct=pct,
                min_cluster_size=memory_cfg["min_cluster_size"],
            )
        else:
            thr, mem_dists = compute_threshold_and_mem_dists(
                mem_bank=mem_banks["__all__"],
                k=K,
                distance=distance,
                pct=pct,
                chunk=chunk,
            )
        thresholds["__all__"] = float(thr)
        mu = float(np.mean(mem_dists))
        sig = float(np.std(mem_dists))
        if not np.isfinite(sig) or sig < 1e-12:
            sig = 1.0
        dom_stats["__all__"] = (mu, sig)

        # If a domain split is missing, fallback will point to __all__ at use time.
        cache[cache_key] = {
            "mem_banks": mem_banks,
            "thresholds": thresholds,
            "dom_stats": dom_stats,
            "bank_path": str(bank_path),
            "cluster_labels": cluster_labels,
            "centroids": centroids,
        }
        return cache[cache_key]

    # 6) Gather test WAVs
    root    = cfg["data"]["root"]
    if args.machine:
        pattern = f"{root}/{args.stage}/raw/{args.machine}/test/**/*.wav"
    else:
        pattern = f"{root}/{args.stage}/raw/*/test/**/*.wav"
    wavs    = sorted(glob.glob(pattern, recursive=True))
    print(f"▶ Found {len(wavs)} test clips under: {pattern}")
    if not wavs:
        print("⚠️  No test files found for this stage/machine.")
        sys.exit(1)

    def machine_from_path(path: Path) -> str:
        parts = path.parts
        if "raw" in parts:
            raw_idx = parts.index("raw")
            if raw_idx + 1 < len(parts):
                return parts[raw_idx + 1]
        if "test" in parts:
            test_idx = parts.index("test")
            if test_idx > 0:
                return parts[test_idx - 1]
        return path.parent.parent.name

    # 7) Inference loop with live GPU-knn scoring
    writers = {}
    clustered_fallback_warned = set()
    try:
        for path in tqdm(
            wavs,
            desc="Scoring eval clips",
            unit="clip",
            file=sys.stdout,
            leave=True,
            dynamic_ncols=True,
        ):
            wav, sr = load_audio_mono(path)
            wav = wav.to(device)
            sr = int(sr)

            p = Path(path)
            machine = machine_from_path(p)
            pack = prepare_machine(machine)

            # Unified scoring: always use the combined bank and single calibration.
            mem_bank = pack["mem_banks"]["__all__"]
            threshold = pack["thresholds"]["__all__"]
            mu, sig = pack["dom_stats"]["__all__"]
            cluster_labels = pack.get("cluster_labels")
            centroids = pack.get("centroids")

            if pipeline_mode == "clip":
                # backbone→embedding (1, D)
                feat = backbone(wav, sr)
                if normalize:
                    feat = F.normalize(feat, dim=-1)

                # GPU k‐NN scoring
                if memory_mode == "clustered":
                    score, debug = score_clustered_knn(
                        query=feat,
                        mem_bank=mem_bank,
                        cluster_labels=cluster_labels,
                        centroids=centroids,
                        k=K,
                        distance=distance,
                        min_cluster_size=memory_cfg["min_cluster_size"],
                    )
                    if verbose_debug:
                        print(
                            f"DEBUG {p.name}: cluster={debug['cluster_id']} size={debug['cluster_size']} "
                            f"centroid_dist={debug['centroid_distance']:.6f} fallback={debug['fallback_global']} "
                            f"score={debug['score']:.6f}"
                        )
                    elif debug["fallback_global"] and machine not in clustered_fallback_warned:
                        print(
                            f"Clustered kNN fallback to global bank for {machine}: "
                            f"selected cluster {debug['cluster_id']} size={debug['cluster_size']}."
                        )
                        clustered_fallback_warned.add(machine)
                else:
                    if distance == "cosine":
                        d = 1.0 - (feat @ mem_bank.T)
                    else:
                        d = torch.cdist(feat, mem_bank)  # (1, N)
                    topk = torch.topk(d, k=K, dim=1, largest=False).values  # (1, K)
                    score = float(topk.mean().item())
            else:
                # window-level scoring + top-N aggregation
                temporal = backbone(wav, sr, return_temporal=True)  # (1,T,D)
                win_emb = make_window_embeddings(temporal, win_frames, hop_frames)  # (W,D)
                if normalize:
                    win_emb = F.normalize(win_emb, dim=-1)

                if distance == "cosine":
                    d = 1.0 - (win_emb @ mem_bank.T)  # (W,N)
                else:
                    d = torch.cdist(win_emb, mem_bank)  # (W,N)

                topk = torch.topk(d, k=K, dim=1, largest=False).values  # (W,K)
                win_scores = topk.mean(dim=1)  # (W,)

                n_top = top_windows
                n_top = min(n_top, win_scores.numel())
                top_vals = torch.topk(win_scores, k=n_top, largest=True).values
                score = float(top_vals.mean().item())

            if aug_config.enabled and test_aug_views > 0:
                view_scores = [float(score)]
                for wav_i, _, is_augmented in augmentor.waveform_views(wav, sr, test_aug_views):
                    if not is_augmented:
                        continue
                    wav_i = wav_i.to(device)
                    mel_aug = augmentor.augment_spectrogram if spec_augment_active else None

                    if pipeline_mode == "clip":
                        feat = backbone(wav_i, sr, spec_augment=mel_aug)
                        if normalize:
                            feat = F.normalize(feat, dim=-1)

                        if memory_mode == "clustered":
                            view_score, debug = score_clustered_knn(
                                query=feat,
                                mem_bank=mem_bank,
                                cluster_labels=cluster_labels,
                                centroids=centroids,
                                k=K,
                                distance=distance,
                                min_cluster_size=memory_cfg["min_cluster_size"],
                            )
                            if verbose_debug:
                                print(
                                    f"DEBUG {p.name}: cluster={debug['cluster_id']} size={debug['cluster_size']} "
                                    f"centroid_dist={debug['centroid_distance']:.6f} fallback={debug['fallback_global']} "
                                    f"score={debug['score']:.6f}"
                                )
                            elif debug["fallback_global"] and machine not in clustered_fallback_warned:
                                print(
                                    f"Clustered kNN fallback to global bank for {machine}: "
                                    f"selected cluster {debug['cluster_id']} size={debug['cluster_size']}."
                                )
                                clustered_fallback_warned.add(machine)
                            view_scores.append(float(view_score))
                        else:
                            if distance == "cosine":
                                d = 1.0 - (feat @ mem_bank.T)
                            else:
                                d = torch.cdist(feat, mem_bank)
                            topk = torch.topk(d, k=K, dim=1, largest=False).values
                            view_scores.append(float(topk.mean().item()))
                    else:
                        temporal = backbone(
                            wav_i,
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

                        topk = torch.topk(d, k=K, dim=1, largest=False).values
                        win_scores = topk.mean(dim=1)
                        n_top = top_windows
                        n_top = min(n_top, win_scores.numel())
                        top_vals = torch.topk(win_scores, k=n_top, largest=True).values
                        view_scores.append(float(top_vals.mean().item()))

                score = aggregate_augmented_scores(view_scores, aug_config.test_score_agg)

            # Score normalization (calibration)
            score_n = (float(score) - float(mu)) / float(sig)
            thr_n = (float(threshold) - float(mu)) / float(sig)
            decision = 1 if score_n > thr_n else 0

            section = p.stem.split("_")[1]  # "section_XX_####" → ["section","XX","####"]
            tag = f"{machine}_section_{section}"

            # open CSVs on first use
            if tag not in writers:
                asc = csv_dir / f"anomaly_score_{tag}_test.csv"
                dec = csv_dir / f"decision_result_{tag}_test.csv"
                writers[tag] = (asc.open("w", buffering=1), dec.open("w", buffering=1))

            asc_fp, dec_fp = writers[tag]
            asc_fp.write(f"{p.name},{score_n:.6f}\n")
            dec_fp.write(f"{p.name},{decision}\n")
            asc_fp.flush()
            dec_fp.flush()
    finally:
        # 8) Close filehandles even on interrupt/error
        for asc_fp, dec_fp in writers.values():
            try:
                asc_fp.close()
            finally:
                dec_fp.close()

    print(f"✅ All CSVs written to {csv_dir}/")


if __name__ == "__main__":
    main()
