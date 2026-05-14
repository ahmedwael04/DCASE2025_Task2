"""Pipeline-mode helpers for clip/window ASD experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any


VALID_PIPELINE_MODES = {"clip", "window"}
VALID_MEMORY_BANK_MODES = {"plain", "clustered"}


def get_pipeline_mode(cfg: dict[str, Any], override: str | None = None) -> str:
    """Resolve the active embedding/scoring mode."""
    mode = override
    if mode is None:
        mode = cfg.get("PIPELINE_MODE")
    if mode is None:
        mode = cfg.get("pipeline", {}).get("mode", "clip")
    mode = str(mode).lower()
    if mode not in VALID_PIPELINE_MODES:
        raise ValueError("PIPELINE_MODE/pipeline.mode must be 'clip' or 'window'")
    return mode


def get_window_params(cfg: dict[str, Any], win_override: int | None = None, hop_override: int | None = None) -> tuple[int, int]:
    """Resolve window extraction parameters."""
    pipe = cfg.get("pipeline", {})
    win_frames = int(win_override if win_override is not None else pipe.get("win_frames", 50))
    hop_frames = int(hop_override if hop_override is not None else pipe.get("hop_frames", 50))
    if win_frames <= 0 or hop_frames <= 0:
        raise ValueError("pipeline.win_frames and pipeline.hop_frames must be > 0")
    return win_frames, hop_frames


def get_top_windows(cfg: dict[str, Any], override: int | None = None) -> int:
    pipe = cfg.get("pipeline", {})
    value = int(override if override is not None else pipe.get("top_windows", 5))
    return max(1, value)


def get_memory_bank_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve memory-bank scoring mode and clustering options."""
    bank_cfg = cfg.get("memory_bank", {})
    if not isinstance(bank_cfg, dict):
        bank_cfg = {}

    mode = str(bank_cfg.get("mode", "plain")).lower()
    if mode not in VALID_MEMORY_BANK_MODES:
        raise ValueError("memory_bank.mode must be 'plain' or 'clustered'")

    method = str(bank_cfg.get("cluster_method", "kmeans")).lower()
    if method != "kmeans":
        raise ValueError("Only memory_bank.cluster_method='kmeans' is currently supported")

    score_mode = str(bank_cfg.get("cluster_score_mode", "nearest_cluster_knn")).lower()
    if score_mode != "nearest_cluster_knn":
        raise ValueError("Only memory_bank.cluster_score_mode='nearest_cluster_knn' is currently supported")

    return {
        "mode": mode,
        "num_clusters": max(1, int(bank_cfg.get("num_clusters", 6))),
        "cluster_method": method,
        "cluster_score_mode": score_mode,
        "min_cluster_size": max(1, int(bank_cfg.get("min_cluster_size", 3))),
        "cleanup_stale_banks": bool(bank_cfg.get("cleanup_stale_banks", True)),
    }


def validate_pipeline_memory_compatibility(pipeline_mode: str, memory_mode: str) -> None:
    if pipeline_mode == "window" and memory_mode == "clustered":
        raise ValueError(
            "Window pipeline is experimental and not part of the finalized AutoTrash configuration."
        )


def get_bank_path(
    bank_dir: str | Path,
    machine: str | None,
    mode: str,
    memory_mode: str = "plain",
    num_clusters: int | None = None,
) -> Path:
    """Return the canonical bank path for a machine/mode pair."""
    mode = get_pipeline_mode({}, mode)
    memory_mode = str(memory_mode).lower()
    if memory_mode not in VALID_MEMORY_BANK_MODES:
        raise ValueError("memory_mode must be 'plain' or 'clustered'")
    validate_pipeline_memory_compatibility(mode, memory_mode)

    stem = "memory_bank" if machine is None else f"memory_bank_{machine}"
    if memory_mode == "clustered":
        k = max(1, int(num_clusters or 1))
        return Path(bank_dir) / f"{stem}_{mode}_clustered_k{k}.pt"
    return Path(bank_dir) / f"{stem}_{mode}.pt"


def get_legacy_bank_paths(bank_dir: str | Path, machine: str | None, mode: str | None = None) -> list[Path]:
    """Known pre-pipeline-mode bank names to warn about or clean up."""
    root = Path(bank_dir)
    paths: list[Path] = []
    if machine is None:
        paths.append(root / "memory_bank.pt")
    else:
        paths.append(root / f"memory_bank_{machine}.pt")
    if mode == "window" and machine is not None:
        paths.append(root / f"memory_bank_{machine}_window.pt")
    return paths


def bank_matches_pipeline(
    meta: dict[str, Any],
    mode: str,
    win_frames: int | None = None,
    hop_frames: int | None = None,
) -> tuple[bool, str]:
    """Validate saved bank metadata against the requested pipeline."""
    bank_level = str(meta.get("bank_level", "")).lower()
    pipeline_mode = str(meta.get("pipeline_mode", bank_level)).lower()

    if bank_level != mode or pipeline_mode != mode:
        return False, f"bank metadata says mode={pipeline_mode or bank_level!r}, requested mode={mode!r}"

    if mode == "window":
        saved_win = meta.get("win_frames")
        saved_hop = meta.get("hop_frames")
        if saved_win is not None and win_frames is not None and int(saved_win) != int(win_frames):
            return False, f"bank win_frames={saved_win}, requested win_frames={win_frames}"
        if saved_hop is not None and hop_frames is not None and int(saved_hop) != int(hop_frames):
            return False, f"bank hop_frames={saved_hop}, requested hop_frames={hop_frames}"

    return True, "ok"


def bank_matches_memory_config(
    meta: dict[str, Any],
    memory_mode: str,
    num_clusters: int | None = None,
    feature_dim: int | None = None,
) -> tuple[bool, str]:
    saved_mode = str(meta.get("memory_bank_mode", "plain")).lower()
    if saved_mode != memory_mode:
        return False, f"bank memory mode={saved_mode!r}, requested memory mode={memory_mode!r}"

    if memory_mode == "clustered":
        saved_clusters = meta.get("requested_num_clusters", meta.get("num_clusters"))
        if saved_clusters is not None and num_clusters is not None and int(saved_clusters) != int(num_clusters):
            return False, f"bank num_clusters={saved_clusters}, requested num_clusters={num_clusters}"

    saved_dim = meta.get("feature_dim")
    if saved_dim is not None and feature_dim is not None and int(saved_dim) != int(feature_dim):
        return False, f"bank feature_dim={saved_dim}, embedding feature_dim={feature_dim}"

    return True, "ok"


def get_stale_bank_paths(
    bank_dir: str | Path,
    machine: str | None,
    pipeline_mode: str,
    memory_mode: str,
    num_clusters: int | None = None,
) -> list[Path]:
    """Return same-machine bank files that can conflict with the active bank."""
    root = Path(bank_dir)
    stem = "memory_bank" if machine is None else f"memory_bank_{machine}"
    active = get_bank_path(root, machine, pipeline_mode, memory_mode, num_clusters)

    candidates = [
        root / f"{stem}_clip.pt",
        root / f"{stem}_window.pt",
        root / f"{stem}.pt",
        root / "memory_bank.pt" if machine is None else root / f"memory_bank_{machine}.pt",
    ]
    candidates.extend(root.glob(f"{stem}_clip_clustered_k*.pt"))
    candidates.extend(root.glob(f"{stem}_window_clustered_k*.pt"))

    stale: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate.resolve() != active.resolve():
            stale.append(candidate)
    return sorted(set(stale))


def describe_pipeline(mode: str, win_frames: int, hop_frames: int, top_windows: int | None = None) -> list[str]:
    lines = [f"Pipeline mode: {mode}"]
    if mode == "window":
        suffix = "" if top_windows is None else f", top_windows={top_windows}"
        lines.append(f"  window params: win_frames={win_frames}, hop_frames={hop_frames}{suffix}")
    else:
        lines.append("  clip-level embeddings and clip-level kNN scoring")
    return lines
