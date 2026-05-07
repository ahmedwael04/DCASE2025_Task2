#!/usr/bin/env python
"""
GPU-accelerated inference for DCASE-2025 Task 2 (eval set only).

Produces:
  • anomaly_score_<machine>_section_<XX>_test.csv
  • decision_result_<machine>_section_<XX>_test.csv
under your csv_out_dir (configs/default.yaml).
"""

import argparse, sys, glob
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


def compute_threshold(mem_bank: torch.Tensor, k: int, distance: str, pct: float, chunk: int = 1024) -> float:
    """Compute kNN mean-distance threshold from within-bank distances."""
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
    return float(np.percentile(mem_dists, pct))


def knn_score_from_frames(frames: torch.Tensor, mem_bank: torch.Tensor, k: int, distance: str) -> torch.Tensor:
    """Return per-frame kNN mean-distance scores, shape (T,)."""
    if frames.ndim != 2:
        raise ValueError(f"frames must be (T,D), got {tuple(frames.shape)}")
    if distance == "cosine":
        d = 1.0 - (frames @ mem_bank.T)  # (T, N)
    else:
        d = torch.cdist(frames, mem_bank)  # (T, N)
    topk = torch.topk(d, k=k, dim=1, largest=False).values  # (T, K)
    return topk.mean(dim=1)

def main():
    # 1) Argparse & config
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--machine",
        default=None,
        help="If set, run inference only for this machine under eval_data/raw/<machine>/test.",
    )
    args   = p.parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Paths
    bank_dir = Path(cfg["logging"]["bank_out"])
    csv_dir  = Path(cfg["logging"]["csv_out_dir"])
    csv_dir.mkdir(parents=True, exist_ok=True)

    # 3) Load backbone
    backbone = BEATsBackbone(cfg["model"]["embedding"]).to(device).eval()

    # 4) Detector settings
    K = cfg.get("detector", {}).get("k", cfg.get("model", {}).get("k", 3))
    distance = str(cfg.get("detector", {}).get("distance", "cosine")).lower()
    normalize = bool(cfg.get("model", {}).get("normalize", False))
    pct = cfg.get("threshold", {}).get("percentile", 90)
    chunk = 1024  # adjust to fit your GPU

    temporal_cfg = cfg.get("detector", {}).get("temporal", {})
    temporal_enabled = bool(temporal_cfg.get("enabled", False))
    temporal_agg = str(temporal_cfg.get("aggregation", "p95")).lower()
    temporal_pct = float(temporal_cfg.get("percentile", 95))
    temporal_stride = int(temporal_cfg.get("frame_stride", 1))
    threshold_train_samples = int(cfg.get("threshold", {}).get("train_samples", 200))

    def aggregate_frame_scores(frame_scores: torch.Tensor) -> float:
        if frame_scores.numel() == 0:
            return 0.0
        if temporal_agg in {"max", "maximum"}:
            return float(frame_scores.max().item())
        if temporal_agg in {"p95", "percentile", "quantile"}:
            q = max(0.0, min(1.0, temporal_pct / 100.0))
            return float(torch.quantile(frame_scores, q=q).item())
        if temporal_agg in {"mean", "avg", "average"}:
            return float(frame_scores.mean().item())
        raise ValueError(f"Unknown temporal aggregation '{temporal_agg}'.")

    def score_clip(wav: torch.Tensor, sr: int, mem_bank: torch.Tensor) -> float:
        if not temporal_enabled:
            feat = backbone(wav, sr)
            if normalize:
                feat = F.normalize(feat, dim=-1)
            if distance == "cosine":
                d = 1.0 - (feat @ mem_bank.T)
            else:
                d = torch.cdist(feat, mem_bank)
            topk = torch.topk(d, k=K, dim=1, largest=False).values
            return float(topk.mean().item())

        frames = backbone(wav, sr, return_frames=True)  # (1,T,D)
        frames = frames.squeeze(0)
        if temporal_stride > 1:
            frames = frames[::temporal_stride]
        if normalize:
            frames = F.normalize(frames, dim=-1)

        frame_scores = knn_score_from_frames(frames=frames, mem_bank=mem_bank, k=K, distance=distance)
        return aggregate_frame_scores(frame_scores)

    # 5) Per-machine cache: {machine: {mem_bank, threshold}}
    cache = {}

    def prepare_machine(machine: str):
        specific = bank_dir / f"memory_bank_{machine}.pt"
        default = bank_dir / "memory_bank.pt"
        use_specific = specific.exists()
        bank_path = specific if use_specific else default

        cache_key = machine if use_specific else "__default__"
        if cache_key in cache:
            return cache[cache_key]["mem_bank"], cache[cache_key]["threshold"]

        ckpt = torch.load(bank_path, map_location="cpu")
        raw_mem = ckpt["memory"]
        train_paths = ckpt.get("paths", [])
        mem_bank = torch.stack([x.squeeze(0) for x in raw_mem], dim=0).to(device)
        if normalize:
            mem_bank = F.normalize(mem_bank, dim=-1)

        print(f"▶ Computing threshold for {machine} using {bank_path.name} …")
        if temporal_enabled and train_paths:
            sample_paths = train_paths[:threshold_train_samples] if threshold_train_samples > 0 else train_paths
            scores = []
            for tp in tqdm(sample_paths, desc="  threshold clips", unit="clip", file=sys.stdout, dynamic_ncols=True):
                w, s = load_audio_mono(tp)
                w = w.to(device)
                scores.append(score_clip(w, int(s), mem_bank))
            threshold = float(np.percentile(np.asarray(scores, dtype=np.float32), pct))
        else:
            threshold = compute_threshold(mem_bank=mem_bank, k=K, distance=distance, pct=pct, chunk=chunk)
        print(f"▶ {machine}: threshold @ {pct}th percentile = {threshold:.6f}")

        cache[cache_key] = {"mem_bank": mem_bank, "threshold": threshold, "bank_path": str(bank_path)}
        return mem_bank, threshold

    # 6) Gather eval_data test WAVs
    root    = cfg["data"]["root"]
    if args.machine:
        pattern = f"{root}/eval_data/raw/{args.machine}/test/**/*.wav"
    else:
        pattern = f"{root}/eval_data/raw/*/test/**/*.wav"
    wavs    = sorted(glob.glob(pattern, recursive=True))
    print(f"▶ Found {len(wavs)} eval clips under: {pattern}")
    if not wavs:
        print("⚠️  No eval_data test files found! Did you run download_task2_data.sh?")
        sys.exit(1)

    # 7) Inference loop with live GPU-knn scoring
    writers = {}
    for path in tqdm(wavs,
                     desc="Scoring eval clips",
                     unit="clip",
                     file=sys.stdout,
                     leave=True,
                     dynamic_ncols=True):
        wav, sr = load_audio_mono(path)
        wav = wav.to(device); sr = int(sr)

        p       = Path(path)
        machine = p.parent.parent.name      # e.g. CoffeeGrinder
        mem_bank, threshold = prepare_machine(machine)

        score = float(score_clip(wav, sr, mem_bank))
        decision = 1 if score > threshold else 0

        section = p.stem.split("_")[1]         # take "section_XX_####" → ["section","XX","####"]
        tag     = f"{machine}_section_{section}"

        # open CSVs on first use
        if tag not in writers:
            asc = csv_dir / f"anomaly_score_{tag}_test.csv"
            dec = csv_dir / f"decision_result_{tag}_test.csv"
            writers[tag] = (asc.open("w"), dec.open("w"))

        asc_fp, dec_fp = writers[tag]
        asc_fp.write(f"{p.name},{score:.6f}\n")
        dec_fp.write(f"{p.name},{decision}\n")

    # 8) Close filehandles
    for asc_fp, dec_fp in writers.values():
        asc_fp.close()
        dec_fp.close()

    print(f"✅ All CSVs written to {csv_dir}/")


if __name__ == "__main__":
    main()
