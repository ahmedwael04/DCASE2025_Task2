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

def main():
    # 1) Argparse & config
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--stage",
        default="eval_data",
        choices=["dev_data", "eval_data"],
        help="Which dataset stage to score (eval_data is unlabeled; dev_data is labeled via filenames).",
    )
    p.add_argument(
        "--machine",
        default=None,
        help="If set, run inference only for this machine under eval_data/raw/<machine>/test.",
    )
    p.add_argument(
        "--score-level",
        default="clip",
        choices=["clip", "window"],
        help="Score clips directly, or score windows and aggregate to clip score.",
    )
    p.add_argument(
        "--win-frames",
        type=int,
        default=50,
        help="Window size in frames when --score-level=window.",
    )
    p.add_argument(
        "--hop-frames",
        type=int,
        default=50,
        help="Hop size in frames when --score-level=window (default: non-overlapping).",
    )
    p.add_argument(
        "--top-windows",
        type=int,
        default=5,
        help="Aggregate by mean of top-N window scores when --score-level=window.",
    )
    p.add_argument(
        "--default-domain",
        default="target",
        choices=["source", "target"],
        help="Domain to assume when filenames don't encode it (eval_data typically doesn't).",
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
    if distance == "cosine" and not normalize:
        print("⚠️  distance=cosine but model.normalize=false; forcing L2 normalization.")
        normalize = True
    pct = cfg.get("threshold", {}).get("percentile", 90)
    chunk = 1024  # adjust to fit your GPU

    # 5) Per-machine cache: {machine: {mem_banks, thresholds, dom_stats}}
    cache = {}

    def prepare_machine(machine: str):
        # Prefer window-level bank if user requested window scoring.
        specific_window = bank_dir / f"memory_bank_{machine}_window.pt"
        specific_clip = bank_dir / f"memory_bank_{machine}.pt"
        default = bank_dir / "memory_bank.pt"

        if args.score_level == "window" and specific_window.exists():
            bank_path = specific_window
            cache_key = f"{machine}__window"
        else:
            use_specific = specific_clip.exists()
            bank_path = specific_clip if use_specific else default
            cache_key = machine if use_specific else "__default__"
        if cache_key in cache:
            return cache[cache_key]

        ckpt = None
        try:
            ckpt = torch.load(bank_path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = None
        except Exception:
            ckpt = None

        # We need non-tensor metadata (`paths`) for per-domain normalization; reload if missing.
        if not (isinstance(ckpt, dict) and "memory" in ckpt and "paths" in ckpt):
            ckpt = torch.load(bank_path, map_location="cpu")
        raw_mem = ckpt["memory"]
        raw_paths = ckpt.get("paths", [""] * len(raw_mem))

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

        print(
            f"▶ Preparing {machine} bank ({bank_path.name}): "
            f"source={mem_banks['counts']['source']}, target={mem_banks['counts']['target']}, all={mem_banks['counts']['__all__']}"
        )

        thresholds: dict[str, float] = {}
        dom_stats: dict[str, tuple[float, float]] = {}

        for dom_key in ("source", "target", "__all__"):
            bank = mem_banks[dom_key]
            # Threshold/stats computed within-domain bank only
            print(f"▶ Computing threshold for {machine} ({dom_key}) …")
            thr, mem_dists = compute_threshold_and_mem_dists(
                mem_bank=bank,
                k=K,
                distance=distance,
                pct=pct,
                chunk=chunk,
            )
            thresholds[dom_key] = float(thr)
            mu = float(np.mean(mem_dists))
            sig = float(np.std(mem_dists))
            if not np.isfinite(sig) or sig < 1e-12:
                sig = 1.0
            dom_stats[dom_key] = (mu, sig)

        # If a domain split is missing, fallback will point to __all__ at use time.
        cache[cache_key] = {
            "mem_banks": mem_banks,
            "thresholds": thresholds,
            "dom_stats": dom_stats,
            "bank_path": str(bank_path),
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

    # 7) Inference loop with live GPU-knn scoring
    writers = {}
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
            machine = p.parent.parent.name  # e.g. CoffeeGrinder
            pack = prepare_machine(machine)

            dom = infer_domain_from_name(p.name) or args.default_domain
            mem_bank = pack["mem_banks"][dom] if dom in pack["mem_banks"] else pack["mem_banks"]["__all__"]
            threshold = pack["thresholds"][dom] if dom in pack["thresholds"] else pack["thresholds"]["__all__"]
            mu, sig = pack["dom_stats"][dom] if dom in pack["dom_stats"] else pack["dom_stats"]["__all__"]

            if args.score_level == "clip":
                # backbone→embedding (1, D)
                feat = backbone(wav, sr)
                if normalize:
                    feat = F.normalize(feat, dim=-1)

                # GPU k‐NN scoring
                if distance == "cosine":
                    d = 1.0 - (feat @ mem_bank.T)
                else:
                    d = torch.cdist(feat, mem_bank)  # (1, N)
                topk = torch.topk(d, k=K, dim=1, largest=False).values  # (1, K)
                score = float(topk.mean().item())
            else:
                # window-level scoring + top-N aggregation
                temporal = backbone(wav, sr, return_temporal=True)  # (1,T,D)
                win_emb = make_window_embeddings(temporal, args.win_frames, args.hop_frames)  # (W,D)
                if normalize:
                    win_emb = F.normalize(win_emb, dim=-1)

                if distance == "cosine":
                    d = 1.0 - (win_emb @ mem_bank.T)  # (W,N)
                else:
                    d = torch.cdist(win_emb, mem_bank)  # (W,N)

                topk = torch.topk(d, k=K, dim=1, largest=False).values  # (W,K)
                win_scores = topk.mean(dim=1)  # (W,)

                n_top = max(1, int(args.top_windows))
                n_top = min(n_top, win_scores.numel())
                top_vals = torch.topk(win_scores, k=n_top, largest=True).values
                score = float(top_vals.mean().item())

            # Per-domain score normalization (calibration)
            score_n = (float(score) - float(mu)) / float(sig)
            if dom == "source":
                score_n = -score_n
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
