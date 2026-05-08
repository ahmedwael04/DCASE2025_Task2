#!/usr/bin/env python
"""
Build k-NN memory bank for DCASE-2025 Task-2.
"""

import argparse
import sys
from pathlib import Path
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running as: python scripts/train_knn.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.file_utils import load_config
from src.data.dcase_dataset import DCASETask2Dataset
from src.models.beats_backbone import BEATsBackbone


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
        default="clip",
        choices=["clip", "window"],
        help="Store clip-level embeddings or window-level embeddings in the memory bank.",
    )
    ap.add_argument(
        "--win-frames",
        type=int,
        default=50,
        help="Window size in frames when --bank-level=window.",
    )
    ap.add_argument(
        "--hop-frames",
        type=int,
        default=50,
        help="Hop size in frames when --bank-level=window (default: non-overlapping).",
    )
    ap.add_argument(
        "--stage",
        default="both",
        choices=["dev_data", "eval_data", "both"],
        help="Which dataset stage to use for building the bank.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = cfg["data"]["root"]

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
    ).to(device).eval()

    k = cfg.get("detector", {}).get("k", cfg.get("model", {}).get("k", 3))
    distance = str(cfg.get("detector", {}).get("distance", "cosine")).lower()
    normalize = bool(cfg.get("model", {}).get("normalize", False))
    if distance == "cosine" and not normalize:
        print("⚠️  distance=cosine but model.normalize=false; forcing L2 normalization.")
        normalize = True

    feats, paths = [], []
    for batch in tqdm(loader, desc="Extracting embeddings"):
        wav, sr, path = batch[0]

        if args.bank_level == "clip":
            feat = backbone(wav.to(device), sr)
            if normalize:
                feat = F.normalize(feat, dim=-1)
            feats.append(feat.cpu())
            paths.append(path)
        else:
            temporal = backbone(wav.to(device), sr, return_temporal=True)  # (1,T,D)
            win_emb = make_window_embeddings(temporal, args.win_frames, args.hop_frames)  # (W,D)
            if normalize:
                win_emb = F.normalize(win_emb, dim=-1)

            # store each window as its own bank entry
            for w_i in range(win_emb.size(0)):
                feats.append(win_emb[w_i].detach().cpu())
                paths.append(f"{path}#win={w_i}")

    out_dir = Path(cfg["logging"]["bank_out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.machine is None:
        out_name = "memory_bank.pt"
    else:
        suffix = "" if args.bank_level == "clip" else "_window"
        out_name = f"memory_bank_{args.machine}{suffix}.pt"
    out_path = out_dir / out_name
    torch.save(
        {
            "memory": feats,
            "paths": paths,
            "meta": {
                "machine": args.machine,
                "stage": args.stage,
                "split": "train",
                "bank_level": args.bank_level,
                "win_frames": args.win_frames if args.bank_level == "window" else None,
                "hop_frames": args.hop_frames if args.bank_level == "window" else None,
            },
        },
        out_path,
    )
    print(f"✅ memory bank → {out_path}")


if __name__ == "__main__":
    main()
