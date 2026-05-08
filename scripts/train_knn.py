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
from src.models.detector import KNNDetector


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--machine",
        default=None,
        help="If set, build a memory bank only for this machine (e.g., AutoTrash).",
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

    detector = KNNDetector(k=k, normalize=normalize)

    feats, paths = [], []
    for batch in tqdm(loader, desc="Extracting embeddings"):
        wav, sr, path = batch[0]
        feat = backbone(wav.to(device), sr)
        if normalize:
            feat = F.normalize(feat, dim=-1)
        feats.append(feat.cpu())
        paths.append(path)

    detector.fit(feats)

    out_dir = Path(cfg["logging"]["bank_out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = "memory_bank.pt" if args.machine is None else f"memory_bank_{args.machine}.pt"
    out_path = out_dir / out_name
    torch.save(
        {
            "memory": feats,
            "paths": paths,
            "meta": {"machine": args.machine, "stage": args.stage, "split": "train"},
        },
        out_path,
    )
    print(f"✅ memory bank → {out_path}")


if __name__ == "__main__":
    main()
