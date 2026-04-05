#!/usr/bin/env python
"""
Build k-NN memory bank for DCASE-2025 Task-2.
"""
import random
import numpy as np
import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = cfg["data"]["root"]
    dataset = DCASETask2Dataset(root, split="train")
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
    detector = KNNDetector(k=k)
    normalize = bool(cfg.get("model", {}).get("normalize", False))

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
    torch.save({"memory": feats, "paths": paths}, out_dir / "memory_bank.pt")
    print(f"✅ memory bank → {out_dir/'memory_bank.pt'}")


if __name__ == "__main__":
    main()
