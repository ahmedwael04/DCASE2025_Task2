import sys
from pathlib import Path

import torch

# Allow running as: python tests/test_detector.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.detector import KNNDetector
import numpy as np


def test_detector():
    torch.manual_seed(0)
    dummy = [torch.randn(10, 768) for _ in range(5)]
    det = KNNDetector(k=3, normalize=True)
    det.fit(dummy)
    score = det.score(torch.randn(8, 768))
    assert isinstance(score, float)


if __name__ == "__main__":
    test_detector()
    print("✓ all tests passed")
