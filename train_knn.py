#!/usr/bin/env python
"""Convenience wrapper for running the training script from repo root.

Usage (equivalent):
  python train_knn.py ...
  python scripts/train_knn.py ...
"""

from scripts.train_knn import main


if __name__ == "__main__":
    main()
