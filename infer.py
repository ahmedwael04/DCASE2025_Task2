#!/usr/bin/env python
"""Convenience wrapper for running inference from repo root.

Usage (equivalent):
  python infer.py ...
  python scripts/infer.py ...
"""

from scripts.infer import main


if __name__ == "__main__":
    main()
