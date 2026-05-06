"""
Recursive loader for DCASE-2025 Task-2 audio.

Matches:
    <root>/<dev|eval>_data/raw/**/<train|test>/**.wav
so it works for layouts like
    fan/train/section_00/*.wav
    gearbox/test/section_01/*.wav
"""

from pathlib import Path
from typing import Tuple, List
import glob

import torch
from torch.utils.data import Dataset

from src.utils.audio_utils import load_audio_mono


class DCASETask2Dataset(Dataset):
    def __init__(self, root: str, split: str = "train", files: List[str] | None = None):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        if files is not None:
            self.files = sorted(files)
        else:
            self.files: List[str] = []
            for stage in ("dev_data", "eval_data"):
                patt = Path(root, stage, "raw", "**", split, "**", "*.wav")
                self.files += glob.glob(str(patt), recursive=True)

            self.files.sort()

        if not self.files:
            raise RuntimeError(
                f"No wavs found beneath {root} for split '{split}'. "
                "Check folder names."
            )

    # -------------------------------------------------------- #
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.files[idx]
        wav, sr = load_audio_mono(path)
        return wav, sr, path
