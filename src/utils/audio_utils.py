"""Audio loading utilities.

This repo previously relied on `torchaudio.load`, but some Windows/conda
installations ship torchaudio without any I/O backends enabled.

We use `soundfile` as a reliable WAV reader fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch


def load_audio_mono(path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
    """Load audio as (1, T) float32 torch tensor and sample rate."""
    # 1) torchaudio (fast when backend is available)
    try:
        import torchaudio  # type: ignore

        wav, sr = torchaudio.load(str(path))
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.to(dtype=torch.float32), int(sr)
    except Exception:
        pass

    # 2) soundfile (reliable, but requires libsndfile to be present)
    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = np.transpose(audio, (1, 0))  # (T, C) -> (C, T)
        wav = torch.from_numpy(audio)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav, int(sr)
    except Exception:
        pass

    # 3) scipy.io.wavfile (no external DLLs; supports PCM WAV)
    from scipy.io import wavfile

    sr, audio = wavfile.read(str(path))
    audio = np.asarray(audio)
    if audio.ndim == 1:
        audio = audio[:, None]

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(1.0, float(info.max))
    else:
        audio = audio.astype(np.float32)

    audio = np.transpose(audio, (1, 0))  # (T, C) -> (C, T)
    wav = torch.from_numpy(audio)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, int(sr)
