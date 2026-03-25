#!/usr/bin/env python
"""
Optionally pre‑compute log‑Mel spectrograms to speed up training.
This script is **optional** because the BEATs wrapper transforms audio on the fly.
"""
import argparse
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcase_root", required=True)
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for cached mel tensors. Default: <dcase_root>/melspec",
    )
    args = ap.parse_args()

    root = Path(args.dcase_root)
    out_root = Path(args.out_dir) if args.out_dir is not None else (root / "melspec")
    out_root.mkdir(parents=True, exist_ok=True)

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=128
    )

    wavs = list(root.rglob("*.wav"))
    for wav in tqdm(wavs):
        waveform, sr = torchaudio.load(wav)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        mel = mel_spec(waveform).log1p()
        out_path = out_root / wav.relative_to(root).with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mel.cpu(), out_path)

if __name__ == "__main__":
    main()
