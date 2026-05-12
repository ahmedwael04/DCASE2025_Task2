"""Light augmentation utilities for normal-training memory bank construction."""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.audio_utils import load_audio_mono


def set_random_seed(seed: int | None) -> None:
    """Seed common RNGs for reproducible augmentation runs."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cfg(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _rand(generator: torch.Generator | None) -> float:
    return float(torch.rand((), generator=generator).item())


def _uniform(lo: float, hi: float, generator: torch.Generator | None) -> float:
    return float(torch.empty(()).uniform_(float(lo), float(hi), generator=generator).item())


def _randint(high: int, generator: torch.Generator | None) -> int:
    return int(torch.randint(high, (1,), generator=generator).item())


def _as_path_list(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    if isinstance(value, Iterable):
        return [Path(v) for v in value if v]
    return []


def discover_noise_files(data_root: str | Path, noise_dirs: Any = None) -> list[Path]:
    """Find background-noise wav files, returning an empty list if none exist."""
    root = Path(data_root)
    dirs = _as_path_list(noise_dirs)
    if not dirs:
        dirs = [
            root / "noise",
            root / "noises",
            root / "background_noise",
            root / "background_noises",
        ]

    wavs: list[Path] = []
    for directory in dirs:
        if not directory.is_absolute():
            directory = root / directory
        if directory.exists() and directory.is_dir():
            wavs.extend(directory.rglob("*.wav"))
    return sorted(set(wavs))


class WaveformAugmentor:
    """Mild waveform-level augmentation for machine-condition ASD."""

    def __init__(
        self,
        config: dict[str, Any],
        noise_files: list[Path] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.config = config
        self.noise_files = noise_files or []
        self.generator = generator

        self.gain_enabled = bool(_cfg(config, "gain", "enabled", default=True))
        self.gain_p = float(_cfg(config, "gain", "probability", default=0.0))
        self.gain_min = float(_cfg(config, "gain", "min", default=1.0))
        self.gain_max = float(_cfg(config, "gain", "max", default=1.0))

        self.noise_enabled = bool(_cfg(config, "noise", "enabled", default=True))
        self.noise_p = float(_cfg(config, "noise", "probability", default=0.0))
        self.snr_min = float(_cfg(config, "noise", "snr_min_db", default=10.0))
        self.snr_max = float(_cfg(config, "noise", "snr_max_db", default=25.0))

        self.eq_enabled = bool(_cfg(config, "eq", "enabled", default=True))
        self.eq_p = float(_cfg(config, "eq", "probability", default=0.0))

    def __call__(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        x = wav.detach().clone()
        if self.gain_enabled and _rand(self.generator) < self.gain_p:
            x = self._apply_gain(x)
        if self.noise_enabled and self.noise_files and _rand(self.generator) < self.noise_p:
            x = self._mix_noise(x, int(sr))
        if self.eq_enabled and _rand(self.generator) < self.eq_p:
            x = self._apply_mild_eq(x, int(sr))
        return x.clamp(-1.0, 1.0)

    def _apply_gain(self, wav: torch.Tensor) -> torch.Tensor:
        gain = _uniform(self.gain_min, self.gain_max, self.generator)
        return wav * gain

    def _mix_noise(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        path = self.noise_files[_randint(len(self.noise_files), self.generator)]
        try:
            noise, noise_sr = load_audio_mono(path)
        except Exception:
            return wav

        noise = self._match_sample_rate(noise, int(noise_sr), sr)
        if noise is None or noise.numel() == 0:
            return wav

        noise = self._fit_length(noise.to(dtype=wav.dtype), wav.size(-1))
        snr_db = _uniform(self.snr_min, self.snr_max, self.generator)

        eps = 1e-8
        signal_rms = torch.sqrt(torch.mean(wav.pow(2)) + eps)
        noise_rms = torch.sqrt(torch.mean(noise.pow(2)) + eps)
        target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        return wav + noise * (target_noise_rms / noise_rms)

    def _match_sample_rate(self, wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor | None:
        if src_sr == dst_sr:
            return wav
        try:
            import torchaudio  # type: ignore

            return torchaudio.functional.resample(wav, src_sr, dst_sr)
        except Exception:
            return None

    def _fit_length(self, wav: torch.Tensor, length: int) -> torch.Tensor:
        if wav.size(-1) >= length:
            max_start = wav.size(-1) - length
            start = _randint(max_start + 1, self.generator) if max_start > 0 else 0
            return wav[..., start : start + length]

        repeats = int(np.ceil(length / max(1, wav.size(-1))))
        return wav.repeat(1, repeats)[..., :length]

    def _apply_mild_eq(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            import torchaudio  # type: ignore

            nyquist = sr / 2.0
            x = wav

            highpass_cutoff = _uniform(20.0, min(120.0, nyquist * 0.2), self.generator)
            x = torchaudio.functional.highpass_biquad(x, sr, highpass_cutoff)

            low_hi = max(1000.0, nyquist * 0.95)
            low_lo = min(low_hi, max(1000.0, nyquist * 0.75))
            lowpass_cutoff = _uniform(low_lo, low_hi, self.generator)
            x = torchaudio.functional.lowpass_biquad(x, sr, lowpass_cutoff)

            center = _uniform(300.0, min(4000.0, nyquist * 0.8), self.generator)
            gain_db = _uniform(-3.0, 3.0, self.generator)
            return torchaudio.functional.equalizer_biquad(x, sr, center, gain_db, Q=0.8)
        except Exception:
            return wav


class SpecAugment:
    """Light frequency/time masking for log-mel or mel-like spectrograms."""

    def __init__(self, config: dict[str, Any], generator: torch.Generator | None = None) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.probability = float(config.get("probability", 0.0))
        self.freq_mask_param = int(config.get("freq_mask_param", 8))
        self.time_mask_param = int(config.get("time_mask_param", 24))
        self.max_freq_masks = int(config.get("max_freq_masks", 1))
        self.max_time_masks = int(config.get("max_time_masks", 1))
        self.max_time_fraction = float(config.get("max_time_fraction", 0.05))
        self.generator = generator

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.enabled or _rand(self.generator) >= self.probability:
            return spec

        x = spec.clone()
        fill = float(x.mean().item())
        freq_bins = x.size(-2)
        time_steps = x.size(-1)

        max_f = max(0, min(self.freq_mask_param, max(1, freq_bins // 12)))
        max_t = max(0, min(self.time_mask_param, max(1, int(time_steps * self.max_time_fraction))))

        for _ in range(max(0, self.max_freq_masks)):
            if max_f <= 0:
                break
            width = _randint(max_f, self.generator) + 1
            start = _randint(max(1, freq_bins - width + 1), self.generator)
            x[..., start : start + width, :] = fill

        for _ in range(max(0, self.max_time_masks)):
            if max_t <= 0:
                break
            width = _randint(max_t, self.generator) + 1
            start = _randint(max(1, time_steps - width + 1), self.generator)
            x[..., :, start : start + width] = fill

        return x


def describe_augmentation(
    config: dict[str, Any],
    noise_count: int,
    spectrogram_active: bool,
) -> list[str]:
    """Human-readable augmentation status lines for training logs."""
    enabled = bool(config.get("enabled", False))
    copies = int(config.get("copies_per_sample", 0))
    wave = config.get("waveform", {})
    spec = config.get("spectrogram", {})

    if not enabled or copies <= 0:
        return ["Augmentation: disabled"]

    lines = [f"Augmentation: enabled, clean originals + {copies} augmented copy/copies per clip"]
    lines.append(
        "  gain: "
        f"{_cfg(wave, 'gain', 'probability', default=0.0)} prob, "
        f"{_cfg(wave, 'gain', 'min', default=1.0)}-{_cfg(wave, 'gain', 'max', default=1.0)}x"
    )
    lines.append(
        "  background noise: "
        f"{_cfg(wave, 'noise', 'probability', default=0.0)} prob, "
        f"{_cfg(wave, 'noise', 'snr_min_db', default=10)}-{_cfg(wave, 'noise', 'snr_max_db', default=25)} dB SNR, "
        f"{noise_count} file(s)"
    )
    if noise_count == 0:
        lines.append("  background noise: no noise folder/files found, skipping safely")
    lines.append(f"  mild EQ/filter: {_cfg(wave, 'eq', 'probability', default=0.0)} prob")
    spec_enabled = bool(spec.get("enabled", True)) and float(spec.get("probability", 0.0)) > 0.0
    if not spec_enabled:
        spec_status = "disabled"
    elif spectrogram_active:
        spec_status = "active"
    else:
        spec_status = "configured, inactive for waveform-input backbone"
    lines.append(f"  SpecAugment: {spec.get('probability', 0.0)} prob ({spec_status})")
    return lines
