"""Domain-generalization augmentations for first-shot ASD.

The augmentations are intentionally mild: they simulate microphone gain,
noise floor, time alignment, broad EQ, and short room coloration without
trying to change the machine state itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def set_augmentation_seed(seed: int | None) -> None:
    """Seed common RNGs for reproducible augmentation runs."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Read either AUG_STYLE or lower-case YAML-style config keys."""
    for key in keys:
        if key in config:
            return config[key]
        lower = key.lower()
        if lower in config:
            return config[lower]
    return default


def _pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return default


def _rand(generator: torch.Generator | None) -> float:
    return float(torch.rand((), generator=generator).item())


def _uniform(lo: float, hi: float, generator: torch.Generator | None) -> float:
    return float(torch.empty(()).uniform_(float(lo), float(hi), generator=generator).item())


def _randint(low: int, high: int, generator: torch.Generator | None) -> int:
    if high <= low:
        return int(low)
    return int(torch.randint(low, high, (1,), generator=generator).item())


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = False
    seed: int | None = 42
    num_views_train: int = 0
    num_views_test: int = 0
    gain_db: tuple[float, float] = (-6.0, 6.0)
    noise_snr_db: tuple[float, float] = (15.0, 35.0)
    time_shift_max: float = 0.05
    freq_mask_max: int = 8
    time_mask_max: int = 20
    memory_mode: str = "expand"
    test_score_agg: str = "median"
    colored_noise: bool = True
    eq: bool = True
    reverb: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AugmentationConfig":
        aug = cfg.get("augmentation", {})
        if not isinstance(aug, dict):
            aug = {}

        # Backward-compatible fallback from the previous config layout.
        old_copies = _get(aug, "copies_per_sample", default=None)
        waveform = aug.get("waveform", {}) if isinstance(aug.get("waveform", {}), dict) else {}
        spectrogram = aug.get("spectrogram", {}) if isinstance(aug.get("spectrogram", {}), dict) else {}

        gain_db_default = (-6.0, 6.0)
        if "gain" in waveform and isinstance(waveform["gain"], dict):
            gain_min = float(waveform["gain"].get("min", 0.5))
            gain_max = float(waveform["gain"].get("max", 2.0))
            gain_db_default = (
                20.0 * np.log10(max(gain_min, 1e-6)),
                20.0 * np.log10(max(gain_max, 1e-6)),
            )

        noise_db_default = (15.0, 35.0)
        if "noise" in waveform and isinstance(waveform["noise"], dict):
            noise_db_default = (
                float(waveform["noise"].get("snr_min_db", 15.0)),
                float(waveform["noise"].get("snr_max_db", 35.0)),
            )

        enabled = bool(_get(cfg, "AUG_ENABLE", default=_get(aug, "AUG_ENABLE", "enabled", default=False)))
        train_views = _get(
            cfg,
            "AUG_NUM_VIEWS_TRAIN",
            default=_get(aug, "AUG_NUM_VIEWS_TRAIN", "num_views_train", default=old_copies or 0),
        )
        test_views = _get(
            cfg,
            "AUG_NUM_VIEWS_TEST",
            default=_get(aug, "AUG_NUM_VIEWS_TEST", "num_views_test", default=0),
        )

        memory_mode = str(
            _get(cfg, "AUG_MEMORY_MODE", default=_get(aug, "AUG_MEMORY_MODE", "memory_mode", default="expand"))
        ).lower()
        if memory_mode not in {"expand", "average"}:
            raise ValueError("AUG_MEMORY_MODE must be 'expand' or 'average'")

        score_agg = str(
            _get(cfg, "AUG_TEST_SCORE_AGG", default=_get(aug, "AUG_TEST_SCORE_AGG", "test_score_agg", default="median"))
        ).lower()
        if score_agg not in {"median", "mean"}:
            raise ValueError("AUG_TEST_SCORE_AGG must be 'median' or 'mean'")

        return cls(
            enabled=enabled,
            seed=_get(cfg, "AUG_SEED", default=_get(aug, "AUG_SEED", "seed", default=42)),
            num_views_train=max(0, int(train_views)) if enabled else 0,
            num_views_test=max(0, int(test_views)) if enabled else 0,
            gain_db=_pair(_get(cfg, "AUG_GAIN_DB", default=_get(aug, "AUG_GAIN_DB", "gain_db", default=None)), gain_db_default),
            noise_snr_db=_pair(
                _get(cfg, "AUG_NOISE_SNR_DB", default=_get(aug, "AUG_NOISE_SNR_DB", "noise_snr_db", default=None)),
                noise_db_default,
            ),
            time_shift_max=float(
                _get(cfg, "AUG_TIME_SHIFT_MAX", default=_get(aug, "AUG_TIME_SHIFT_MAX", "time_shift_max", default=0.05))
            ),
            freq_mask_max=int(
                _get(
                    cfg,
                    "AUG_FREQ_MASK_MAX",
                    default=_get(
                        aug,
                        "AUG_FREQ_MASK_MAX",
                        "freq_mask_max",
                        default=spectrogram.get("freq_mask_param", 8),
                    ),
                )
            ),
            time_mask_max=int(
                _get(
                    cfg,
                    "AUG_TIME_MASK_MAX",
                    default=_get(
                        aug,
                        "AUG_TIME_MASK_MAX",
                        "time_mask_max",
                        default=spectrogram.get("time_mask_param", 20),
                    ),
                )
            ),
            memory_mode=memory_mode,
            test_score_agg=score_agg,
            colored_noise=bool(_get(aug, "colored_noise", "AUG_COLORED_NOISE", default=True)),
            eq=bool(_get(aug, "eq", "AUG_EQ", default=True)),
            reverb=bool(_get(aug, "reverb", "AUG_REVERB", default=True)),
        )


class DomainGeneralizationAugmentor:
    """Create waveform and spectrogram views for domain-generalized embeddings."""

    def __init__(
        self,
        config: AugmentationConfig,
        generator: torch.Generator | None = None,
    ) -> None:
        self.config = config
        self.generator = generator

    def waveform_views(
        self,
        wav: torch.Tensor,
        sr: int,
        num_augmented_views: int,
    ) -> list[tuple[torch.Tensor, str, bool]]:
        """Return original plus N augmented waveform views."""
        views = [(wav, "clean", False)]
        if not self.config.enabled or num_augmented_views <= 0:
            return views
        for idx in range(int(num_augmented_views)):
            views.append((self.augment_waveform(wav, int(sr)), f"aug={idx}", True))
        return views

    def augment_waveform(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        x = wav.detach().clone()
        original_peak = x.abs().max().clamp_min(1e-6)

        x = self._random_gain(x)
        x = self._add_noise(x)
        x = self._time_shift(x, int(sr))
        if self.config.eq:
            x = self._mild_eq(x, int(sr))
        if self.config.reverb:
            x = self._small_reverb(x, int(sr))

        # Preserve the broad amplitude range while avoiding clipping artifacts.
        peak = x.abs().max().clamp_min(1e-6)
        if peak > 1.0:
            x = x / peak * min(float(original_peak.item()) * 1.25, 1.0)
        return x.clamp(-1.0, 1.0)

    def augment_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.config.enabled:
            return spec
        x = spec.clone()
        fill = float(x.mean().item())
        freq_bins = int(x.size(-2))
        time_steps = int(x.size(-1))

        max_f = max(0, min(int(self.config.freq_mask_max), max(1, freq_bins // 10)))
        max_t = max(0, min(int(self.config.time_mask_max), max(1, time_steps // 20)))

        if max_f > 0:
            width = _randint(1, max_f + 1, self.generator)
            start = _randint(0, max(1, freq_bins - width + 1), self.generator)
            x[..., start : start + width, :] = fill

        if max_t > 0:
            width = _randint(1, max_t + 1, self.generator)
            start = _randint(0, max(1, time_steps - width + 1), self.generator)
            x[..., :, start : start + width] = fill

        return x

    def _random_gain(self, wav: torch.Tensor) -> torch.Tensor:
        lo, hi = self.config.gain_db
        gain_db = _uniform(lo, hi, self.generator)
        return wav * (10.0 ** (gain_db / 20.0))

    def _add_noise(self, wav: torch.Tensor) -> torch.Tensor:
        if self.config.colored_noise and _rand(self.generator) < 0.5:
            noise = self._colored_noise_like(wav)
        else:
            noise = torch.randn(wav.shape, dtype=wav.dtype, device=wav.device)

        snr_db = _uniform(*self.config.noise_snr_db, self.generator)
        eps = 1e-8
        signal_rms = torch.sqrt(torch.mean(wav.pow(2)) + eps)
        noise_rms = torch.sqrt(torch.mean(noise.pow(2)) + eps)
        target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        return wav + noise * (target_noise_rms / noise_rms)

    def _colored_noise_like(self, wav: torch.Tensor) -> torch.Tensor:
        white = torch.randn(wav.shape, dtype=wav.dtype, device=wav.device)
        if white.size(-1) < 4:
            return white
        spec = torch.fft.rfft(white, dim=-1)
        freqs = torch.fft.rfftfreq(white.size(-1), d=1.0, device=wav.device)
        freqs = freqs.clamp_min(1.0 / white.size(-1))
        slope = 0.5 if _rand(self.generator) < 0.5 else 1.0
        shaped = spec / freqs.pow(slope).view(*([1] * (spec.ndim - 1)), -1)
        noise = torch.fft.irfft(shaped, n=white.size(-1), dim=-1)
        return noise / noise.std().clamp_min(1e-6)

    def _time_shift(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        max_samples = int(max(0.0, self.config.time_shift_max) * float(sr))
        max_samples = min(max_samples, max(0, wav.size(-1) - 1))
        if max_samples <= 0:
            return wav
        shift = _randint(-max_samples, max_samples + 1, self.generator)
        if shift == 0:
            return wav

        out = torch.zeros_like(wav)
        if shift > 0:
            out[..., shift:] = wav[..., :-shift]
        else:
            out[..., :shift] = wav[..., -shift:]
        return out

    def _mild_eq(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            import torchaudio  # type: ignore

            nyquist = sr / 2.0
            x = wav
            highpass = _uniform(20.0, min(140.0, nyquist * 0.25), self.generator)
            lowpass = _uniform(max(1000.0, nyquist * 0.72), max(1000.0, nyquist * 0.98), self.generator)
            center = _uniform(250.0, min(4500.0, nyquist * 0.8), self.generator)
            gain_db = _uniform(-3.0, 3.0, self.generator)

            x = torchaudio.functional.highpass_biquad(x, sr, highpass)
            x = torchaudio.functional.lowpass_biquad(x, sr, lowpass)
            return torchaudio.functional.equalizer_biquad(x, sr, center, gain_db, Q=0.8)
        except Exception:
            return wav

    def _small_reverb(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        max_delay = max(1, int(0.035 * sr))
        min_delay = max(1, int(0.008 * sr))
        delay1 = _randint(min_delay, max_delay + 1, self.generator)
        delay2 = _randint(delay1, max_delay + 1, self.generator)
        decay1 = _uniform(0.04, 0.12, self.generator)
        decay2 = _uniform(0.01, 0.06, self.generator)

        out = wav.clone()
        if delay1 < wav.size(-1):
            out[..., delay1:] = out[..., delay1:] + decay1 * wav[..., :-delay1]
        if delay2 < wav.size(-1):
            out[..., delay2:] = out[..., delay2:] + decay2 * wav[..., :-delay2]
        return out


def aggregate_augmented_scores(scores: list[float], method: str = "median") -> float:
    if not scores:
        raise ValueError("Cannot aggregate an empty score list")
    if method == "mean":
        return float(np.mean(scores))
    if method == "median":
        return float(np.median(scores))
    raise ValueError("AUG_TEST_SCORE_AGG must be 'median' or 'mean'")


def describe_augmentation(config: AugmentationConfig, phase: str) -> list[str]:
    if not config.enabled:
        return [f"Augmentation: disabled ({phase})"]

    lines = [
        f"Augmentation: enabled ({phase})",
        f"  train views: {config.num_views_train}",
        f"  test views: {config.num_views_test}",
        f"  gain dB: [{config.gain_db[0]}, {config.gain_db[1]}]",
        f"  noise SNR dB: [{config.noise_snr_db[0]}, {config.noise_snr_db[1]}]",
        f"  time shift max: {config.time_shift_max}s",
        f"  log-mel masks: freq<={config.freq_mask_max}, time<={config.time_mask_max}",
        f"  colored noise: {config.colored_noise}, EQ: {config.eq}, reverb: {config.reverb}",
        f"  memory mode: {config.memory_mode}",
        f"  test score aggregation: {config.test_score_agg}",
    ]
    return lines
