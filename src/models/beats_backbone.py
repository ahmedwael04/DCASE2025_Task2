"""
Backbone wrapper: works with HuBERT, Wav2Vec2, BEATs.
"""

from pathlib import Path
import torch, torchaudio


VALID_EMBEDDING_MODES = {
    "last_layer_mean",
    "last_layer_cls",
    "last_layer_mean_std",
    "last4_layers_mean",
    "middle_layer_mean",
    "middle_layer_mean_std",
}


class BEATsBackbone(torch.nn.Module):
    def __init__(self, checkpoint="HUBERT_BASE", use_layer_stack=False, embedding_mode="last_layer_mean"):
        super().__init__()

        self.bundle = getattr(torchaudio.pipelines, checkpoint)
        self.model = self.bundle.get_model().eval()
        self.sample_rate = self.bundle.sample_rate
        self.use_layer_stack = use_layer_stack
        self.embedding_mode = str(embedding_mode).lower()
        if self.embedding_mode not in VALID_EMBEDDING_MODES:
            valid = ", ".join(sorted(VALID_EMBEDDING_MODES))
            raise ValueError(f"embedding_mode must be one of: {valid}")
        self._warned_fallbacks: set[str] = set()
        self._logged_embedding_dim: bool = False

        print(f"[INFO] Selected embedding mode: {self.embedding_mode}")

        # waveform vs. spectrogram input
        self.expect_waveform = checkpoint.startswith(
            ("HUBERT", "WAV2VEC", "XLSR")
        )
        if not self.expect_waveform:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
            )

        # load optional fine-tuned weights
        finetuned = Path("finetuned_beats_large.pt")
        if finetuned.exists():
            self.model.load_state_dict(
                torch.load(finetuned, map_location="cpu"), strict=False
            )
            print("✔ loaded fine-tuned backbone")

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned_fallbacks:
            print(f"WARNING: {message}")
            self._warned_fallbacks.add(key)

    def _has_cls_token(self) -> bool:
        return any(
            hasattr(self.model, name)
            for name in ("cls_token", "class_token", "classification_token")
        )

    def _select_temporal_features(self, layers):
        mode = self.embedding_mode
        if self.use_layer_stack and isinstance(layers, list):
            return torch.cat(layers[::3], dim=-1)
        if not isinstance(layers, list):
            if mode in {"last4_layers_mean", "middle_layer_mean"}:
                self._warn_once(
                    mode,
                    f"{mode} requested but hidden-layer outputs are unavailable; using last_layer_mean.",
                )
            if mode == "middle_layer_mean_std":
                self._warn_once(
                    mode,
                    "middle_layer_mean_std requested but hidden-layer outputs are unavailable; falling back to last_layer_mean_std.",
                )
            return layers
        if mode == "last4_layers_mean":
            if len(layers) >= 4:
                # Average hidden layers before time pooling for the last-4 BEATs variant.
                return torch.stack(layers[-4:], dim=0).mean(dim=0)
            self._warn_once(
                "last4_layers_mean",
                "last4_layers_mean requested but fewer than 4 hidden layers are available; using last_layer_mean.",
            )
            return layers[-1]
        if mode == "middle_layer_mean":
            # Use the middle hidden layer before time pooling for the middle-layer variant.
            return layers[len(layers) // 2]
        if mode == "middle_layer_mean_std":
            # Use the middle hidden layer before time pooling for the middle-layer mean-std variant.
            return layers[len(layers) // 2]
        return layers[-1]

    def _pool_clip_embedding(self, temporal_feats: torch.Tensor) -> torch.Tensor:
        mode = self.embedding_mode
        if mode == "last_layer_cls":
            if self._has_cls_token() and temporal_feats.size(1) > 0:
                return temporal_feats[:, 0, :]
            self._warn_once(
                "last_layer_cls",
                "last_layer_cls requested but no CLS token was detected; using last_layer_mean.",
            )
            return temporal_feats.mean(dim=1)
        if mode in {"last_layer_mean_std", "middle_layer_mean_std"}:
            # Mean-std pooling concatenates temporal mean and temporal standard deviation.
            mean = temporal_feats.mean(dim=1)
            std = temporal_feats.std(dim=1, unbiased=False)
            return torch.cat([mean, std], dim=-1)
        return temporal_feats.mean(dim=1)

    # -------------------------------------------------------- #
    @torch.no_grad()
    def forward(
        self,
        wav: torch.Tensor,
        sr: int | torch.Tensor,
        return_temporal: bool = False,
        spec_augment=None,
    ):
        sr = int(sr) if torch.is_tensor(sr) else sr
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # extract features
        if self.expect_waveform:
            out = self.model.extract_features(wav)
        else:
            mel = self.mel(wav)
            if spec_augment is not None:
                mel = spec_augment(mel)
            out = self.model.extract_features(mel)

        # normalise return types
        x = out[0] if isinstance(out, tuple) else out
        feats = self._select_temporal_features(x)

        if return_temporal:
            return feats  # (B, T, D)

        emb = self._pool_clip_embedding(feats)    # (B, D)
        if not self._logged_embedding_dim:
            try:
                dim = int(emb.size(-1))
            except Exception:
                dim = None
            if dim is not None:
                print(f"[INFO] Embedding dimensionality: {dim}")
            self._logged_embedding_dim = True
        return emb
