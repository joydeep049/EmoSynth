from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Service-wide configuration loaded from environment variables."""

    dataset_dir: Path = Field(
        default=PROJECT_ROOT / "model" / "data" / "noto-128",
        description="Directory containing raw emoji PNG files.",
    )
    model_path: Path = Field(
        default=PROJECT_ROOT / "model" / "trained_models" / "vae_model_300.pth",
        description="Path to the trained VAE weights.",
    )
    latent_dim: int = Field(
        default=6,
        description="Latent dimensionality of the trained VAE.",
    )
    sample_size: int = Field(
        default=100,
        description="Number of emoji thumbnails to cache at startup.",
    )
    thumbnail_size: int = Field(
        default=128,
        description="Side length (pixels) for cached thumbnails.",
    )
    gif_cache_size: int = Field(
        default=32,
        description="Maximum number of synthesized outputs to keep in memory.",
    )
    interpolation_max_frames: int = Field(
        default=256,
        description="Upper bound on generated frames to guard against long jobs.",
    )
    gif_frame_size: int = Field(
        default=512,
        description="Output resolution (square) for synthesized GIF frames.",
    )
    device_preference: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="Force device selection or let the service decide automatically.",
    )
    random_seed: int = Field(
        default=42,
        description="Seed used when sampling cached emojis for deterministic startups.",
    )

    class Config:
        env_prefix = "EMOSYNTH_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        protected_namespaces = ("settings_",)

    @property
    def device(self) -> torch.device:
        if self.device_preference == "cpu":
            return torch.device("cpu")
        if self.device_preference == "cuda":
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()

