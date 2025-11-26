from __future__ import annotations

import asyncio
import base64
import io
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms as T

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.skeleton import VariationalAutoEncoder

from .cache import EmojiCache, LRUCache, pil_to_png_bytes
from .config import Settings, get_settings
from .interpolation import interpolate_latents, limit_frames
from .schemas import FramesPayload, GifPayload, SynthesisRequest


class AppState:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.device = self.settings.device
        self.model: VariationalAutoEncoder | None = None
        self.cache = EmojiCache()
        self.gif_cache = LRUCache(self.settings.gif_cache_size)
        self._nn: NearestNeighbors | None = None
        self._id_to_index: Dict[str, int] = {}
        self._latent_matrix: np.ndarray | None = None
        self._id_list: List[str] = []
        self._lock = asyncio.Lock()
        self._ready_event = asyncio.Event()
        self._tensor_transform = T.Compose(
            [
                T.Resize((128, 128)),
                T.ToTensor(),
            ]
        )
        self._thumbnail_transform = T.Compose(
            [
                T.Resize((self.settings.thumbnail_size, self.settings.thumbnail_size)),
            ]
        )

    async def initialize(self) -> None:
        if self._ready_event.is_set():
            return
        async with self._lock:
            if self._ready_event.is_set():
                return
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._blocking_initialize)
            self._ready_event.set()

    def _blocking_initialize(self) -> None:
        random.seed(self.settings.random_seed)
        torch.manual_seed(self.settings.random_seed)
        self.model = self._load_model()
        self._populate_cache()
        self._build_index()

    def _load_model(self) -> VariationalAutoEncoder:
        if not self.settings.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.settings.model_path}")
        model = VariationalAutoEncoder(self.settings.latent_dim)
        raw_state = torch.load(self.settings.model_path, map_location=self.device)
        state_dict = raw_state.get("model_state_dict", raw_state)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        return model

    def _sample_emoji_files(self) -> List[Path]:
        dataset_dir = self.settings.dataset_dir
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory missing: {dataset_dir}")
        files = [p for p in dataset_dir.iterdir() if p.suffix.lower() == ".png"]
        if len(files) < self.settings.sample_size:
            return files
        return random.sample(files, self.settings.sample_size)

    def _populate_cache(self) -> None:
        assert self.model is not None
        sampled_files = self._sample_emoji_files()
        pil_to_tensor = self._tensor_transform
        to_pil = T.ToPILImage()
        device = self.device

        for path in sampled_files:
            emoji_id = path.stem
            pil_img = Image.open(path).convert("RGBA").convert("RGB")
            tensor = pil_to_tensor(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = self.model.encode(tensor)
            latent = mu.squeeze(0).cpu().numpy()

            thumb_img = self._thumbnail_transform(pil_img)
            thumb_bytes = pil_to_png_bytes(thumb_img)
            self.cache.add(emoji_id, latent, thumb_bytes, tensor.cpu().numpy())

    def _build_index(self) -> None:
        latents = list(self.cache.latents.values())
        if not latents:
            raise RuntimeError("Cache is empty; cannot build kNN index.")
        self._latent_matrix = np.stack(latents)
        self._id_list = self.cache.ids()
        self._nn = NearestNeighbors(metric="cosine")
        self._nn.fit(self._latent_matrix)
        self._id_to_index = {emoji_id: idx for idx, emoji_id in enumerate(self._id_list)}


    async def wait_ready(self) -> None:
        await self._ready_event.wait()

    @property
    def ready(self) -> bool:
        return self._ready_event.is_set()

    def health(self) -> dict:
        return {
            "initialized": self.ready,
            "device": str(self.device),
            "cached_count": len(self.cache.latents),
            "model_path": str(self.settings.model_path),
            "dataset_dir": str(self.settings.dataset_dir),
        }

    def random_emoji(self) -> Tuple[str, str]:
        return self.cache.random_choice()

    def _neighbors(self, emoji_id: str, k: int) -> List[str]:
        if self._nn is None or self._latent_matrix is None:
            raise RuntimeError("Nearest neighbor index is not built.")
        if emoji_id not in self.cache.latents:
            raise KeyError(f"Unknown emoji id: {emoji_id}")
        idx = self._id_to_index[emoji_id]
        n_neighbors = min(k + 1, len(self.cache.latents))
        _, indices = self._nn.kneighbors(self._latent_matrix[idx : idx + 1], n_neighbors=n_neighbors)
        ordered_ids = []
        for neighbor_idx in indices[0]:
            neighbor_id = self._id_list[neighbor_idx]
            if neighbor_id != emoji_id:
                ordered_ids.append(neighbor_id)
        return ordered_ids[:k]

    def _decode_latents(self, latents: List[np.ndarray]) -> List[Image.Image]:
        assert self.model is not None
        if not latents:
            return []
        latent_tensor = torch.from_numpy(np.stack(latents)).float().to(self.device)
        with torch.no_grad():
            decoded = self.model.decoder(latent_tensor).cpu().clamp(0.0, 1.0)
        to_pil = T.ToPILImage()
        return [to_pil(frame) for frame in decoded]

    def _frames_to_base64(self, frames: List[Image.Image]) -> List[str]:
        encoded = []
        for frame in frames:
            buffer = io.BytesIO()
            frame.save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
        return encoded

    def _frames_to_gif(self, frames: List[Image.Image], duration_ms: int) -> str:
        if not frames:
            raise ValueError("No frames to convert to GIF.")
        processed_frames = [self._prep_frame_for_gif(frame) for frame in frames]
        buffer = io.BytesIO()
        processed_frames[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=processed_frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
        )
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _generate_frames(self, request: SynthesisRequest) -> Tuple[List[Image.Image], List[str]]:
        source_latent = self.cache.get_latent(request.id)
        neighbor_ids = self._neighbors(request.id, request.k)
        if not neighbor_ids:
            neighbor_ids = [request.id]

        all_latents: List[np.ndarray] = []
        current = source_latent

        for idx, neighbor_id in enumerate(neighbor_ids):
            target = self.cache.get_latent(neighbor_id)
            segment = interpolate_latents(current, target, request.steps_between, request.mode)
            if idx > 0 and segment:
                segment = segment[1:]
            all_latents.extend(segment)
            current = target

        bounded_latents = limit_frames(all_latents, self.settings.interpolation_max_frames)
        frames = self._decode_latents(bounded_latents)
        return frames, neighbor_ids

    def _prep_frame_for_gif(self, frame: Image.Image) -> Image.Image:
        target_size = (
            self.settings.gif_frame_size,
            self.settings.gif_frame_size,
        )
        if frame.size != target_size:
            frame = frame.resize(target_size, Image.LANCZOS)
        return frame.convert("P", palette=Image.ADAPTIVE, colors=256, dither=Image.FLOYDSTEINBERG)

    async def synthesize(self, request: SynthesisRequest) -> GifPayload | FramesPayload:
        key = request.cache_key()
        cached = self.gif_cache.get(key)
        if cached:
            return cached

        loop = asyncio.get_event_loop()
        frames, neighbor_ids = await loop.run_in_executor(None, self._generate_frames, request)

        payload: GifPayload | FramesPayload
        if request.return_type == "gif":
            gif_b64 = await loop.run_in_executor(None, self._frames_to_gif, frames, request.duration_ms)
            payload = GifPayload(
                gif_base64=gif_b64,
                neighbor_ids=neighbor_ids,
                frame_count=len(frames),
                duration_ms=request.duration_ms,
                mode=request.mode,
            )
        else:
            frame_b64 = await loop.run_in_executor(None, self._frames_to_base64, frames)
            payload = FramesPayload(
                frames=frame_b64,
                neighbor_ids=neighbor_ids,
                frame_count=len(frames),
                duration_ms=request.duration_ms,
                mode=request.mode,
            )

        self.gif_cache.set(key, payload)
        return payload


