# Copyright (C) 2025 Joydeep Tripathy
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import base64
import io
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class EmojiEntry:
    emoji_id: str
    latent: np.ndarray
    thumbnail_b64: str


@dataclass
class EmojiCache:
    """Stores tensors, latents, and thumbnails for a sampled emoji subset."""

    latents: Dict[str, np.ndarray] = field(default_factory=dict)
    thumbnails: Dict[str, str] = field(default_factory=dict)
    tensors: Dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, emoji_id: str, latent: np.ndarray, thumbnail: bytes, tensor: np.ndarray) -> None:
        thumb_b64 = base64.b64encode(thumbnail).decode("utf-8")
        self.latents[emoji_id] = latent
        self.thumbnails[emoji_id] = thumb_b64
        self.tensors[emoji_id] = tensor

    def ids(self) -> List[str]:
        return list(self.latents.keys())

    def random_choice(self) -> Tuple[str, str]:
        if not self.latents:
            raise RuntimeError("Emoji cache is empty.")
        emoji_id = random.choice(self.ids())
        return emoji_id, self.thumbnails[emoji_id]

    def get_latent(self, emoji_id: str) -> np.ndarray:
        return self.latents[emoji_id]


def pil_to_png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class LRUCache:
    """Small in-memory cache for synthesized outputs."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._entries: OrderedDict[str, object] = OrderedDict()

    def _key(self, parts: Iterable) -> str:
        return "|".join(map(str, parts))

    def get(self, key_parts: Iterable):
        key = self._key(key_parts)
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return self._entries[key]

    def set(self, key_parts: Iterable, value) -> None:
        key = self._key(key_parts)
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = value
        if len(self._entries) > self.max_size:
            self._entries.popitem(last=False)

