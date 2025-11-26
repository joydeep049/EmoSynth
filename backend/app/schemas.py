from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    initialized: bool
    device: str
    cached_count: int
    model_path: str
    dataset_dir: str


class RandomEmojiResponse(BaseModel):
    id: str
    thumbnail_base64: str


class SynthesisRequest(BaseModel):
    id: str = Field(..., description="Source emoji identifier (filename stem).")
    k: int = Field(3, ge=1, le=20, description="Number of nearest neighbors.")
    steps_between: int = Field(8, ge=1, le=32, description="Interpolation steps per neighbor pair.")
    mode: Literal["lerp", "slerp"] = Field("lerp")
    duration_ms: int = Field(80, ge=10, le=500, description="Frame duration used for GIF playback.")
    return_type: Literal["gif", "frames"] = Field("gif")

    def cache_key(self) -> tuple:
        return (self.id, self.k, self.steps_between, self.mode, self.duration_ms, self.return_type)


class FramesPayload(BaseModel):
    frames: List[str]
    neighbor_ids: List[str]
    frame_count: int
    duration_ms: int
    mode: Literal["lerp", "slerp"]
    type: Literal["frames"] = "frames"


class GifPayload(BaseModel):
    gif_base64: str
    neighbor_ids: List[str]
    frame_count: int
    duration_ms: int
    mode: Literal["lerp", "slerp"]
    type: Literal["gif"] = "gif"


SynthesisResponse = GifPayload | FramesPayload


