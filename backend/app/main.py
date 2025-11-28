# Copyright (C) 2025 Joydeep Tripathy
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import (
    HealthResponse,
    RandomEmojiResponse,
    SynthesisRequest,
    SynthesisResponse,
)
from .state import AppState


settings = get_settings()
state = AppState(settings)

app = FastAPI(title="EmoSynth API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    await state.initialize()


async def require_ready() -> AppState:
    if not state.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model cache is still warming up.",
        )
    return state


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(**state.health())


@app.get("/random-emoji", response_model=RandomEmojiResponse)
async def random_emoji(app_state: AppState = Depends(require_ready)) -> RandomEmojiResponse:
    emoji_id, thumbnail = app_state.random_emoji()
    return RandomEmojiResponse(id=emoji_id, thumbnail_base64=thumbnail)


@app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize(
    payload: SynthesisRequest,
    app_state: AppState = Depends(require_ready),
) -> SynthesisResponse:

    if payload.id not in app_state.cache.latents:
        raise HTTPException(status_code=404, detail=f"Emoji id '{payload.id}' is not cached.")
    return await app_state.synthesize(payload)



