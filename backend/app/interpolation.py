# Copyright (C) 2025 Joydeep Tripathy
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np


def lerp(z0: np.ndarray, z1: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * z0 + t * z1


def slerp(z0: np.ndarray, z1: np.ndarray, t: float) -> np.ndarray:
    z0_norm = z0 / np.linalg.norm(z0)
    z1_norm = z1 / np.linalg.norm(z1)
    dot = np.clip(np.dot(z0_norm, z1_norm), -1.0, 1.0)
    if dot > 0.9995:
        return lerp(z0, z1, t)
    omega = math.acos(dot)
    sin_omega = math.sin(omega)
    return (
        math.sin((1.0 - t) * omega) / sin_omega * z0
        + math.sin(t * omega) / sin_omega * z1
    )


def interpolate_latents(
    z0: np.ndarray,
    z1: np.ndarray,
    steps_between: int,
    mode: str,
) -> List[np.ndarray]:
    """Generate a list of latent vectors from z0 to z1 inclusive."""
    num_points = max(steps_between, 1) + 2
    ts = np.linspace(0.0, 1.0, num_points)
    interp_fn = slerp if mode == "slerp" else lerp
    return [interp_fn(z0, z1, float(t)) for t in ts]


def limit_frames(latents: Iterable[np.ndarray], max_frames: int) -> List[np.ndarray]:
    latents_list = list(latents)
    return latents_list[:max_frames]



