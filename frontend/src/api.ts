// Copyright (C) 2025 Joydeep Tripathy
//
// SPDX-License-Identifier: GPL-3.0-or-later

import type { HealthStatus, RandomEmojiResponse, SynthesisRequestPayload, SynthesisResponse } from './types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options?.headers ?? {}),
    },
    ...options,
  })

  if (!response.ok) {
    const message = await response.text()
    throw new Error(message || `Request failed with status ${response.status}`)
  }

  return response.json() as Promise<T>
}

export const fetchHealth = () => request<HealthStatus>('/health')

export const fetchRandomEmoji = () => request<RandomEmojiResponse>('/random-emoji')

export const synthesize = (payload: SynthesisRequestPayload) =>
  request<SynthesisResponse>('/synthesize', {
    method: 'POST',
    body: JSON.stringify(payload),
  })

