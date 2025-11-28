export type Mode = 'lerp' | 'slerp'
export type ReturnType = 'gif' | 'frames'

export interface HealthStatus {
  initialized: boolean
  device: string
  cached_count: number
  model_path: string
  dataset_dir: string
}

export interface RandomEmojiResponse {
  id: string
  thumbnail_base64: string
}

export interface SynthesisRequestPayload {
  id: string
  k: number
  steps_between: number
  mode: Mode
  duration_ms: number
  return_type: ReturnType
}

export interface GifResponse {
  type: 'gif'
  gif_base64: string
  neighbor_ids: string[]
  frame_count: number
  duration_ms: number
  mode: Mode
  caption?: string
}

export interface FramesResponse {
  type: 'frames'
  frames: string[]
  neighbor_ids: string[]
  frame_count: number
  duration_ms: number
  mode: Mode
  caption?: string
}

export type SynthesisResponse = GifResponse | FramesResponse

export interface SynthesisConfig {
  k: number
  stepsBetween: number
  mode: Mode
  durationMs: number
  returnType: ReturnType
  caption: string
}

