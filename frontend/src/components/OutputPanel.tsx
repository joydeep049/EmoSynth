// Copyright (C) 2025 Joydeep Tripathy
//
// SPDX-License-Identifier: GPL-3.0-or-later

import { FramePlayer } from './FramePlayer'
import type { SynthesisResponse } from '../types'

interface Props {
  result: SynthesisResponse | null
}

export function OutputPanel({ result }: Props) {
  if (!result) {
    return (
      <div className="card">
        <h2>Output</h2>
        <p>Run synthesis to view results.</p>
      </div>
    )
  }

  return (
    <div className="card">
      <h2>Output ({result.mode.toUpperCase()})</h2>
      {result.caption && <p className="caption">{result.caption}</p>}
      {result.type === 'gif' ? (
        <>
          <img
            className="output-gif"
            src={`data:image/gif;base64,${result.gif_base64}`}
            alt="Generated interpolation"
          />
          <a
            className="primary"
            href={`data:image/gif;base64,${result.gif_base64}`}
            download="emosynth.gif"
          >
            Download GIF
          </a>
        </>
      ) : (
        <FramePlayer frames={result.frames} durationMs={result.duration_ms} />
      )}
      <p>{result.frame_count} frames · neighbors: {result.neighbor_ids.join(', ') || 'n/a'}</p>
    </div>
  )
}

