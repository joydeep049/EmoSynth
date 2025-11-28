// Copyright (C) 2025 Joydeep Tripathy
//
// SPDX-License-Identifier: GPL-3.0-or-later

import { useEffect, useMemo, useState } from 'react'

import { fetchHealth, fetchRandomEmoji, synthesize } from './api'
import { ControlPanel } from './components/ControlPanel'
import { OutputPanel } from './components/OutputPanel'
import { RandomEmojiCard } from './components/RandomEmojiCard'
import type {
  HealthStatus,
  RandomEmojiResponse,
  SynthesisConfig,
  SynthesisResponse,
} from './types'

const defaultConfig: SynthesisConfig = {
  k: 3,
  stepsBetween: 8,
  mode: 'lerp',
  durationMs: 80,
  returnType: 'gif',
  caption: '',
}

export function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [randomEmoji, setRandomEmoji] = useState<RandomEmojiResponse | null>(null)
  const [config, setConfig] = useState<SynthesisConfig>(() => {
    const stored = localStorage.getItem('emosynth-config')
    return stored ? { ...defaultConfig, ...JSON.parse(stored) } : defaultConfig
  })
  const [result, setResult] = useState<SynthesisResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    localStorage.setItem('emosynth-config', JSON.stringify(config))
  }, [config])

  useEffect(() => {
    const bootstrap = async () => {
      try {
        const [healthRes, emojiRes] = await Promise.all([fetchHealth(), fetchRandomEmoji()])
        setHealth(healthRes)
        setRandomEmoji(emojiRes)
      } catch (err) {
        console.error(err)
        setError('Failed to contact backend. Make sure the API is running.')
      }
    }
    bootstrap()
  }, [])

  const handleRandom = async () => {
    setError(null)
    try {
      const emoji = await fetchRandomEmoji()
      setRandomEmoji(emoji)
    } catch (err) {
      console.error(err)
      setError('Unable to fetch a random emoji.')
    }
  }

  const handleSynthesize = async () => {
    if (!randomEmoji) {
      setError('Please fetch an emoji first.')
      return
    }
    setIsLoading(true)
    setError(null)
    try {
      const response = await synthesize({
        id: randomEmoji.id,
        k: config.k,
        steps_between: config.stepsBetween,
        mode: config.mode,
        duration_ms: config.durationMs,
        return_type: config.returnType,
      })
      setResult({ ...response, caption: config.caption })
    } catch (err) {
      console.error(err)
      setError('Synthesis failed. Check backend logs for details.')
    } finally {
      setIsLoading(false)
    }
  }

  const neighborSummary = useMemo(() => {
    if (!result || result.neighbor_ids.length === 0) {
      return 'Awaiting synthesis...'
    }
    return `Interpolated with: ${result.neighbor_ids.join(', ')}`
  }, [result])

  return (
    <div className="app-shell">
      <header className="header">
        <div>
          <h1>Emoji Latent Interpolation Explorer</h1>
          <p>Blend cached emojis with VAE-powered interpolation.</p>
        </div>
        <div className={`health ${health?.initialized ? 'ok' : 'warn'}`}>
          {health ? `${health.initialized ? 'Ready' : 'Warming up'} · ${health.device}` : 'Checking health...'}
        </div>
      </header>

      <main className="layout">
        <section className="panel">
          <RandomEmojiCard emoji={randomEmoji} onRefresh={handleRandom} disabled={isLoading} />
          <ControlPanel config={config} onChange={setConfig} onSynthesize={handleSynthesize} busy={isLoading} />
          <p className="neighbor-summary">{neighborSummary}</p>
          {error && <div className="error">{error}</div>}
        </section>

        <section className="panel">
          <OutputPanel result={result} />
        </section>
      </main>
    </div>
  )
}

