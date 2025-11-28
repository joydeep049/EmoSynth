import type { Dispatch, SetStateAction } from 'react'

import type { Mode, ReturnType, SynthesisConfig } from '../types'

interface Props {
  config: SynthesisConfig
  onChange: Dispatch<SetStateAction<SynthesisConfig>>
  onSynthesize: () => void
  busy: boolean
}

const kOptions = Array.from({ length: 9 }, (_, i) => i + 2)
const stepsOptions = [2, 4, 6, 8, 10, 12, 16, 24, 32]
const durationOptions = [40, 60, 80, 100, 140, 200]
const modes: Mode[] = ['lerp', 'slerp']
const returnTypes: ReturnType[] = ['gif', 'frames']

export function ControlPanel({ config, onChange, onSynthesize, busy }: Props) {
  const setField = <K extends keyof SynthesisConfig>(key: K, value: SynthesisConfig[K]) => {
    onChange((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div className="card">
      <h2>Control Panel</h2>

      <label>
        Neighbors (k)
        <select
          value={config.k}
          onChange={(e) => setField('k', Number(e.target.value))}
          disabled={busy}
        >
          {kOptions.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
      </label>

      <label>
        Steps between points
        <select
          value={config.stepsBetween}
          onChange={(e) => setField('stepsBetween', Number(e.target.value))}
          disabled={busy}
        >
          {stepsOptions.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
      </label>

      <label>
        Duration per frame (ms)
        <select
          value={config.durationMs}
          onChange={(e) => setField('durationMs', Number(e.target.value))}
          disabled={busy}
        >
          {durationOptions.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
      </label>

      <label>
        Mode
        <div className="inline-options">
          {modes.map((value) => (
            <button
              key={value}
              type="button"
              className={value === config.mode ? 'active' : ''}
              onClick={() => setField('mode', value)}
              disabled={busy}
            >
              {value.toUpperCase()}
            </button>
          ))}
        </div>
      </label>

      <label>
        Return type
        <div className="inline-options">
          {returnTypes.map((value) => (
            <button
              key={value}
              type="button"
              className={value === config.returnType ? 'active' : ''}
              onClick={() => setField('returnType', value)}
              disabled={busy}
            >
              {value.toUpperCase()}
            </button>
          ))}
        </div>
      </label>

      <label>
        Caption
        <input
          type="text"
          placeholder="Describe this interpolation"
          value={config.caption}
          onChange={(e) => setField('caption', e.target.value)}
          disabled={busy}
        />
      </label>

      <button className="primary" type="button" onClick={onSynthesize} disabled={busy}>
        {busy ? 'Synthesizing...' : 'Synthesize'}
      </button>
    </div>
  )
}

