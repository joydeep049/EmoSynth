import { useEffect, useMemo, useState } from 'react'

interface Props {
  frames: string[]
  durationMs: number
}

export function FramePlayer({ frames, durationMs }: Props) {
  const [index, setIndex] = useState(0)

  useEffect(() => {
    if (!frames.length) return
    const timer = setInterval(() => {
      setIndex((prev) => (prev + 1) % frames.length)
    }, durationMs)
    return () => clearInterval(timer)
  }, [frames, durationMs])

  useEffect(() => {
    setIndex(0)
  }, [frames])

  const frameSrc = useMemo(() => (frames[index] ? `data:image/png;base64,${frames[index]}` : null), [frames, index])

  if (!frameSrc) {
    return <div className="output-placeholder">No frames decoded yet.</div>
  }

  return (
    <div className="frame-player">
      <img src={frameSrc} alt={`Frame ${index + 1}`} />
      <input
        type="range"
        min={0}
        max={Math.max(frames.length - 1, 0)}
        value={index}
        onChange={(e) => setIndex(Number(e.target.value))}
      />
      <p>
        Frame {index + 1}/{frames.length}
      </p>
    </div>
  )
}



