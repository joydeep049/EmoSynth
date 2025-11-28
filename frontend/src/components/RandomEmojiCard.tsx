import type { RandomEmojiResponse } from '../types'

interface Props {
  emoji: RandomEmojiResponse | null
  onRefresh: () => void
  disabled?: boolean
}

export function RandomEmojiCard({ emoji, onRefresh, disabled }: Props) {
  return (
    <div className="card">
      <h2>Source Emoji</h2>
      {emoji ? (
        <>
          <img
            src={`data:image/png;base64,${emoji.thumbnail_base64}`}
            alt={`emoji ${emoji.id}`}
            className="emoji-thumb"
          />
          <p>ID: {emoji.id}</p>
        </>
      ) : (
        <p>Loading emoji preview...</p>
      )}
      <button type="button" onClick={onRefresh} disabled={disabled}>
        Random Emoji
      </button>
    </div>
  )
}



