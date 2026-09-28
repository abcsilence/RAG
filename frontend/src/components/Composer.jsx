import { useEffect, useRef, useState } from 'react'

// The question box. Enter sends, Shift+Enter starts a new line.
export default function Composer({ busy, onSend, onStop }) {
  const [text, setText] = useState('')
  const box = useRef(null)

  useEffect(() => {
    // Grow with the text (CSS sets the maximum height). When empty, go back to one line.
    box.current.style.height = ''
    if (text) box.current.style.height = `${box.current.scrollHeight}px`
  }, [text])

  useEffect(() => {
    if (!busy) box.current.focus()
  }, [busy])

  function submit(event) {
    event.preventDefault()
    const question = text.trim()
    if (!question || busy) return
    onSend(question)
    setText('')
  }

  function onKeyDown(event) {
    // isComposing: don't send while a Nepali (or other) input method is still typing a word.
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) submit(event)
  }

  return (
    <form className="composer" onSubmit={submit}>
      <textarea
        ref={box}
        rows={1}
        value={text}
        maxLength={2000}
        placeholder="Ask about the Constitution…"
        aria-label="Your question"
        onChange={(event) => setText(event.target.value)}
        onKeyDown={onKeyDown}
      />
      {busy ? (
        <button type="button" className="icon-button" onClick={onStop} aria-label="Stop answering">
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <rect width="14" height="14" rx="3" fill="currentColor" />
          </svg>
        </button>
      ) : (
        <button type="submit" className="icon-button" disabled={!text.trim()} aria-label="Send">
          <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
            <path
              d="M12 19V5M5 12l7-7 7 7"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      )}
    </form>
  )
}
