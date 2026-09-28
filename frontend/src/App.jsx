import { useEffect, useRef, useState } from 'react'
import { streamChat } from './api'
import Composer from './components/Composer'
import Logo from './components/Logo'
import Message from './components/Message'
import Welcome from './components/Welcome'

export default function App() {
  // Messages alternate question / answer: {id, role, content}, and answers also have
  // sources, error and a status: "searching" -> "writing" -> "done" (or "stopped" / "error").
  const [messages, setMessages] = useState([])
  const [busy, setBusy] = useState(false)
  const controller = useRef(null) // lets the Stop button cancel the answer being written
  const lastId = useRef(0)
  const followOutput = useRef(true) // keep scrolling down while the reader is at the bottom

  useEffect(() => {
    const onScroll = () => {
      const distance = document.documentElement.scrollHeight - window.innerHeight - window.scrollY
      followOutput.current = distance < 120
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    if (followOutput.current) window.scrollTo({ top: document.documentElement.scrollHeight })
  }, [messages])

  function updateAnswer(id, change) {
    setMessages((list) => list.map((m) => (m.id === id ? { ...m, ...change(m) } : m)))
  }

  async function send(question) {
    // Earlier questions with finished answers, so the server can understand follow-ups.
    const history = []
    for (let i = 0; i + 1 < messages.length; i += 2) {
      const [asked, answer] = [messages[i], messages[i + 1]]
      if (answer.status === 'done') {
        history.push({ role: 'user', content: asked.content }, { role: 'assistant', content: answer.content })
      }
    }

    const id = (lastId.current += 2)
    setMessages((list) => [
      ...list,
      { id: id - 1, role: 'user', content: question },
      { id, role: 'assistant', content: '', sources: [], status: 'searching', error: null },
    ])
    followOutput.current = true
    setBusy(true)
    const request = new AbortController()
    controller.current = request

    try {
      await streamChat({
        question,
        history,
        signal: request.signal,
        onEvent: (event) => {
          if (event.type === 'sources') updateAnswer(id, () => ({ sources: event.sources, status: 'writing' }))
          if (event.type === 'text') updateAnswer(id, (m) => ({ content: m.content + event.text }))
          if (event.type === 'done') updateAnswer(id, () => ({ status: 'done' }))
          if (event.type === 'error') updateAnswer(id, () => ({ status: 'error', error: event.message }))
        },
      })
      // The reply ended without "done" or "error", so the server stopped half-way.
      updateAnswer(id, (m) =>
        m.status === 'done' || m.status === 'error'
          ? {}
          : { status: 'error', error: 'The answer stopped unexpectedly. Check the terminal running the server.' },
      )
    } catch (error) {
      updateAnswer(id, () =>
        error.name === 'AbortError' ? { status: 'stopped' } : { status: 'error', error: error.message },
      )
    } finally {
      if (controller.current === request) setBusy(false)
    }
  }

  function newChat() {
    controller.current?.abort()
    controller.current = null
    setBusy(false)
    setMessages([])
  }

  return (
    <>
      <header className="header">
        <div className="brand">
          <Logo size={26} />
          <div>
            <div className="brand-title">Constitution of Nepal</div>
            <div className="brand-subtitle">Answers with Article citations</div>
          </div>
        </div>
        {messages.length > 0 && (
          <button type="button" className="ghost-button" onClick={newChat}>
            <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 5v14M5 12h14" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
            </svg>
            New chat
          </button>
        )}
      </header>

      <main className="conversation" aria-busy={busy}>
        {messages.length === 0 ? (
          <Welcome onPick={send} />
        ) : (
          messages.map((message) => <Message key={message.id} message={message} />)
        )}
      </main>

      <footer className="composer-area">
        <Composer busy={busy} onSend={send} onStop={() => controller.current?.abort()} />
        <p className="disclaimer">
          Answers come only from the text of the Constitution and can be wrong. Check the sources. Not legal
          advice.
        </p>
      </footer>
    </>
  )
}
