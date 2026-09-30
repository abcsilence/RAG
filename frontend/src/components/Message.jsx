import { useState } from 'react'
import Markdown from 'react-markdown'
import Logo from './Logo'

export default function Message({ message }) {
  if (message.role === 'user') {
    return (
      <div className="message user">
        <p>{message.content}</p>
      </div>
    )
  }

  const { content, status, error } = message
  return (
    <div className="message assistant">
      <div className="avatar">
        <Logo size={18} />
      </div>
      <div className="message-body">
        {status === 'searching' && <Working text="Searching the Constitution" />}
        {content && (
          <div className={`markdown ${status === 'writing' ? 'writing' : ''}`}>
            <Markdown>{content}</Markdown>
          </div>
        )}
        {status === 'stopped' && <p className="note">Stopped.</p>}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        {status === 'done' && content && <CopyButton text={content} />}
      </div>
    </div>
  )
}

function Working({ text }) {
  return (
    <p className="working">
      <span className="dots" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      {text}…
    </p>
  )
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  async function copy() {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }
  return (
    <button type="button" className="text-button" onClick={copy}>
      {copied ? 'Copied' : 'Copy answer'}
    </button>
  )
}
