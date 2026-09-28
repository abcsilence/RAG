import { useState } from 'react'
import Markdown from 'react-markdown'
import Logo from './Logo'
import Sources from './Sources'

export default function Message({ message }) {
  if (message.role === 'user') {
    return (
      <div className="message user">
        <p>{message.content}</p>
      </div>
    )
  }

  const { content, sources, status, error } = message
  const finished = ['done', 'stopped', 'error'].includes(status)
  return (
    <div className="message assistant">
      <div className="avatar">
        <Logo size={18} />
      </div>
      <div className="message-body">
        {status === 'searching' && <Working text="Searching the Constitution" />}
        {status === 'writing' && !content && <Working text="Writing the answer" />}
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
        {finished && sources.length > 0 && <Sources sources={sources} />}
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
