// The Articles and Schedules an answer was based on. Click one to read its text.
export default function Sources({ sources }) {
  return (
    <div className="sources">
      <p className="sources-title">Sources</p>
      {sources.map((source) => (
        <details key={source.title} className="source">
          <summary>
            <span>{source.title}</span>
            {source.page && <span className="source-page">page {source.page}</span>}
          </summary>
          <p className="source-text">{source.text}</p>
        </details>
      ))}
    </div>
  )
}
