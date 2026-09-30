const SERVER_DOWN = "Can't reach the chatbot server. Start it with: uvicorn server:app --reload"

// Sends a question to the Python API (server.py) and calls onEvent for each line of the reply:
// {type: "text"} pieces, then {type: "done"} or {type: "error"}.
export async function streamChat({ question, history, signal, onEvent }) {
  let response
  try {
    response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history }),
      signal,
    })
  } catch (error) {
    if (error.name === 'AbortError') throw error
    throw new Error(SERVER_DOWN)
  }
  if (!response.ok) {
    throw new Error(response.status >= 500 ? SERVER_DOWN : `The request failed (error ${response.status}).`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() // an unfinished line waits for the next piece
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line))
    }
  }
}
