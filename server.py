"""API for the React chat in frontend/. The Groq API key is read from the .env file only.

Development (two terminals, the page reloads when you edit the code):
    uvicorn server:app --reload            # API on http://localhost:8000
    cd frontend && npm run dev             # chat on http://localhost:5173

Or build the chat once (cd frontend && npm run build) and run a single server:
    uvicorn server:app                     # chat + API on http://localhost:8000
"""
import json
from typing import Literal

import groq
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from rag import Retriever, ask, explain_error, load_api_key

api_key = load_api_key()
if not api_key:
    raise SystemExit("GROQ_API_KEY is missing. Add it to the .env file (see .env.example).")

client = groq.Groq(api_key=api_key)
retriever = Retriever()  # loads the index and the embedding model once, at startup
app = FastAPI(title="Constitution of Nepal Chatbot")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = []


def group_sources(chunks):
    """One entry per Article or Schedule, with the text of every chunk that was used."""
    grouped = {}
    for chunk in chunks:
        entry = grouped.setdefault(chunk["source"], {"title": chunk["source"], "page": chunk["page"], "texts": []})
        entry["texts"].append(chunk["text"])
    return [{"title": e["title"], "page": e["page"], "text": "\n\n".join(e["texts"])} for e in grouped.values()]


@app.post("/api/chat")
def chat(request: ChatRequest):
    """Streams the answer as JSON lines: {"type": "sources"} first, then {"type": "text"}
    pieces, then {"type": "done"}, or {"type": "error"} if Groq fails."""

    def events():
        try:
            history = [message.model_dump() for message in request.history]
            stream, sources = ask(client, retriever, request.question, history)
            yield json.dumps({"type": "sources", "sources": group_sources(sources)}) + "\n"
            for text in stream:
                yield json.dumps({"type": "text", "text": text}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except groq.APIError as error:
            yield json.dumps({"type": "error", "message": explain_error(error)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


# Serve the built chat page (frontend/dist) from this server too, once it has been built.
DIST = config.ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
