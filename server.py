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


@app.post("/api/chat")
def chat(request: ChatRequest):
    """Streams the answer as JSON lines: {"type": "text"} pieces, then {"type": "done"},
    or {"type": "error"} if Groq fails."""

    def events():
        try:
            history = [message.model_dump() for message in request.history]
            for text in ask(client, retriever, request.question, history):
                yield json.dumps({"type": "text", "text": text}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except groq.APIError as error:
            yield json.dumps({"type": "error", "message": explain_error(error)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


# Serve the built chat page (frontend/dist) from this server too, once it has been built.
DIST = config.ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
