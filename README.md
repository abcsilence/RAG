# Constitution of Nepal Chatbot

Ask questions about the Constitution of Nepal (2015, with the 2016 and 2020 amendments) and get
answers that cite the Articles they come from. It uses retrieval-augmented generation (RAG): an LLM
writes the answer, but only from the parts of the Constitution that match the question.

## How it works

1. **`ingest.py`** reads the PDF and splits it into Articles, clauses and Schedules (596 chunks),
   embeds them with `BAAI/bge-base-en-v1.5` and saves a FAISS index in `data/`.
2. **`rag.py`** finds the chunks that match a question with hybrid search: embeddings (meaning)
   plus BM25 (keywords). A question that names an Article, Part or Schedule ("What does Article 17
   say?") fetches it directly. Follow-up questions are rewritten into full questions first.
3. The chunks go to **`openai/gpt-oss-120b`** on Groq, which answers only from them and cites Articles.
4. **`server.py`** (FastAPI) streams the answer to the React chat in **`frontend/`**.

## Setup

You need Python 3.10+ (built with 3.14), Node.js 20.19+ and a free [Groq API key](https://console.groq.com/keys).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then put your Groq API key in .env
python ingest.py        # builds data/ (takes about a minute)
cd frontend && npm install && npm run build && cd ..
```

## Run

```bash
uvicorn server:app
```

Then open http://localhost:8000.

To work on the UI with live reload, run the API and the Vite dev server in two terminals and open
http://localhost:5173:

```bash
uvicorn server:app --reload
cd frontend && npm run dev
```

You can also chat in the terminal with `python rag.py`.

## Check search quality

```bash
python eval.py
```

It asks 52 questions with known answers and checks that search finds the right Article (no API key
needed). Run it after changing the chunking, the embedding model or `TOP_K` in `config.py`.

## Project structure

```text
config.py      settings: models, chunk size, number of chunks per answer
ingest.py      PDF -> chunks -> embeddings -> data/
rag.py         search, question rewriting, answers, terminal chat
server.py      API for the web chat (the Groq key is read from .env only)
eval.py        search tests
frontend/      React + Vite chat
```

## Limits

- Answers can be wrong: check the Articles each answer cites. This is not legal advice.
- Search works in English. Nepali only works for number lookups like "धारा 17".
- The Constitution calls the provinces "Province No. 1" to "No. 7", so province names are not found.
